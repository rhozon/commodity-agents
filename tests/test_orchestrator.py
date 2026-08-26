import re

import numpy as np
import pandas as pd
import pytest
from agents import orchestrator
from agents.econometrician import Econometrista
from agents.llm import LLMFake
from agro import guard
from agro.types import ModelFit, SeriesBundle


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2018-01-01", periods=500, freq="B")
    df = pd.DataFrame({"cbot": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))),
                       "usdbrl": 5 + rng.normal(0, 0.05, 500)}, index=idx)
    df.index.name = "data"
    p = tmp_path / "milho.parquet"
    df.to_parquet(p)
    bundle = SeriesBundle("milho", "2018-01-01", "2019-12-01", ["cbot", "usdbrl"], 500, str(p))
    monkeypatch.setattr(orchestrator.data, "fetch_series", lambda *a, **k: bundle)
    monkeypatch.setattr(orchestrator.report, "plot_series", lambda b, d: d)
    return bundle


def _llm(familias, corpo="Analise sem numeros novos."):
    respostas = [{"inicio": "2018-01-01", "fim": "2019-12-01", "justificativa": "j"}]
    respostas += [{"familia": f, "justificativa": "j"} for f in familias]
    respostas += [{"corpo": corpo, "confianca": "media"}]
    return LLMFake(respostas)


# Divergencia entre o brief (`.superpowers/sdd/task-9-brief.md`) e o codigo
# atual: `models.diagnose` hoje reprova qualquer ajuste que nao traga
# `ljung_box_pvalor`/`arch_lm_pvalor` (ausencia de teste de residuo conta
# como motivo de reprovacao, nao so p-valor baixo -- ver
# `agro.models.diagnose` e `tests/test_models.py::test_diagnose_reprova_por_
# ausencia_de_testes_de_residuo`). O brief mockava `fit_model` sem esses dois
# campos e esperava aprovacao de primeira; com o codigo atual isso reprovaria
# sempre. Os mocks abaixo incluem os dois p-valores acima do limiar para que
# o Critico realmente aprove -- o codigo manda.
def _fit_aprovado(familia, log_lik=900.0, aic=-1800.0):
    return ModelFit(familia, True, {"a": 0.01}, log_lik, aic,
                    ljung_box_pvalor=0.6, arch_lm_pvalor=0.7)


def test_caminho_feliz_uma_tentativa(ambiente, monkeypatch):
    monkeypatch.setattr(orchestrator.models, "fit_model", lambda s, f: _fit_aprovado(f))
    res = orchestrator.rodar("o que move o preco?", "milho", _llm(["msgarch"]))
    assert res.tentativas == 1 and res.teto_estourado is False
    assert res.diagnosis.aprovado and res.relatorio_md


def test_reprovacao_faz_recuar_de_modelo(ambiente, monkeypatch):
    def fit(serie, familia):
        if familia == "msgarch":
            return ModelFit(familia, False, {}, None, None, "nao convergiu")
        return _fit_aprovado(familia, 800.0, -1600.0)
    monkeypatch.setattr(orchestrator.models, "fit_model", fit)

    res = orchestrator.rodar("p", "milho", _llm(["msgarch", "garch"]))
    assert res.tentativas == 2
    assert res.fit.familia == "garch"
    assert res.teto_estourado is False


def test_teto_estourado_marca_e_nao_levanta(ambiente, monkeypatch):
    monkeypatch.setattr(orchestrator.models, "fit_model",
                        lambda s, f: ModelFit(f, False, {}, None, None, "nao convergiu"))
    res = orchestrator.rodar("p", "milho", _llm(["msgarch", "garch", "arima"]))
    assert res.tentativas == 3
    assert res.teto_estourado is True
    assert "teto de tentativas" in res.relatorio_md.lower()


def test_reprovacoes_respeitam_contrato_da_escada(ambiente, monkeypatch):
    """Prova que a lista `reprovacoes` construida por `rodar` sempre respeita
    o contrato de `Econometrista.escolher` (cada motivo contem o nome de uma
    familia conhecida como palavra inteira) -- se nao respeitasse,
    `escolher` levantaria `ValueError` e este teste falharia com erro, nao
    com uma asserção. O espiao grava cada lista de reprovacoes tal como
    chega em `escolher`, para conferir o conteudo alem de so "nao explodiu".
    """
    def fit(serie, familia):
        if familia in ("msgarch", "garch"):
            return ModelFit(familia, False, {}, None, None, "nao convergiu")
        return _fit_aprovado(familia)
    monkeypatch.setattr(orchestrator.models, "fit_model", fit)

    chamadas: list[list[str]] = []
    original = Econometrista.escolher

    def espiao(self, pergunta, tentativa, reprovacoes):
        chamadas.append(list(reprovacoes))
        return original(self, pergunta, tentativa, reprovacoes)

    monkeypatch.setattr(Econometrista, "escolher", espiao)

    res = orchestrator.rodar("p", "milho", _llm(["msgarch", "garch", "arima"]))

    assert res.tentativas == 3
    assert res.fit.familia == "arima"
    nao_vazias = [r for r in chamadas if r]
    assert len(nao_vazias) == 2, "tentativas 2 e 3 deveriam chegar com reprovacoes anteriores"
    for reprovacoes in nao_vazias:
        for motivo in reprovacoes:
            assert re.search(r"\b(msgarch|garch|arima)\b", motivo, re.IGNORECASE), (
                f"motivo sem nome de familia conhecida: {motivo!r}"
            )


def test_trava_falha_retenta_e_sucede(ambiente, monkeypatch):
    """Decisao sobre a falha da trava anti-alucinacao: quando o Redator
    levanta `NumeroInventado`/`NumeroAmbiguo`, o orquestrador tenta escrever
    de novo (ate `MAX_TENTATIVAS_REDACAO` vezes) em vez de abortar a
    execucao inteira -- o ajuste de modelo esta bom, so o texto saiu errado,
    e uma nova rodada de LLM e barata."""
    monkeypatch.setattr(orchestrator.models, "fit_model", lambda s, f: _fit_aprovado(f))
    respostas = [
        {"inicio": "2018-01-01", "fim": "2019-12-01", "justificativa": "j"},
        {"familia": "msgarch", "justificativa": "j"},
        {"corpo": "O modelo aponta sensibilidade de 314159.265 pontos.", "confianca": "media"},
        {"corpo": "Analise sem numeros novos.", "confianca": "media"},
    ]
    llm = LLMFake(respostas)

    res = orchestrator.rodar("o que move o preco?", "milho", llm)

    assert res.relatorio_md
    assert "314159" not in res.relatorio_md
    assert not llm._respostas, "as duas respostas de corpo deveriam ter sido consumidas"


def test_trava_falha_esgota_tentativas_e_propaga(ambiente, monkeypatch):
    """Se a trava reprovar em TODAS as tentativas de redacao, a excecao sobe
    sem ser capturada -- publicar relatorio com numero alucinado e pior do
    que falhar alto (mesmo principio de `agro.guard`)."""
    monkeypatch.setattr(orchestrator.models, "fit_model", lambda s, f: _fit_aprovado(f))
    respostas = [
        {"inicio": "2018-01-01", "fim": "2019-12-01", "justificativa": "j"},
        {"familia": "msgarch", "justificativa": "j"},
        {"corpo": "Numero inventado: 314159.265.", "confianca": "media"},
        {"corpo": "Outro numero inventado: 271828.182.", "confianca": "media"},
    ]
    llm = LLMFake(respostas)

    with pytest.raises(guard.NumeroInventado):
        orchestrator.rodar("p", "milho", llm)


def test_relatorio_sem_backtest_quando_serie_curta(tmp_path, monkeypatch):
    """`models.backtest` levanta `ValueError` com serie curta demais para o
    horizonte; `rodar` captura e segue com `backtest=None`, e o relatorio
    final nao deve trazer a secao de backtest nem quebrar."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2018-01-01", periods=90, freq="B")
    df = pd.DataFrame({"cbot": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 90))),
                       "usdbrl": 5 + rng.normal(0, 0.05, 90)}, index=idx)
    df.index.name = "data"
    p = tmp_path / "milho.parquet"
    df.to_parquet(p)
    bundle = SeriesBundle("milho", "2018-01-01", "2018-05-11", ["cbot", "usdbrl"], 90, str(p))
    monkeypatch.setattr(orchestrator.data, "fetch_series", lambda *a, **k: bundle)
    monkeypatch.setattr(orchestrator.report, "plot_series", lambda b, d: d)
    # Serie curta demais reprova SEMPRE (models.diagnose reprova por
    # "serie curta demais" independente da familia), entao o teto estoura.
    monkeypatch.setattr(orchestrator.models, "fit_model", lambda s, f: _fit_aprovado(f))

    res = orchestrator.rodar("p", "milho", _llm(["msgarch", "garch", "arima"]))

    assert res.teto_estourado is True
    assert res.backtest is None
    assert res.relatorio_md
    assert "## Backtest" not in res.relatorio_md
