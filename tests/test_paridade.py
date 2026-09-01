"""Paridade entre os dois motores de orquestracao.

Sem estes testes, "reimplementei em LangGraph" e uma afirmacao. Com eles, e
uma propriedade verificada: os dois motores recebem a MESMA entrada e
produzem o MESMO `RunResult`, campo a campo.

Por que isso importa mais que o codigo do grafo em si: uma reimplementacao
que se comporta *quase* igual e pior que nao ter reimplementacao nenhuma --
ela parece uma alternativa e e uma armadilha. A paridade e o que autoriza
trocar de motor sem reler o relatorio.

Estes testes NAO tocam a rede nem gastam credito de API: usam serie
sintetica, `LLMFake` e `monkeypatch` em `data.fetch_series`, exatamente como
`tests/test_orchestrator.py`. Exigem R instalado, como o resto da suite.
"""
import numpy as np
import pandas as pd
import pytest
from agents import orchestrator
from agents.llm import LLMFake  # noqa: F401  (mantido para paridade de imports)
from agents_langgraph import graph as motor_grafo
from agro.types import SeriesBundle

# A escada real desce sozinha conforme o R reprova; para a paridade importa
# que os DOIS motores recebam a mesma sequencia de respostas do LLM.
FAMILIAS = ["msgarch", "garch", "arima"]


def _bundle(tmp_path, semente=11, n=500):
    rng = np.random.default_rng(semente)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"cbot": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
         "usdbrl": 5 + rng.normal(0, 0.05, n)}, index=idx)
    df.index.name = "data"
    p = tmp_path / "milho.parquet"
    df.to_parquet(p)
    return SeriesBundle("milho", "2018-01-01", "2019-12-01",
                        ["cbot", "usdbrl"], n, str(p))


class _LLMRoteado:
    """Fake que roteia pelo AGENTE que pergunta, nao por ordem de chamada.

    `agents.llm.LLMFake` consome uma lista na ordem em que as respostas sao
    pedidas, e isso nao serve aqui: quantas vezes o Econometrista e chamado
    depende de quantos degraus da escada o ajuste REAL em R precisa descer,
    o que so se sabe rodando. Com lista fixa, uma aprovacao de primeira faz o
    Redator receber a resposta do Econometrista e o teste falha por motivo
    errado -- foi o que aconteceu na primeira versao deste arquivo.

    Mesma solucao que `run.py` usa em `_LLMDemonstracao`, e pelo mesmo
    motivo. Cada agente se identifica no proprio prompt.
    """

    def __init__(self, familias=FAMILIAS):
        self._familias = list(familias)
        self._n_econ = 0

    def perguntar(self, prompt: str, esquema: dict) -> dict:
        if "agente Coletor" in prompt:
            return {"inicio": "2018-01-01", "fim": "2019-12-01", "justificativa": "j"}
        if "agente Econometrista" in prompt:
            i = min(self._n_econ, len(self._familias) - 1)
            self._n_econ += 1
            return {"familia": self._familias[i], "justificativa": "j"}
        if "agente Redator" in prompt:
            # Sem digito no corpo, de proposito: assim a trava
            # anti-alucinacao nunca reprova por texto do fake, e o teste mede
            # paridade de orquestracao, nao sorte do gerador de prosa.
            return {"previsao": "Analise sem numeros novos.",
                    "drivers": "Drivers sem numeros novos.",
                    "implicacao": "Implicacao sem numeros novos.",
                    "confianca": "media"}
        raise AssertionError(f"prompt de agente desconhecido: {prompt[:80]!r}")


def _llm():
    """Uma instancia nova a cada motor, para os dois partirem do mesmo estado."""
    return _LLMRoteado()


@pytest.fixture
def ambiente(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    # Os dois motores importam `data` e `report` do mesmo modulo `agro`, mas
    # cada um pelo seu namespace; parchear so um deixaria o outro batendo na
    # rede. Por isso o patch e aplicado nos dois.
    for modulo in (orchestrator, motor_grafo):
        monkeypatch.setattr(modulo.data, "fetch_series", lambda *a, **k: bundle)
        monkeypatch.setattr(modulo.report, "plot_series", lambda b, d: d)
    return bundle


def _rodar_ambos(pergunta="o que move o preco?"):
    a = orchestrator.rodar(pergunta, "milho", _llm(), usar_cache=True,
                           destino_grafico="/tmp/x.png")
    b = motor_grafo.rodar(pergunta, "milho", _llm(), usar_cache=True,
                          destino_grafico="/tmp/x.png")
    return a, b


def test_mesma_familia_e_mesmo_numero_de_tentativas(ambiente):
    a, b = _rodar_ambos()
    assert a.fit.familia == b.fit.familia
    assert a.tentativas == b.tentativas
    assert a.teto_estourado == b.teto_estourado


def test_mesmo_historico_de_reprovacoes(ambiente):
    """O recuo de modelo e a parte que o relatorio publica -- se os dois
    motores divergirem aqui, divergem no que o leitor ve."""
    a, b = _rodar_ambos()
    assert a.historico_reprovacoes == b.historico_reprovacoes


def test_mesmo_diagnostico_e_mesmo_backtest(ambiente):
    a, b = _rodar_ambos()
    assert a.diagnosis.aprovado == b.diagnosis.aprovado
    assert a.diagnosis.motivos == b.diagnosis.motivos
    assert (a.backtest is None) == (b.backtest is None)
    if a.backtest is not None:
        assert a.backtest.mape == pytest.approx(b.backtest.mape)
        assert a.backtest.rmse == pytest.approx(b.backtest.rmse)
        assert a.backtest.bateu_baseline == b.backtest.bateu_baseline


def test_mesmo_relatorio_markdown(ambiente):
    """O teste mais forte do conjunto: o artefato publicavel sai identico."""
    a, b = _rodar_ambos()
    assert a.relatorio_md == b.relatorio_md


def test_mesmos_numeros_autorizados(ambiente):
    """A trava anti-alucinacao aprova o mesmo conjunto de numeros nos dois --
    se divergisse, um motor publicaria texto que o outro reprovaria."""
    a, b = _rodar_ambos()
    assert a.valores_permitidos() == b.valores_permitidos()


def test_grafo_expoe_os_nos_esperados():
    """Prende a topologia: um no somindo do grafo e uma etapa do pipeline
    desaparecendo em silencio."""
    g = motor_grafo.construir_grafo(_llm())
    nos = set(g.get_graph().nodes)
    assert {"coletar", "ajustar", "julgar", "consolidar", "redigir"} <= nos
