import pandas as pd
import pytest
from agents.collector import Coletor
from agents.critic import Critico
from agents.econometrician import Econometrista
from agents.llm import LLMFake
from agents.writer import Redator
from agro import guard
from agro.types import Backtest, Diagnosis, ModelFit, RunResult, SeriesBundle


def test_coletor_decide_janela():
    llm = LLMFake([{"inicio": "2018-01-01", "fim": "2024-12-31",
                    "justificativa": "sete anos cobrem dois ciclos"}])
    d = Coletor(llm).decidir("o que move o preco do milho?", "milho")
    assert d["inicio"] == "2018-01-01"
    assert "milho" in llm.prompts[0]


def test_econometrista_comeca_pelo_msgarch():
    llm = LLMFake([{"familia": "msgarch", "justificativa": "volatilidade muda de regime"}])
    assert Econometrista(llm).escolher("p", tentativa=1, reprovacoes=[]) == "msgarch"


def test_econometrista_recebe_as_reprovacoes_anteriores():
    llm = LLMFake([{"familia": "garch", "justificativa": "recuo"}])
    Econometrista(llm).escolher("p", tentativa=2, reprovacoes=["msgarch nao convergiu"])
    assert "msgarch nao convergiu" in llm.prompts[0]


def test_econometrista_forca_escada_se_llm_repetir_familia_reprovada():
    llm = LLMFake([{"familia": "msgarch", "justificativa": "insisto"}])
    fam = Econometrista(llm).escolher("p", tentativa=2, reprovacoes=["msgarch nao convergiu"])
    assert fam == "garch"


def test_critico_reprova_e_da_motivo():
    serie = pd.Series(range(1, 300), index=pd.date_range("2020-01-01", periods=299, freq="B"),
                      dtype=float)
    fit = ModelFit("msgarch", False, {}, None, None, "nao convergiu")
    d = Critico().julgar(fit, serie)
    assert d.aprovado is False and d.motivos


def test_redator_falha_se_inventar_numero(tmp_path):
    p = tmp_path / "s.parquet"
    pd.DataFrame({"cbot": [1, 2, 3]}).to_parquet(p)
    res = RunResult("milho", "?",
                    SeriesBundle("milho", "2020-01-01", "2020-06-16", ["cbot"], 120, str(p)),
                    ModelFit("msgarch", True, {"a": 0.01}, 900.0, -1800.0),
                    Diagnosis(True, [], {}), Backtest(20, 4.53, 3.21, 0.9, 5.10, 3.80), 1, False,
                    numeros={"preco_atual": 62.75})
    llm = LLMFake([{"previsao": "O preco sobe 37.9% ate dezembro.",
                    "drivers": "d", "implicacao": "i", "confianca": "media"}])
    with pytest.raises(guard.NumeroInventado):
        Redator(llm).escrever(res)


def test_redator_devolve_as_tres_camadas_separadas(tmp_path):
    """O esquema tem um campo por camada: com um campo so, duas das tres
    secoes do relatorio viravam carimbo fixo."""
    p = tmp_path / "s.parquet"
    pd.DataFrame({"cbot": [1, 2, 3]}).to_parquet(p)
    res = RunResult("milho", "?",
                    SeriesBundle("milho", "2020-01-01", "2020-06-16", ["cbot"], 120, str(p)),
                    ModelFit("msgarch", True, {"a": 0.01}, 900.0, -1800.0),
                    Diagnosis(True, [], {}), Backtest(20, 4.53, 3.21, 0.9, 5.10, 3.80), 1, False,
                    numeros={"preco_atual": 62.75})
    llm = LLMFake([{"previsao": "camada um", "drivers": "camada dois",
                    "implicacao": "camada tres", "confianca": "media"}])
    corpo = Redator(llm).escrever(res)
    assert (corpo.previsao, corpo.drivers, corpo.implicacao) == (
        "camada um", "camada dois", "camada tres")
    # A trava vale sobre o texto inteiro, nao so sobre a primeira camada.
    assert "camada tres" in corpo.texto_completo()


def test_trava_vale_para_todas_as_camadas(tmp_path):
    """Numero inventado na TERCEIRA camada tem de derrubar do mesmo jeito:
    separar o esquema em tres campos nao pode abrir uma porta lateral."""
    p = tmp_path / "s.parquet"
    pd.DataFrame({"cbot": [1, 2, 3]}).to_parquet(p)
    res = RunResult("milho", "?",
                    SeriesBundle("milho", "2020-01-01", "2020-06-16", ["cbot"], 120, str(p)),
                    ModelFit("msgarch", True, {"a": 0.01}, 900.0, -1800.0),
                    Diagnosis(True, [], {}), Backtest(20, 4.53, 3.21, 0.9, 5.10, 3.80), 1, False,
                    numeros={"preco_atual": 62.75})
    llm = LLMFake([{"previsao": "sem numero", "drivers": "sem numero",
                    "implicacao": "O preco sobe 37.9% ate dezembro.",
                    "confianca": "media"}])
    with pytest.raises(guard.NumeroInventado):
        Redator(llm).escrever(res)


def test_econometrista_forca_escada_com_maiuscula():
    """Reprovacao em maiusculas (comum em erro vindo do R) tem de forcar a escada."""
    llm = LLMFake([{"familia": "msgarch", "justificativa": "insisto"}])
    # Simular mensagem de erro do R com MSGARCH em maiúsculas
    fam = Econometrista(llm).escolher("p", tentativa=2, reprovacoes=["MSGARCH nao convergiu"])
    assert fam == "garch", "Reprovação em maiúsculas deve forçar o próximo degrau"


def test_econometrista_detecta_violacao_contrato_reprovacoes():
    """Reprovacao sem nome de familia viola o contrato de `escolher`."""
    llm = LLMFake([{"familia": "garch", "justificativa": "recuo"}])
    # Motivo sem nenhum nome de família conhecido — viola o contrato
    with pytest.raises(ValueError, match="contrato"):
        Econometrista(llm).escolher("p", tentativa=2,
                                   reprovacoes=["ajuste ruim mas sem mencionar familia"])
