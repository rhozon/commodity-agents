"""Testes para agro.report: grafico da serie e relatorio em tres camadas.

As fixtures cobrem os campos de `agro.types` que o relatorio precisa saber
mostrar: em ModelFit, vol_por_regime/vol_atual (a contribuicao dos modelos de
volatilidade) e ljung_box_pvalor/arch_lm_pvalor (a prova de que a premissa
foi checada); em Backtest, mape_baseline/rmse_baseline/bateu_baseline (MAPE
sem referencia nao diz se o modelo presta) e `nota` (por que o modelo empatou
com o passeio aleatorio).
"""
from pathlib import Path

import pandas as pd
import pytest

from agro import report
from agro.types import (
    Backtest,
    CorpoRelatorio,
    Diagnosis,
    ModelFit,
    RunResult,
    SeriesBundle,
)


def _corpo(texto: str = "corpo") -> CorpoRelatorio:
    """As tres camadas que o Redator devolve, uma por campo do esquema.

    Cada camada carrega um texto distinto para que os testes consigam provar
    que ela chegou na SUA secao -- com um texto so, um relatorio que
    imprimisse a previsao nas tres secoes passaria despercebido.
    """
    return CorpoRelatorio(previsao=texto,
                          drivers=f"{texto} -- camada de drivers",
                          implicacao=f"{texto} -- camada de implicacao")


@pytest.fixture
def bundle(tmp_path):
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    df = pd.DataFrame({"cbot": range(120), "usdbrl": [5.0] * 120}, index=idx)
    df.index.name = "data"
    p = tmp_path / "milho.parquet"
    df.to_parquet(p)
    return SeriesBundle("milho", "2020-01-01", "2020-06-16", ["cbot", "usdbrl"], 120, str(p))


@pytest.fixture
def fit_arima():
    """Modelo de media (arima): sem vol_por_regime/vol_atual, com p-valores
    de residuo presentes -- caso comum quando o ajuste converge bem."""
    return ModelFit(
        "arima", True, {"ar1": 0.4}, -1800.0, 3610.0,
        ljung_box_pvalor=0.32, arch_lm_pvalor=0.18,
    )


@pytest.fixture
def fit_msgarch():
    """Modelo de volatilidade: contribui vol_por_regime (varios regimes) e
    vol_atual (condicional mais recente)."""
    return ModelFit(
        "msgarch", True, {"alpha0_1": 0.01}, 900.0, -1800.0,
        vol_por_regime=[0.008, 0.021],
        vol_atual=0.014,
        ljung_box_pvalor=0.41,
        arch_lm_pvalor=0.27,
    )


@pytest.fixture
def diagnosis():
    """Diagnosis.testes carrega n_retornos sempre; ljung_box_pvalor e
    arch_lm_pvalor entram quando o ajuste convergiu e o R os devolveu (ver
    agro.models.diagnose) -- e por isso que vem aqui, nao em ModelFit."""
    return Diagnosis(True, [], {
        "n_retornos": 119.0,
        "ljung_box_pvalor": 0.41,
        "arch_lm_pvalor": 0.27,
    })


@pytest.fixture
def backtest_normal():
    """Backtest de um modelo de media que bateu a referencia de fato."""
    return Backtest(20, 4.5, 3.2, 0.9, 5.1, 3.8, nota="")


@pytest.fixture
def backtest_empate_construcao():
    """Empate por construcao: modelo de volatilidade previu zero por
    especificacao. bateu_baseline e False, mas isso NAO e derrota."""
    return Backtest(
        20, 5.1, 3.8, 0.9, 5.1, 3.8,
        nota="o modelo previu zero por construcao (media zero na "
             "especificacao) e empatou com o passeio aleatorio no ponto",
    )


@pytest.fixture
def backtest_nao_convergiu():
    return Backtest(
        20, 5.1, 3.8, 0.9, 5.1, 3.8,
        nota="o refit do backtest nao convergiu e a previsao pontual "
             "caiu na referencia",
    )


@pytest.fixture
def backtest_sem_vol_atual():
    return Backtest(
        20, 4.9, 3.6, 0.85, 5.1, 3.8,
        nota="o modelo nao devolveu vol_atual utilizavel: a banda caiu "
             "no desvio-padrao historico dos retornos de treino",
    )


def _resultado(bundle, fit, diagnosis, backtest, **kw):
    defaults = dict(
        commodity="milho", pergunta="o que move o preco?", bundle=bundle,
        fit=fit, diagnosis=diagnosis, backtest=backtest,
        tentativas=1, teto_estourado=False,
        numeros={"preco_atual": 119.0},
    )
    defaults.update(kw)
    return RunResult(**defaults)


@pytest.fixture
def resultado(bundle, fit_msgarch, diagnosis, backtest_normal):
    return _resultado(bundle, fit_msgarch, diagnosis, backtest_normal)


def test_plot_series_gera_png(resultado, tmp_path):
    destino = tmp_path / "g.png"
    caminho = report.plot_series(resultado.bundle, str(destino))
    assert Path(caminho).exists() and Path(caminho).stat().st_size > 1000


def test_render_report_tem_as_tres_camadas(resultado):
    md = report.render_report(resultado, _corpo("corpo do analista"))
    for cabecalho in ("## Previsao", "## Drivers", "## Implicacao de decisao"):
        assert cabecalho in md
    assert "corpo do analista" in md


def test_cada_camada_cai_na_sua_secao(resultado):
    """Tres camadas de verdade, nao tres titulos: "Drivers" e "Implicacao de
    decisao" eram carimbos fixos que apontavam de volta para a primeira
    secao. Cada campo do `CorpoRelatorio` tem de aparecer sob o seu proprio
    cabecalho, e na ordem."""
    corpo = CorpoRelatorio(previsao="camada um", drivers="camada dois",
                           implicacao="camada tres")
    md = report.render_report(resultado, corpo)

    assert md.index("## Previsao") < md.index("camada um") < md.index("## Drivers")
    assert md.index("## Drivers") < md.index("camada dois") < md.index("## Implicacao de decisao")
    assert md.index("## Implicacao de decisao") < md.index("camada tres")
    # Os carimbos antigos nao podem voltar.
    assert "Ver decomposicao no corpo acima" not in md


def test_render_report_declara_teto_estourado(resultado):
    resultado.teto_estourado = True
    resultado.tentativas = 3
    md = report.render_report(resultado, _corpo())
    assert "teto de tentativas" in md.lower()


def test_render_report_declara_o_recuo_de_modelo(resultado):
    """O recuo sai declarado sempre que houve reprovacao -- nao apenas
    quando o teto estourou."""
    resultado.tentativas = 2
    resultado.historico_reprovacoes = ["msgarch: o ajuste nao convergiu: MSGARCH nao convergiu"]
    md = report.render_report(resultado, _corpo())
    assert "## Recuo de modelo" in md
    assert "msgarch" in md
    assert "nao convergiu" in md


def test_render_report_sem_secao_de_recuo_quando_aprovou_de_primeira(resultado):
    md = report.render_report(resultado, _corpo())
    assert "## Recuo de modelo" not in md


def test_render_report_lista_troca_de_fonte(resultado):
    resultado.bundle.trocas_de_fonte = ["CEPEA indisponivel; seguiu so com CBOT"]
    md = report.render_report(resultado, _corpo())
    assert "CEPEA indisponivel" in md


def test_render_report_mostra_nota_do_backtest(bundle, fit_arima, diagnosis,
                                                backtest_empate_construcao):
    res = _resultado(bundle, fit_arima, diagnosis, backtest_empate_construcao)
    md = report.render_report(res, _corpo())
    assert backtest_empate_construcao.nota in md


def test_render_report_nao_afirma_derrota_em_empate_por_construcao(
        bundle, fit_msgarch, diagnosis, backtest_empate_construcao):
    """bateu_baseline e False no empate por construcao, mas o texto nao pode
    dizer que o passeio aleatorio venceu o modelo -- isso seria uma leitura
    invertida da teoria (media zero e a especificacao correta)."""
    res = _resultado(bundle, fit_msgarch, diagnosis, backtest_empate_construcao)
    md = report.render_report(res, _corpo())
    assert not backtest_empate_construcao.bateu_baseline
    baixo = md.lower()
    assert "nao foi batido" not in baixo
    assert "não foi batido" not in baixo
    assert "modelo perdeu" not in baixo
    assert "perdeu para" not in baixo


def test_render_report_distingue_nota_de_refit_nao_convergido(
        bundle, fit_arima, diagnosis, backtest_nao_convergiu):
    res = _resultado(bundle, fit_arima, diagnosis, backtest_nao_convergiu)
    md = report.render_report(res, _corpo())
    assert "nao convergiu" in md.lower() or "não convergiu" in md.lower()


def test_render_report_distingue_nota_de_banda_sem_vol_atual(
        bundle, fit_arima, diagnosis, backtest_sem_vol_atual):
    res = _resultado(bundle, fit_arima, diagnosis, backtest_sem_vol_atual)
    md = report.render_report(res, _corpo())
    assert "desvio-padrao historico" in md.lower() or "desvio-padrão histórico" in md.lower()


def test_render_report_mostra_volatilidade_do_modelo(bundle, fit_msgarch,
                                                       diagnosis, backtest_normal):
    res = _resultado(bundle, fit_msgarch, diagnosis, backtest_normal)
    md = report.render_report(res, _corpo())
    # vol_por_regime e vol_atual precisam aparecer quando o fit os fornece.
    assert "0.008" in md or "0,008" in md
    assert "0.021" in md or "0,021" in md
    assert "0.014" in md or "0,014" in md


def test_render_report_sem_volatilidade_quando_fit_nao_fornece(
        bundle, fit_arima, diagnosis, backtest_normal):
    """fit_arima nao tem vol_por_regime/vol_atual: o relatorio nao deve
    inventar uma secao de volatilidade vazia ou com placeholder."""
    res = _resultado(bundle, fit_arima, diagnosis, backtest_normal)
    md = report.render_report(res, _corpo())
    assert "vol_atual" not in md  # nome de campo cru nao deve vazar pro texto


def test_render_report_mostra_pvalores_de_residuo(resultado):
    md = report.render_report(resultado, _corpo())
    assert "0.41" in md or "0,41" in md  # ljung_box_pvalor
    assert "0.27" in md or "0,27" in md  # arch_lm_pvalor


def test_cobertura_sai_com_nivel_nominal_n_e_resolucao(bundle, fit_arima, diagnosis):
    """Cobertura sem referencia nao diz nada: "1.00" pode ser calibracao
    perfeita ou banda larga demais. Com o nivel nominal, o n e a resolucao
    (1/n, o menor passo que a medida consegue dar), 1.00 sobre 20 pontos num
    intervalo de 95% se le como o achado de ma calibracao que e."""
    b = Backtest(20, 4.5, 3.2, 1.0, 5.1, 3.8, nota="")
    res = _resultado(bundle, fit_arima, diagnosis, b)
    md = report.render_report(res, _corpo())
    assert "Cobertura do intervalo de 95%: 1.00 (20 pontos, resolucao 0.05)" in md


def test_render_report_funciona_sem_backtest(bundle, fit_arima, diagnosis):
    res = _resultado(bundle, fit_arima, diagnosis, None)
    md = report.render_report(res, _corpo())
    assert "## Previsao" in md
