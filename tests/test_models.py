"""Testes para agro.models: ajuste, diagnostico e backtest.

Os testes usam monkeypatch em models.rbridge.chamar_r para nao chamar o R de
verdade, exceto os de backtest com convergencia, que chamam o R via
rbridge (arima e rapido o bastante para rodar em teste), e o de reprovacao
por autocorrelacao residual, que chama o R via rbridge com familia garch
(tambem rapido o bastante, ~2s) porque a violacao so e inequivoca com um
ajuste de verdade -- ver comentario no proprio teste.
"""
import numpy as np
import pandas as pd
import pytest
from agro import models
from agro.types import ModelFit


@pytest.fixture
def serie():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)


# --- fit_model --------------------------------------------------------------


def test_fit_model_devolve_modelfit(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "msgarch", "parametros": {"alpha0_1": 0.01},
        "log_lik": 900.0, "aic": -1800.0, "vol_por_regime": [0.005, 0.03],
        "vol_atual": 0.01})
    fit = models.fit_model(serie, "msgarch")
    assert isinstance(fit, ModelFit)
    assert fit.convergiu and fit.familia == "msgarch"
    assert fit.aic == -1800.0
    assert fit.parametros == {"alpha0_1": 0.01}


def test_fit_nao_convergido_nao_levanta(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": False, "familia": "msgarch", "mensagem": "nao convergiu"})
    fit = models.fit_model(serie, "msgarch")
    assert fit.convergiu is False
    assert "convergiu" in fit.mensagem


def test_fit_model_serie_curta_nao_chama_r(monkeypatch):
    def _explode(*a, **k):
        raise AssertionError("fit_model nao deveria chamar o R com serie curta")
    monkeypatch.setattr(models.rbridge, "chamar_r", _explode)
    curta = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
    fit = models.fit_model(curta, "msgarch")
    assert fit.convergiu is False
    assert "curta" in fit.mensagem


def test_fit_model_ignora_parametros_nao_numericos(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "arima",
        "parametros": {"ar1": 0.5, "ordem": "auto"},
        "log_lik": 100.0, "aic": -200.0})
    fit = models.fit_model(serie, "arima")
    assert fit.parametros == {"ar1": 0.5}


# --- diagnose ----------------------------------------------------------------


def test_diagnose_reprova_fit_que_nao_convergiu(serie):
    fit = ModelFit("msgarch", False, {}, None, None, "nao convergiu")
    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    assert any("convergiu" in m for m in d.motivos)


def test_diagnose_reprova_serie_curta():
    curta = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
    fit = ModelFit("msgarch", True, {"a": 1.0}, 10.0, -20.0)
    d = models.diagnose(fit, curta)
    assert d.aprovado is False


def test_diagnose_reprova_aic_nao_finito(serie):
    fit = ModelFit("arima", True, {"a": 1.0}, 10.0, float("inf"))
    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    assert any("AIC" in m or "aic" in m for m in d.motivos)


def test_diagnose_reprova_parametro_explodiu(serie):
    fit = ModelFit("garch", True, {"alpha1": 1e6}, 10.0, -20.0)
    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    assert any("instavel" in m or "explodiu" in m for m in d.motivos)


def test_diagnose_aprova_fit_valido(serie):
    # Antes desta correcao este ModelFit (sem p-valores de residuo) bastava
    # para aprovar: era exatamente o buraco que este fix fecha -- "aprovado"
    # sem nenhum teste de residuo ter rodado. Agora aprovar exige tambem
    # ljung_box_pvalor/arch_lm_pvalor presentes e acima do limiar; um
    # ModelFit sem eles cai no motivo de ausencia de informacao (ver teste
    # test_diagnose_reprova_por_ausencia_de_testes_de_residuo), entao este
    # fixture passa a fornece-los para continuar sendo o caso "aprova de
    # verdade".
    fit = ModelFit("arima", True, {"ar1": 0.3}, 900.0, -1800.0,
                    ljung_box_pvalor=0.6, arch_lm_pvalor=0.7)
    d = models.diagnose(fit, serie)
    assert d.aprovado is True
    assert d.motivos == []
    assert d.testes["n_retornos"] == float(len(serie) - 1)


def test_diagnose_registra_pvalores_dos_testes_de_residuo(serie):
    fit = ModelFit("arima", True, {"ar1": 0.3}, 900.0, -1800.0,
                    ljung_box_pvalor=0.6123, arch_lm_pvalor=0.789)
    d = models.diagnose(fit, serie)
    assert d.testes["ljung_box_pvalor"] == 0.6123
    assert d.testes["arch_lm_pvalor"] == 0.789


def test_diagnose_reprova_ljung_box_abaixo_do_limiar(serie):
    fit = ModelFit("arima", True, {"ar1": 0.3}, 900.0, -1800.0,
                    ljung_box_pvalor=0.001, arch_lm_pvalor=0.7)
    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    assert any("ljung-box" in m.lower() and "autocorrela" in m.lower() for m in d.motivos)


def test_diagnose_reprova_arch_lm_abaixo_do_limiar(serie):
    fit = ModelFit("garch", True, {"omega": 1e-6}, 900.0, -1800.0,
                    ljung_box_pvalor=0.7, arch_lm_pvalor=0.001)
    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    assert any("arch-lm" in m.lower() and "heterocedasticidade" in m.lower() for m in d.motivos)


def test_diagnose_reprova_por_ausencia_de_testes_de_residuo(serie):
    """Convergiu, AIC finito, parametro normal -- mas o R nao devolveu os
    p-valores de residuo (ex.: versao antiga do script, ou falha isolada no
    calculo). Isso reprova, mas por um motivo *diferente* do de premissa
    violada: aqui nao ha confirmacao de nada, so falta de informacao."""
    fit = ModelFit("arima", True, {"ar1": 0.3}, 900.0, -1800.0)
    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    motivo_ausencia = [m for m in d.motivos if "ausencia de informacao" in m]
    assert motivo_ausencia, d.motivos
    # o motivo de ausencia nao pode se confundir com o de violacao de premissa
    assert not any("rejeita" in m.lower() for m in d.motivos)


def test_diagnose_reprova_ajuste_com_autocorrelacao_residual_forte():
    """Violacao inequivoca e real (nao fabricada): GARCH fixa a media em
    zero por especificacao (nao modela autocorrelacao nenhuma), entao
    alimentado com uma serie AR(1) de phi alto toda a autocorrelacao da
    media sobra nos residuos padronizados. O Ljung-Box tem de flagrar isso
    e diagnose() tem de reprovar citando qual premissa falhou."""
    rng = np.random.default_rng(5)
    n = 500
    ruido = rng.normal(0, 0.01, n)
    ret = np.empty(n)
    ret[0] = ruido[0]
    for i in range(1, n):
        ret[i] = 0.85 * ret[i - 1] + ruido[i]
    precos = 100 * np.exp(np.cumsum(ret))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    serie_ar1 = pd.Series(precos, index=idx)

    fit = models.fit_model(serie_ar1, "garch")
    assert fit.convergiu is True
    assert fit.ljung_box_pvalor is not None

    d = models.diagnose(fit, serie_ar1)
    assert d.aprovado is False
    assert any("ljung-box" in m.lower() and "autocorrela" in m.lower() for m in d.motivos)


def test_serie_degenerada_reprova_com_motivo_e_nao_levanta():
    """Serie constante (item explicito de verificacao do spec): o R converge
    o auto.arima e devolve `aic = "-Inf"`, `log_lik = "Inf"` e
    `ljung_box_pvalor = "NaN"` -- strings, porque jsonlite nao tem literal
    JSON para NaN/Inf. Antes desta correcao, `np.isfinite("-Inf")` levantava
    TypeError exatamente no ramo cujo trabalho e REPROVAR o ajuste."""
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    constante = pd.Series([100.0] * n, index=idx)

    fit = models.fit_model(constante, "arima")
    d = models.diagnose(fit, constante)

    assert d.aprovado is False
    assert d.motivos and all(isinstance(m, str) and m.strip() for m in d.motivos)
    assert any("aic" in m.lower() for m in d.motivos), d.motivos


def test_preco_nao_positivo_erra_em_portugues_antes_de_chamar_o_r(monkeypatch):
    """Sem a guarda, `np.log(0)` vira `-inf`, `json.dumps` escreve o literal
    `-Infinity` (que nao existe em JSON), o jsonlite recusa e o usuario ve um
    RuntimeError do R longe da causa real."""
    def _explode(*a, **k):
        raise AssertionError("nao deveria chegar a chamar o R com preco invalido")
    monkeypatch.setattr(models.rbridge, "chamar_r", _explode)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    com_zero = pd.Series([100.0] * 199 + [0.0], index=idx)

    with pytest.raises(ValueError, match="menor ou igual a zero"):
        models.fit_model(com_zero, "garch")


def test_fit_model_nomeia_parametro_nao_finito_em_vez_de_descartar(monkeypatch, serie):
    """Parametro NaN/Inf nao pode sumir junto com os nao numericos: se
    sumisse, `parametros` ficaria vazio, o teste de magnitude nem rodaria e o
    ajuste degenerado escapava por um segundo caminho."""
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "garch",
        "parametros": {"omega": "NaN", "alpha1": 0.2, "beta1": "Inf"},
        "log_lik": 100.0, "aic": -200.0,
        "ljung_box_pvalor": 0.6, "arch_lm_pvalor": 0.7})
    fit = models.fit_model(serie, "garch")
    assert fit.parametros == {"alpha1": 0.2}
    assert set(fit.parametros_nao_finitos) == {"omega", "beta1"}

    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    assert any("nao finito" in m for m in d.motivos), d.motivos


def test_aic_nao_finito_do_r_vira_none_e_reprova(monkeypatch, serie):
    """`"-Inf"` (o que o jsonlite manda) nao chega cru ate o `if`: `_num` o
    converte na fronteira, e o motivo de reprovacao sai escrito."""
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "arima", "parametros": {"intercept": 0.0},
        "log_lik": "Inf", "aic": "-Inf",
        "ljung_box_pvalor": 0.6, "arch_lm_pvalor": 0.7})
    fit = models.fit_model(serie, "arima")
    assert fit.aic is None and fit.log_lik is None

    d = models.diagnose(fit, serie)
    assert d.aprovado is False
    assert any("AIC" in m for m in d.motivos), d.motivos


# --- backtest ------------------------------------------------------------


def test_backtest_devolve_modelo_e_baseline(serie):
    bt = models.backtest(serie, "arima", horizonte=10)
    assert bt.horizonte == 10
    assert bt.mape >= 0 and bt.rmse >= 0
    assert bt.mape_baseline >= 0 and bt.rmse_baseline >= 0
    assert 0.0 <= bt.cobertura_ic <= 1.0
    assert isinstance(bt.bateu_baseline, bool)


def test_backtest_sem_previsao_do_modelo_empata_com_baseline(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r",
                        lambda *a, **k: {"convergiu": False, "familia": "arima"})
    bt = models.backtest(serie, "arima", horizonte=10)
    assert bt.mape == bt.mape_baseline
    assert bt.rmse == bt.rmse_baseline
    # Apos a correcao, essa situacao deve registrar uma nota sobre a nao-convergencia
    assert "nao convergiu" in bt.nota.lower() or "não convergiu" in bt.nota.lower()


def test_backtest_modelo_volatilidade_com_previsao_zero_empata_com_baseline(monkeypatch, serie):
    """msgarch/garch convergem mas a previsao de retorno e zero por
    construcao (media zero na especificacao) -- isso e o comportamento
    correto, nao um bug, e o backtest deve empatar com o passeio aleatorio
    no ponto previsto (a contribuicao real esta no intervalo)."""
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "msgarch",
        "parametros": {"alpha0_1": 0.01}, "log_lik": 500.0, "aic": -900.0,
        "vol_por_regime": [0.01, 0.02], "vol_atual": 0.015,
        "previsao": [0.0] * 10})
    bt = models.backtest(serie, "msgarch", horizonte=10)
    assert bt.mape == bt.mape_baseline
    assert bt.rmse == bt.rmse_baseline


def test_backtest_serie_curta_levanta_erro(serie):
    curta = serie.iloc[:50]
    with pytest.raises(ValueError):
        models.backtest(curta, "arima", horizonte=10)


# --- Correcao 1: a volatilidade do R precisa chegar ao ModelFit -------------


def test_fit_model_traz_volatilidade_do_r(monkeypatch, serie):
    """O R calcula vol_por_regime e vol_atual; antes desta correcao os dois
    morriam na fachada e nunca chegavam ao relatorio."""
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "msgarch", "parametros": {"alpha0_1": 0.01},
        "log_lik": 900.0, "aic": -1800.0,
        "vol_por_regime": [0.005, 0.03], "vol_atual": 0.012})
    fit = models.fit_model(serie, "msgarch")
    assert fit.vol_por_regime == [0.005, 0.03]
    assert fit.vol_atual == 0.012


def test_fit_model_aceita_vol_por_regime_escalar(monkeypatch, serie):
    """garch/arima podem devolver um unico numero em vez de lista."""
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "garch", "parametros": {"omega": 1e-6},
        "log_lik": 900.0, "aic": -1800.0,
        "vol_por_regime": 0.02, "vol_atual": 0.03})
    fit = models.fit_model(serie, "garch")
    assert fit.vol_por_regime == [0.02]


def test_fit_model_sem_volatilidade_usa_padroes(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "arima", "parametros": {"ar1": 0.5},
        "log_lik": 100.0, "aic": -200.0})
    fit = models.fit_model(serie, "arima")
    assert fit.vol_por_regime == []
    assert fit.vol_atual is None


def test_fit_model_descarta_volatilidade_nao_finita(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "garch", "parametros": {"omega": 1e-6},
        "log_lik": 900.0, "aic": -1800.0,
        "vol_por_regime": [0.01, None], "vol_atual": float("nan")})
    fit = models.fit_model(serie, "garch")
    assert fit.vol_por_regime == [0.01]
    assert fit.vol_atual is None


# --- Correcao 2: a banda do backtest tem de vir do modelo ajustado ----------


def _saida_volatilidade(vol_atual, horizonte=10):
    """Fabrica uma saida de msgarch convergido com a vol_atual pedida."""
    def _chamar(*a, **k):
        return {"convergiu": True, "familia": "msgarch",
                "parametros": {"alpha0_1": 0.01}, "log_lik": 500.0, "aic": -900.0,
                "vol_por_regime": [0.01, 0.02], "vol_atual": vol_atual,
                "previsao": [0.0] * horizonte}
    return _chamar


def test_backtest_banda_usa_vol_atual_do_modelo(monkeypatch, serie):
    """Com a banda vindo do desvio historico bruto, MSGARCH, GARCH e ARIMA
    davam a MESMA cobertura_ic e o modelo nao influenciava nada. Vindo do
    ajuste, a cobertura passa a refletir o modelo mesmo quando a previsao
    pontual empata com a referencia -- e o que torna verdadeira a frase 'a
    contribuicao do modelo de volatilidade e o intervalo'."""
    monkeypatch.setattr(models.rbridge, "chamar_r", _saida_volatilidade(1e-9))
    apertada = models.backtest(serie, "msgarch", horizonte=10)
    monkeypatch.setattr(models.rbridge, "chamar_r", _saida_volatilidade(1.0))
    larga = models.backtest(serie, "msgarch", horizonte=10)

    # mesma previsao pontual nos dois casos: so a banda muda.
    assert apertada.mape == larga.mape
    assert apertada.rmse == larga.rmse
    assert apertada.cobertura_ic == 0.0
    assert larga.cobertura_ic == 1.0


def test_backtest_sem_vol_atual_cai_no_desvio_historico_e_registra(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "arima", "parametros": {"ar1": 0.1},
        "log_lik": 500.0, "aic": -900.0, "previsao": [0.001] * 10})
    bt = models.backtest(serie, "arima", horizonte=10)
    assert "historico" in bt.nota.lower()


def test_backtest_vol_atual_invalida_cai_no_desvio_historico(monkeypatch, serie):
    monkeypatch.setattr(models.rbridge, "chamar_r",
                        _saida_volatilidade(float("nan")))
    bt = models.backtest(serie, "msgarch", horizonte=10)
    assert "historico" in bt.nota.lower()


# --- Correção 3: notas de Backtest para distinguir empate por construção ---
# --- de refit que não convergiu ---


def test_backtest_refit_nao_convergiu_registra_nota_do_ponto(monkeypatch, serie):
    """Quando o refit truncado não converge, a previsão cai na referência.
    Deve haver nota explícita sobre isso, não só silêncio."""
    monkeypatch.setattr(models.rbridge, "chamar_r",
                        lambda *a, **k: {"convergiu": False, "familia": "arima"})
    bt = models.backtest(serie, "arima", horizonte=10)
    assert bt.mape == bt.mape_baseline
    assert bt.rmse == bt.rmse_baseline
    # A nota deve citar que não convergiu E que usou a referência para o ponto previsto
    assert "nao convergiu" in bt.nota.lower() or "não convergiu" in bt.nota.lower()
    assert bt.nota != ""


def test_backtest_previsao_de_zeros_por_construcao_registra(monkeypatch, serie):
    """msgarch/garch convergem, previsão de retorno é zero por especificação
    (média zero), e o backtest empata com a referência no ponto POR CONSTRUÇÃO.
    Isso não é um problema -- é o comportamento correto. A nota deve deixar claro
    que o empate é por construção, não por falha."""
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "msgarch",
        "parametros": {"alpha0_1": 0.01}, "log_lik": 500.0, "aic": -900.0,
        "vol_por_regime": [0.01, 0.02], "vol_atual": 0.015,
        "previsao": [0.0] * 10})
    bt = models.backtest(serie, "msgarch", horizonte=10)
    assert bt.mape == bt.mape_baseline
    # A nota deve citar que é empate por construção
    assert "construcao" in bt.nota.lower() or "construção" in bt.nota.lower()
    assert bt.nota != ""


def test_backtest_convergiu_com_vol_invalida_e_previsao_zeros_registra_ambas(monkeypatch, serie):
    """Caso em que AMBOS os problemas ocorrem: vol_atual inválida (banda cai no
    histórico) E previsão é zeros (ponto cai na referência). As duas notas devem
    aparecer juntas no campo `nota`."""
    monkeypatch.setattr(models.rbridge, "chamar_r", lambda *a, **k: {
        "convergiu": True, "familia": "msgarch",
        "parametros": {"alpha0_1": 0.01}, "log_lik": 500.0, "aic": -900.0,
        "vol_por_regime": [0.01, 0.02], "vol_atual": None,  # vol_atual inválida
        "previsao": [0.0] * 10})  # previsão de zeros
    bt = models.backtest(serie, "msgarch", horizonte=10)
    assert bt.mape == bt.mape_baseline
    # Ambas as notas devem estar presentes
    nota_lower = bt.nota.lower()
    assert "construcao" in nota_lower or "construção" in nota_lower, \
        f"Falta menção a construção: {bt.nota}"
    assert "historico" in nota_lower, \
        f"Falta menção ao desvio histórico: {bt.nota}"
    # E devem ser separadas (viajam juntas como uma lista)
    assert ";" in bt.nota or len(bt.nota.split()) > 10, \
        f"Notas devem coexistir ou ser descritivas: {bt.nota}"
