"""Testes para agro.models: ajuste, diagnostico e backtest.

Os testes usam monkeypatch em models.rbridge.chamar_r para nao chamar o R de
verdade, exceto os de backtest com convergencia, que chamam o R via
rbridge (arima e rapido o bastante para rodar em teste).
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
    fit = ModelFit("arima", True, {"ar1": 0.3}, 900.0, -1800.0)
    d = models.diagnose(fit, serie)
    assert d.aprovado is True
    assert d.motivos == []
    assert d.testes["n_retornos"] == float(len(serie) - 1)


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
