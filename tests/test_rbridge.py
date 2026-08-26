import numpy as np
import pytest
from agro import rbridge


def _retornos_com_dois_regimes(n=600, seed=7):
    rng = np.random.default_rng(seed)
    calmo = rng.normal(0, 0.005, n // 2)
    agitado = rng.normal(0, 0.03, n - n // 2)
    return np.concatenate([calmo, agitado]).tolist()


def test_msgarch_recupera_dois_regimes():
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "msgarch",
        "retornos": _retornos_com_dois_regimes(),
    })
    assert saida["convergiu"] is True
    assert saida["familia"] == "msgarch"
    assert saida["log_lik"] is not None
    # com dois regimes plantados, o ajuste tem de achar volatilidades distintas
    vols = sorted(saida["vol_por_regime"])
    assert vols[1] > 2 * vols[0]


def test_horizonte_devolve_previsao_de_retornos():
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "arima",
        "retornos": _retornos_com_dois_regimes(),
        "horizonte": 10,
    })
    assert saida["convergiu"] is True
    assert len(saida["previsao"]) == 10


def test_erro_do_r_vira_excecao_python():
    with pytest.raises(RuntimeError) as e:
        rbridge.chamar_r("fit_model.R", {"familia": "inexistente", "retornos": [0.1, 0.2]})
    assert "inexistente" in str(e.value)


def test_script_ausente_erra_cedo():
    with pytest.raises(FileNotFoundError):
        rbridge.chamar_r("nao_existe.R", {})
