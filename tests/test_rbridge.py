import math

import numpy as np
import pytest
from agro import rbridge


def _retornos_com_dois_regimes(n=600, seed=7):
    rng = np.random.default_rng(seed)
    calmo = rng.normal(0, 0.005, n // 2)
    agitado = rng.normal(0, 0.03, n - n // 2)
    return np.concatenate([calmo, agitado]).tolist()


def _retornos_garch_agrupados(n=500, seed=11, omega=0.000002, alpha1=0.15, beta1=0.80):
    """Serie sintetica de um GARCH(1,1) de verdade: volatilidade agrupada
    (clusters de retornos grandes seguidos por retornos grandes, e pequenos
    seguidos por pequenos), gerada pela propria equacao de recorrencia do
    modelo -- nao apenas dois blocos de variancia constante como em
    `_retornos_com_dois_regimes`.
    """
    rng = np.random.default_rng(seed)
    ruido = rng.normal(0, 1, n)
    sigma2 = np.empty(n)
    y = np.empty(n)
    sigma2[0] = omega / (1 - alpha1 - beta1)
    y[0] = math.sqrt(sigma2[0]) * ruido[0]
    for i in range(1, n):
        sigma2[i] = omega + alpha1 * y[i - 1] ** 2 + beta1 * sigma2[i - 1]
        y[i] = math.sqrt(sigma2[i]) * ruido[i]
    return y.tolist()


def test_msgarch_recupera_dois_regimes():
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "msgarch",
        "retornos": _retornos_com_dois_regimes(),
    })
    assert saida["convergiu"] is True
    assert saida["familia"] == "msgarch"
    assert saida["log_lik"] is not None
    # com dois regimes plantados, o ajuste tem de achar volatilidades distintas.
    # o limiar (>3x) tem folga sobre a razao real (~6x) mas ainda detecta uma
    # extracao de parametros capenga (limiar antigo de >2x era frouxo demais).
    vols = sorted(saida["vol_por_regime"])
    assert vols[1] > 3 * vols[0]


def test_horizonte_devolve_previsao_de_retornos():
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "arima",
        "retornos": _retornos_com_dois_regimes(),
        "horizonte": 10,
    })
    assert saida["convergiu"] is True
    assert len(saida["previsao"]) == 10
    assert all(math.isfinite(v) for v in saida["previsao"])


def test_erro_do_r_vira_excecao_python():
    with pytest.raises(RuntimeError) as e:
        rbridge.chamar_r("fit_model.R", {"familia": "inexistente", "retornos": [0.1, 0.2]})
    assert "inexistente" in str(e.value)


def test_script_ausente_erra_cedo():
    with pytest.raises(FileNotFoundError):
        rbridge.chamar_r("nao_existe.R", {})


# --- Achado 1: familia "garch" nao tinha nenhum teste ---------------------

def test_garch_converge_em_serie_com_volatilidade_agrupada():
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "garch",
        "retornos": _retornos_garch_agrupados(),
    })
    assert saida["convergiu"] is True
    assert saida["familia"] == "garch"
    assert saida["log_lik"] is not None


def test_garch_previsao_tem_horizonte_e_e_zero():
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "garch",
        "retornos": _retornos_garch_agrupados(),
        "horizonte": 10,
    })
    assert saida["convergiu"] is True
    assert len(saida["previsao"]) == 10
    # GARCH e um modelo de volatilidade (media zero por construcao): a
    # previsao de retorno tem de ser zero em todos os passos, e finita.
    assert all(math.isfinite(v) for v in saida["previsao"])
    assert all(v == 0 for v in saida["previsao"])


def test_garch_vol_por_regime_preenchido_positivo_finito():
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "garch",
        "retornos": _retornos_garch_agrupados(),
    })
    assert saida["convergiu"] is True
    vol = saida["vol_por_regime"]
    assert len(vol) >= 1
    assert math.isfinite(vol[0])
    assert vol[0] > 0


# --- Achado 2: "vol_por_regime" tem que significar a mesma coisa em ---------
# --- todas as familias (estrutural de longo prazo), com "vol_atual" ---------
# --- separado para a volatilidade condicional do ultimo instante. ----------

def test_vol_atual_presente_e_positivo_nas_tres_familias():
    for familia, retornos in (
        ("msgarch", _retornos_com_dois_regimes()),
        ("garch", _retornos_garch_agrupados()),
        ("arima", _retornos_com_dois_regimes()),
    ):
        saida = rbridge.chamar_r("fit_model.R", {"familia": familia, "retornos": retornos})
        assert saida["convergiu"] is True
        assert "vol_atual" in saida, f"vol_atual ausente para familia={familia}"
        assert math.isfinite(saida["vol_atual"])
        assert saida["vol_atual"] > 0


def test_garch_vol_por_regime_e_estrutural_nao_condicional_recente():
    # serie com uma quebra de regime abrupta (calmo -> agitado) faz a
    # volatilidade estrutural de longo prazo (vol_por_regime, que enxerga a
    # serie inteira) e a condicional do ultimo instante (vol_atual, presa no
    # trecho mais recente) divergirem bastante -- nao importa a direcao da
    # diferenca, o que importa e que os dois campos carregam numeros
    # claramente distintos em vez de reciclar o mesmo valor sob dois nomes.
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "garch",
        "retornos": _retornos_com_dois_regimes(),
    })
    assert saida["convergiu"] is True
    estrutural = saida["vol_por_regime"][0]
    atual = saida["vol_atual"]
    assert math.isfinite(estrutural) and estrutural > 0
    assert math.isfinite(atual) and atual > 0
    diferenca_relativa = abs(atual - estrutural) / estrutural
    assert diferenca_relativa > 0.2


# --- Achado 3: timeout do subprocess tinha que virar RuntimeError em pt-br -

# --- Correcao final: diagnosticos de residuo (Ljung-Box e ARCH-LM) --------

def _retornos_ar1_forte(n=500, seed=5, phi=0.85, sigma=0.01):
    """AR(1) de phi alto: autocorrelacao forte e inequivoca na media, algo
    que um modelo de volatilidade pura (garch, media fixada em zero) nunca
    captura -- usada para forcar uma rejeicao real do Ljung-Box."""
    rng = np.random.default_rng(seed)
    ruido = rng.normal(0, sigma, n)
    y = np.empty(n)
    y[0] = ruido[0]
    for i in range(1, n):
        y[i] = phi * y[i - 1] + ruido[i]
    return y.tolist()


def test_pvalores_de_residuo_presentes_nas_tres_familias():
    for familia, retornos in (
        ("msgarch", _retornos_com_dois_regimes()),
        ("garch", _retornos_garch_agrupados()),
        ("arima", _retornos_com_dois_regimes()),
    ):
        saida = rbridge.chamar_r("fit_model.R", {"familia": familia, "retornos": retornos})
        assert saida["convergiu"] is True
        for campo in ("ljung_box_pvalor", "arch_lm_pvalor"):
            assert campo in saida, f"{campo} ausente para familia={familia}"
            assert math.isfinite(saida[campo]), f"{campo} nao finito para familia={familia}"
            assert 0.0 <= saida[campo] <= 1.0, f"{campo} fora de [0,1] para familia={familia}"


def test_garch_em_serie_ar1_forte_reprova_no_ljung_box():
    """GARCH so modela variancia (media fixada em zero na especificacao);
    alimentado com uma serie AR(1) de autocorrelacao forte, a estrutura de
    media inteira sobra nos residuos padronizados -- o Ljung-Box tem de
    flagrar isso com p-valor bem abaixo de qualquer limiar usual."""
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": "garch", "retornos": _retornos_ar1_forte()})
    assert saida["convergiu"] is True
    assert saida["ljung_box_pvalor"] < 0.01


def test_timeout_vira_runtime_error_em_portugues():
    with pytest.raises(RuntimeError) as e:
        rbridge.chamar_r("fit_model.R", {
            "familia": "arima",
            "retornos": _retornos_com_dois_regimes(),
        }, timeout=0.5)
    msg = str(e.value)
    assert "fit_model.R" in msg
    assert "0.5" in msg
