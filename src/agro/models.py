"""Fachada sobre o R: ajuste, diagnostico e backtest.

O diagnostico e a peca que o Critico usa para reprovar. Ele e deliberadamente
conservador: na duvida, reprova, porque o custo de um relatorio errado e maior
que o de uma tentativa a mais.

Sobre o contrato de saida do R (r/fit_model.R): `vol_por_regime` e a
volatilidade ESTRUTURAL de longo prazo (mesma semantica nas tres familias);
`vol_atual` e a volatilidade CONDICIONAL mais recente. Sao grandezas
diferentes e esta fachada nao as mistura -- alias, nenhuma das duas faz parte
do contrato de `ModelFit` (que so tem parametros/log_lik/aic), entao aqui elas
simplesmente nao sao extraidas.

Para msgarch e garch, `previsao` vem do R como uma lista de zeros por
construcao: sao modelos de volatilidade com media zero na especificacao, e a
previsao de retorno e zero em qualquer horizonte. Isso e o comportamento
correto, nao a ausencia de previsao -- por isso o backtest abaixo so cai no
ramo "sem previsao" quando o ajuste NAO convergiu ou o R nao devolveu o campo,
nunca por causa do conteudo (zeros) da lista. A consequencia esperada e que um
modelo de volatilidade empata com o passeio aleatorio no ponto previsto; o que
ele contribui de fato e o intervalo, e e por isso que `Backtest` carrega
`mape_baseline`/`rmse_baseline` lado a lado com `mape`/`rmse`.
"""
import numpy as np
import pandas as pd

from agro import rbridge
from agro.types import Backtest, Diagnosis, ModelFit

MIN_OBS = 100


def _retornos(serie: pd.Series) -> np.ndarray:
    return np.diff(np.log(serie.to_numpy(dtype=float)))


def fit_model(serie: pd.Series, familia: str) -> ModelFit:
    """Ajusta a familia pedida chamando o R. Nao convergir nao e excecao."""
    ret = _retornos(serie)
    if len(ret) < MIN_OBS:
        return ModelFit(familia, False, {}, None, None,
                        f"serie curta demais: {len(ret)} retornos, minimo {MIN_OBS}")

    saida = rbridge.chamar_r("fit_model.R", {"familia": familia, "retornos": ret.tolist()})
    if not saida.get("convergiu"):
        return ModelFit(familia, False, {}, None, None,
                        saida.get("mensagem", "nao convergiu"))

    pars = {k: float(v) for k, v in (saida.get("parametros") or {}).items()
            if isinstance(v, (int, float))}
    return ModelFit(familia, True, pars, saida.get("log_lik"), saida.get("aic"))


def diagnose(fit: ModelFit, serie: pd.Series) -> Diagnosis:
    """Aprova ou reprova o ajuste, sempre com motivo escrito."""
    motivos: list[str] = []
    testes: dict[str, float] = {}

    if not fit.convergiu:
        motivos.append(f"o ajuste nao convergiu: {fit.mensagem}")

    ret = _retornos(serie)
    testes["n_retornos"] = float(len(ret))
    if len(ret) < MIN_OBS:
        motivos.append(f"serie curta demais: {len(ret)} retornos, minimo {MIN_OBS}")

    if fit.convergiu and fit.aic is not None:
        testes["aic"] = float(fit.aic)
        if not np.isfinite(fit.aic):
            motivos.append("AIC nao finito, sinal de ajuste degenerado")

    if fit.convergiu and fit.parametros:
        maior = max(abs(v) for v in fit.parametros.values())
        testes["maior_parametro_abs"] = float(maior)
        if maior > 1e4:
            motivos.append("parametro explodiu, ajuste instavel")

    return Diagnosis(aprovado=not motivos, motivos=motivos, testes=testes)


def _metricas(teste: np.ndarray, previsao: np.ndarray) -> tuple[float, float]:
    erro = teste - previsao
    return (float(np.mean(np.abs(erro / teste)) * 100),
            float(np.sqrt(np.mean(erro ** 2))))


def backtest(serie: pd.Series, familia: str, horizonte: int = 20) -> Backtest:
    """Backtest de origem fixa: ajusta ate T-h e projeta h passos.

    Mede o modelo ajustado E o passeio aleatorio, para que o relatorio possa
    dizer se valeu a pena modelar. Modelos de volatilidade (msgarch/garch)
    devolvem previsao de retorno zero por construcao -- isso empata de
    proposito com a referencia no ponto previsto, o ganho deles esta na
    cobertura do intervalo, nao no MAPE/RMSE do ponto.
    """
    valores = serie.to_numpy(dtype=float)
    if len(valores) <= horizonte + MIN_OBS:
        raise ValueError(f"serie curta demais para backtest de {horizonte} passos")

    treino, teste = valores[:-horizonte], valores[-horizonte:]
    ret_treino = np.diff(np.log(treino))
    sigma = float(np.std(ret_treino, ddof=1))

    # Referencia: passeio aleatorio, o ultimo preco repetido.
    base = np.full(horizonte, treino[-1])
    mape_base, rmse_base = _metricas(teste, base)

    # Modelo: retornos previstos pelo R, acumulados sobre o ultimo preco.
    saida = rbridge.chamar_r("fit_model.R", {
        "familia": familia, "retornos": ret_treino.tolist(), "horizonte": horizonte})
    previsao = saida.get("previsao")
    if saida.get("convergiu") and previsao is not None and len(previsao) > 0:
        ret_prev = np.asarray(previsao, dtype=float)
        prev = treino[-1] * np.exp(np.cumsum(ret_prev))
    else:
        prev = base                       # sem previsao do modelo, empata com a referencia
    mape_mod, rmse_mod = _metricas(teste, prev)

    passos = np.arange(1, horizonte + 1)
    banda = 1.96 * sigma * np.sqrt(passos) * treino[-1]
    dentro = np.abs(teste - prev) <= banda

    return Backtest(horizonte, mape_mod, rmse_mod, float(np.mean(dentro)),
                    mape_base, rmse_base)
