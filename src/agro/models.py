"""Fachada sobre o R: ajuste, diagnostico e backtest.

O diagnostico e a peca que o Critico usa para reprovar. Ele e deliberadamente
conservador: na duvida, reprova, porque o custo de um relatorio errado e maior
que o de uma tentativa a mais.

Alem de convergencia/AIC/magnitude de parametro, `diagnose` checa premissa de
residuo de verdade: Ljung-Box (autocorrelacao remanescente) e ARCH-LM
(heterocedasticidade nao capturada), calculados no R (r/fit_model.R) e
carregados em `ModelFit.ljung_box_pvalor`/`arch_lm_pvalor`. Um p-valor abaixo
de `ALPHA_TESTES_RESIDUO` reprova por premissa violada, com o motivo
nomeando qual teste falhou; a AUSENCIA do p-valor (R nao devolveu, por
exemplo porque o ajuste nao convergiu) reprova por motivo diferente e
explicito -- ausencia de informacao nao e a mesma coisa que confirmar que a
premissa foi violada, e misturar os dois enganaria o leitor do relatorio.

Sobre o contrato de saida do R (r/fit_model.R): `vol_por_regime` e a
volatilidade ESTRUTURAL de longo prazo (mesma semantica nas tres familias);
`vol_atual` e a volatilidade CONDICIONAL mais recente. Sao grandezas
diferentes e esta fachada nao as mistura, mas as duas sao extraidas e viajam
em `ModelFit` ate `RunResult.valores_permitidos()`. Sem isso elas morriam
aqui, e a trava anti-alucinacao -- que so deixa citar numero presente nos
resultados -- acabava PROIBINDO o Redator de mencionar volatilidade.

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
from agro.types import Backtest, Diagnosis, ModelFit, Z_IC_95

MIN_OBS = 100

# Nivel de significancia dos testes de residuo (Ljung-Box e ARCH-LM) usados
# em diagnose(). 5% e o nivel convencional para testes de diagnostico de
# residuo em series temporais aplicadas (Tsay, "Analysis of Financial Time
# Series"; Enders, "Applied Econometric Time Series") -- conservador o
# bastante para pegar autocorrelacao/heterocedasticidade relevante sem
# reprovar ajustes bons por ruido amostral (o que 1% deixaria passar com
# mais frequencia, e o que 10% reprovaria com falso positivo demais).
# Constante nomeada para nao virar numero solto dentro de um `if`.
ALPHA_TESTES_RESIDUO = 0.05


def _retornos(serie: pd.Series) -> np.ndarray:
    """Log-retornos da serie de precos.

    O preco tem de ser estritamente positivo. Sem esta guarda, `np.log` de um
    zero ou negativo produz `-inf`/`nan`, `json.dumps` os escreve como os
    literais `-Infinity`/`NaN`, que nao existem em JSON, o jsonlite recusa a
    entrada e o usuario ve um `RuntimeError` obscuro vindo do R -- longe da
    causa, que e um dado ruim na serie.
    """
    valores = serie.to_numpy(dtype=float)
    if not np.all(np.isfinite(valores)):
        raise ValueError(
            "a serie de precos tem valor ausente ou nao finito: o log-retorno "
            "so existe para preco valido"
        )
    if (valores <= 0).any():
        primeiro = int(np.argmax(valores <= 0))
        raise ValueError(
            f"a serie de precos tem valor menor ou igual a zero (posicao "
            f"{primeiro}, valor {valores[primeiro]:g}): o log-retorno exige "
            f"preco estritamente positivo"
        )
    return np.diff(np.log(valores))


def _num(valor) -> float | None:
    """Converte para float finito, ou None. NaN/inf do R nao passam."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    f = float(valor)
    return f if np.isfinite(f) else None


# jsonlite nao tem literal JSON para NaN/Inf: serializa os dois como as
# STRINGS "NaN", "Inf" e "-Inf". Sem reconhece-las, um ajuste degenerado
# chega aqui parecendo "valor nao numerico" e some em silencio -- que e
# exatamente o oposto do que se quer, porque ajuste degenerado tem de
# REPROVAR com motivo escrito.
_NAO_FINITOS_DO_R = frozenset({"nan", "inf", "+inf", "-inf", "infinity",
                               "-infinity", "+infinity"})


def _e_nao_finito(valor) -> bool:
    """`valor` e um numero que nao e finito (NaN/Inf), inclusive na forma de
    string que o jsonlite usa? Distingue ajuste DEGENERADO de campo que o R
    simplesmente nao mandou ou mandou como texto (ex.: `ordem = "auto"`)."""
    if isinstance(valor, str):
        return valor.strip().lower() in _NAO_FINITOS_DO_R
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return False
    return not np.isfinite(float(valor))


def _lista_de_vol(bruto) -> list[float]:
    """Normaliza `vol_por_regime`: o R manda lista no msgarch e pode mandar
    escalar nas demais familias (auto_unbox). Aqui vira sempre lista."""
    if bruto is None:
        return []
    itens = bruto if isinstance(bruto, (list, tuple)) else [bruto]
    return [v for v in (_num(x) for x in itens) if v is not None]


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

    # Parametro nao finito (NaN/Inf) NAO e parametro ausente: e a assinatura
    # de um ajuste degenerado, e sai daqui nomeado em `parametros_nao_finitos`
    # para `diagnose` reprovar por escrito. Antes ele era descartado junto com
    # os valores nao numericos, e o ajuste degenerado escapava porque
    # `if fit.convergiu and fit.parametros:` nem chegava a rodar.
    pars: dict[str, float] = {}
    pars_nao_finitos: list[str] = []
    for chave, bruto in (saida.get("parametros") or {}).items():
        convertido = _num(bruto)
        if convertido is not None:
            pars[chave] = convertido
        elif _e_nao_finito(bruto):
            pars_nao_finitos.append(chave)

    # `aic` e `log_lik` passam por `_num` AQUI, na fronteira com o R, e nao
    # la adiante num `np.isfinite`: o jsonlite manda "NaN"/"Inf" como string,
    # e `np.isfinite("NaN")` levanta TypeError -- justamente no ramo cujo
    # trabalho e reprovar o ajuste degenerado.
    return ModelFit(familia, True, pars,
                    _num(saida.get("log_lik")), _num(saida.get("aic")),
                    parametros_nao_finitos=pars_nao_finitos,
                    vol_por_regime=_lista_de_vol(saida.get("vol_por_regime")),
                    vol_atual=_num(saida.get("vol_atual")),
                    ljung_box_pvalor=_num(saida.get("ljung_box_pvalor")),
                    arch_lm_pvalor=_num(saida.get("arch_lm_pvalor")))


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

    if fit.convergiu:
        # Le o valor CONVERTIDO. `fit.aic` pode chegar aqui como o `float`
        # normal de um ajuste bom, como `None` (o R nao devolveu, ou devolveu
        # NaN/Inf e `fit_model` ja converteu) ou -- quando alguem constroi o
        # `ModelFit` na mao -- como um float nao finito. `_num` cobre os tres.
        aic = _num(fit.aic)
        if aic is not None:
            testes["aic"] = aic
        else:
            motivos.append("AIC ausente ou nao finito (NaN/Inf), "
                           "sinal de ajuste degenerado")

    if fit.convergiu and fit.parametros_nao_finitos:
        motivos.append(
            "parametro(s) nao finito(s) no ajuste: "
            f"{', '.join(fit.parametros_nao_finitos)} -- NaN ou infinito e "
            "ajuste degenerado, nao parametro ausente"
        )

    if fit.convergiu and fit.parametros:
        maior = max(abs(v) for v in fit.parametros.values())
        testes["maior_parametro_abs"] = float(maior)
        if maior > 1e4:
            motivos.append("parametro explodiu, ajuste instavel")

    # Testes de residuo (Ljung-Box e ARCH-LM): so fazem sentido quando o
    # ajuste convergiu -- sem convergencia o R nem calcula residuo, e o
    # motivo de reprovacao ja e o de nao ter convergido, escrito acima. So
    # entram aqui os dois casos que a tarefa pede para distinguir por
    # escrito: p-valor baixo (premissa violada, motivo cita qual) e p-valor
    # ausente (o R nao devolveu, ausencia de informacao -- nao e o mesmo
    # que confirmar que a premissa foi violada).
    if fit.convergiu:
        testes_ausentes: list[str] = []

        if fit.ljung_box_pvalor is not None:
            testes["ljung_box_pvalor"] = float(fit.ljung_box_pvalor)
            if fit.ljung_box_pvalor < ALPHA_TESTES_RESIDUO:
                motivos.append(
                    f"Ljung-Box rejeita ausencia de autocorrelacao nos residuos "
                    f"(p={fit.ljung_box_pvalor:.4f} < {ALPHA_TESTES_RESIDUO}): "
                    f"o modelo nao capturou toda a estrutura temporal da serie"
                )
        else:
            testes_ausentes.append("Ljung-Box")

        if fit.arch_lm_pvalor is not None:
            testes["arch_lm_pvalor"] = float(fit.arch_lm_pvalor)
            if fit.arch_lm_pvalor < ALPHA_TESTES_RESIDUO:
                motivos.append(
                    f"ARCH-LM rejeita homocedasticidade dos residuos "
                    f"(p={fit.arch_lm_pvalor:.4f} < {ALPHA_TESTES_RESIDUO}): "
                    f"ha heterocedasticidade nao capturada pelo modelo"
                )
        else:
            testes_ausentes.append("ARCH-LM")

        if testes_ausentes:
            motivos.append(
                f"o R nao devolveu o(s) teste(s) de residuo "
                f"{', '.join(testes_ausentes)}: ausencia de informacao, "
                f"nao violacao de premissa confirmada"
            )

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

    LIMITACAO DA BANDA -- a largura NAO e a variancia multi-passo do modelo.
    Ela e calculada como `Z_IC_95 * vol_atual * sqrt(h) * P_T`: a
    volatilidade condicional CORRENTE escalada por raiz de h, que e a
    aproximacao de passeio aleatorio na variancia. Num GARCH a variancia
    h-passos-a-frente reverte a media de longo prazo e o intervalo acumulado
    e a SOMA das variancias condicionais previstas, nao `h` copias da
    corrente. A reversao a media, portanto, nao esta capturada aqui, e a
    diferenca e material quando a volatilidade corrente esta longe da
    estrutural (nos exemplos publicados, 0.0121 contra 0.0173). Fechar essa
    lacuna exige `ugarchforecast` (e o equivalente no MSGARCH), o que muda a
    banda de todos os resultados ja publicados -- fica registrado, nao
    implementado.

    LIMITACAO DO DESENHO -- e UMA janela de origem fixa, com `horizonte`
    pontos. Nao ha rolling origin: a cobertura e o MAPE medidos aqui saem de
    um unico corte da serie e ilustram o comportamento do modelo, nao o
    estimam com precisao.
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
    notas: list[str] = []

    # Ramo do ponto previsto: verifica se convergiu E se tem previsao valida
    if saida.get("convergiu") and previsao is not None and len(previsao) > 0:
        ret_prev = np.asarray(previsao, dtype=float)
        # Verifica se a previsao e zeros por construcao (modelos de volatilidade)
        if np.allclose(ret_prev, 0.0):
            notas.append("o modelo previu zero por construcao (media zero na "
                        "especificacao) e empatou com o passeio aleatorio no ponto")
        prev = treino[-1] * np.exp(np.cumsum(ret_prev))
    else:
        # Refit do backtest nao convergiu ou nao devolveu previsao
        if not saida.get("convergiu"):
            notas.append("o refit do backtest nao convergiu e a previsao pontual "
                        "caiu na referencia")
        prev = base                       # sem previsao do modelo, empata com a referencia
    mape_mod, rmse_mod = _metricas(teste, prev)

    # A banda vem da volatilidade do MODELO AJUSTADO. Com o desvio-padrao
    # historico bruto dos retornos de treino ela era identica para msgarch,
    # garch e arima -- nem a cobertura_ic refletia o ajuste, e o modelo
    # econometrico nao influenciava nada do que saia. So com vol_atual a
    # frase "a contribuicao do modelo de volatilidade e o intervalo" passa a
    # ser verdadeira: a previsao pontual empata com a referencia, mas a
    # cobertura nao.
    vol_modelo = _num(saida.get("vol_atual"))
    if vol_modelo is not None and vol_modelo > 0:
        sigma_banda = vol_modelo
    else:
        sigma_banda = sigma
        notas.append("o modelo nao devolveu vol_atual utilizavel: a banda caiu "
                     "no desvio-padrao historico dos retornos de treino")

    passos = np.arange(1, horizonte + 1)
    banda = Z_IC_95 * sigma_banda * np.sqrt(passos) * treino[-1]
    dentro = np.abs(teste - prev) <= banda

    return Backtest(horizonte, mape_mod, rmse_mod, float(np.mean(dentro)),
                    mape_base, rmse_base, nota="; ".join(notas))
