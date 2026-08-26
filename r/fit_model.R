# Ajusta um modelo de volatilidade a uma serie de retornos.
# Contrato: JSON no stdin, JSON no stdout. Roda sozinho no RStudio para depurar
# (basta atribuir `entrada <- list(familia = "msgarch", retornos = c(...))`
# antes da linha `entrada <- fromJSON(...)` e rodar o resto do arquivo).
suppressPackageStartupMessages({
  library(jsonlite)
})

entrada <- fromJSON(file("stdin"))
familia <- entrada$familia
retornos <- as.numeric(entrada$retornos)
horizonte <- if (is.null(entrada$horizonte)) 0L else as.integer(entrada$horizonte)

falhar <- function(msg) {
  cat(msg, file = stderr())
  quit(status = 1)
}

familias_validas <- c("msgarch", "garch", "arima")
if (!familia %in% familias_validas) falhar(sprintf("familia desconhecida: %s", familia))
if (length(retornos) < 100) falhar("serie curta demais: minimo de 100 observacoes")

resultado <- switch(
  familia,
  "msgarch" = {
    suppressPackageStartupMessages(library(MSGARCH))
    spec <- CreateSpec(
      variance.spec = list(model = c("sGARCH", "sGARCH")),
      distribution.spec = list(distribution = c("norm", "norm")),
      switch.spec = list(do.mix = FALSE)
    )
    ajuste <- try(FitML(spec = spec, data = retornos), silent = TRUE)
    if (inherits(ajuste, "try-error")) {
      list(convergiu = FALSE, mensagem = "MSGARCH nao convergiu")
    } else {
      pars <- ajuste$par
      vol <- sapply(1:2, function(k) {
        nomes <- grep(sprintf("_%d$", k), names(pars), value = TRUE)
        a <- pars[grep("alpha1", nomes, value = TRUE)]
        b <- pars[grep("beta", nomes, value = TRUE)]
        o <- pars[grep("alpha0", nomes, value = TRUE)]
        if (length(o) && length(a) && length(b)) sqrt(o / max(1e-8, 1 - a - b)) else NA_real_
      })
      # MSGARCH (como o GARCH abaixo) e um modelo de VOLATILIDADE: a
      # especificacao nao tem termo de media, entao o retorno esperado e
      # zero em qualquer horizonte por construcao. A contribuicao real do
      # modelo e o intervalo (via vol_por_regime), nao o ponto previsto.
      # Devolvemos zeros explicitos em vez de usar predict()$vol so para
      # depois multiplicar por zero -- isso deixaria a intencao ilegivel.
      prev <- if (horizonte > 0) rep(0, horizonte) else numeric(0)
      # vol_por_regime: volatilidade ESTRUTURAL de longo prazo de cada
      # regime (sqrt(alpha0/(1-alpha1-beta))) -- nao muda a cada observacao
      # nova. vol_atual: volatilidade CONDICIONAL do ultimo instante
      # (mistura dos regimes pesada pela probabilidade filtrada), que
      # varia a cada nova observacao. Sao duas grandezas diferentes; ver
      # achado 2 da revisao da Task 3.
      list(convergiu = TRUE, parametros = as.list(pars),
           log_lik = as.numeric(logLik(ajuste)), aic = as.numeric(AIC(ajuste)),
           vol_por_regime = I(as.numeric(vol)),
           vol_atual = as.numeric(tail(as.numeric(Volatility(ajuste)), 1)),
           previsao = prev)
    }
  },
  "garch" = {
    suppressPackageStartupMessages(library(rugarch))
    # include.mean = FALSE: fixamos a media em zero de proposito. Um GARCH
    # simples nao tem poder preditivo sobre o sinal/tamanho do retorno
    # medio; ele modela a volatilidade condicional. Com media zero na
    # especificacao, a previsao de retorno e zero por construcao -- o preco
    # projetado a partir dela vira o ultimo preco (o mesmo passeio aleatorio
    # usado como referencia no backtest). O que o GARCH agrega e o
    # vol_por_regime (a volatilidade estrutural), nao o ponto previsto.
    spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                       mean.model = list(armaOrder = c(0, 0), include.mean = FALSE))
    ajuste <- try(ugarchfit(spec, retornos, solver = "hybrid"), silent = TRUE)
    if (inherits(ajuste, "try-error")) {
      list(convergiu = FALSE, mensagem = "GARCH nao convergiu")
    } else {
      prev <- if (horizonte > 0) rep(0, horizonte) else numeric(0)
      pars <- coef(ajuste)
      # vol_por_regime: volatilidade ESTRUTURAL de longo prazo
      # (sqrt(omega/(1-alpha1-beta1))) -- o analogo direto da formula usada
      # no MSGARCH acima, para que o campo signifique a mesma coisa nas
      # duas familias (achado 2 da revisao da Task 3). vol_atual: a
      # volatilidade CONDICIONAL do ultimo instante (tail(sigma(ajuste),1)),
      # que muda a cada observacao nova -- e o que o campo vol_por_regime
      # trazia antes da correcao.
      vol_estrutural <- sqrt(pars["omega"] / max(1e-8, 1 - pars["alpha1"] - pars["beta1"]))
      list(convergiu = TRUE, parametros = as.list(pars),
           log_lik = as.numeric(likelihood(ajuste)),
           aic = as.numeric(infocriteria(ajuste)[1]),
           vol_por_regime = I(as.numeric(vol_estrutural)),
           vol_atual = as.numeric(tail(as.numeric(sigma(ajuste)), 1)),
           previsao = prev)
    }
  },
  "arima" = {
    suppressPackageStartupMessages(library(forecast))
    # ARIMA e um modelo de MEDIA: aqui a previsao de retorno e informativa
    # de verdade (nao e zero por construcao como em msgarch/garch acima).
    ajuste <- try(auto.arima(retornos), silent = TRUE)
    if (inherits(ajuste, "try-error")) {
      list(convergiu = FALSE, mensagem = "ARIMA nao convergiu")
    } else {
      prev <- if (horizonte > 0) {
        as.numeric(forecast(ajuste, h = horizonte)$mean)
      } else numeric(0)
      # ARIMA nao modela heterocedasticidade: nao ha distincao entre
      # "estrutural de longo prazo" e "condicional do ultimo instante" como
      # em msgarch/garch, entao os dois campos usam a mesma medida (o
      # desvio-padrao dos residuos do ajuste), mantida por consistencia de
      # contrato com as outras familias (achado 2 da revisao da Task 3).
      vol_residuos <- as.numeric(sd(residuals(ajuste)))
      list(convergiu = TRUE, parametros = as.list(coef(ajuste)),
           log_lik = as.numeric(logLik(ajuste)), aic = as.numeric(AIC(ajuste)),
           vol_por_regime = I(vol_residuos), vol_atual = vol_residuos, previsao = prev)
    }
  },
  falhar(sprintf("familia desconhecida: %s", familia))
)

resultado$familia <- familia
cat(toJSON(resultado, auto_unbox = TRUE, digits = 8, null = "null"))
