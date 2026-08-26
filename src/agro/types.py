"""Contratos entre as pecas. Objeto tipado, nunca texto solto."""
from dataclasses import dataclass, field
from typing import Literal

Familia = Literal["msgarch", "garch", "arima"]

# Nivel do intervalo de previsao usado no backtest. `models.Z_IC_95 = 1.96` e
# o quantil de 97,5%: as duas caudas somam 5%, entao a banda e de 95%. A
# constante vive aqui (e nao e importada de `models`) porque `models` importa
# `types` -- o caminho inverso seria ciclo. Ela entra em `valores_rotulados()`
# porque `cobertura_ic` guarda a cobertura OBSERVADA (ex.: 0.90) e nunca o
# NIVEL do intervalo: sem isto a trava anti-alucinacao proibiria o Redator de
# escrever "intervalo de 95%", que e um fato deterministico do pipeline.
NIVEL_IC = 0.95


@dataclass(frozen=True)
class Commodity:
    chave: str
    nome_exibicao: str
    ticker_cbot: str          # Yahoo Finance
    cepea_id: str             # identificador do indicador no CEPEA


@dataclass
class SeriesBundle:
    """Series brutas ja alinhadas por data."""
    commodity: str
    inicio: str               # ISO, ex "2015-01-01"
    fim: str
    colunas: list[str]        # ex ["cbot", "cepea", "usdbrl"]
    n_obs: int
    caminho_parquet: str
    fontes_usadas: dict[str, str] = field(default_factory=dict)
    trocas_de_fonte: list[str] = field(default_factory=list)


@dataclass
class ModelFit:
    """Ajuste devolvido pelo R.

    `vol_por_regime` e a volatilidade ESTRUTURAL de longo prazo (um valor por
    regime no msgarch, um valor so nas outras familias); `vol_atual` e a
    volatilidade CONDICIONAL do ultimo instante. Sao grandezas diferentes e
    ficam em campos diferentes de proposito -- mas as duas precisam existir
    aqui, senao morrem na fachada e a trava anti-alucinacao acaba PROIBINDO o
    Redator de citar volatilidade, que e justamente o que o modelo tem a dizer.

    Os campos novos vao depois de `mensagem` para nao quebrar a construcao
    posicional de seis argumentos ja usada nos testes e na propria fachada.

    `parametros_nao_finitos` guarda o NOME dos parametros que o R devolveu
    como NaN ou infinito (o jsonlite serializa os dois como string: "NaN",
    "Inf", "-Inf"). Eles nao entram em `parametros`, que so aceita float
    finito, mas tambem nao podem sumir: parametro nao finito e a assinatura
    de um ajuste degenerado e `models.diagnose` reprova por causa dele. A
    ausencia de um parametro e coisa diferente, e nao aparece nesta lista.

    `ljung_box_pvalor` e `arch_lm_pvalor` sao os diagnosticos de residuo que
    o Critico usa para checar premissa de verdade (ver `models.diagnose`):
    autocorrelacao remanescente e heterocedasticidade nao capturada. Vem do
    R (r/fit_model.R) so quando o ajuste converge; `None` quando o R nao
    devolveu (ajuste nao convergiu, ou R antigo/degenerado) -- essa ausencia
    e distinta de um p-valor baixo, e `diagnose` trata os dois casos com
    motivo diferente.
    """
    familia: Familia
    convergiu: bool
    parametros: dict[str, float]
    log_lik: float | None
    aic: float | None
    mensagem: str = ""
    vol_por_regime: list[float] = field(default_factory=list)
    vol_atual: float | None = None
    ljung_box_pvalor: float | None = None
    arch_lm_pvalor: float | None = None
    parametros_nao_finitos: list[str] = field(default_factory=list)


@dataclass
class CorpoRelatorio:
    """As tres camadas que o Redator escreve, uma em cada campo.

    O relatorio se anuncia "em tres camadas" desde o spec: previsao, drivers
    e implicacao de decisao. Enquanto o esquema do Redator tinha um campo so
    (`corpo`), duas das tres secoes do markdown eram carimbos fixos
    apontando de volta para a primeira -- havia tres titulos e uma camada. O
    Redator ja escrevia tres paragrafos; o esquema e que nao os separava.

    A trava anti-alucinacao roda sobre `texto_completo()`, isto e, sobre as
    tres camadas juntas: cada uma delas e texto de LLM e nenhuma escapa da
    conferencia por estar em outro campo.
    """
    previsao: str
    drivers: str
    implicacao: str

    def texto_completo(self) -> str:
        """As tres camadas concatenadas -- o que a trava precisa varrer."""
        return "\n\n".join([self.previsao, self.drivers, self.implicacao])


@dataclass
class Diagnosis:
    aprovado: bool
    motivos: list[str] = field(default_factory=list)
    testes: dict[str, float] = field(default_factory=dict)


@dataclass
class Backtest:
    """Modelo ajustado E passeio aleatorio, lado a lado.

    MAPE sem referencia nao diz se o modelo presta: em preco de commodity o
    passeio aleatorio e um adversario dificil de bater, e mostrar a comparacao
    e o que separa analise de exibicao de numero.

    `nota` existe porque MAPE igual ao baseline tem duas causas OPOSTAS que
    antes produziam o mesmo silencio: (a) modelo de volatilidade com media
    zero na especificacao, que empata no ponto POR CONSTRUCAO e entrega o
    ganho no intervalo, e (b) refit truncado que NAO CONVERGIU e obrigou o
    backtest a usar a referencia como recurso. Sem esse campo, um ARIMA que
    falhou em silencio ficava indistinguivel de um GARCH que empatou de
    proposito.
    """
    horizonte: int
    mape: float                 # do modelo ajustado
    rmse: float
    cobertura_ic: float         # fracao de vezes que o real caiu no intervalo
    mape_baseline: float        # passeio aleatorio
    rmse_baseline: float
    nota: str = ""              # como ler o resultado; ver docstring

    @property
    def bateu_baseline(self) -> bool:
        return self.mape < self.mape_baseline


@dataclass
class RunResult:
    """Resultado de uma execucao completa do pipeline.

    `historico_reprovacoes` guarda uma entrada por tentativa REPROVADA pelo
    Critico, no formato "familia: motivo; motivo". E o que sustenta o recuo
    de modelo no relatorio: sem ele, quem le so ve "Tentativas: 2" e nao
    fica sabendo nem qual familia foi tentada primeiro nem por que ela caiu
    -- que e exatamente a informacao que o recuo de MSGARCH para GARCH tem a
    dar. Fica vazio quando o primeiro ajuste passou de primeira.
    """
    commodity: str
    pergunta: str
    bundle: SeriesBundle
    fit: ModelFit
    diagnosis: Diagnosis
    backtest: Backtest | None
    tentativas: int
    teto_estourado: bool
    grafico: str = ""
    numeros: dict[str, float] = field(default_factory=dict)
    relatorio_md: str = ""
    historico_reprovacoes: list[str] = field(default_factory=list)

    def valores_rotulados(self) -> list[tuple[str, float]]:
        """Todo numero que o Redator pode citar, COM o rotulo de onde ele vem.

        O rotulo existe para a mensagem de erro da trava anti-alucinacao
        (`guard.verificar_numeros`): uma lista crua de floats nao diz ao
        Redator qual grandeza ele confundiu, e o consumidor da mensagem e um
        laco de nova tentativa que precisa saber o que corrigir.

        `fit.aic` e `fit.log_lik` entram aqui porque citar o AIC e coisa
        natural num relatorio econometrico -- sem eles a trava PROIBIA o
        Redator de mencionar o criterio de informacao do proprio ajuste que o
        relatorio esta descrevendo. Sao `None` quando o ajuste nao convergiu,
        e so entram quando existem.
        """
        vals: list[tuple[str, float]] = [
            (f"numeros.{k}", float(v)) for k, v in self.numeros.items()
        ]
        vals += [(f"fit.parametros.{k}", float(v))
                 for k, v in self.fit.parametros.items()]
        vals += [(f"fit.vol_por_regime[{i + 1}]", float(v))
                 for i, v in enumerate(self.fit.vol_por_regime)]
        if self.fit.vol_atual is not None:
            vals.append(("fit.vol_atual", float(self.fit.vol_atual)))
        if self.fit.log_lik is not None:
            vals.append(("fit.log_lik", float(self.fit.log_lik)))
        if self.fit.aic is not None:
            vals.append(("fit.aic", float(self.fit.aic)))
        vals += [(f"diagnosis.testes.{k}", float(v))
                 for k, v in self.diagnosis.testes.items()]
        if self.backtest:
            b = self.backtest
            vals += [("backtest.horizonte", float(b.horizonte)),
                     ("backtest.mape", float(b.mape)),
                     ("backtest.rmse", float(b.rmse)),
                     ("backtest.cobertura_ic", float(b.cobertura_ic)),
                     ("backtest.mape_baseline", float(b.mape_baseline)),
                     ("backtest.rmse_baseline", float(b.rmse_baseline))]
        vals.append(("bundle.n_obs", float(self.bundle.n_obs)))
        vals.append(("nivel_ic", NIVEL_IC))
        return vals

    def valores_permitidos(self) -> set[float]:
        """Todo numero que o Redator pode citar, sem rotulo."""
        return {valor for _, valor in self.valores_rotulados()}
