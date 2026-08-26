"""Contratos entre as pecas. Objeto tipado, nunca texto solto."""
from dataclasses import dataclass, field
from typing import Literal

Familia = Literal["msgarch", "garch", "arima"]


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

    def valores_permitidos(self) -> set[float]:
        """Todo numero que o Redator pode citar."""
        vals: set[float] = set(self.numeros.values())
        vals.update(self.fit.parametros.values())
        vals.update(self.fit.vol_por_regime)
        if self.fit.vol_atual is not None:
            vals.add(float(self.fit.vol_atual))
        vals.update(self.diagnosis.testes.values())
        if self.backtest:
            b = self.backtest
            vals.update({b.mape, b.rmse, b.cobertura_ic, float(b.horizonte),
                         b.mape_baseline, b.rmse_baseline})
        vals.add(float(self.bundle.n_obs))
        return vals
