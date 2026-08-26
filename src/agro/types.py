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
    familia: Familia
    convergiu: bool
    parametros: dict[str, float]
    log_lik: float | None
    aic: float | None
    mensagem: str = ""


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
    """
    horizonte: int
    mape: float                 # do modelo ajustado
    rmse: float
    cobertura_ic: float         # fracao de vezes que o real caiu no intervalo
    mape_baseline: float        # passeio aleatorio
    rmse_baseline: float

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
        vals.update(self.diagnosis.testes.values())
        if self.backtest:
            b = self.backtest
            vals.update({b.mape, b.rmse, b.cobertura_ic, float(b.horizonte),
                         b.mape_baseline, b.rmse_baseline})
        vals.add(float(self.bundle.n_obs))
        return vals
