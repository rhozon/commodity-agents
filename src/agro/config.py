"""Configuracao: commodities, caminhos e localizacao do Rscript."""
import os
from pathlib import Path

from agro.types import Commodity

RAIZ = Path(__file__).resolve().parents[2]
CACHE_DIR = RAIZ / "cache"
EXAMPLES_DIR = RAIZ / "examples"
R_DIR = RAIZ / "r"

# Rscript nao esta no PATH desta maquina.
RSCRIPT_PADRAO = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"

COMMODITIES: dict[str, Commodity] = {
    "milho": Commodity("milho", "Milho", "ZC=F", "milho-esalq"),
    "soja": Commodity("soja", "Soja", "ZS=F", "soja-parana"),
}

TICKER_CAMBIO = "BRL=X"
MAX_TENTATIVAS = 3
ESCADA_MODELOS = ("msgarch", "garch", "arima")


def rscript_path() -> str:
    """Caminho do Rscript. AGRO_RSCRIPT tem prioridade sobre o padrao."""
    caminho = os.environ.get("AGRO_RSCRIPT", RSCRIPT_PADRAO)
    if not Path(caminho).exists():
        raise FileNotFoundError(
            f"Rscript nao encontrado em {caminho!r}. "
            "Defina a variavel de ambiente AGRO_RSCRIPT com o caminho do Rscript.exe."
        )
    return caminho
