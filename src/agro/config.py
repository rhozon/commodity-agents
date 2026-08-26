"""Configuracao: commodities, caminhos e localizacao do Rscript."""
import os
import shutil
from pathlib import Path

from agro.types import Commodity

RAIZ = Path(__file__).resolve().parents[2]
CACHE_DIR = RAIZ / "cache"
EXAMPLES_DIR = RAIZ / "examples"
R_DIR = RAIZ / "r"

# Ultimo recurso, e so no Windows: o instalador do R nao registra o PATH por
# padrao, entao o caminho de instalacao tipico e um palpite util. Em Linux e
# macOS o Rscript quase sempre esta no PATH, e e o PATH que responde primeiro
# (ver `rscript_path`) -- nao esta constante, que carrega versao pinada e
# separador de Windows.
RSCRIPT_PADRAO = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"

COMMODITIES: dict[str, Commodity] = {
    "milho": Commodity("milho", "Milho", "ZC=F", "milho-esalq"),
    "soja": Commodity("soja", "Soja", "ZS=F", "soja-parana"),
}

TICKER_CAMBIO = "BRL=X"
MAX_TENTATIVAS = 3
ESCADA_MODELOS = ("msgarch", "garch", "arima")


def _instrucao_de_erro() -> str:
    """As tres formas de o projeto achar o Rscript, ditas por extenso."""
    return (
        "O projeto procura o Rscript em tres lugares, nesta ordem: "
        "(1) a variavel de ambiente AGRO_RSCRIPT; "
        "(2) 'Rscript' ou 'Rscript.exe' no PATH; "
        f"(3) o caminho padrao de instalacao no Windows ({RSCRIPT_PADRAO}). "
        "Instale o R e deixe-o no PATH, ou defina AGRO_RSCRIPT com o caminho "
        "completo do executavel."
    )


def rscript_path() -> str:
    """Caminho do Rscript, procurado em tres formas, nesta ordem.

    1. `AGRO_RSCRIPT`, quando definida. Um override explicito e AUTORITATIVO:
       se ele apontar para um caminho que nao existe, isso e erro de
       configuracao e o erro sobe. Cair calado no PATH nesse caso esconderia
       o engano de quem definiu a variavel e faria o projeto rodar com um R
       diferente do pedido.
    2. `Rscript`/`Rscript.exe` no PATH, via `shutil.which`. E o caso normal
       de Linux e macOS -- e de Windows quando o instalador do R foi marcado
       para registrar o PATH. Sem este passo, o projeto morria numa maquina
       com R instalado normalmente, exibindo um caminho de Windows com versao
       pinada.
    3. `RSCRIPT_PADRAO`, o caminho de instalacao tipico do R no Windows.
    """
    do_ambiente = os.environ.get("AGRO_RSCRIPT")
    if do_ambiente:
        if Path(do_ambiente).exists():
            return do_ambiente
        raise FileNotFoundError(
            f"Rscript nao encontrado em {do_ambiente!r}, caminho vindo de "
            f"AGRO_RSCRIPT. {_instrucao_de_erro()}"
        )

    for nome in ("Rscript", "Rscript.exe"):
        no_path = shutil.which(nome)
        if no_path:
            return no_path

    if Path(RSCRIPT_PADRAO).exists():
        return RSCRIPT_PADRAO

    raise FileNotFoundError(f"Rscript nao encontrado. {_instrucao_de_erro()}")
