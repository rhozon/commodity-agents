"""Coleta de series com cache em parquet.

O cache e a espinha dorsal da reprodutibilidade: uma vez baixado e commitado,
o grafico publicado nao muda quando o mercado mexe, e os testes rodam sem rede.
"""
import json
from pathlib import Path

import pandas as pd

from agro import config
from agro.types import SeriesBundle


def _caminho_cache(commodity: str, inicio: str, fim: str) -> Path:
    return config.CACHE_DIR / f"{commodity}_{inicio}_{fim}.parquet"


def _caminho_meta(parquet_path: Path) -> Path:
    """Calcula o caminho do arquivo _meta.json irmao ao parquet."""
    return parquet_path.parent / (parquet_path.stem + "_meta.json")


def _gravar_meta(parquet_path: Path, fontes: dict, trocas: list) -> None:
    """Grava a proveniencia (fontes e trocas) em arquivo JSON irmao ao parquet."""
    meta = {
        "fontes_usadas": fontes,
        "trocas_de_fonte": trocas
    }
    meta_path = _caminho_meta(parquet_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _ler_meta(parquet_path: Path) -> tuple[dict, list]:
    """Le a proveniencia do arquivo JSON irmao ao parquet.

    Se o arquivo nao existir (cache antigo), retorna aviso sobre desconhecimento.
    """
    meta_path = _caminho_meta(parquet_path)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("fontes_usadas", {}), meta.get("trocas_de_fonte", [])
    else:
        # Parquet antigo sem meta: registra que a proveniencia e desconhecida
        return {}, ["Proveniencia deste cache e desconhecida (arquivo de metadados nao encontrado)"]


def _baixar_yahoo(ticker: str, inicio: str, fim: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=inicio, end=fim, progress=False, auto_adjust=True)
    if df.empty:
        raise ConnectionError(f"Yahoo Finance nao devolveu dados para {ticker}")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.rename(ticker)


def _baixar_cepea(cepea_id: str, inicio: str, fim: str) -> pd.Series:
    """Serie do CEPEA. Sem API oficial; o download vive atras desta funcao
    justamente para que a troca de fonte fique isolada num lugar so."""
    raise ConnectionError("coleta CEPEA nao implementada nesta versao")


def _baixar(commodity: str, inicio: str, fim: str) -> tuple[pd.DataFrame, dict, list]:
    c = config.COMMODITIES[commodity]
    colunas: dict[str, pd.Series] = {}
    fontes: dict[str, str] = {}
    trocas: list[str] = []

    colunas["cbot"] = _baixar_yahoo(c.ticker_cbot, inicio, fim)
    fontes["cbot"] = f"Yahoo Finance ({c.ticker_cbot})"

    try:
        colunas["cepea"] = _baixar_cepea(c.cepea_id, inicio, fim)
        fontes["cepea"] = f"CEPEA ({c.cepea_id})"
    except ConnectionError as e:
        trocas.append(f"CEPEA indisponivel ({e}); relatorio segue so com a serie internacional")

    colunas["usdbrl"] = _baixar_yahoo(config.TICKER_CAMBIO, inicio, fim)
    fontes["usdbrl"] = f"Yahoo Finance ({config.TICKER_CAMBIO})"

    df = pd.concat(colunas.values(), axis=1)
    df.columns = list(colunas)
    df = df.dropna()
    df.index.name = "data"
    return df, fontes, trocas


def fetch_series(commodity: str, inicio: str, fim: str, usar_cache: bool = True) -> SeriesBundle:
    """Devolve as series alinhadas da commodity, do cache quando existir."""
    if commodity not in config.COMMODITIES:
        raise KeyError(f"commodity desconhecida: {commodity!r}. "
                       f"Conhecidas: {sorted(config.COMMODITIES)}")

    caminho = _caminho_cache(commodity, inicio, fim)
    if usar_cache and caminho.exists():
        df = pd.read_parquet(caminho)
        fontes, trocas = _ler_meta(caminho)
        return SeriesBundle(commodity, inicio, fim, list(df.columns), len(df),
                            str(caminho), fontes, trocas)

    df, fontes, trocas = _baixar(commodity, inicio, fim)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(caminho)
    _gravar_meta(caminho, fontes, trocas)
    return SeriesBundle(commodity, inicio, fim, list(df.columns), len(df),
                        str(caminho), fontes, trocas)
