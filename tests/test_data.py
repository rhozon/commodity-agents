import json
import pandas as pd
import pytest
from agro import data


@pytest.fixture
def cache_falso(tmp_path, monkeypatch):
    monkeypatch.setattr(data.config, "CACHE_DIR", tmp_path)
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    df = pd.DataFrame({"cbot": range(50), "cepea": range(50), "usdbrl": [5.0] * 50}, index=idx)
    df.index.name = "data"
    df.to_parquet(tmp_path / "milho_2020-01-01_2020-03-11.parquet")
    return tmp_path


def test_le_do_cache_sem_rede(cache_falso, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("nao pode tocar a rede quando ha cache")
    monkeypatch.setattr(data, "_baixar", explode)

    b = data.fetch_series("milho", "2020-01-01", "2020-03-11")
    assert b.n_obs == 50
    assert b.colunas == ["cbot", "cepea", "usdbrl"]
    assert b.commodity == "milho"


def test_commodity_desconhecida_erra():
    with pytest.raises(KeyError) as e:
        data.fetch_series("cafe", "2020-01-01", "2020-03-11")
    assert "cafe" in str(e.value)


def test_registra_troca_de_fonte(cache_falso, monkeypatch):
    def cepea_fora_do_ar(*a, **k):
        raise ConnectionError("CEPEA indisponivel")
    monkeypatch.setattr(data, "_baixar_cepea", cepea_fora_do_ar)
    monkeypatch.setattr(data, "_baixar_yahoo",
                        lambda t, i, f: pd.Series([1.0, 2.0], name=t,
                                                  index=pd.to_datetime(["2021-01-04", "2021-01-05"])))

    b = data.fetch_series("milho", "2021-01-04", "2021-01-05", usar_cache=False)
    assert any("CEPEA" in t for t in b.trocas_de_fonte)
    assert "cepea" not in b.colunas


def test_cache_hit_com_meta_restaura_proveniencia(cache_falso, monkeypatch):
    """Cache-hit com _meta.json: as trocas de fonte gravadas na descida sao devolvidas na leitura."""
    # Arrange: cria parquet e meta.json com uma troca de fonte
    idx = pd.date_range("2021-01-04", periods=2, freq="D")
    df = pd.DataFrame({"cbot": [1.0, 2.0], "usdbrl": [5.0, 5.0]}, index=idx)
    df.index.name = "data"
    parquet_path = cache_falso / "milho_2021-01-04_2021-01-05.parquet"
    df.to_parquet(parquet_path)

    meta = {
        "fontes_usadas": {"cbot": "Yahoo Finance (ZWC=F)", "usdbrl": "Yahoo Finance (USDBRL=X)"},
        "trocas_de_fonte": ["CEPEA indisponivel (coleta CEPEA nao implementada nesta versao); relatorio segue so com a serie internacional"]
    }
    meta_path = cache_falso / "milho_2021-01-04_2021-01-05_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # Act: faz fetch_series que le do cache
    def explode(*a, **k):
        raise AssertionError("nao pode tocar a rede quando ha cache com meta")
    monkeypatch.setattr(data, "_baixar", explode)
    b = data.fetch_series("milho", "2021-01-04", "2021-01-05")

    # Assert: as trocas de fonte foram restauradas
    assert b.trocas_de_fonte == meta["trocas_de_fonte"]
    assert b.fontes_usadas == meta["fontes_usadas"]


def test_cache_hit_sem_meta_registra_desconhecimento(cache_falso, monkeypatch):
    """Cache-hit sem _meta.json: trocas_de_fonte contem aviso sobre proveniencia desconhecida."""
    # Arrange: cria apenas parquet, sem meta.json
    idx = pd.date_range("2021-01-04", periods=2, freq="D")
    df = pd.DataFrame({"cbot": [1.0, 2.0], "cepea": [1.0, 2.0], "usdbrl": [5.0, 5.0]}, index=idx)
    df.index.name = "data"
    parquet_path = cache_falso / "milho_2021-01-04_2021-01-05.parquet"
    df.to_parquet(parquet_path)
    # Deliberadamente nao cria meta_path

    # Act: faz fetch_series que le do cache
    def explode(*a, **k):
        raise AssertionError("nao pode tocar a rede quando ha cache")
    monkeypatch.setattr(data, "_baixar", explode)
    b = data.fetch_series("milho", "2021-01-04", "2021-01-05")

    # Assert: ha mensagem avisando que a proveniencia e desconhecida
    assert any("desconhecida" in msg.lower() or "proveniencia" in msg.lower()
               for msg in b.trocas_de_fonte)


def test_ida_volta_com_cepea_falhando(cache_falso, monkeypatch):
    """Ida e volta: baixar com CEPEA falhando grava o meta, leitura seguinte devolve a mesma troca."""
    # Arrange: mock das funcoes de download
    def cepea_fora_do_ar(*a, **k):
        raise ConnectionError("CEPEA indisponivel")
    monkeypatch.setattr(data, "_baixar_cepea", cepea_fora_do_ar)
    monkeypatch.setattr(data, "_baixar_yahoo",
                        lambda t, i, f: pd.Series([1.0, 2.0], name=t,
                                                  index=pd.to_datetime(["2021-01-04", "2021-01-05"])))

    # Act: primeira chamada faz download (sem cache)
    b1 = data.fetch_series("milho", "2021-01-04", "2021-01-05", usar_cache=False)
    trocas_esperadas = b1.trocas_de_fonte

    # Assert: trocas registradas na primeira descida
    assert any("CEPEA" in t for t in trocas_esperadas)

    # Act: segunda chamada le do cache
    def explode(*a, **k):
        raise AssertionError("nao pode tocar a rede na segunda chamada")
    monkeypatch.setattr(data, "_baixar", explode)
    b2 = data.fetch_series("milho", "2021-01-04", "2021-01-05", usar_cache=True)

    # Assert: as mesmas trocas foram restauradas do meta.json
    assert b2.trocas_de_fonte == trocas_esperadas
