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
