import os
from pathlib import Path
import pytest
from agro import config
from agro.types import Commodity


def test_commodities_tem_milho_e_soja():
    assert set(config.COMMODITIES) == {"milho", "soja"}
    milho = config.COMMODITIES["milho"]
    assert isinstance(milho, Commodity)
    assert milho.ticker_cbot == "ZC=F"
    assert milho.nome_exibicao == "Milho"


def test_rscript_path_respeita_variavel_de_ambiente(monkeypatch, tmp_path):
    falso = tmp_path / "Rscript.exe"
    falso.write_text("")
    monkeypatch.setenv("AGRO_RSCRIPT", str(falso))
    assert config.rscript_path() == str(falso)


def test_rscript_path_erra_com_mensagem_util(monkeypatch):
    monkeypatch.setenv("AGRO_RSCRIPT", r"C:\nao\existe\Rscript.exe")
    with pytest.raises(FileNotFoundError) as e:
        config.rscript_path()
    assert "AGRO_RSCRIPT" in str(e.value)
