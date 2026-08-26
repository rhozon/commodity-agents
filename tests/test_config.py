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
    msg = str(e.value)
    # A mensagem cita as TRES formas de achar o Rscript, nao so a variavel.
    assert "AGRO_RSCRIPT" in msg
    assert "PATH" in msg
    assert config.RSCRIPT_PADRAO in msg


def test_rscript_path_acha_no_path_sem_variavel(monkeypatch, tmp_path):
    """Sem AGRO_RSCRIPT, o PATH responde antes do caminho padrao de Windows.

    E o caso de qualquer Linux ou macOS com R instalado normalmente: sem esta
    via o projeto morria exibindo um caminho de Windows com versao pinada.
    """
    falso = tmp_path / "Rscript"
    falso.write_text("")
    monkeypatch.delenv("AGRO_RSCRIPT", raising=False)
    monkeypatch.setattr(config.shutil, "which",
                        lambda nome: str(falso) if nome == "Rscript" else None)
    assert config.rscript_path() == str(falso)


def test_rscript_path_erra_quando_nao_ha_r_em_lugar_nenhum(monkeypatch):
    monkeypatch.delenv("AGRO_RSCRIPT", raising=False)
    monkeypatch.setattr(config.shutil, "which", lambda nome: None)
    monkeypatch.setattr(config, "RSCRIPT_PADRAO", r"C:\nao\existe\Rscript.exe")
    with pytest.raises(FileNotFoundError) as e:
        config.rscript_path()
    assert "PATH" in str(e.value) and "AGRO_RSCRIPT" in str(e.value)
