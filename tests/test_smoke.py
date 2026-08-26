"""Fumaca de ponta a ponta: cache congelado, LLM fake, zero rede."""
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def test_cache_congelado_existe():
    parquets = list((RAIZ / "cache").glob("*.parquet"))
    assert parquets, "rode scripts/congelar_cache.py e commite o cache"
    assert any("milho" in p.name for p in parquets)
    assert any("soja" in p.name for p in parquets)


@pytest.mark.parametrize("commodity", ["milho", "soja"])
def test_run_py_gera_relatorio(commodity, tmp_path):
    saida = tmp_path / f"{commodity}.md"
    proc = subprocess.run(
        [sys.executable, "run.py", "--commodity", commodity,
         "--pergunta", "o que move o preco?", "--fake-llm", "--saida", str(saida)],
        cwd=RAIZ, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    texto = saida.read_text(encoding="utf-8")
    assert "## Previsao" in texto and "## Drivers" in texto

    # O grafico vai para o lado do relatorio pedido, nao para `examples/`:
    # rodar com `--saida` nao pode sobrescrever o artefato publicado.
    png = saida.with_suffix(".png")
    assert png.exists(), "o grafico deveria ser gravado ao lado do relatorio"
    assert f"({png.name})" in texto
    # E o markdown referencia a imagem pelo nome, sem caminho absoluto.
    assert str(RAIZ) not in texto
    assert str(png.parent) not in texto
