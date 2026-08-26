"""Fronteira Python-R: subprocess com JSON, sem rpy2.

rpy2 e a fonte numero um de dor de instalacao em projeto R+Python. Com
subprocess o contrato entre as linguagens e um arquivo de texto, e o script R
roda sozinho no RStudio quando for preciso depurar o modelo.
"""
import json
import subprocess

from agro import config


def chamar_r(script: str, payload: dict, timeout: int = 300) -> dict:
    """Roda r/<script> passando JSON na entrada e lendo JSON na saida."""
    caminho = config.R_DIR / script
    if not caminho.exists():
        raise FileNotFoundError(f"script R nao encontrado: {caminho}")

    try:
        proc = subprocess.run(
            [config.rscript_path(), "--vanilla", str(caminho)],
            input=json.dumps(payload),
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"R estourou o tempo limite rodando {script} (apos {timeout}s)"
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(f"R falhou ({proc.returncode}): {proc.stderr.strip()}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"R devolveu saida invalida.\nstdout: {proc.stdout[:500]}\nstderr: {proc.stderr[:500]}"
        ) from e
