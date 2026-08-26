"""Baixa as series uma vez e commita o parquet.

Rodar so quando quiser atualizar a janela. O cache commitado e o que faz os
testes rodarem sem rede e o grafico publicado nao mudar sozinho.

Precisa de rede (unica parte deste projeto que precisa). Baixa do Yahoo
Finance as series publicas de milho (ZC=F), soja (ZS=F) e cambio (BRL=X). O
coletor do CEPEA levanta `ConnectionError` de proposito nesta versao
(`agro.data._baixar_cepea`) -- a troca de fonte fica registrada nos metadados
irmaos do parquet (`<commodity>_<inicio>_<fim>_meta.json`), e isso e esperado,
nao e erro: o aviso abaixo, "aviso: CEPEA indisponivel ...", aparece a cada
execucao normal deste script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agro import config, data  # noqa: E402

INICIO, FIM = "2018-01-01", "2025-12-31"

for chave in config.COMMODITIES:
    b = data.fetch_series(chave, INICIO, FIM, usar_cache=False)
    print(f"{chave}: {b.n_obs} observacoes -> {b.caminho_parquet}")
    for troca in b.trocas_de_fonte:
        print(f"  aviso: {troca}")
