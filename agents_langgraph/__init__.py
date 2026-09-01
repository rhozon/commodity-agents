"""Orquestracao alternativa da mesma camada de agentes, sobre LangGraph.

Existe para responder uma pergunta concreta: o que um framework de grafos
compra, e o que cobra, num sistema agentico que ja funciona sem ele?

`agents/orchestrator.py` e `agents_langgraph/graph.py` resolvem o MESMO
problema, com os MESMOS agentes e o MESMO nucleo deterministico em `agro/`.
So a orquestracao muda: um laco `for` com `break` de um lado, um grafo de
estado com arestas condicionais do outro. `tests/test_paridade.py` prende as
duas ao mesmo resultado.
"""
from agents_langgraph.graph import construir_grafo, rodar

__all__ = ["rodar", "construir_grafo"]
