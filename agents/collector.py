"""Decide quais series a pergunta exige e com que janela."""
from agents.llm import LLM

ESQUEMA = {"inicio": str, "fim": str, "justificativa": str}


class Coletor:
    def __init__(self, llm: LLM):
        self._llm = llm

    def decidir(self, pergunta: str, commodity: str) -> dict:
        prompt = (
            f"Voce e o agente Coletor de um sistema de analise de commodities.\n"
            f"Commodity: {commodity}. Pergunta do usuario: {pergunta!r}\n\n"
            "Escolha a janela historica adequada para responder. Considere que "
            "modelos de volatilidade com mudanca de regime precisam de varios anos "
            "para identificar os regimes. Datas no formato ISO (AAAA-MM-DD)."
        )
        return self._llm.perguntar(prompt, ESQUEMA)
