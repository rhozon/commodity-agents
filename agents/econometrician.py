"""Escolhe a familia de modelo, respeitando a escada de recuo.

A escada (`agro.config.ESCADA_MODELOS`) e regra do sistema, nao sugestao do
LLM: se o LLM insistir numa familia ja reprovada -- ou responder algo fora da
escada -- o codigo forca o proximo degrau que ainda nao foi tentado. Sem essa
correcao, um LLM teimoso queimaria as tres tentativas ajustando a mesma
familia repetidas vezes.

As reprovacoes anteriores (motivo escrito pelo Critico) entram no prompt: sem
elas o Econometrista escolhe as cegas na tentativa 2 e 3, e a escada vira
sorteio em vez de recuo informado.
"""
import re

from agents.llm import LLM
from agro.config import ESCADA_MODELOS

ESQUEMA = {"familia": str, "justificativa": str}


class Econometrista:
    def __init__(self, llm: LLM):
        self._llm = llm

    def escolher(self, pergunta: str, tentativa: int, reprovacoes: list[str]) -> str:
        historico = ("\n".join(f"- {r}" for r in reprovacoes)
                     if reprovacoes else "nenhuma tentativa anterior")
        prompt = (
            f"Voce e o agente Econometrista.\nPergunta: {pergunta!r}\n"
            f"Tentativa numero {tentativa}.\nReprovacoes anteriores:\n{historico}\n\n"
            f"Escolha uma familia entre {list(ESCADA_MODELOS)}. "
            "Comece por msgarch quando nao houver motivo contrario: ele captura "
            "mudanca de regime de volatilidade, comum em serie de grao."
        )
        escolha = self._llm.perguntar(prompt, ESQUEMA)["familia"]

        # A escada e regra do sistema, nao sugestao: familia ja reprovada nao repete.
        # Palavra inteira, nao substring -- "garch" e substring de "msgarch",
        # entao `f in r` marcaria garch como reprovada so por causa do msgarch
        # que apareceu na mesma frase de reprovacao.
        reprovadas = {
            f for f in ESCADA_MODELOS
            if any(re.search(rf"\b{re.escape(f)}\b", r) for r in reprovacoes)
        }
        if escolha in reprovadas or escolha not in ESCADA_MODELOS:
            for f in ESCADA_MODELOS:
                if f not in reprovadas:
                    return f
            return ESCADA_MODELOS[-1]
        return escolha
