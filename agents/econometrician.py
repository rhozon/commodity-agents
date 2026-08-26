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
        """Escolhe a proxima familia de modelo respeitando a escada de recuo.

        A escada (agro.config.ESCADA_MODELOS) e regra do sistema, nao sugestao
        do LLM: se o LLM insistir numa familia ja reprovada, o codigo forca o
        proximo degrau que ainda nao foi tentado.

        **Contrato de entrada (IMPORTANTE)**:
            Se `reprovacoes` nao estiver vazia, cada string DEVE conter o nome
            completo de uma familia ja reprovada (msgarch/garch/arima), como
            palavra inteira. Exemplos corretos:
            - "msgarch nao convergiu" (contém "msgarch")
            - "GARCH(1,1) falhou em Ljung-Box" (contém "garch" ou "GARCH")
            - "arima parametros explodiram" (contém "arima")

            Se quem chama passar motivos sem mencionar nenhuma familia conhecida,
            isso viola o contrato e levanta ValueError. A escada deixa de ser
            garantida sem esse prefixo, porque nao ha como forcar o proximo
            degrau se a familia anterior ficou invisivel.

        **Comportamento**:
            - Se a escolha do LLM ja foi reprovada (detectado na string de
              reprovacao, case-insensitive), forcamos o proximo degrau nao
              tentado ainda.
            - Se a escolha nao esta na escada, tambem forcamos o proximo degrau.
            - Se nenhum degrau esta disponivel, retornamos o ultimo (fallback).

        Args:
            pergunta: Pergunta/contexto do usuario.
            tentativa: Numero da tentativa (1, 2 ou 3).
            reprovacoes: Lista de motivos de reprovacao de tentativas anteriores.
                         Cada motivo DEVE conter o nome da familia reprovada.

        Returns:
            Nome da familia (msgarch, garch ou arima).

        Raises:
            ValueError: Se `reprovacoes` nao estiver vazia e nenhuma string
                        contiver um nome de familia conhecido (violacao do
                        contrato).
        """
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
        # Case-INSENSITIVE: erros de R podem vir em maiusculas (MSGARCH, GARCH(1,1))
        reprovadas = {
            f for f in ESCADA_MODELOS
            if any(re.search(rf"\b{re.escape(f)}\b", r, re.IGNORECASE) for r in reprovacoes)
        }

        # Valida contrato: se ha reprovacoes, todas DEVEM mencionar uma familia.
        if reprovacoes and not reprovadas:
            raise ValueError(
                f"Violacao do contrato: `reprovacoes` nao estao vazias "
                f"({len(reprovacoes)} motivo(s)), mas nenhum menciona uma familia conhecida. "
                f"Familias conhecidas: {list(ESCADA_MODELOS)}. "
                f"Cada motivo de reprovacao deve conter o nome da familia reprovada."
            )

        if escolha in reprovadas or escolha not in ESCADA_MODELOS:
            for f in ESCADA_MODELOS:
                if f not in reprovadas:
                    return f
            return ESCADA_MODELOS[-1]
        return escolha
