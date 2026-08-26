"""Cliente de LLM com validacao de esquema, e um fake para os testes.

Todo agente conversa com o LLM por aqui. Os testes usam LLMFake e nunca tocam a
rede nem gastam credito de API.
"""
import json
import os
from typing import Any, Protocol

MODELO_PADRAO = "claude-sonnet-5"


class EsquemaInvalido(ValueError):
    """A resposta do LLM nao bate com o esquema pedido."""


def validar(dados: dict, esquema: dict[str, type]) -> dict:
    """Confere presenca e tipo de cada campo. Campos extras sao ignorados."""
    for campo, tipo in esquema.items():
        if campo not in dados:
            raise EsquemaInvalido(f"campo ausente na resposta: {campo!r}")
        if not isinstance(dados[campo], tipo):
            raise EsquemaInvalido(
                f"campo {campo!r} veio como {type(dados[campo]).__name__}, "
                f"esperado {tipo.__name__}")
    return dados


class LLM(Protocol):
    def perguntar(self, prompt: str, esquema: dict[str, type]) -> dict[str, Any]: ...


class LLMFake:
    """Devolve respostas pre-programadas, em ordem. Usado nos testes."""

    def __init__(self, respostas: list[dict]):
        self._respostas = list(respostas)
        self.prompts: list[str] = []

    def perguntar(self, prompt: str, esquema: dict[str, type]) -> dict:
        self.prompts.append(prompt)
        assert self._respostas, "LLMFake sem respostas restantes"
        return validar(self._respostas.pop(0), esquema)


class ClaudeLLM:
    """Cliente real. Uma nova tentativa quando o esquema nao bate; depois falha."""

    def __init__(self, modelo: str = MODELO_PADRAO):
        import anthropic
        self._cliente = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._modelo = modelo

    def perguntar(self, prompt: str, esquema: dict[str, type]) -> dict:
        campos = ", ".join(f"{k} ({v.__name__})" for k, v in esquema.items())
        instrucao = (f"{prompt}\n\nResponda SOMENTE com um objeto JSON contendo "
                     f"exatamente estes campos: {campos}. Sem texto fora do JSON.")
        ultimo_erro = None
        for _ in range(2):
            msg = self._cliente.messages.create(
                model=self._modelo, max_tokens=2000,
                messages=[{"role": "user", "content": instrucao}])
            bruto = msg.content[0].text.strip().removeprefix("```json").removesuffix("```")
            try:
                return validar(json.loads(bruto), esquema)
            except (json.JSONDecodeError, EsquemaInvalido) as e:
                ultimo_erro = e
                instrucao += f"\n\nSua resposta anterior falhou: {e}. Corrija."
        raise EsquemaInvalido(f"o LLM falhou duas vezes: {ultimo_erro}")
