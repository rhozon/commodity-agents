"""Trava anti-alucinacao: o Redator nao inventa numero.

Esta e a peca mais importante do projeto. O corpo do relatorio (secao
"Previsao") e escrito por um LLM; tudo o mais no pipeline e determinístico.
Um numero que o LLM cita sem que ele exista nos resultados calculados pelo
nucleo (`R`/`models.py`) nao e um erro de estilo -- e uma alucinacao, e num
artefato que serve de prova publica de competencia em IA isso derruba a
credibilidade de quem assina.

`verificar_numeros` varre todo numero presente no texto do Redator e confere
cada um contra `RunResult.valores_permitidos()` (ver `types.py` -- nao
alterado aqui). Numero sem origem levanta `NumeroInventado` e interrompe a
execucao: e melhor falhar alto do que publicar um numero fabricado.

ESCOPO -- isto valida o CORPO escrito pelo LLM, nao o markdown final.
`render_report` (`report.py`, tambem nao alterado aqui) monta a moldura do
relatorio ao redor do corpo, e essa moldura imprime numeros deterministicos
que NAO estao em `valores_permitidos()` -- o numero de tentativas
(`res.tentativas`), por exemplo, vem direto do controlador do pipeline, nao
do LLM. Rodar esta trava sobre o documento montado por `render_report`
derrubaria a execucao por um numero correto, so que fora do dicionario que
a trava conhece. Quem for compor esta trava com o restante do pipeline
(Tasks 8 e 9) deve chamar `verificar_numeros` ANTES de `render_report`,
com o `corpo_md` que sairia do Redator -- nunca com a string que
`render_report` devolve.

Formatos de numero aceitos: brasileiro (decimal com virgula, "4,53") e
ingles (decimal com ponto, "4.53"). Nao aceitamos separador de milhar
("1.234,56" ou "1,234.56"): o mesmo caractere (ponto ou virgula) e separador
decimal num formato e separador de milhar no outro, e tentar desambiguar os
dois teria uma taxa de erro maior do que o beneficio -- nenhum numero deste
dominio (MAPE, RMSE, precos, parametros do ajuste, p-valores, contagem de
observacoes) chega a milhares com separador no texto do Redator.
"""
import re

from agro.types import RunResult

# Um sinal opcional, digitos, e no maximo UM separador decimal (virgula ou
# ponto) seguido de mais digitos. Sem grupo de milhar: ver docstring do
# modulo sobre por que isso e deliberado, nao um descuido.
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")

# Percentuais redondos que funcionam como ancora retorica em portugues --
# "nada mudou" (0%), "a metade" (50%), "tudo" (100%) -- e que aparecem em
# prosa comum sem vir de nenhum numero calculado pelo nucleo. A isencao so
# vale quando o proprio texto marca o numero como percentual (o digito e
# seguido de "%" na frase); sem essa condicao, um preco ou parametro real
# que por coincidencia vale 0, 50 ou 100 ficaria imune a trava, o que abriria
# um buraco desnecessario.
_PERCENTUAIS_RETORICOS = {0.0, 50.0, 100.0}

# Piso absoluto de tolerancia. Existe porque tolerancia PURAMENTE relativa
# degenera perto de zero: um parametro de ajuste como alpha0_1 = 0.0134 tem
# 1% relativo igual a 0.000134, um limiar mais apertado do que o
# arredondamento de 2 a 4 casas decimais usado normalmente no relatorio --
# isso faria uma citacao legitima como "0.013" ser rejeitada como
# alucinacao. 0.005 e metade da menor casa decimal usada de forma
# consistente nos numeros deste dominio (MAPE e RMSE em 2 casas, precos em
# 2 casas), entao cobre arredondamento legitimo sem abrir espaco para
# qualquer valor pequeno e desconexo passar (ver teste
# `test_valor_pequeno_perto_de_zero_...`).
_TOLERANCIA_ABSOLUTA_MINIMA = 0.005


class NumeroInventado(RuntimeError):
    """O texto cita um numero que nao existe nos resultados do nucleo."""


def _para_float(token: str) -> float | None:
    """Converte um token numerico (BR ou EN) para float.

    Como `_NUM` casa no maximo um separador, bastar trocar virgula por
    ponto cobre os dois formatos sem ambiguidade.
    """
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def extrair_numeros(texto: str) -> list[float]:
    """Extrai todo numero do texto, em ordem, sem aplicar nenhuma isencao.

    Isencoes (ano, percentual redondo, tolerancia) sao responsabilidade de
    `verificar_numeros` -- esta funcao e extracao pura, para quem (Tasks 8
    e 9) precisar so dos numeros citados, sem julgar se sao validos.
    """
    numeros = []
    for m in _NUM.finditer(texto):
        v = _para_float(m.group())
        if v is not None:
            numeros.append(v)
    return numeros


def _eh_ano(valor: float, token_termina_em: int, texto: str) -> bool:
    """Um inteiro entre 1900 e 2100 e quase certamente um ano de calendario
    em prosa sobre commodities ("em 2024...", "desde 2020") -- nenhum MAPE,
    RMSE, preco, parametro ou contagem de observacoes deste dominio cai
    naturalmente nessa faixa de 4 digitos. E uma isencao estreita o
    suficiente para nao valer a pena restringir mais (ex.: exigir que nao
    seja seguido de "%") porque um "ano percentual" nunca ocorre na pratica.
    """
    return valor.is_integer() and 1900 <= valor <= 2100


def _eh_percentual_retorico(valor: float, token_termina_em: int, texto: str) -> bool:
    """0%, 50% e 100% sao ancoras retoricas (nada / metade / tudo), so
    isentas quando o texto de fato as marca como percentual -- ver
    `_PERCENTUAIS_RETORICOS` acima para a justificativa completa."""
    if valor not in _PERCENTUAIS_RETORICOS:
        return False
    return token_termina_em < len(texto) and texto[token_termina_em] == "%"


def _dentro_da_tolerancia(valor: float, permitido: float, tolerancia: float) -> bool:
    folga = max(_TOLERANCIA_ABSOLUTA_MINIMA, tolerancia * abs(permitido))
    return abs(valor - permitido) <= folga


def verificar_numeros(texto: str, res: RunResult, tolerancia: float = 0.01) -> None:
    """Levanta `NumeroInventado` no primeiro numero do CORPO sem origem nos
    resultados do nucleo (`res.valores_permitidos()`).

    `texto` deve ser o corpo escrito pelo LLM -- nao o markdown que
    `render_report` monta. Ver docstring do modulo, secao ESCOPO.
    """
    permitidos = res.valores_permitidos()
    for m in _NUM.finditer(texto):
        valor = _para_float(m.group())
        if valor is None:
            continue
        if _eh_ano(valor, m.end(), texto):
            continue
        if _eh_percentual_retorico(valor, m.end(), texto):
            continue
        if any(_dentro_da_tolerancia(valor, p, tolerancia) for p in permitidos):
            continue
        raise NumeroInventado(
            f"o texto cita {valor!r}, que nao existe nos resultados do nucleo. "
            f"Valores disponiveis: {sorted(permitidos)}"
        )
