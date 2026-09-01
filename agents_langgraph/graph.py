"""O mesmo pipeline de `agents/orchestrator.py`, expresso como StateGraph.

O QUE MUDA E O QUE NAO MUDA
---------------------------
Nao muda: os quatro agentes (`Coletor`, `Econometrista`, `Critico`,
`Redator`), o nucleo deterministico em `agro/`, a escada de modelos, a trava
anti-alucinacao, o teto de tentativas. Sao importados, nao reescritos.

Muda: o controle de fluxo. No orquestrador escrito a mao, o laco do Critico e
um `for` com `break` e a retentativa do Redator e um `for` com `try/except`.
Aqui os dois viram ARESTAS CONDICIONAIS entre nos de um grafo, e o estado
que atravessa o pipeline vira um `TypedDict` explicito em vez de variaveis
locais.

DUAS DECISOES QUE NAO SE LEEM NO CODIGO SOZINHAS
------------------------------------------------
1. OS AGENTES NAO VIRARAM TOOLS. LangGraph oferece `ToolNode` e o padrao
   ReAct, em que o LLM decide qual ferramenta chamar. Nao e o que este
   sistema faz, e transformar `fit_model` numa tool seria uma mudanca de
   SEMANTICA disfarcada de mudanca de framework: a ordem Coletor -> ajuste
   -> critica -> redacao e fixa de proposito, e o LLM escolhe PARAMETROS
   (janela, familia), nunca a proxima etapa. Um grafo de nos explicitos
   descreve isso com honestidade; um agente ReAct descreveria um sistema
   diferente. A comparacao so vale se os dois lados fizerem a mesma coisa.

2. AS CONSTANTES SAO IMPORTADAS, NAO COPIADAS. `MAX_TENTATIVAS_REDACAO` vem
   de `agents.orchestrator` e `MAX_TENTATIVAS` de `agro.config`. Redeclarar
   qualquer uma aqui criaria duas politicas que divergem no primeiro ajuste
   feito num lado so -- e o teste de paridade passaria a comparar dois
   sistemas que ja nao sao o mesmo.
"""
from typing import Any, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.collector import Coletor
from agents.critic import Critico
from agents.econometrician import Econometrista
from agents.llm import LLM
from agents.orchestrator import MAX_TENTATIVAS_REDACAO, _grafico_padrao
from agents.writer import Redator
from agro import config, data, guard, models, report
from agro.types import RunResult


class Estado(TypedDict, total=False):
    """O que atravessa o grafo.

    No orquestrador a mao isso e um punhado de variaveis locais de `rodar`.
    Aqui precisa ser declarado, porque cada no recebe o estado e devolve so
    as chaves que mudou. Esse e o primeiro custo visivel do framework -- e
    tambem o primeiro ganho: o estado do pipeline deixa de ser implicito.
    """
    pergunta: str
    commodity: str
    usar_cache: bool
    destino_grafico: str | None
    # coleta
    bundle: Any
    # laco do Critico
    familia: str
    fit: Any
    diag: Any
    reprovacoes: list[str]
    tentativa: int
    # saida
    res: RunResult
    corpo: Any
    tentativas_redacao: int


def _serie_principal(bundle) -> pd.Series:
    """A serie NAO vive no estado do grafo, e isso e uma concessao ao framework.

    `MemorySaver` serializa o estado com msgpack a cada no, e `pandas.Series`
    nao e serializavel -- a primeira versao deste modulo carregava a serie no
    estado e quebrava com `TypeError: Type is not msgpack serializable`.

    O laco `for` do orquestrador a mao nunca enfrenta isso: la a serie e uma
    variavel local, lida uma vez. Aqui, para manter o checkpointing (que e a
    principal vantagem do framework), o estado carrega so o `SeriesBundle` --
    um dataclass de primitivos, com o caminho do parquet -- e cada no que
    precisa da serie a rele do disco.

    O custo e real e mensuravel: o parquet e lido uma vez por no que usa a
    serie, em vez de uma vez por execucao. Num arquivo de ~2.000 linhas isso
    e barato; numa serie grande, deixaria de ser -- e a saida seria abrir mao
    do checkpointer ou escrever um serializador proprio.
    """
    df = pd.read_parquet(bundle.caminho_parquet)
    return df["cbot"].astype(float)


# --------------------------------------------------------------- nos

def no_coletar(estado: Estado, llm: LLM) -> dict:
    janela = Coletor(llm).decidir(estado["pergunta"], estado["commodity"])
    bundle = data.fetch_series(
        estado["commodity"], janela["inicio"], janela["fim"], estado["usar_cache"]
    )
    return {
        "bundle": bundle,
        "reprovacoes": [],
        "tentativa": 0,
        "tentativas_redacao": 0,
    }


def no_ajustar(estado: Estado, llm: LLM) -> dict:
    tentativa = estado["tentativa"] + 1
    familia = Econometrista(llm).escolher(
        estado["pergunta"], tentativa, estado["reprovacoes"]
    )
    return {
        "tentativa": tentativa,
        "familia": familia,
        "fit": models.fit_model(_serie_principal(estado["bundle"]), familia),
    }


def no_julgar(estado: Estado) -> dict:
    diag = Critico().julgar(estado["fit"], _serie_principal(estado["bundle"]))
    if diag.aprovado:
        return {"diag": diag}
    # Mesmo contrato do orquestrador a mao: cada motivo PRECISA carregar o
    # nome da familia como palavra inteira, senao `Econometrista.escolher`
    # levanta ValueError. `models.diagnose` nao escreve a familia nos motivos.
    motivo = f"{estado['familia']}: {'; '.join(diag.motivos)}"
    return {"diag": diag, "reprovacoes": estado["reprovacoes"] + [motivo]}


def no_consolidar(estado: Estado) -> dict:
    serie, diag = _serie_principal(estado["bundle"]), estado["diag"]
    teto_estourado = not diag.aprovado
    try:
        bt = models.backtest(serie, estado["fit"].familia)
    except ValueError:
        bt = None

    destino = estado.get("destino_grafico") or _grafico_padrao(estado["commodity"])
    reprovacoes = estado["reprovacoes"]
    res = RunResult(
        commodity=estado["commodity"], pergunta=estado["pergunta"],
        bundle=estado["bundle"], fit=estado["fit"], diagnosis=diag, backtest=bt,
        tentativas=len(reprovacoes) + (0 if teto_estourado else 1),
        teto_estourado=teto_estourado,
        grafico=report.plot_series(estado["bundle"], destino),
        numeros={"preco_final": float(serie.iloc[-1]),
                 "preco_inicial": float(serie.iloc[0])},
        historico_reprovacoes=list(reprovacoes),
    )
    return {"res": res}


def no_redigir(estado: Estado, llm: LLM) -> dict:
    """Escreve o corpo e deixa a trava anti-alucinacao falhar.

    Nao captura `NumeroInventado`/`NumeroAmbiguo` aqui: quem decide se vale
    uma nova tentativa e a aresta condicional, nao o no -- o mesmo desenho do
    orquestrador a mao, onde a decisao e de `rodar` e nao do `Redator`.
    """
    tentativas = estado["tentativas_redacao"] + 1
    try:
        corpo = Redator(llm).escrever(estado["res"])
    except (guard.NumeroInventado, guard.NumeroAmbiguo):
        if tentativas >= MAX_TENTATIVAS_REDACAO:
            # Teto de redacao estourado: a excecao sobe e derruba a execucao.
            # Publicar relatorio cujo corpo alucina numero e pior que falhar.
            raise
        return {"tentativas_redacao": tentativas}
    res = estado["res"]
    res.relatorio_md = report.render_report(res, corpo)
    return {"corpo": corpo, "tentativas_redacao": tentativas, "res": res}


# ------------------------------------------------- arestas condicionais

def _apos_julgar(estado: Estado) -> str:
    """O laco do Critico. No orquestrador a mao isto e `if diag.aprovado: break`
    dentro de um `for range(1, MAX_TENTATIVAS + 1)`."""
    if estado["diag"].aprovado:
        return "consolidar"
    if estado["tentativa"] >= config.MAX_TENTATIVAS:
        return "consolidar"          # teto estourado: relatorio sai assim mesmo
    return "ajustar"


def _apos_redigir(estado: Estado) -> str:
    return END if estado.get("corpo") is not None else "redigir"


# --------------------------------------------------------------- grafo

def construir_grafo(llm: LLM, checkpointer=None):
    """Monta o StateGraph. `checkpointer` habilita retomada e inspecao do
    estado a cada no -- coisa que o laco `for` a mao nao oferece de graca."""
    g = StateGraph(Estado)
    g.add_node("coletar", lambda e: no_coletar(e, llm))
    g.add_node("ajustar", lambda e: no_ajustar(e, llm))
    g.add_node("julgar", no_julgar)
    g.add_node("consolidar", no_consolidar)
    g.add_node("redigir", lambda e: no_redigir(e, llm))

    g.add_edge(START, "coletar")
    g.add_edge("coletar", "ajustar")
    g.add_edge("ajustar", "julgar")
    g.add_conditional_edges("julgar", _apos_julgar,
                            {"ajustar": "ajustar", "consolidar": "consolidar"})
    g.add_edge("consolidar", "redigir")
    g.add_conditional_edges("redigir", _apos_redigir, {"redigir": "redigir", END: END})
    return g.compile(checkpointer=checkpointer)


def rodar(pergunta: str, commodity: str, llm: LLM, usar_cache: bool = True,
          destino_grafico: str | None = None) -> RunResult:
    """Mesma assinatura e mesmo contrato de saida de `agents.orchestrator.rodar`.

    A troca de motor tem de ser invisivel para quem chama -- e o que permite
    `tests/test_paridade.py` rodar os dois lado a lado com a mesma entrada e
    comparar campo a campo.
    """
    grafo = construir_grafo(llm, checkpointer=MemorySaver())
    final = grafo.invoke(
        {"pergunta": pergunta, "commodity": commodity, "usar_cache": usar_cache,
         "destino_grafico": destino_grafico},
        # `recursion_limit` precisa acomodar o pior caso: MAX_TENTATIVAS voltas
        # no laco do Critico mais MAX_TENTATIVAS_REDACAO no do Redator. O
        # padrao do LangGraph (25) ja daria, mas deixar implicito significaria
        # que mexer em `config.MAX_TENTATIVAS` poderia estourar o limite sem
        # que nada aqui indicasse a relacao.
        config={"configurable": {"thread_id": f"{commodity}-{hash(pergunta) & 0xFFFF}"},
                "recursion_limit": 4 * (config.MAX_TENTATIVAS + MAX_TENTATIVAS_REDACAO) + 10},
    )
    return final["res"]
