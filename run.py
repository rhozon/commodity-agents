"""CLI do analista de commodities multiagente.

Em modo real (sem `--fake-llm`), exige `ANTHROPIC_API_KEY` no ambiente e
chama a API a cada rodada do laco Coletor/Econometrista/Critico/Redator. Em
modo demonstracao (`--fake-llm`), nenhuma rede e tocada e nenhuma chave e
lida: o LLM e trocado por `_LLMDemonstracao` (abaixo), que devolve respostas
fixas sem chamar `agents.llm.LLMFake` diretamente.

`LLMFake` pop respostas de uma lista na ORDEM em que sao pedidas -- isso
exigiria adivinhar de fora quantas vezes o Econometrista vai ser chamado
(depende de quantos degraus da escada o ajuste real em R precisa descer, o
que so se sabe rodando) e quantas vezes o Redator vai ser chamado (depende
de a trava anti-alucinacao aprovar o corpo de primeira). `_LLMDemonstracao`
roteia pelo AGENTE que pergunta, lendo o proprio texto do prompt (cada
agente se identifica: "agente Coletor", "agente Econometrista", "agente
Redator") -- assim funciona com qualquer numero real de tentativas ou
retentativas, sem lista de tamanho fixo para acertar. O corpo fixo do
Redator nao contem nenhum digito de proposito, entao nunca cai no proprio
buraco que a trava existe para fechar, nao importa qual familia de modelo
tenha passado no Critico desta vez.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agents.llm import ClaudeLLM, LLM, validar  # noqa: E402
from agents.orchestrator import rodar  # noqa: E402
from agro import config  # noqa: E402


def _motor(nome: str):
    """Escolhe a orquestracao. O import do LangGraph e adiado de proposito:
    o motor `manual` e o padrao e nao depende do framework, entao quem nao
    pedir `--engine langgraph` nao precisa te-lo instalado."""
    if nome == "manual":
        return rodar
    from agents_langgraph import rodar as rodar_grafo
    return rodar_grafo


def _corpo_demonstracao(commodity: str) -> dict[str, str]:
    """As tres camadas fixas do modo demonstracao, uma por campo do esquema.

    Nenhuma delas contem digito, de proposito: assim o corpo nunca cai no
    proprio buraco que a trava anti-alucinacao existe para fechar, seja qual
    for a familia de modelo que passou no Critico desta vez.
    """
    nome = config.COMMODITIES[commodity].nome_exibicao
    return {
        "previsao": (
            "Modo de demonstracao (--fake-llm): este texto e uma resposta fixa, "
            "nao uma chamada real ao modelo de linguagem -- por isso ele nao "
            "cita nenhum numero alem do que o nucleo ja calculou, exatamente "
            "como a trava anti-alucinacao exige de uma resposta de verdade. "
            "Modelos de volatilidade (MSGARCH, GARCH) tem media zero na "
            "especificacao, entao o ponto previsto empata com o passeio "
            "aleatorio por construcao; a contribuicao real deles esta na "
            "largura do intervalo do backtest, nao no valor pontual. Quando a "
            "escada recua ate o ARIMA, o modelo passa a produzir previsao "
            "pontual de verdade, ao custo de nao capturar mudanca de regime de "
            "volatilidade."
        ),
        "drivers": (
            f"O preco de {nome} no CBOT e a serie principal deste ajuste; o "
            "cambio USD/BRL entra como referencia para quem converte a serie "
            "internacional em preco domestico -- sem ele, analise de preco "
            "domestico de grao no Brasil esta errada. O CEPEA nao esta "
            "disponivel nesta versao, e o aviso de troca de fonte acima do "
            "grafico documenta isso. As secoes deterministicas abaixo trazem o "
            "que o modelo de fato estimou: volatilidade, testes de residuo e "
            "backtest."
        ),
        "implicacao": (
            "A decisao de manter, reduzir ou ampliar exposicao cabe a quem le o "
            "relatorio: este texto informa, nao recomenda. O que o sistema "
            "oferece para essa decisao e o alcance do modelo que passou no "
            "Critico, o motivo escrito de cada familia reprovada antes dele e a "
            "largura do intervalo medida no backtest -- e nao uma projecao de "
            "preco, que nenhuma das familias de volatilidade produz."
        ),
    }


class _LLMDemonstracao:
    """LLM fake para --fake-llm: roteia pelo prompt, nao por uma ordem fixa.

    Ver docstring do modulo para o porque de nao usar `agents.llm.LLMFake`
    diretamente aqui.
    """

    ESQUEMA_COLETOR = {"inicio": str, "fim": str, "justificativa": str}
    ESQUEMA_ECONOMETRISTA = {"familia": str, "justificativa": str}
    ESQUEMA_REDATOR = {"previsao": str, "drivers": str, "implicacao": str,
                       "confianca": str}

    def __init__(self, commodity: str):
        self._escada = list(config.ESCADA_MODELOS)  # ("msgarch", "garch", "arima")
        self._proximo_degrau = 0
        self._corpo = _corpo_demonstracao(commodity)

    def perguntar(self, prompt: str, esquema: dict[str, type]) -> dict:
        if "agente Coletor" in prompt:
            resp = {"inicio": "2018-01-01", "fim": "2025-12-31",
                    "justificativa": "cobre varios ciclos e da ao msgarch janela "
                                     "suficiente para identificar regime de "
                                     "volatilidade"}
        elif "agente Econometrista" in prompt:
            # Uma familia por tentativa, na ordem da escada; se a escada
            # inteira ja foi oferecida (nao deveria, o orquestrador para em
            # config.MAX_TENTATIVAS), repete o ultimo degrau em vez de
            # estourar indice.
            familia = self._escada[min(self._proximo_degrau, len(self._escada) - 1)]
            self._proximo_degrau += 1
            resp = {"familia": familia,
                    "justificativa": f"degrau da escada: {familia}"}
        elif "agente Redator" in prompt:
            resp = {**self._corpo, "confianca": "baixa"}
        else:
            raise AssertionError(
                f"--fake-llm nao reconhece este prompt (nenhum agente conhecido "
                f"se identificou nele): {prompt[:120]!r}"
            )
        return validar(resp, esquema)


def main() -> int:
    p = argparse.ArgumentParser(description="Analista de commodities multiagente")
    p.add_argument("--commodity", required=True, choices=sorted(config.COMMODITIES))
    p.add_argument("--pergunta", required=True)
    p.add_argument("--saida", default="")
    p.add_argument("--fake-llm", action="store_true",
                   help="usa respostas fixas; nao chama a API nem toca a rede")
    p.add_argument("--engine", default="manual", choices=["manual", "langgraph"],
                   help="orquestracao: laco escrito a mao (padrao) ou StateGraph "
                        "do LangGraph. Os dois produzem o mesmo relatorio -- "
                        "ver tests/test_paridade.py")
    a = p.parse_args()

    llm: LLM = _LLMDemonstracao(a.commodity) if a.fake_llm else ClaudeLLM()

    # O destino do relatorio manda no destino do grafico: o markdown
    # referencia a imagem pelo nome (`report.render_report` emite o
    # basename), entao a imagem tem de cair ao lado dele. Assim
    # `--saida /tmp/x.md` escreve `/tmp/x.png` em vez de sobrescrever o
    # artefato publicado em `examples/`.
    destino = Path(a.saida) if a.saida else config.EXAMPLES_DIR / f"{a.commodity}.md"
    destino.parent.mkdir(parents=True, exist_ok=True)

    res = _motor(a.engine)(a.pergunta, a.commodity, llm,
                           destino_grafico=str(destino.with_suffix(".png")))

    destino.write_text(res.relatorio_md, encoding="utf-8")

    print(f"motor: {a.engine}")
    print(f"modelo: {res.fit.familia} | tentativas: {res.tentativas} | "
          f"teto estourado: {res.teto_estourado}")
    print(f"relatorio: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
