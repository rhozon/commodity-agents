"""O laco: Econometrista propoe, Critico reprova, repete ate o teto.

E aqui que o sistema e agentico de verdade. O resto e pipeline.

Duas decisoes de desenho deste modulo que nao se leem no codigo sozinhas:

1. CONTRATO DA ESCADA (`Econometrista.escolher`). Cada string em
   `reprovacoes` PRECISA conter o nome da familia reprovada como palavra
   inteira -- se nao contiver, `escolher` levanta `ValueError` de proposito
   (ver docstring de `agents.econometrician.Econometrista.escolher`).
   `models.diagnose` NAO escreve o nome da familia nos motivos que produz
   ("o ajuste nao convergiu: ...", "Ljung-Box rejeita ..."); por isso este
   modulo prefixa cada motivo com a familia que o gerou antes de guardar
   em `reprovacoes`, no unico lugar em que a lista e construida.

2. FALHA DA TRAVA ANTI-ALUCINACAO NO REDATOR (`agents.writer.Redator`). O
   Redator chama `guard.verificar_numeros` e deixa `NumeroInventado`/
   `NumeroAmbiguo` subirem sem capturar -- a decisao de o que fazer com a
   falha e deste orquestrador, nao do agente (ver docstring de
   `agents.writer`). A falha da trava e categoricamente diferente de uma
   reprovacao do Critico: o AJUSTE esta bom, foi o TEXTO que citou um numero
   sem origem (ou ambiguo). Uma nova chamada ao LLM para reescrever o corpo
   e barata comparada a reajustar um modelo GARCH/MSGARCH via R, e
   plausivel que resolva sozinha -- o mesmo raciocinio que motiva o retry de
   esquema em `agents.llm.ClaudeLLM.perguntar`. Por isso `rodar` tenta
   escrever o corpo ate `MAX_TENTATIVAS_REDACAO` vezes; se a trava continuar
   reprovando depois disso, a excecao sobe sem ser capturada -- publicar um
   relatorio cujo corpo alucina numero e pior do que falhar alto, e "melhor
   falhar alto do que publicar um numero fabricado" e a propria justificativa
   escrita em `agro.guard`.
"""
import pandas as pd

from agents.collector import Coletor
from agents.critic import Critico
from agents.econometrician import Econometrista
from agents.llm import LLM
from agents.writer import Redator
from agro import config, data, guard, models, report
from agro.types import CorpoRelatorio, RunResult

# Ver ponto 2 na docstring do modulo: uma tentativa extra de redacao e barata
# e provavelmente resolve uma falha da trava anti-alucinacao. Nao usa
# `config.MAX_TENTATIVAS` (esse e o teto da escada de MODELO, uma decisao
# totalmente diferente) nem vive em `agro.config` -- o numero de vezes que se
# repete uma chamada de LLM e politica da camada de agentes, e `agro/` e o
# nucleo deterministico, que nao sabe que existe LLM.
MAX_TENTATIVAS_REDACAO = 2


def _grafico_padrao(commodity: str) -> str:
    """Onde o grafico cai quando quem chama nao escolhe: ao lado do relatorio
    de exemplo versionado. Quem passa `--saida` recebe o grafico ao lado do
    proprio relatorio -- ver `destino_grafico` em `rodar`."""
    return str(config.EXAMPLES_DIR / f"{commodity}.png")


def _serie_principal(bundle) -> pd.Series:
    df = pd.read_parquet(bundle.caminho_parquet)
    return df["cbot"].astype(float)


def _escrever_com_retentativa(llm: LLM, res: RunResult) -> CorpoRelatorio:
    """Chama o Redator ate `MAX_TENTATIVAS_REDACAO` vezes.

    So retenta quando a FALHA e da trava anti-alucinacao (`NumeroInventado`/
    `NumeroAmbiguo`) -- essas sao as unicas excecoes que `Redator.escrever`
    deixa subir de proposito. Qualquer outra excecao (erro do LLM, esquema
    invalido etc.) sobe na primeira tentativa, sem retry: o retry aqui e uma
    aposta especifica em "o LLM escreveu prosa com numero sem origem, uma
    nova redacao provavelmente evita o mesmo erro" -- nao um retry generico.
    """
    ultimo_erro: Exception | None = None
    for _ in range(MAX_TENTATIVAS_REDACAO):
        try:
            return Redator(llm).escrever(res)
        except (guard.NumeroInventado, guard.NumeroAmbiguo) as e:
            ultimo_erro = e
    assert ultimo_erro is not None
    raise ultimo_erro


def rodar(pergunta: str, commodity: str, llm: LLM, usar_cache: bool = True,
          destino_grafico: str | None = None) -> RunResult:
    """Executa o pipeline completo: Coletor -> laco Econometrista/Critico -> Redator.

    `destino_grafico` e o caminho do PNG da serie. Quem chama deriva esse
    caminho do destino do RELATORIO, para que a imagem caia ao lado do
    markdown que a referencia (`report.render_report` emite so o basename).
    Sem esse parametro, o grafico ia sempre para `examples/<commodity>.png`:
    uma execucao com `--saida /tmp/x.md` sobrescrevia o artefato publicado e
    versionado, e o markdown resultante apontava para um arquivo que nao
    estava ao lado dele. Omitido, cai no destino de exemplo.

    Contrato de saida (o que qualquer chamador -- a CLI, um notebook --
    pode assumir do `RunResult` devolvido):
        - Sempre devolve um `RunResult` ou levanta excecao -- nunca devolve
          `None` nem um resultado parcialmente preenchido.
        - `res.relatorio_md` so vem no retorno depois de passar pela trava
          anti-alucinacao (`agro.guard.verificar_numeros`, chamada dentro de
          `agro.report.render_report`); se `rodar` retornar sem levantar,
          `relatorio_md` esta seguro para publicar.
        - `res.tentativas` conta exatamente as tentativas de AJUSTE de
          modelo (1 na aprovacao de primeira, N em qualquer cenario de
          recuo ou teto estourado) -- nao inclui tentativas de redacao.
        - `res.teto_estourado` e `True` quando as `config.MAX_TENTATIVAS`
          tentativas se esgotam sem aprovacao do Critico; mesmo assim o
          relatorio SAI, com o ULTIMO ajuste tentado (a escada e uma
          progressao fixa -- proximo degrau ainda nao tentado -- sem
          nenhuma comparacao de AIC ou verossimilhança entre as tentativas;
          o que sobra nao e necessariamente o melhor ajuste possivel) e o
          aviso explicito no corpo do markdown.
        - `res.backtest` e `None` quando a serie e curta demais para o
          horizonte do backtest (`models.backtest` levanta `ValueError`);
          o relatorio se comporta bem nesse caso (secao "## Backtest" so
          aparece quando `res.backtest` existe).
        - Excecoes que PODEM escapar de `rodar`: `guard.NumeroInventado`/
          `guard.NumeroAmbiguo` se a trava reprovar o corpo em todas as
          `MAX_TENTATIVAS_REDACAO` tentativas de redacao; `ValueError` de
          `Econometrista.escolher` so se o proprio orquestrador quebrar o
          contrato da escada (nao deveria acontecer em uso normal, ja que
          este modulo sempre prefixa a familia no motivo -- ver ponto 1 na
          docstring do modulo); erros de rede/coleta de `agro.data`; erros
          de subprocess de `agro.rbridge` (R ausente, timeout etc.).
    """
    janela = Coletor(llm).decidir(pergunta, commodity)
    bundle = data.fetch_series(commodity, janela["inicio"], janela["fim"], usar_cache)
    serie = _serie_principal(bundle)

    economista, critico = Econometrista(llm), Critico()
    reprovacoes: list[str] = []
    fit = diag = None

    for tentativa in range(1, config.MAX_TENTATIVAS + 1):
        familia = economista.escolher(pergunta, tentativa, reprovacoes)
        fit = models.fit_model(serie, familia)
        diag = critico.julgar(fit, serie)
        if diag.aprovado:
            break
        # Contrato de `Econometrista.escolher`: cada motivo PRECISA conter o
        # nome da familia que o gerou, como palavra inteira -- `diag.motivos`
        # nao traz isso sozinho (ver docstring do modulo, ponto 1).
        reprovacoes.append(f"{familia}: {'; '.join(diag.motivos)}")

    teto_estourado = not diag.aprovado
    try:
        bt = models.backtest(serie, fit.familia)
    except ValueError:
        bt = None

    grafico = report.plot_series(bundle, destino_grafico or _grafico_padrao(commodity))
    res = RunResult(
        commodity=commodity, pergunta=pergunta, bundle=bundle, fit=fit, diagnosis=diag,
        backtest=bt, tentativas=len(reprovacoes) + (0 if teto_estourado else 1),
        teto_estourado=teto_estourado, grafico=grafico,
        numeros={"preco_final": float(serie.iloc[-1]), "preco_inicial": float(serie.iloc[0])},
        # A mesma lista que alimenta o prompt do Econometrista viaja para o
        # relatorio: e ela que faz o recuo de modelo aparecer para quem le.
        # Descartada, o MSGARCH reprovado sumia do texto final -- e ele e o
        # modelo-assinatura, a razao declarada de o projeto existir.
        historico_reprovacoes=list(reprovacoes),
    )
    corpo = _escrever_com_retentativa(llm, res)
    res.relatorio_md = report.render_report(res, corpo)
    return res
