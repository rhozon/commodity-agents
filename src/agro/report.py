"""Grafico da serie e montagem do relatorio em tres camadas.

Le `types.py`/`models.py` no estado ATUAL (pos onda de correcao), nao no
estado do brief original. Diferencas relevantes:

- `ModelFit` ganhou `vol_por_regime` (lista, volatilidade estrutural por
  regime) e `vol_atual` (volatilidade condicional do ultimo instante). Sao a
  contribuicao real dos modelos de volatilidade (msgarch/garch) e precisam
  aparecer no relatorio quando existirem -- senao a correcao de
  `models.fit_model` fica invisivel para quem le o texto final.
- `Backtest` ganhou `mape_baseline`, `rmse_baseline`, a propriedade
  `bateu_baseline` e o campo `nota`. `nota` explica POR QUE o modelo empatou
  ou nao com o passeio aleatorio -- ela cobre tres casos que antes eram
  silencio: empate por construcao (modelo de volatilidade previu zero por
  especificacao), refit que nao convergiu, e banda que caiu no
  desvio-padrao historico por falta de vol_atual. Sempre que `nota` nao for
  vazia, o relatorio precisa mostra-la em vez de deixar o leitor concluir
  "o modelo falhou" quando ele funcionou como a teoria manda.
- `Diagnosis.testes` agora pode carregar `ljung_box_pvalor` e
  `arch_lm_pvalor` (testes de residuo que o Critico usa para reprovar). Sao
  a prova de que as premissas do ajuste foram checadas, e por isso aparecem
  no relatorio quando presentes.

Cuidado especifico de redacao: quando `nota` descreve um empate por
construcao, `bateu_baseline` e `False` (MAPE do modelo nao e ESTRITAMENTE
menor que o do baseline -- sao iguais). Isso NAO e derrota do modelo, e o
texto do veredito nunca deve dizer "o passeio aleatorio nao foi batido"
nesse caso: le-se a nota e se diz o que de fato aconteceu.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agro.guard import verificar_numeros  # noqa: E402
from agro.types import (  # noqa: E402
    Backtest,
    CorpoRelatorio,
    ModelFit,
    RunResult,
    SeriesBundle,
)


def plot_series(bundle: SeriesBundle, destino: str) -> str:
    """Um grafico que se entende em cinco segundos: preco e cambio."""
    df = pd.read_parquet(bundle.caminho_parquet)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=140)
    ax.plot(df.index, df["cbot"], linewidth=1.4, label="CBOT")
    if "cepea" in df.columns:
        ax.plot(df.index, df["cepea"], linewidth=1.4, label="CEPEA")
    ax.set_title(f"{bundle.commodity.capitalize()} — {bundle.inicio} a {bundle.fim}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino)
    plt.close(fig)
    return destino


def _formatar_volatilidade(fit: ModelFit) -> list[str]:
    """Volatilidade e a contribuicao real dos modelos de volatilidade
    (msgarch/garch); so aparece no relatorio quando o fit realmente a
    forneceu -- nada de secao vazia ou placeholder quando um ARIMA (media)
    nao tem essa grandeza."""
    if not fit.vol_por_regime and fit.vol_atual is None:
        return []

    linhas = ["## Volatilidade", ""]
    if fit.vol_por_regime:
        if len(fit.vol_por_regime) > 1:
            regimes = "; ".join(
                f"regime {i + 1}: {v:.4f}" for i, v in enumerate(fit.vol_por_regime)
            )
            linhas.append(f"Volatilidade estrutural de longo prazo por regime: {regimes}.")
        else:
            linhas.append(
                f"Volatilidade estrutural de longo prazo: {fit.vol_por_regime[0]:.4f}."
            )
    if fit.vol_atual is not None:
        linhas.append(f"Volatilidade condicional do ultimo instante: {fit.vol_atual:.4f}.")
    linhas.append("")
    return linhas


def _formatar_recuo_de_modelo(res: RunResult) -> list[str]:
    """Uma linha por tentativa reprovada, com a familia e o motivo escrito.

    "Tentativas: 2" sozinho nao diz nada: nao nomeia a familia que caiu nem
    por que ela caiu. Quando o MSGARCH e reprovado e o sistema recua para o
    GARCH, e AQUI que o recuo fica declarado -- e ele fica declarado sempre
    que houve reprovacao, nao apenas quando o teto estourou.
    """
    if not res.historico_reprovacoes:
        return []

    linhas = [
        "## Recuo de modelo",
        "",
        "O Critico reprovou as tentativas abaixo antes do ajuste que o "
        "relatorio descreve. Cada linha traz a familia tentada e o motivo "
        "escrito da reprovacao.",
        "",
    ]
    for i, entrada in enumerate(res.historico_reprovacoes, start=1):
        familia, separador, motivos = entrada.partition(": ")
        if separador and motivos:
            linhas.append(f"{i}. **{familia}** — {motivos}")
        else:
            linhas.append(f"{i}. {entrada}")
    linhas.append("")
    return linhas


def _formatar_testes_residuo(res: RunResult) -> list[str]:
    """Ljung-Box e ARCH-LM sao a prova de que as premissas do ajuste foram
    checadas -- aparecem quando o Critico (via `diagnose`) os carregou em
    `Diagnosis.testes`."""
    testes = res.diagnosis.testes
    lb = testes.get("ljung_box_pvalor")
    arch = testes.get("arch_lm_pvalor")
    if lb is None and arch is None:
        return []

    linhas = ["## Testes de residuo", ""]
    if lb is not None:
        linhas.append(f"Ljung-Box (autocorrelacao remanescente): p-valor {lb:.4f}.")
    if arch is not None:
        linhas.append(f"ARCH-LM (heterocedasticidade nao capturada): p-valor {arch:.4f}.")
    linhas.append("")
    return linhas


def _veredito_backtest(b: Backtest) -> str:
    """O texto do veredito nunca pode ler `bateu_baseline` sozinho: para
    modelos de volatilidade os dois MAPEs sao identicos por construcao e
    `bateu_baseline` e `False`, mas isso e empate, nao derrota. `nota` e
    quem desempata as tres leituras possiveis; so cai no texto generico
    quando `nota` vem vazia (caso normal de modelo de media que de fato
    perdeu ou ganhou do passeio aleatorio no ponto)."""
    if b.nota:
        return b.nota
    if b.bateu_baseline:
        return "o modelo bateu o passeio aleatorio"
    return "o modelo nao superou o passeio aleatorio no ponto previsto"


def render_report(res: RunResult, corpo: CorpoRelatorio) -> str:
    """Monta o markdown final. O corpo vem do Redator; a moldura e nossa.

    `corpo` traz UMA CAMADA POR CAMPO e cada uma vai para a sua secao. Antes,
    o Redator devolvia um texto so e as secoes "Drivers" e "Implicacao de
    decisao" eram carimbos fixos ("Ver decomposicao no corpo acima.") --
    tres titulos e uma camada, num relatorio que se anuncia em tres camadas.

    A PRIMEIRA coisa que acontece aqui e a trava anti-alucinacao rodar sobre
    as tres camadas juntas (`corpo.texto_completo()`). Antes, o contrato
    "chame `verificar_numeros` antes de montar o relatorio" vivia so em prosa
    nas docstrings -- e a forma mais facil de burlar uma trava e nao
    chama-la. Com a chamada aqui dentro, a omissao deixou de ser possivel
    por esquecimento.

    A trava roda sobre o CORPO, nunca sobre o markdown montado: a moldura
    imprime numeros deterministicos (`res.tentativas`, por exemplo) que nao
    estao em `valores_permitidos()`, e conferi-los derrubaria a execucao por
    um numero correto. Ver `guard.py`, secao ESCOPO.

    Levanta `guard.NumeroInventado` (ou `guard.NumeroAmbiguo`) e nao devolve
    markdown nenhum quando o corpo cita numero sem origem.
    """
    verificar_numeros(corpo.texto_completo(), res)

    linhas = [
        f"# {res.commodity.capitalize()} — analise quantitativa",
        "",
        f"**Pergunta:** {res.pergunta}",
        "",
        f"Serie de {res.bundle.inicio} a {res.bundle.fim}, {res.bundle.n_obs} observacoes. "
        f"Modelo: {res.fit.familia}. Tentativas: {res.tentativas}.",
        "",
    ]
    linhas += _formatar_recuo_de_modelo(res)
    if res.teto_estourado:
        linhas += ["> **Aviso:** o teto de tentativas foi atingido sem ajuste aprovado. "
                   "O que segue e o ultimo ajuste tentado, sem comparacao de AIC ou "
                   "verossimilhança com as tentativas anteriores, e deve ser lido com "
                   "reserva.", ""]
    for troca in res.bundle.trocas_de_fonte:
        linhas += [f"> **Fonte trocada:** {troca}", ""]
    if res.grafico:
        # BASENAME, nunca o caminho absoluto: o grafico e sempre gravado ao
        # lado do relatorio (quem chama `orchestrator.rodar` deriva um
        # destino do outro), e um caminho absoluto quebraria a imagem para
        # quem le o markdown no GitHub ou depois de mover a pasta. Este e o
        # unico lugar que decide como a imagem e referenciada -- antes havia
        # uma reescrita de string em `run.py` fazendo o mesmo por fora.
        linhas += [f"![Serie de {res.commodity}]({Path(res.grafico).name})", ""]

    linhas += ["## Previsao", "", corpo.previsao, "",
               "## Drivers", "", corpo.drivers, "",
               "## Implicacao de decisao", "", corpo.implicacao, ""]

    linhas += _formatar_volatilidade(res.fit)
    linhas += _formatar_testes_residuo(res)

    if res.backtest:
        b = res.backtest
        veredito = _veredito_backtest(b)
        linhas += ["## Backtest", "",
                   f"Horizonte de {b.horizonte} passos. Modelo: MAPE {b.mape:.2f}%, "
                   f"RMSE {b.rmse:.4f}. Passeio aleatorio: MAPE {b.mape_baseline:.2f}%, "
                   f"RMSE {b.rmse_baseline:.4f}. Cobertura do intervalo "
                   f"{b.cobertura_ic:.2f}. Em resumo, {veredito}.", ""]
    return "\n".join(linhas)
