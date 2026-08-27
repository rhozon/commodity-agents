"""Trava anti-alucinacao: o Redator nao inventa numero.

Esta e a peca mais importante do projeto. O corpo do relatorio (secao
"Previsao") e escrito por um LLM; tudo o mais no pipeline e determinístico.
Um numero que o LLM cita sem que ele exista nos resultados calculados pelo
nucleo (`R`/`models.py`) nao e um erro de estilo -- e uma alucinacao, e num
artefato que serve de prova publica de competencia em IA isso derruba a
credibilidade de quem assina.

`verificar_numeros` varre todo numero presente no texto do Redator e confere
cada um contra `RunResult.valores_rotulados()` (ver `types.py`). Numero sem
origem levanta `NumeroInventado` e interrompe a execucao: e melhor falhar
alto do que publicar um numero fabricado.

ESCOPO -- isto valida o CORPO escrito pelo LLM, nao o markdown final.
`render_report` (`report.py`) monta a moldura do relatorio ao redor do corpo,
e essa moldura imprime numeros deterministicos que NAO estao em
`valores_permitidos()` -- o numero de tentativas (`res.tentativas`), por
exemplo, vem direto do controlador do pipeline, nao do LLM. Rodar esta trava
sobre o documento montado por `render_report` derrubaria a execucao por um
numero correto, so que fora do dicionario que a trava conhece. Por isso a
composicao com o resto do pipeline nao e mais um contrato escrito em prosa:
`render_report` chama `verificar_numeros(corpo_md, res)` na primeira linha,
sobre o corpo, antes de montar coisa alguma. Pular a trava deixou de ser
possivel por esquecimento.


COMO CADA DECISAO DE DESENHO SE SUSTENTA
----------------------------------------

1. TOLERANCIA PELA PRECISAO ESCRITA, nao por epsilon fixo nem por percentual
   relativo. Um numero escrito com `k` casas decimais AFIRMA precisao de `k`
   casas; ele so e aceito se for um arredondamento correto de algum valor
   autorizado nessa precisao, isto e, se `|valor - permitido| <= 0.5 * 10^-k`.
   Isso resolve de uma vez tres problemas do desenho anterior:
     - o piso absoluto de 0.005 era metade de casa de MAPE, mas era aplicado
       tambem a p-valor, onde 0.005 e diferenca MATERIAL: com p real 0.046
       (rejeita a premissa) o texto podia escrever 0.051 (nao rejeita) e
       passar, invertendo a conclusao do relatorio;
     - a tolerancia relativa de 1% aceitava 0.0170 no lugar de 0.0134 (27% de
       erro) porque a distancia absoluta era pequena, e aceitava "2000
       observacoes" quando n_obs era 1987 (1% de 1987 e 19.87);
     - a degeneracao perto de zero que motivava o piso desaparece sozinha:
       "0.013" para alpha0_1 = 0.0134 passa porque 0.013 E o arredondamento
       em tres casas, nao porque a distancia e pequena.
   O efeito colateral e conhecido e desejado: escrever mais casas decimais
   passou a ser mais exigente, nao menos. "62.76" para um preco de 62.75 nao
   e mais aceito -- e um digito fabricado na ultima casa.

2. ANO SO E ISENTO COM CONTEXTO DE DATA. A faixa 1900..2100 tem 201 inteiros;
   isenta-la em qualquer contexto era abrir 201 numeros livres, e `n_obs` de
   uma serie diaria de oito anos cai exatamente ali. A isencao agora exige
   que o numero PARECA data: nao seguido de "%", nao seguido de substantivo
   de coisa contada ("2000 observacoes" e contagem, nao ano), e precedido de
   preposicao de data ("em 2024", "desde 2015").

2b. SIMBOLO DE MOEDA ANTES DO NUMERO CANCELA AS TRES ISENCOES DE PROSA (ano,
   percentual retorico, inteiro pequeno). "R$ 2" e preco, nao a contagem
   "2 regimes"; preco vem do nucleo. E a mesma logica do teste do "%": quando
   o texto MARCA o numero como grandeza, ele deixa de ser numero de prosa.

3. DATAS COMPOSTAS E ORDEM DE MODELO SAO MASCARADAS. "2020-01-01" nao e tres
   numeros a conferir, e "GARCH(1,1)" nao e o numero 1,1 -- sao literais.
   `_MASCARAS` reconhece essas formas RIGIDAS (data com mes 01-12 e dia
   01-31, intervalo de anos, safra "2024/25", ordem de modelo com no maximo
   dois digitos por posicao) e pula os numeros dentro delas. A rigidez e o
   que impede a mascara de virar canal: nenhum valor arbitrario cabe nelas.

4. INTEIROS PEQUENOS (1, 2, 3) SAO ISENTOS EXCETO SEGUIDOS DE "%" -- o
   inverso exato da regra dos percentuais retoricos (0, 50, 100), que so sao
   isentos QUANDO seguidos de "%". "o modelo identificou 2 regimes" e prosa
   sobre a estrutura do MSGARCH, o modelo principal do projeto; "cresceu 3%"
   e afirmacao quantitativa e tem de vir do nucleo.

5. SEPARADOR DE MILHAR E SUPORTADO POR ALTERNACAO ANCORADA, e a ambiguidade
   que sobra FALHA ALTO. A regex antiga (`-?\\d+(?:[.,]\\d+)?`) lia
   "2.600 observacoes" como 2.6 e REPROVAVA um numero correto, e rachava
   "R$ 1.234,56" em 1.234 e 56 -- com o pedaco podendo casar por acidente com
   um parametro permitido, deixando trava e leitor lendo numeros diferentes.
   A alternacao com ancoras (`[1-9]\\d{0,2}` na frente, grupos de exatamente
   tres digitos, e lookahead que proibe digito adiante) restaura o suporte sem
   quebrar "-1800.0". Quando o token tem UM grupo so ("2.600"), as duas
   leituras -- 2600 e 2.6 -- sao de fato possiveis: se as duas estiverem
   autorizadas, `NumeroAmbiguo` pede reescrita sem separador de milhar em vez
   de escolher em silencio qual numero o leitor vai ver.

6. ESCALA PERCENTUAL E TRATADA NO COMPARADOR, nao autorizando 90.0. Quando o
   token e seguido de "%", o comparador tambem tenta `valor/100` -- mas so
   contra as grandezas que o nucleo guarda COMO FRACAO DE 1 e o texto escreve
   como percentual (`backtest.cobertura_ic` e `nivel_ic`, ver
   `_ROTULOS_EM_FRACAO`). Isso deixa o Redator escrever "90%" para
   `cobertura_ic = 0.90` sem colocar 90.0 no conjunto autorizado -- o que
   liberaria o inteiro 90 em QUALQUER contexto (um preco de 90, um MAPE de
   90), exatamente o tipo de numero livre que o resto deste modulo fecha. A
   restricao aos dois rotulos e o que impede a leitura de escala de virar
   canal por conta propria: sem ela, "MAPE de 4.6%" casaria com o p-valor
   0.046 do ARCH-LM, duas grandezas que nao tem nada a ver uma com a outra.

Formatos de numero aceitos: brasileiro (decimal com virgula) e ingles
(decimal com ponto), com ou sem separador de milhar.


LIMITACOES -- O QUE ESTA TRAVA NAO FAZ
---------------------------------------
Uma leitura otimista desta docstring superestimaria a protecao. Registrado
aqui porque isso so existia antes no relatorio de uma revisao manual, que
nao viaja com o codigo.

- PROVENIENCIA DE DIGITO, NAO VINCULO SEMANTICO. A trava confere se o
  DIGITO citado existe em algum resultado do nucleo -- ela nao confere se a
  GRANDEZA a que o Redator atribui esse digito e a mesma grandeza de onde o
  digito veio. "o preco subiu 3%" PASSA quando `backtest.rmse = 3.21` e o
  unico valor autorizado perto de 3, porque "3" escrito com zero casas
  decimais afirma tolerancia de +-0.5 (decisao 1) e 3.21 cai dentro dela --
  mesmo RMSE e "o preco subiu X%" sendo duas grandezas sem nenhuma relacao.
  A trava nao sabe, e nao tem como saber sem entender a frase, que o numero
  citado deveria vir de uma variacao percentual de preco e nao de um erro
  de previsao.

- INTEIROS CURTOS SAO ESTRUTURALMENTE LIVRES. Todo numero escrito SEM casa
  decimal herda a tolerancia larga da decisao 1 (+-0.5 de qualquer valor
  autorizado). Com um `RunResult` realista de MSGARCH (varios parametros,
  AIC, log-verossimilhanca, horizonte, MAPE, RMSE, n_obs etc.), isso libera
  algo da ordem de DEZ inteiros distintos -- um por valor autorizado, mais
  os que caem no meio do caminho entre dois valores proximos -- que o
  Redator pode citar em QUALQUER contexto de prosa sem que a trava reclame.
  Nao ha correcao facil: apertar a tolerancia de inteiros quebraria a
  decisao 1 (tolerancia pela precisao ESCRITA) para todo numero curto
  legitimo.

- CITAR O QUANTIL DERRUBA A EXECUCAO. Se o Redator escrever "o quantil de
  1,96", a trava reprova -- `Z_IC_95` fica fora do conjunto autorizado de
  proposito, porque autoriza-lo libertaria o inteiro "2" pela tolerancia de
  +-0.5, inclusive marcado ("subiu 2%"). E falso positivo aceito: a
  alternativa afrouxa a trava.

- NADA IMPEDE TROCAR O ROTULO. "o RMSE do backtest foi 4.53" passa mesmo que
  4.53 seja o MAPE, nao o RMSE -- a trava so confere se o numero existe em
  ALGUM lugar de `valores_rotulados()`, nao se o rotulo que o texto atribui
  a ele bate com o rotulo verdadeiro. Trocar duas grandezas de mesmo valor
  (ou de valores que colidem por tolerancia) e invisivel para este modulo.

- ESCOPO E SO NUMERO. A trava nao verifica direcao (alta vs. queda), unidade
  (R$ vs. USD, nivel vs. variacao) nem coerencia entre frases do mesmo
  corpo -- so se cada numero, isolado, tem alguma origem no nucleo.

Quem le esta docstring e conclui que nenhum numero fabricado passa esta
lendo mais protecao do que a que existe. A trava fecha o canal de digito
totalmente inventado (a maioria das alucinacoes reais); ela nao fecha
digito real com significado trocado.
"""
import re

from agro.types import RunResult

# Um numero, em cinco formas, na ordem em que precisam ser tentadas. As
# ancoras -- `[1-9]\d{0,2}` na frente dos grupos de milhar, `\d{3}` exato em
# cada grupo, e `(?![.,]?\d)` proibindo digito depois -- sao o que impede a
# alternativa de milhar de morder um decimal legitimo ("-1800.0" nao vira
# -180 e 0.0; "0.0134" e "3.2100" continuam decimais).
# `(?<![\w.,])` impede que o "-" de "2020-2025" ou de "2020-01-01" seja lido
# como sinal negativo, e impede tambem que a varredura comece no meio de um
# numero ja consumido.
_NUM = re.compile(
    r"""(?<![\w.,])-?(?:
          [1-9]\d{0,2}(?:\.\d{3})+,\d+       # 1.234,56  milhar BR + decimal
        | [1-9]\d{0,2}(?:,\d{3})+\.\d+       # 1,234.56  milhar EN + decimal
        | [1-9]\d{0,2}(?:\.\d{3})+(?![.,]?\d)  # 1.234     milhar BR
        | [1-9]\d{0,2}(?:,\d{3})+(?![.,]?\d)   # 1,234     milhar EN
        | \d+(?:[.,]\d+)?                    # 4.53 / 4,53 / 120
    )""",
    re.VERBOSE,
)

# Token com UM unico grupo de tres digitos: "2.600" pode ser 2600 (milhar) ou
# 2.600 (decimal). O `[1-9]` inicial e o que separa esse caso de "0.123", que
# nunca e milhar (grupo de milhar nao comeca com zero).
_MILHAR_DE_UM_GRUPO = re.compile(r"^[1-9]\d{0,2}([.,])\d{3}$")

# Nomes de familia de modelo conhecidos do projeto (ver `ESCADA_MODELOS` em
# `config.py`). A mascara de ordem de modelo so reconhece ESTES nomes -- um
# identificador generico ("aumento(50)", "IC(95)") nao e ordem de modelo, e
# restringir a mascara a nomes conhecidos fecha esse canal sem quebrar
# "GARCH(1,1)". `(?i:...)` porque o Redator pode escrever em qualquer caixa.
_NOME_MODELO = r"(?i:msgarch|garch|arima)"

# Formas RIGIDAS cujos numeros sao literais, nao grandezas a conferir. A
# rigidez e deliberada: mes 01-12, dia 01-31, ano 19xx/20xx, ordem de modelo
# com no maximo dois digitos e nome de familia conhecido -- nenhum valor
# arbitrario cabe aqui dentro.
_MASCARAS = tuple(re.compile(p) for p in (
    r"(?<!\d)(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])(?!\d)",
    r"(?<!\d)(?:0[1-9]|[12]\d|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}(?!\d)",
    r"(?<!\d)(?:0[1-9]|1[0-2])[-/](?:19|20)\d{2}(?!\d)",          # 03/2024
    r"(?<!\d)(?:19|20)\d{2}\s*[-–/]\s*(?:19|20)\d{2}(?!\d)",  # 2020-2025
    r"(?<!\d)(?:19|20)\d{2}\s*/\s*\d{2}(?!\d)",                    # safra 2024/25
    rf"\b{_NOME_MODELO}\(\s*\d{{1,2}}(?:\s*,\s*\d{{1,2}})*\s*\)",  # GARCH(1,1)
))

# Percentuais redondos que funcionam como ancora retorica em portugues --
# "nada mudou" (0%), "a metade" (50%), "tudo" (100%). A isencao so vale
# quando o proprio texto marca o numero como percentual; sem essa condicao um
# preco ou parametro real que por coincidencia vale 0, 50 ou 100 ficaria imune.
_PERCENTUAIS_RETORICOS = {0.0, 50.0, 100.0}

# Inteiros pequenos que aparecem como CONTAGEM em prosa tecnica ("2 regimes",
# "1 defasagem"). Regra simetrica a de cima, e no sentido inverso: isentos
# EXCETO quando seguidos de "%", porque ai deixam de ser contagem e viram
# afirmacao quantitativa ("cresceu 3%").
_INTEIROS_PEQUENOS = {1.0, 2.0, 3.0}

# Preposicao de data imediatamente antes do numero. Inclui "a" e "e" porque
# sao as duas construcoes idiomaticas de periodo em portugues ("de 2015 a
# 2024", "entre 2015 e 2024") -- sem elas, prosa correta sobre a janela
# amostral era derrubada por alarme falso, e o consumidor e um laco de nova
# tentativa com teto (falso positivo recorrente esgota tentativas). O "("
# cobre citacao academica ("Bollerslev (1986)"): o parenteses e contexto de
# data tanto quanto uma preposicao -- a correcao fica aqui, na lista de
# contexto, e nao na mascara de ordem de modelo (alargar a mascara para
# aceitar "Hamilton(1989)" abriria canal para qualquer literal disfarçado
# de citacao).
_PREPOSICAO_DE_DATA = re.compile(
    r"(?:(?<!\w)(?:em|desde|de|do|da|at[eé]|ao\s+longo\s+de|entre|para|no|na"
    r"|ao|a|e|ano|anos|safra|in[ií]cio|inicio|fim|dezembro|janeiro)\s+"
    r"|\(\s*)\Z",
    re.IGNORECASE,
)

# Simbolo de moeda imediatamente antes do numero: nao e ano, e preco.
_MOEDA = re.compile(r"(?:R\$|US\$|USD|BRL|\$)\s*\Z", re.IGNORECASE)

# Substantivo de coisa contada logo depois: "2000 observacoes" e contagem,
# nao ano. Sem isto, "uma serie de 2000 observacoes" reabriria o canal de
# contrabando por cima da preposicao "de".
_SUBSTANTIVO_CONTADO = re.compile(
    r"\A\s*(?:observ\w*|obs|dias?|meses|mes|semanas?|pontos?|amostras?|dados"
    r"|registros|reais|d[oó]lares|toneladas|sacas|contratos|vezes|unidades?"
    r"|passos?|defasagens?|regimes?)\b",
    re.IGNORECASE,
)

# Rotulos de `RunResult.valores_rotulados()` que o nucleo guarda como FRACAO
# DE 1 e o Redator escreve como percentual ("cobertura de 90%" para 0.90).
# Sao os unicos valores contra os quais a leitura /100 e tentada -- ver
# decisao 6 na docstring do modulo.
_ROTULOS_EM_FRACAO = frozenset({"backtest.cobertura_ic", "nivel_ic"})

# Folga numerica para nao perder um empate exato por representacao binaria
# (|4.535 - 4.54| sai 0.0050000000000004 em ponto flutuante).
_FOLGA_BINARIA = 1e-9


class NumeroInventado(RuntimeError):
    """O texto cita um numero que nao existe nos resultados do nucleo."""


class NumeroAmbiguo(RuntimeError):
    """O texto escreve um numero que tem mais de uma leitura autorizada.

    Acontece com separador de milhar de um grupo so ("2.600" = 2600 ou 2.6).
    Nao ha como decidir sem adivinhar, e adivinhar aqui significa trava e
    leitor lendo numeros diferentes -- entao a execucao para e pede reescrita.
    """


# "por cento" por extenso e portugues normal ("intervalo de 95 por cento"),
# tao percentual quanto o glifo "%" -- ver `_seguido_de_percentual`.
_POR_CENTO = re.compile(r"\s+por\s+cento\b", re.IGNORECASE)


def _seguido_de_percentual(fim: int, texto: str) -> bool:
    resto = texto[fim:fim + 2]
    if resto.startswith("%") or resto.startswith(" %"):
        return True
    return bool(_POR_CENTO.match(texto, fim))


def _leituras(token: str) -> tuple[list[tuple[float, int]], bool]:
    """Leituras possiveis de um token, como pares (valor, casas decimais).

    Devolve tambem se o token e ambiguo (mais de uma leitura possivel). Casas
    decimais e o que o texto AFIRMOU de precisao -- e o que fixa a tolerancia
    em `_casa_com_valor`.
    """
    sinal = -1.0 if token.startswith("-") else 1.0
    corpo = token.lstrip("-")
    tem_ponto, tem_virgula = "." in corpo, "," in corpo

    if tem_ponto and tem_virgula:
        # Os dois separadores presentes: o ultimo e o decimal, sem ambiguidade.
        decimal = "." if corpo.rfind(".") > corpo.rfind(",") else ","
        milhar = "," if decimal == "." else "."
        limpo = corpo.replace(milhar, "")
        casas = len(limpo.split(decimal)[1])
        return [(sinal * float(limpo.replace(",", ".")), casas)], False

    m = _MILHAR_DE_UM_GRUPO.match(corpo)
    if m:
        sep = m.group(1)
        return (
            [(sinal * float(corpo.replace(sep, "")), 0),      # 2.600 -> 2600
             (sinal * float(corpo.replace(sep, ".")), 3)],    # 2.600 -> 2.6
            True,
        )

    if tem_ponto or tem_virgula:
        sep = "." if tem_ponto else ","
        partes = corpo.split(sep)
        if len(partes) > 2:                 # 1.234.567: milhar, sem ambiguidade
            return [(sinal * float(corpo.replace(sep, "")), 0)], False
        return [(sinal * float(corpo.replace(",", ".")), len(partes[1]))], False

    return [(sinal * float(corpo), 0)], False


def extrair_numeros(texto: str) -> list[float]:
    """Extrai todo numero do texto, em ordem, sem aplicar nenhuma isencao.

    Isencoes (mascara, ano, percentual redondo, tolerancia) sao
    responsabilidade de `verificar_numeros` -- esta funcao e extracao pura,
    para quem precisar so dos numeros citados, sem julgar se sao validos.
    Token com separador de milhar ambiguo sai aqui na leitura de MILHAR (a
    unica leitura que a forma "2.600" tem quando lida isoladamente); a
    ambiguidade e resolvida, ou levantada, em `verificar_numeros`, que e quem
    conhece os valores autorizados.
    """
    numeros = []
    for m in _NUM.finditer(texto):
        leituras, _ = _leituras(m.group())
        numeros.append(leituras[0][0])
    return numeros


def _spans_mascarados(texto: str) -> list[tuple[int, int]]:
    """Trechos cujos numeros sao literais de data ou de ordem de modelo.

    A mascara so vale se o CONTEXTO ao redor do trecho inteiro nao denuncia
    que os numeros ali dentro sao uma grandeza de verdade, e nao uma data: os
    mesmos tres vetos usados para o ano solto (moeda antes, "%" depois,
    substantivo de coisa contada depois) tambem cancelam a mascara. Sem isso
    "2000-2010 observacoes", "2000/25 observacoes" e "10/2000 observacoes"
    sao o mesmo contrabando de "2000 observacoes" (ver `_eh_ano`), so que
    escapando por dentro da mascara em vez de pelo numero solto.
    """
    spans = []
    for padrao in _MASCARAS:
        for m in padrao.finditer(texto):
            inicio, fim = m.span()
            if _precedido_de_moeda(inicio, texto):
                continue
            if _seguido_de_percentual(fim, texto):
                continue
            if _SUBSTANTIVO_CONTADO.match(texto[fim:fim + 24]):
                continue
            spans.append((inicio, fim))
    return spans


def _eh_ano(valor: float, inicio: int, fim: int, texto: str) -> bool:
    """Um inteiro entre 1900 e 2100 SO e ano quando o contexto e de data.

    Sem essa exigencia a faixa e um canal de contrabando de 201 numeros --
    inclusive `n_obs` de qualquer serie diaria de alguns anos (ver decisao 2
    na docstring do modulo).
    """
    if not (valor.is_integer() and 1900 <= valor <= 2100):
        return False
    if _seguido_de_percentual(fim, texto):
        return False
    if _SUBSTANTIVO_CONTADO.match(texto[fim:fim + 24]):
        return False
    if _inicio_de_linha(inicio, texto):
        return True
    return bool(_PREPOSICAO_DE_DATA.search(texto, max(0, inicio - 12), inicio))


def _inicio_de_linha(inicio: int, texto: str) -> bool:
    """Ano em inicio de linha, sem preposicao antes, e prosa normal ("2024
    foi um ano de alta."). Conta o INICIO da linha, nao da frase -- so passa
    aqui o numero que nao tem nada alem de espaco em branco entre a quebra
    de linha anterior (ou o inicio do texto) e ele."""
    ultima_quebra = texto.rfind("\n", 0, inicio)
    return texto[ultima_quebra + 1:inicio].strip() == ""


def _eh_percentual_retorico(valor: float, fim: int, texto: str) -> bool:
    """0%, 50% e 100% sao ancoras retoricas (nada / metade / tudo), so
    isentas quando o texto de fato as marca como percentual."""
    return valor in _PERCENTUAIS_RETORICOS and _seguido_de_percentual(fim, texto)


def _eh_inteiro_pequeno_contado(valor: float, fim: int, texto: str) -> bool:
    """1, 2 e 3 aparecem como contagem em prosa tecnica ("2 regimes"). A
    isencao cai no instante em que o texto os marca como percentual."""
    return valor in _INTEIROS_PEQUENOS and not _seguido_de_percentual(fim, texto)


def _precedido_de_moeda(inicio: int, texto: str) -> bool:
    """Simbolo de moeda logo antes: o numero e PRECO, e preco vem do nucleo.

    O teste vale para as tres isencoes de prosa (ano, percentual retorico,
    inteiro pequeno) de uma vez, e nao so para o ano: "R$ 2" nao e a contagem
    "2 regimes", e "R$ 2000" nao e o ano de 2000. E a mesma logica do teste do
    "%": quando o texto MARCA o numero como grandeza, ele deixa de ser numero
    de prosa e precisa ter origem.
    """
    return bool(_MOEDA.search(texto, max(0, inicio - 8), inicio))


def _casa_com_valor(valor: float, casas: int, permitidos: list[float]) -> bool:
    """`valor`, escrito com `casas` decimais, e o arredondamento correto de
    algum valor autorizado nessa precisao?"""
    folga = 0.5 * (10.0 ** -casas)
    folga += folga * _FOLGA_BINARIA + 1e-12
    return any(abs(valor - p) <= folga for p in permitidos)


def _leitura_autorizada(valor: float, casas: int, inicio: int, fim: int,
                        texto: str, permitidos: list[float],
                        fracionarios: list[float]) -> bool:
    if not _precedido_de_moeda(inicio, texto):
        if _eh_ano(valor, inicio, fim, texto):
            return True
        if _eh_percentual_retorico(valor, fim, texto):
            return True
        if _eh_inteiro_pequeno_contado(valor, fim, texto):
            return True
    if _casa_com_valor(valor, casas, permitidos):
        return True
    # Escala de apresentacao: "90%" para cobertura_ic = 0.90. So com o "%"
    # escrito, e so contra as grandezas guardadas como fracao de 1 -- ver
    # decisao 6 na docstring do modulo.
    if _seguido_de_percentual(fim, texto):
        return _casa_com_valor(valor / 100.0, casas + 2, fracionarios)
    return False


def _trecho(texto: str, inicio: int, fim: int, folga: int = 40) -> str:
    """A frase ao redor do numero, para o Redator saber ONDE reescrever."""
    a, b = max(0, inicio - folga), min(len(texto), fim + folga)
    pedaco = " ".join(texto[a:b].split())
    return f"{'...' if a > 0 else ''}{pedaco}{'...' if b < len(texto) else ''}"


def verificar_numeros(texto: str, res: RunResult) -> None:
    """Confere todo numero do CORPO contra os resultados do nucleo.

    Levanta `NumeroAmbiguo` se algum numero tiver mais de uma leitura
    autorizada (separador de milhar), e `NumeroInventado` se algum numero nao
    tiver nenhuma. Nos dois casos TODOS os infratores vao na mesma mensagem:
    o consumidor e um laco de nova tentativa do Redator, e reportar um erro
    por vez custaria uma rodada de LLM por numero errado.

    `texto` deve ser o corpo escrito pelo LLM. `render_report` ja faz essa
    chamada com o corpo certo -- ver docstring do modulo, secao ESCOPO.
    """
    rotulados = res.valores_rotulados()
    permitidos = [v for _, v in rotulados]
    fracionarios = [v for r, v in rotulados if r in _ROTULOS_EM_FRACAO]
    spans = _spans_mascarados(texto)

    ambiguos: list[str] = []
    inventados: list[str] = []

    for m in _NUM.finditer(texto):
        inicio, fim = m.span()
        if any(a <= inicio and fim <= b for a, b in spans):
            continue

        leituras, ambiguo = _leituras(m.group())
        ok = [(v, k) for v, k in leituras
              if _leitura_autorizada(v, k, inicio, fim, texto,
                                     permitidos, fracionarios)]

        if len(ok) > 1 and ambiguo:
            valores = " ou ".join(f"{v:g}" for v, _ in ok)
            ambiguos.append(
                f"  - {m.group()!r} pode ser lido como {valores}, e as duas "
                f"leituras existem nos resultados, em: {_trecho(texto, inicio, fim)}"
            )
        elif not ok:
            valores = " / ".join(f"{v:g}" for v, _ in leituras)
            inventados.append(
                f"  - {m.group()!r} (lido como {valores}) em: "
                f"{_trecho(texto, inicio, fim)}"
            )

    if ambiguos:
        raise NumeroAmbiguo(
            f"{len(ambiguos)} numero(s) com separador de milhar ambiguo. "
            "Reescreva sem separador de milhar (ex.: 2600 em vez de 2.600):\n"
            + "\n".join(ambiguos)
        )

    if inventados:
        raise NumeroInventado(
            f"{len(inventados)} numero(s) do texto sem origem nos resultados "
            "do nucleo:\n" + "\n".join(inventados)
            + "\nValores autorizados: "
            + "; ".join(f"{rotulo}={valor:g}" for rotulo, valor in rotulados)
        )
