import pandas as pd
import pytest

from agro import guard, report
from agro.types import (
    Backtest,
    CorpoRelatorio,
    Diagnosis,
    ModelFit,
    RunResult,
    SeriesBundle,
)


@pytest.fixture
def resultado(tmp_path):
    p = tmp_path / "s.parquet"
    pd.DataFrame({"cbot": [1, 2, 3]}).to_parquet(p)
    bundle = SeriesBundle("milho", "2020-01-01", "2020-06-16", ["cbot"], 120, str(p))
    return RunResult(
        commodity="milho", pergunta="?", bundle=bundle,
        fit=ModelFit("msgarch", True, {"alpha0_1": 0.0134}, 900.0, -1800.0),
        diagnosis=Diagnosis(True, [], {"n_retornos": 119.0}),
        backtest=Backtest(20, 4.53, 3.21, 0.90, 5.10, 3.80), tentativas=1, teto_estourado=False,
        numeros={"preco_atual": 62.75},
    )


def test_extrai_numeros_de_varios_formatos():
    n = guard.extrair_numeros("MAPE de 4,53% e RMSE 3.21 com 120 obs e -1800.0 de AIC")
    assert 4.53 in n and 3.21 in n and 120.0 in n and -1800.0 in n


def test_texto_so_com_numeros_conhecidos_passa(resultado):
    guard.verificar_numeros("O MAPE foi de 4.53% e o preco atual e 62.75.", resultado)


def test_numero_inventado_derruba(resultado):
    with pytest.raises(guard.NumeroInventado) as e:
        guard.verificar_numeros("O preco vai subir 37.9% no trimestre.", resultado)
    assert "37.9" in str(e.value)


def test_ano_e_percentual_redondo_sao_ignorados(resultado):
    guard.verificar_numeros("Em 2024 o mercado mudou; a alta foi de 100%.", resultado)


def test_tolerancia_aceita_arredondamento(resultado):
    guard.verificar_numeros("MAPE de 4.5%.", resultado)


def test_numero_em_formato_brasileiro_e_reconhecido(resultado):
    # 4,53 (formato BR) deve casar com o mape 4.53 ja permitido.
    guard.verificar_numeros("O MAPE foi de 4,53%.", resultado)


def test_numero_a_uma_unidade_da_casa_escrita_derruba(resultado):
    # ESTE TESTE FOI INVERTIDO (era `..._dentro_da_tolerancia_passa`).
    # A tolerancia deixou de ser relativa (1%) e passou a depender da
    # precisao que o proprio texto escreveu: escrever "62.76" afirma duas
    # casas decimais, e 62.75 arredondado em duas casas e 62.75, nao 62.76.
    # Aceitar isso era aceitar um digito fabricado na ultima casa -- exatamente
    # o que a trava existe para impedir. A assercao foi APERTADA, nao afrouxada.
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O preco fechou em 62.76.", resultado)


def test_numero_proximo_mas_fora_da_tolerancia_derruba(resultado):
    # 65.0 escreve uma casa decimal; 62.75 arredondado em uma casa e 62.8.
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O preco fechou em 65.0.", resultado)


def test_valor_pequeno_perto_de_zero_nao_aceita_qualquer_coisa_por_tolerancia_relativa(resultado):
    # alpha0_1 = 0.0134 e permitido. Uma tolerancia puramente relativa
    # (0.0134 * 1%) deixaria passar qualquer numero pequeno por a distancia
    # absoluta ser minuscula -- mas 0.05 nao e o arredondamento de 0.0134 em
    # duas casas (que e 0.01) e deve ser rejeitado.
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O parametro estimado foi 0.05.", resultado)


def test_trava_valida_corpo_do_redator_nao_o_markdown_montado(resultado):
    """Limite de escopo: `verificar_numeros` roda sobre o texto do LLM, nao
    sobre a saida de `render_report`. A moldura (numero de tentativas, por
    exemplo) e determinística e nao esta em `valores_permitidos()` -- se a
    trava fosse aplicada ao markdown final, esse numero causaria falso
    positivo."""
    # 7 tentativas (acima do teto normal) de proposito: 1, 2 e 3 agora sao
    # isentos como inteiros pequenos (ver IMPORTANTE 6), entao um contador
    # baixo nao demonstraria mais a propriedade. A propriedade testada e a
    # mesma: numero da MOLDURA nao esta em valores_permitidos().
    resultado.tentativas = 7
    corpo = CorpoRelatorio(previsao="O MAPE foi de 4.53% e o preco atual e 62.75.",
                           drivers="Sem numero novo aqui.",
                           implicacao="Nem aqui.")
    # O corpo isolado passa -- as tres camadas juntas, que e o que a trava ve.
    guard.verificar_numeros(corpo.texto_completo(), resultado)

    md = report.render_report(resultado, corpo)
    # O markdown final contem "Tentativas: 7." (numero de tentativas), que
    # nao esta em valores_permitidos() -- aplicar a trava a ele derrubaria
    # a execucao por um numero que e correto, so que vem do codigo, nao do LLM.
    assert "Tentativas: 7." in md
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros(md, resultado)


# ---------------------------------------------------------------------------
# Endurecimento da trava (revisao que rodou 26 tentativas de burla).
# Cada teste abaixo nasceu VERMELHO contra o codigo anterior, e cada um prende
# UMA decisao de desenho: revertida a decisao, o teste quebra.
# ---------------------------------------------------------------------------

@pytest.fixture
def parquet(tmp_path):
    p = tmp_path / "s.parquet"
    pd.DataFrame({"cbot": [1, 2, 3]}).to_parquet(p)
    return str(p)


def _fabricar(parquet, *, n_obs=120, numeros=None, parametros=None,
              testes=None, backtest=None, tentativas=1):
    """RunResult sob medida: cada burla precisa do seu proprio cenario."""
    bundle = SeriesBundle("milho", "2020-01-01", "2020-06-16", ["cbot"], n_obs, parquet)
    return RunResult(
        commodity="milho", pergunta="?", bundle=bundle,
        fit=ModelFit("msgarch", True, parametros or {"alpha0_1": 0.0134}, 900.0, -1800.0),
        diagnosis=Diagnosis(True, [], testes or {"n_retornos": 119.0}),
        backtest=backtest or Backtest(20, 4.53, 3.21, 0.90, 5.10, 3.80),
        tentativas=tentativas, teto_estourado=False,
        numeros=numeros if numeros is not None else {"preco_atual": 62.75},
    )


# --- CRITICO 1: a faixa de ano nao pode ser canal de contrabando -----------

def test_ano_nao_isenta_contagem_de_observacoes(parquet):
    """n_obs de serie diaria longa cai dentro de 1900..2100. Sem exigir
    contexto de data, o Redator troca 1987 por 2000 e a trava nao ve."""
    res = _fabricar(parquet, n_obs=1987)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("A amostra tem 2000 observacoes.", res)


def test_ano_precedido_de_preposicao_mas_seguido_de_substantivo_contado(parquet):
    """'de 2000 observacoes' tem preposicao de data na frente, mas o
    substantivo depois denuncia que o numero e uma contagem, nao um ano."""
    res = _fabricar(parquet, n_obs=1987)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("Trabalhamos com uma serie de 2000 observacoes.", res)


def test_ano_seguido_de_percentual_nao_e_isento(parquet):
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("A alta foi de 2000%.", res)


def test_ano_precedido_de_moeda_nao_e_isento(parquet):
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O contrato fechou a R$ 2000.", res)


def test_inteiro_pequeno_precedido_de_moeda_nao_e_isento(parquet):
    """Simbolo de moeda MARCA o numero como grandeza, do mesmo jeito que o
    '%': 'R$ 2' e preco, nao a contagem '2 regimes'. Sem este teste, o teste
    do simbolo de moeda ficaria sem nada que o prendesse -- a exigencia de
    preposicao de data ja rejeita 'R$ 2000' por outro caminho."""
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O contrato fechou a R$ 2.", res)


def test_inteiro_na_faixa_de_ano_sem_contexto_de_data_nao_e_isento(parquet):
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O volume negociado somou 1950.", res)


def test_ano_com_preposicao_de_data_continua_isento(parquet):
    """A isencao existe e precisa continuar existindo: prosa sobre a janela
    amostral cita ano o tempo todo."""
    res = _fabricar(parquet)
    guard.verificar_numeros("Em 2024 o mercado mudou; desde 2015 nao se via isso.", res)


def test_periodo_com_a_e_e_e_isento(parquet):
    """'de 2015 a 2024' e 'entre 2015 e 2024' sao as duas construcoes
    idiomaticas de periodo em portugues. Sem 'a' e 'e' na lista de
    preposicao de data, um relatorio correto e derrubado por alarme falso
    -- e o consumidor e um laco de nova tentativa com teto."""
    res = _fabricar(parquet)
    guard.verificar_numeros("A serie vai de 2015 a 2024.", res)
    guard.verificar_numeros("Entre 2015 e 2024 o mercado mudou.", res)


def test_ano_no_inicio_da_linha_e_isento(parquet):
    """Ano em inicio de frase, sem preposicao antes, e prosa normal."""
    res = _fabricar(parquet)
    guard.verificar_numeros("2024 foi um ano de alta.", res)


def test_ano_em_citacao_academica_e_isento(parquet):
    """'Bollerslev (1986)' e citacao, nao dado do nucleo: o ano dentro do
    parenteses precisa ser isento sem alargar a mascara de ordem de modelo
    (isso abriria o canal de 'Hamilton(1989)' como literal generico)."""
    res = _fabricar(parquet)
    guard.verificar_numeros("Bollerslev (1986) propos o GARCH.", res)


# --- CRITICO 2: tolerancia dependente da precisao escrita ------------------

def test_pvalor_com_erro_na_terceira_casa_derruba(parquet):
    """0.046 rejeita a premissa, 0.051 nao. O piso absoluto de 0.005 deixava
    a conclusao do relatorio se inverter em silencio."""
    res = _fabricar(parquet, testes={"arch_lm_pvalor": 0.046})
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O teste ARCH-LM deu p-valor de 0.051.", res)


def test_parametro_com_erro_de_27_por_cento_derruba(parquet):
    res = _fabricar(parquet, parametros={"alpha0_1": 0.0134})
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O parametro estimado foi 0.0170.", res)


def test_arredondamento_na_casa_escrita_passa(parquet):
    """0.013 e o arredondamento correto de 0.0134 em tres casas -- a
    degeneracao perto de zero que motivava o piso absoluto continua coberta,
    agora sem o piso."""
    res = _fabricar(parquet, parametros={"alpha0_1": 0.0134})
    guard.verificar_numeros("O parametro estimado foi 0.013.", res)


# --- IMPORTANTE 3: mascaras de data composta nao pulam os vetos ------------

def test_intervalo_de_anos_mascarado_seguido_de_contagem_nao_e_isento(parquet):
    """'2000-2010 observacoes' e o mesmo contrabando de
    'test_ano_nao_isenta_contagem_de_observacoes', so que disfarcado de
    intervalo de anos. A mascara de intervalo pula o veto de substantivo
    contado se aplicada sem checar o que vem depois do trecho inteiro."""
    res = _fabricar(parquet, n_obs=1987)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("A amostra tem 2000-2010 observacoes.", res)


def test_safra_mascarada_seguida_de_contagem_nao_e_isenta(parquet):
    res = _fabricar(parquet, n_obs=1987)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("A amostra cobre 2000/25 observacoes.", res)


def test_mes_ano_mascarado_seguido_de_contagem_nao_e_isento(parquet):
    res = _fabricar(parquet, n_obs=1987)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("A amostra tem 10/2000 observacoes.", res)


# --- IMPORTANTE 4: sinal de menos sem lookbehind ---------------------------

def test_intervalo_de_anos_nao_vira_numero_negativo(parquet):
    res = _fabricar(parquet)
    guard.verificar_numeros("A serie cobre o periodo 2020-2025 sem falhas.", res)


def test_data_iso_nao_vira_numero_negativo(parquet):
    res = _fabricar(parquet)
    guard.verificar_numeros("De 2020-01-01 ate 2024-12-31 nao houve falha.", res)


def test_menos_colado_em_digito_nao_e_sinal():
    assert guard.extrair_numeros("periodo 2020-2025") == [2020.0, 2025.0]


# --- IMPORTANTE 5: separador de milhar -------------------------------------

def test_separador_de_milhar_reconhece_numero_correto(parquet):
    res = _fabricar(parquet, n_obs=2600)
    guard.verificar_numeros("A serie tem 2.600 observacoes.", res)


def test_valor_monetario_com_milhar_e_decimal_nao_racha():
    assert guard.extrair_numeros("O contrato vale R$ 1.234,56 hoje.") == [1234.56]


def test_negativo_de_quatro_digitos_nao_racha():
    assert guard.extrair_numeros("AIC de -1800.0 no ajuste") == [-1800.0]


def test_milhar_ambiguo_com_duas_leituras_autorizadas_falha_alto(parquet):
    """2.600 pode ser 2600 (milhar) ou 2.6 (decimal). Quando as DUAS leituras
    existem nos resultados, interpretar em silencio e escolher por conta
    propria qual numero o leitor vai ver."""
    res = _fabricar(parquet, n_obs=2600, numeros={"preco_atual": 2.6})
    with pytest.raises(guard.NumeroAmbiguo):
        guard.verificar_numeros("A serie tem 2.600 observacoes.", res)


# --- IMPORTANTE 6: inteiros pequenos, simetricos ao percentual retorico ----

def test_contagem_de_regimes_nao_derruba(parquet):
    res = _fabricar(parquet)
    guard.verificar_numeros("O modelo identificou 2 regimes de volatilidade.", res)


def test_assinatura_de_ordem_do_modelo_nao_derruba(parquet):
    res = _fabricar(parquet)
    guard.verificar_numeros("Ajustamos um GARCH(1,1) sobre os retornos.", res)


def test_ordem_de_modelo_com_tres_digitos_nao_e_mascarada(parquet):
    """A docstring afirma que o teto de dois digitos por posicao e o que
    impede a mascara de virar canal. '123' nao e uma ordem de modelo
    plausivel e nao esta nos resultados -- tem que derrubar."""
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("Ajustamos um GARCH(123) sobre os retornos.", res)


def test_mascara_de_ordem_nao_aceita_nome_arbitrario(parquet):
    """A mascara de ordem de modelo e restrita aos nomes de familia do
    projeto (msgarch/garch/arima) -- 'aumento(50)', 'IC(95)' e 'alta(37)'
    nao sao ordem de modelo, sao numero de prosa disfarcado de assinatura,
    e tem que ser conferido normalmente contra os resultados."""
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O aumento(50) foi observado no periodo.", res)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O IC(95) ficou acima do esperado.", res)


def test_inteiro_pequeno_seguido_de_percentual_derruba(parquet):
    """O inverso exato do teste aplicado a 0, 50 e 100: 'subiu 2%' e uma
    afirmacao quantitativa e precisa vir do nucleo."""
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O preco subiu 2% no mes.", res)


def test_percentual_retorico_sem_o_simbolo_derruba(parquet):
    """0, 50 e 100 so sao isentos quando o texto os marca como percentual."""
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O preco fechou em 50.", res)


# --- IMPORTANTE 7: conjunto autorizado ------------------------------------

def test_aic_e_log_lik_sao_citaveis(parquet):
    res = _fabricar(parquet)
    guard.verificar_numeros("O AIC foi -1800.0 e a log-verossimilhanca 900.0.", res)


def test_cobertura_em_escala_percentual_passa(parquet):
    """cobertura_ic vale 0.90; o Redator escreve '90%'."""
    res = _fabricar(parquet)
    guard.verificar_numeros("A cobertura do intervalo foi de 90%.", res)


def test_nivel_do_intervalo_e_citavel(parquet):
    res = _fabricar(parquet)
    guard.verificar_numeros("Usamos um intervalo de 95%.", res)


def test_percentual_por_extenso_e_reconhecido(parquet):
    """'por cento' por extenso e portugues normal -- so o glifo '%' era
    reconhecido, e 'intervalo de 95 por cento' barrava um relatorio
    correto."""
    res = _fabricar(parquet)
    guard.verificar_numeros("Usamos um intervalo de 95 por cento.", res)


def test_escala_percentual_barra_vizinhos_de_um_ponto(parquet):
    """A leitura /100 e comparada com a MESMA tolerancia por precisao escrita
    das outras leituras (`casas`, nao `casas + 2`). Sem isso, a folga de duas
    casas extras faz o comparador aceitar qualquer inteiro de 40% a 145% para
    cobertura_ic=0.90 e nivel_ic=0.95 -- 106 inteiros livres em vez dos dois
    unicos autorizados. "89%" e "91%" tem que continuar barrando."""
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("A cobertura do intervalo foi de 89%.", res)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("A cobertura do intervalo foi de 91%.", res)


def test_escala_percentual_nao_casa_percentual_com_pvalor(parquet):
    """A leitura /100 so vale contra grandezas guardadas como fracao de 1
    (cobertura do intervalo, nivel do intervalo). Sem essa restricao, um MAPE
    inventado de 4.6% casaria com o p-valor 0.046 do ARCH-LM -- duas
    grandezas sem relacao nenhuma uma com a outra."""
    res = _fabricar(parquet, testes={"arch_lm_pvalor": 0.046})
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O MAPE ficou em 4.6%.", res)


def test_escala_percentual_so_vale_com_o_simbolo(parquet):
    """A leitura /100 e uma questao de escala de apresentacao, nao uma
    autorizacao nova: sem o '%' o numero 90 continua sem origem."""
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O preco fechou em 90.", res)


# --- IMPORTANTE 8: a trava e chamada pelo proprio render_report -----------

def test_render_report_recusa_corpo_com_numero_inventado(parquet):
    res = _fabricar(parquet)
    with pytest.raises(guard.NumeroInventado):
        report.render_report(res, CorpoRelatorio(
            previsao="O preco vai subir 37.9% no trimestre.",
            drivers="", implicacao=""))


# --- MENOR: contratos documentados sem teste --------------------------

def test_contencao_de_span_mascarado_nao_e_sobreposicao(parquet, monkeypatch):
    """`verificar_numeros` so pula um numero que esteja TOTALMENTE contido
    num span mascarado -- contencao (`a <= inicio and fim <= b`), nao
    sobreposicao. Um numero que so toca a borda do span, comecando antes
    dele, tem que continuar sendo conferido normalmente contra os
    resultados. Usa um span mascarado sintetico via monkeypatch porque as
    mascaras reais tem lookaround que impede construir esse caso por
    texto."""
    res = _fabricar(parquet)
    texto = "O valor foi 999 no periodo."
    inicio_999 = texto.index("999")
    fim_999 = inicio_999 + 3
    # Span que SOBREPOE "999" sem conte-lo: comeca depois do inicio do
    # token e termina depois do fim dele.
    span_sobreposto = (inicio_999 + 1, fim_999 + 5)
    monkeypatch.setattr(guard, "_spans_mascarados", lambda t: [span_sobreposto])
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros(texto, res)


def test_milhar_de_um_grupo_nao_comeca_com_zero(parquet):
    """Grupo de milhar nunca comeca com zero (`[1-9]` inicial de
    `_MILHAR_DE_UM_GRUPO`): '0.123' e sempre um decimal, nunca as duas
    leituras 123 (milhar) e 0.123 (decimal). Sem essa ancora, o token
    viraria ambiguo por engano quando as duas leituras coincidissem com
    valores autorizados."""
    res = _fabricar(parquet, numeros={"preco_atual": 0.123, "outro": 123.0})
    guard.verificar_numeros("O indicador ficou em 0.123.", res)


def test_extrair_numeros_ambiguo_devolve_leitura_de_milhar_primeiro():
    """'2.600' isolado (sem contexto que resolva a ambiguidade) sai como
    leitura de MILHAR (2600) -- a unica leitura que a forma tem quando lida
    sozinha, ver docstring de `extrair_numeros` -- e nao a leitura decimal
    (2.6, que seria `leituras[-1]`)."""
    assert guard.extrair_numeros("A serie tem 2.600 observacoes.") == [2600.0]


# --- MENOR 9: mensagem de erro utilizavel pelo laco de nova tentativa -----

def test_erro_lista_todos_os_infratores_com_rotulo_e_trecho(parquet):
    res = _fabricar(parquet)
    # Os dois infratores ficam longe um do outro de proposito: com eles na
    # mesma frase, o TRECHO do primeiro ja conteria o segundo e o teste
    # passaria mesmo se a trava abortasse no primeiro infrator.
    corpo = (
        "O preco vai a 37.9 nas proximas semanas, segundo a leitura que o "
        "modelo faz da serie completa, e nada indica reversao no curto prazo. "
        "Ja o MAPE seria 88.4 no cenario alternativo."
    )
    with pytest.raises(guard.NumeroInventado) as e:
        guard.verificar_numeros(corpo, res)
    msg = str(e.value)
    assert "37.9" in msg and "88.4" in msg          # todos de uma vez
    assert "backtest.mape" in msg                    # valores rotulados
    assert "MAPE seria 88.4" in msg                  # trecho onde apareceu
