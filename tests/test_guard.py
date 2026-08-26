import pandas as pd
import pytest

from agro import guard, report
from agro.types import Backtest, Diagnosis, ModelFit, RunResult, SeriesBundle


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


def test_numero_muito_proximo_de_permitido_dentro_da_tolerancia_passa(resultado):
    # 62.75 e permitido (preco_atual); 62.76 fica dentro da tolerancia relativa padrao (1%).
    guard.verificar_numeros("O preco fechou em 62.76.", resultado)


def test_numero_proximo_mas_fora_da_tolerancia_derruba(resultado):
    # 62.75 * 1.01 ~= 63.38; 65.0 esta bem fora da tolerancia de 1%.
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O preco fechou em 65.0.", resultado)


def test_valor_pequeno_perto_de_zero_nao_aceita_qualquer_coisa_por_tolerancia_relativa(resultado):
    # alpha0_1 = 0.0134 e permitido. Sem piso absoluto na tolerancia, uma
    # tolerancia puramente relativa (0.0134 * 1%) deixaria passar qualquer
    # numero pequeno por a distancia absoluta ser minuscula -- mas 0.05 e
    # muito distante de 0.0134 em termos absolutos e deve ser rejeitado.
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros("O parametro estimado foi 0.05.", resultado)


def test_trava_valida_corpo_do_redator_nao_o_markdown_montado(resultado):
    """Limite de escopo: `verificar_numeros` roda sobre o texto do LLM, nao
    sobre a saida de `render_report`. A moldura (numero de tentativas, por
    exemplo) e determinística e nao esta em `valores_permitidos()` -- se a
    trava fosse aplicada ao markdown final, esse numero causaria falso
    positivo."""
    corpo = "O MAPE foi de 4.53% e o preco atual e 62.75."
    # O corpo isolado passa.
    guard.verificar_numeros(corpo, resultado)

    md = report.render_report(resultado, corpo)
    # O markdown final contem "Tentativas: 1." (numero de tentativas), que
    # nao esta em valores_permitidos() -- aplicar a trava a ele derrubaria
    # a execucao por um numero que e correto, so que vem do codigo, nao do LLM.
    assert "Tentativas: 1." in md
    with pytest.raises(guard.NumeroInventado):
        guard.verificar_numeros(md, resultado)
