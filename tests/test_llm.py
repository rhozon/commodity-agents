import pytest
from agents.llm import EsquemaInvalido, LLMFake, validar


ESQUEMA = {"familia": str, "justificativa": str}


def test_fake_devolve_respostas_em_ordem():
    f = LLMFake([{"familia": "msgarch", "justificativa": "a"},
                 {"familia": "garch", "justificativa": "b"}])
    assert f.perguntar("x", ESQUEMA)["familia"] == "msgarch"
    assert f.perguntar("x", ESQUEMA)["familia"] == "garch"


def test_fake_acaba_e_avisa():
    f = LLMFake([{"familia": "msgarch", "justificativa": "a"}])
    f.perguntar("x", ESQUEMA)
    with pytest.raises(AssertionError):
        f.perguntar("x", ESQUEMA)


def test_validar_aceita_campos_certos():
    validar({"familia": "msgarch", "justificativa": "porque sim"}, ESQUEMA)


def test_validar_rejeita_campo_faltando():
    with pytest.raises(EsquemaInvalido) as e:
        validar({"familia": "msgarch"}, ESQUEMA)
    assert "justificativa" in str(e.value)


def test_validar_rejeita_tipo_errado():
    with pytest.raises(EsquemaInvalido):
        validar({"familia": 3, "justificativa": "x"}, ESQUEMA)


def test_fake_valida_resposta_pre_programada_contra_esquema():
    """O LLMFake nao deve devolver uma resposta que o cliente real rejeitaria."""
    # Resposta inválida: campo faltando
    with pytest.raises(EsquemaInvalido):
        f = LLMFake([{"familia": "msgarch"}])
        f.perguntar("x", ESQUEMA)

    # Resposta inválida: tipo errado
    with pytest.raises(EsquemaInvalido):
        f = LLMFake([{"familia": 123, "justificativa": "b"}])
        f.perguntar("x", ESQUEMA)
