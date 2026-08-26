"""Testes para contratos de tipos em agro.types."""
import pytest
from agro.types import (
    Backtest,
    Commodity,
    Diagnosis,
    ModelFit,
    RunResult,
    SeriesBundle,
)


@pytest.fixture
def bundle():
    """SeriesBundle com valores distintos e testáveis."""
    return SeriesBundle(
        commodity="milho",
        inicio="2015-01-01",
        fim="2024-12-31",
        colunas=["cbot", "cepea", "usdbrl"],
        n_obs=2520,
        caminho_parquet="cache/milho.parquet",
    )


@pytest.fixture
def fit():
    """ModelFit com parametros distintos."""
    return ModelFit(
        familia="arima",
        convergiu=True,
        parametros={"ar1": 0.5, "ma1": 0.3, "sigma2": 0.02},
        log_lik=-1234.5,
        aic=2475.0,
    )


@pytest.fixture
def diagnosis():
    """Diagnosis com testes de valores distintos."""
    return Diagnosis(
        aprovado=True,
        motivos=[],
        testes={"ljung_box_pval": 0.15, "adf_pval": 0.001},
    )


@pytest.fixture
def backtest():
    """Backtest com valores conhecidos."""
    return Backtest(
        horizonte=10,
        mape=8.5,
        rmse=12.3,
        cobertura_ic=0.92,
        mape_baseline=11.2,
        rmse_baseline=15.7,
    )


@pytest.fixture
def result_com_backtest(bundle, fit, diagnosis, backtest):
    """RunResult completo com backtest."""
    return RunResult(
        commodity="milho",
        pergunta="Qual sera o preco medio em 10 dias?",
        bundle=bundle,
        fit=fit,
        diagnosis=diagnosis,
        backtest=backtest,
        tentativas=3,
        teto_estourado=False,
        numeros={"preco_medio": 550.25, "desvio": 45.8},
    )


@pytest.fixture
def result_sem_backtest(bundle, fit, diagnosis):
    """RunResult sem backtest."""
    return RunResult(
        commodity="soja",
        pergunta="Tendencia de preco?",
        bundle=bundle,
        fit=fit,
        diagnosis=diagnosis,
        backtest=None,
        tentativas=1,
        teto_estourado=False,
        numeros={"preco_inicial": 600.0},
    )


class TestRunResultValoresPermitidos:
    """Testes para RunResult.valores_permitidos()."""

    def test_inclui_numeros(self, result_com_backtest):
        """valores_permitidos() inclui todos os valores de numeros."""
        permitidos = result_com_backtest.valores_permitidos()
        assert 550.25 in permitidos
        assert 45.8 in permitidos

    def test_inclui_fit_parametros(self, result_com_backtest):
        """valores_permitidos() inclui todos os parametros do fit."""
        permitidos = result_com_backtest.valores_permitidos()
        assert 0.5 in permitidos  # ar1
        assert 0.3 in permitidos  # ma1
        assert 0.02 in permitidos  # sigma2

    def test_inclui_diagnosis_testes(self, result_com_backtest):
        """valores_permitidos() inclui todos os testes da diagnosis."""
        permitidos = result_com_backtest.valores_permitidos()
        assert 0.15 in permitidos  # ljung_box_pval
        assert 0.001 in permitidos  # adf_pval

    def test_inclui_backtest_quando_existe(self, result_com_backtest):
        """valores_permitidos() inclui valores do backtest quando presente."""
        permitidos = result_com_backtest.valores_permitidos()
        assert 8.5 in permitidos  # mape
        assert 12.3 in permitidos  # rmse
        assert 0.92 in permitidos  # cobertura_ic
        assert 11.2 in permitidos  # mape_baseline
        assert 15.7 in permitidos  # rmse_baseline
        assert 10.0 in permitidos  # horizonte como float

    def test_inclui_bundle_n_obs_como_float(self, result_com_backtest):
        """valores_permitidos() inclui bundle.n_obs como float."""
        permitidos = result_com_backtest.valores_permitidos()
        assert 2520.0 in permitidos

    def test_funciona_sem_backtest(self, result_sem_backtest):
        """valores_permitidos() funciona quando backtest é None."""
        permitidos = result_sem_backtest.valores_permitidos()
        # Deve incluir numeros
        assert 600.0 in permitidos
        # Deve incluir fit parametros
        assert 0.5 in permitidos  # ar1
        # Deve incluir diagnosis testes
        assert 0.15 in permitidos  # ljung_box_pval
        # Deve incluir bundle.n_obs
        assert 2520.0 in permitidos
        # Não deve incluir valores de backtest (que é None)

    def test_todos_valores_sao_float(self, result_com_backtest):
        """Todos os valores em valores_permitidos() devem ser float."""
        permitidos = result_com_backtest.valores_permitidos()
        for valor in permitidos:
            assert isinstance(valor, float)

    def test_valores_distintos_mantidos(self, result_com_backtest):
        """Valores distintos não são duplicados."""
        result = result_com_backtest
        # Contar quantos valores distintos esperamos
        expected_count = (
            len(result.numeros)  # 2
            + len(result.fit.parametros)  # 3
            + len(result.diagnosis.testes)  # 2
            + 6  # backtest: mape, rmse, cobertura_ic, horizonte(float), mape_baseline, rmse_baseline
            + 1  # bundle.n_obs
        )
        permitidos = result.valores_permitidos()
        # Sem colisões, esperamos exatamente esse número
        assert len(permitidos) == expected_count


class TestBacktestBateuBaseline:
    """Testes para Backtest.bateu_baseline."""

    def test_bateu_baseline_quando_mape_menor(self):
        """bateu_baseline é True quando mape < mape_baseline."""
        b = Backtest(
            horizonte=10,
            mape=5.0,
            rmse=8.0,
            cobertura_ic=0.9,
            mape_baseline=10.0,
            rmse_baseline=12.0,
        )
        assert b.bateu_baseline is True

    def test_nao_bateu_baseline_quando_mape_maior(self):
        """bateu_baseline é False quando mape > mape_baseline."""
        b = Backtest(
            horizonte=10,
            mape=15.0,
            rmse=18.0,
            cobertura_ic=0.9,
            mape_baseline=10.0,
            rmse_baseline=12.0,
        )
        assert b.bateu_baseline is False

    def test_nao_bateu_baseline_quando_mape_igual(self):
        """bateu_baseline é False quando mape == mape_baseline."""
        b = Backtest(
            horizonte=10,
            mape=10.0,
            rmse=12.0,
            cobertura_ic=0.9,
            mape_baseline=10.0,
            rmse_baseline=12.0,
        )
        assert b.bateu_baseline is False

    def test_bateu_baseline_com_numeros_pequenos(self):
        """bateu_baseline funciona com numeros muito pequenos."""
        b = Backtest(
            horizonte=1,
            mape=0.001,
            rmse=0.002,
            cobertura_ic=0.95,
            mape_baseline=0.002,
            rmse_baseline=0.003,
        )
        assert b.bateu_baseline is True

    def test_bateu_baseline_nao_dependente_rmse(self):
        """bateu_baseline depende apenas de mape, não de rmse."""
        # rmse baixo, mape alto -> False
        b1 = Backtest(
            horizonte=10,
            mape=15.0,
            rmse=5.0,  # rmse baixo
            cobertura_ic=0.9,
            mape_baseline=10.0,
            rmse_baseline=8.0,
        )
        assert b1.bateu_baseline is False

        # rmse alto, mape baixo -> True
        b2 = Backtest(
            horizonte=10,
            mape=5.0,
            rmse=15.0,  # rmse alto
            cobertura_ic=0.9,
            mape_baseline=10.0,
            rmse_baseline=12.0,
        )
        assert b2.bateu_baseline is True
