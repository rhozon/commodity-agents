"""Julga o ajuste. Nao usa LLM: criterio de aceitacao e codigo, nao opiniao.

`models.diagnose` roda Ljung-Box e ARCH-LM (entre outros testes) e devolve
sempre um motivo escrito quando reprova. Um critico que "acha" que o ajuste
esta bom e teatro; este roda teste e reprova com motivo, ponto final.
"""
import pandas as pd

from agro import models
from agro.types import Diagnosis, ModelFit


class Critico:
    def julgar(self, fit: ModelFit, serie: pd.Series) -> Diagnosis:
        return models.diagnose(fit, serie)
