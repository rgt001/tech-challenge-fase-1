"""Carga e preparacao do dataset estruturado (Breast Cancer Wisconsin)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config


def carregar_dados(caminho=None) -> pd.DataFrame:
    """Le o CSV e remove colunas nao-informativas (id e coluna vazia)."""
    caminho = caminho or config.BREAST_CANCER_CSV
    df = pd.read_csv(caminho)
    return df.drop(columns=["id", "Unnamed: 32"], errors="ignore")


def preparar_treino_teste(df, test_size=0.20):
    """Separa X/y, codifica o alvo (M=1, B=0) e faz split estratificado."""
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].map({"M": 1, "B": 0})
    return train_test_split(
        X, y, test_size=test_size,
        random_state=config.RANDOM_STATE, stratify=y,
    )
