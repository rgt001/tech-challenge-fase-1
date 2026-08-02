"""Testes das metricas e da estrutura do CSV de radiomica (se existir)."""
import pandas as pd
import pytest

from techchallenge import config
from techchallenge.evaluation import metrics


def test_metricas_retorna_chaves():
    m = metrics.metricas([0, 1, 1, 0], [0, 1, 0, 0])
    assert set(m) == {"Accuracy", "Recall (maligno)", "F1 (maligno)"}
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_metricas_perfeitas():
    m = metrics.metricas([0, 1, 1, 0], [0, 1, 1, 0])
    assert m["Accuracy"] == 1.0
    assert m["Recall (maligno)"] == 1.0


def test_radiomica_csv_estrutura():
    csv = config.DATA_DIR / "radiomica_mamografia.csv"
    if not csv.exists():
        pytest.skip("radiomica_mamografia.csv ausente (gere com --extrair-radiomica)")
    df = pd.read_csv(csv, nrows=50)
    assert "label" in df.columns
    assert "split" in df.columns
    assert set(df["split"].unique()) <= {"train", "test"}
