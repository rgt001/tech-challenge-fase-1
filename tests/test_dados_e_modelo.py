"""Testes do dado estruturado e de um treino rapido (garante que o ML roda)."""
from sklearn.metrics import accuracy_score

from techchallenge.data import estruturado as dados
from techchallenge.models import estruturado as modelos_estr


def test_carregar_dados():
    df = dados.carregar_dados()
    assert df.shape[0] == 569, "esperado 569 casos no Breast Cancer Wisconsin"
    assert "diagnosis" in df.columns
    assert "id" not in df.columns  # coluna nao-informativa deve ter sido removida


def test_split_estratificado():
    df = dados.carregar_dados()
    X_train, X_test, y_train, y_test = dados.preparar_treino_teste(df)
    assert len(X_train) + len(X_test) == len(df)
    assert X_train.shape[1] == 30  # 30 features
    assert set(y_train.unique()) <= {0, 1}
    assert set(y_test.unique()) <= {0, 1}


def test_treino_rapido_atinge_boa_acuracia():
    df = dados.carregar_dados()
    X_train, X_test, y_train, y_test = dados.preparar_treino_teste(df)
    modelo = modelos_estr.construir_modelos()["Regressao Logistica"]
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    assert acc >= 0.90, f"acuracia baixa demais ({acc:.3f}) - algo quebrou no modelo"


def test_stacking_constroi():
    stk = modelos_estr.construir_stacking(modelos_estr.construir_modelos())
    assert hasattr(stk, "fit") and hasattr(stk, "predict_proba")
