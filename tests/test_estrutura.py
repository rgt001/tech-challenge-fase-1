"""Testes de fumaca: imports do pacote e caminhos essenciais."""
import importlib

from techchallenge import config


def test_config_caminhos():
    assert config.PROJETO_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.BREAST_CANCER_CSV.exists(), "data/breast_cancer.csv nao encontrado"


def test_imports_principais():
    # Modulos que NAO exigem torch (o modulo CNN e testado a parte / e opcional)
    modulos = [
        "techchallenge.pipeline",
        "techchallenge.__main__",
        "techchallenge.data.estruturado",
        "techchallenge.models.estruturado",
        "techchallenge.evaluation.metrics",
        "techchallenge.evaluation.limiar",
        "techchallenge.evaluation.explicabilidade",
        "techchallenge.cli.treinar_estruturado",
        "techchallenge.cli.prever_estruturado",
        "techchallenge.cli.treinar_radiomica",
    ]
    for m in modulos:
        importlib.import_module(m)
