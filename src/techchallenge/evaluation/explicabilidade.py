"""Explicabilidade dos modelos estruturados (feature importance + SHAP)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def feature_importance_logistica(pipeline_logit, colunas, caminho):
    """Grafico dos coeficientes da Regressao Logistica."""
    coefs = pd.Series(pipeline_logit.named_steps["clf"].coef_[0],
                      index=colunas).sort_values()
    plt.figure(figsize=(8, 9))
    coefs.plot(kind="barh", color=np.where(coefs > 0, "#d1495b", "#4c9f70"))
    plt.title("Coeficientes da Regressao Logistica (vermelho=maligno)")
    plt.xlabel("Peso (dados padronizados)")
    plt.tight_layout()
    plt.savefig(caminho, dpi=110)
    plt.close()


def shap_arvore(pipeline_arvore, X_test, colunas, caminho):
    """Grafico summary SHAP para a Arvore de Decisao."""
    import shap
    tree = pipeline_arvore.named_steps["clf"]
    scaler = pipeline_arvore.named_steps["scaler"]
    X_scaled = pd.DataFrame(scaler.transform(X_test), columns=colunas, index=X_test.index)
    valores = shap.TreeExplainer(tree).shap_values(X_scaled)
    sv = valores[:, :, 1] if isinstance(valores, np.ndarray) and valores.ndim == 3 else valores
    plt.figure()
    shap.summary_plot(sv, X_scaled, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()
