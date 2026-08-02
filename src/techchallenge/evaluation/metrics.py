"""Metricas e graficos de avaliacao (comuns aos dois pipelines)."""
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)


def metricas(y_true, y_pred):
    """Dict com accuracy, recall (classe 1) e F1 (classe 1)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall (maligno)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (maligno)": f1_score(y_true, y_pred, zero_division=0),
    }


def relatorio_texto(y_true, y_pred, titulo):
    m = metricas(y_true, y_pred)
    return (f"{titulo}\n"
            f"Accuracy: {m['Accuracy']:.4f}\n"
            f"Recall (maligno): {m['Recall (maligno)']:.4f}\n"
            f"F1 (maligno): {m['F1 (maligno)']:.4f}\n\n"
            + classification_report(y_true, y_pred,
                                    target_names=["Benigno", "Maligno"],
                                    zero_division=0))


def salvar_matriz_confusao(y_true, y_pred, caminho, titulo):
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benigno", "Maligno"]).plot(cmap="Blues")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()


def salvar_roc(modelos_ajustados, X_test, y_test, caminho):
    """Curvas ROC de todos os modelos (que tenham predict_proba)."""
    from sklearn.metrics import roc_curve, roc_auc_score
    plt.figure(figsize=(7.5, 7))
    for nome, m in modelos_ajustados.items():
        if hasattr(m, "predict_proba"):
            prob = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            lw = 2.6 if nome == "Stacking" else 1.3
            plt.plot(fpr, tpr, lw=lw, label=f"{nome} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Falso positivo (1 - especificidade)")
    plt.ylabel("Verdadeiro positivo (recall)")
    plt.title("Curvas ROC - modelos estruturados")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(caminho, dpi=130); plt.close()
