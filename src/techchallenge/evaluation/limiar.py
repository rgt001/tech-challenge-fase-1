"""Analise de limiar de decisao orientada ao recall (contexto clinico).

Em diagnostico, o falso negativo (deixar passar um cancer) e o erro mais grave.
Ajustando o limiar de decisao (o corte padrao e 0.5), podemos priorizar o recall
da classe maligna, aceitando mais falsos positivos como custo.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)


def tabela_limiares(y_true, y_prob, limiares=None):
    """DataFrame com recall, precisao, F1, FN e FP para varios limiares."""
    if limiares is None:
        limiares = np.round(np.arange(0.05, 0.96, 0.05), 2)
    linhas = []
    for t in limiares:
        p = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
        linhas.append({
            "limiar": float(t),
            "recall": recall_score(y_true, p, zero_division=0),
            "precisao": precision_score(y_true, p, zero_division=0),
            "f1": f1_score(y_true, p, zero_division=0),
            "FN": int(fn), "FP": int(fp),
        })
    return pd.DataFrame(linhas)


def escolher_limiar_por_recall(y_true, y_prob, recall_alvo=1.0):
    """Maior limiar que ainda atinge o recall alvo (minimiza FP mantendo recall)."""
    melhor = 0.01
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        p = (y_prob >= t).astype(int)
        if recall_score(y_true, p, zero_division=0) >= recall_alvo:
            melhor = float(t)
    return melhor


def plot_trade_off(tab, caminho, limiar_escolhido=None):
    plt.figure(figsize=(8, 5))
    plt.plot(tab["limiar"], tab["recall"], marker="o", label="Recall (sensibilidade)")
    plt.plot(tab["limiar"], tab["precisao"], marker="s", label="Precisao")
    plt.plot(tab["limiar"], tab["f1"], marker="^", label="F1")
    plt.axvline(0.5, color="gray", ls=":", label="limiar padrao = 0.5")
    if limiar_escolhido is not None:
        plt.axvline(limiar_escolhido, color="red", ls="--",
                    label=f"escolhido = {limiar_escolhido}")
    plt.xlabel("Limiar de decisao"); plt.ylabel("Score")
    plt.title("Trade-off por limiar - priorizando recall (nao deixar passar cancer)")
    plt.legend(); plt.tight_layout(); plt.savefig(caminho, dpi=120); plt.close()


def comparar_matrizes(y_true, y_prob, limiar_a, limiar_b, caminho):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axes, [limiar_a, limiar_b]):
        cm = confusion_matrix(y_true, (y_prob >= t).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["Benigno", "Maligno"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        fn = cm[1, 0]
        ax.set_title(f"Limiar {t}  (falsos negativos: {fn})")
    plt.suptitle("Matriz de confusao: limiar padrao vs. priorizando recall")
    plt.tight_layout(); plt.savefig(caminho, dpi=120); plt.close()
