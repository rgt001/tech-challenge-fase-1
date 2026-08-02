"""Graficos de analise exploratoria (EDA) para o dataset estruturado."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def gerar_graficos(df, out):
    """Gera os graficos de EDA em <out> com prefixo estruturado_eda_."""
    sns.set_theme(style="whitegrid")

    # 1. Distribuicao do alvo
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="diagnosis", hue="diagnosis", order=["B", "M"],
                       palette=["#4c9f70", "#d1495b"], legend=False)
    ax.set_title("Distribuicao do diagnostico (B=benigno, M=maligno)")
    for c in ax.containers:
        ax.bar_label(c)
    plt.tight_layout(); plt.savefig(out / "estruturado_eda_distribuicao.png", dpi=120); plt.close()

    # correlacao (alvo numerico)
    d = df.copy(); d["diagnosis"] = d["diagnosis"].map({"M": 1, "B": 0})
    corr_alvo = d.corr()["diagnosis"].drop("diagnosis").sort_values(ascending=False)

    # 2. Correlacao com o alvo
    plt.figure(figsize=(7, 6))
    corr_alvo.head(15).plot(kind="barh", color="#d1495b")
    plt.gca().invert_yaxis()
    plt.title("Correlacao das features com o diagnostico (maligno=1)")
    plt.xlabel("Correlacao de Pearson")
    plt.tight_layout(); plt.savefig(out / "estruturado_eda_correlacao.png", dpi=120); plt.close()

    # 3. Heatmap de correlacao
    plt.figure(figsize=(15, 12))
    sns.heatmap(d.corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6})
    plt.title("Matriz de correlacao")
    plt.tight_layout(); plt.savefig(out / "estruturado_eda_heatmap.png", dpi=110); plt.close()

    # 4. Distribuicoes por diagnostico
    feats = ["radius_mean", "area_mean", "concavity_mean",
             "concave points_mean", "texture_mean", "smoothness_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), feats):
        sns.kdeplot(data=df, x=col, hue="diagnosis", fill=True, common_norm=False,
                    palette=["#4c9f70", "#d1495b"], ax=ax)
        ax.set_title(col)
    plt.suptitle("Distribuicao de features por diagnostico", y=1.02)
    plt.tight_layout(); plt.savefig(out / "estruturado_eda_features.png", dpi=120, bbox_inches="tight"); plt.close()
