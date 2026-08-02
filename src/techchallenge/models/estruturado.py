"""Modelos de classificacao para dados estruturados (scikit-learn).

Inclui seis modelos de base cobrindo os principais paradigmas e um
StackingClassifier que combina todos com um meta-modelo (Regressao Logistica).
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.svm import SVC

from .. import config


def _pipe(clf):
    """Envolve o classificador num Pipeline com padronizacao (sem leakage)."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def construir_modelos(balanced=False):
    """Retorna dict {nome: Pipeline} com os 6 modelos de base.

    Paradigmas cobertos: linear (Regressao Logistica), regras (Arvore),
    distancia (KNN), ensembles (Random Forest, Gradient Boosting) e
    margem maxima (SVM).

    balanced=True aplica class_weight='balanced' nos modelos que suportam
    (LogReg, Arvore, Random Forest, SVM) — util em dados desbalanceados,
    para priorizar a classe minoritaria (ex.: maligno na radiomica).
    """
    rs = config.RANDOM_STATE
    cw = "balanced" if balanced else None
    return {
        "Regressao Logistica": _pipe(LogisticRegression(max_iter=5000, random_state=rs, class_weight=cw)),
        "Arvore de Decisao": _pipe(DecisionTreeClassifier(max_depth=5, random_state=rs, class_weight=cw)),
        "KNN": _pipe(KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": _pipe(RandomForestClassifier(n_estimators=300, random_state=rs, class_weight=cw)),
        "Gradient Boosting": _pipe(GradientBoostingClassifier(random_state=rs)),
        "SVM": _pipe(SVC(kernel="rbf", probability=True, random_state=rs, class_weight=cw)),
    }


def construir_stacking(modelos=None):
    """StackingClassifier com os 6 modelos de base + meta-modelo (Regressao Logistica).

    O StackingClassifier usa validacao cruzada interna (cv=5) para gerar as
    previsoes dos modelos de base sem vazamento de dados.
    """
    modelos = modelos or construir_modelos()
    estimadores = [(nome, pipe) for nome, pipe in modelos.items()]
    return StackingClassifier(
        estimators=estimadores,
        final_estimator=LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE),
        cv=5,
        stack_method="auto",
        n_jobs=-1,
    )
