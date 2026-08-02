"""Treina e avalia os modelos estruturados (Breast Cancer Wisconsin).

Fluxo: carga -> split -> 6 modelos de base + Stacking -> validacao cruzada ->
metricas no teste -> matrizes de confusao -> feature importance -> SHAP.
Salva tudo em outputs/.

Uso:  python -m techchallenge.cli.treinar_estruturado
"""
import argparse
import json
import warnings
warnings.filterwarnings("ignore", message=".*probability.*was deprecated.*")

import matplotlib
matplotlib.use("Agg")  # headless (salva sem display)
import pandas as pd
from joblib import dump
from sklearn.model_selection import cross_validate

from .. import config
from ..data import estruturado as dados
from ..models import estruturado as modelos_estr
from ..evaluation import metrics, explicabilidade, limiar, eda


def main():
    ap = argparse.ArgumentParser(description="Treino dos modelos estruturados (cancer de mama)")
    ap.add_argument("--csv", default=None, help="caminho do CSV (default: data/breast_cancer.csv)")
    ap.add_argument("--sem-cv", action="store_true", help="pula a validacao cruzada")
    args = ap.parse_args()

    out = config.garantir_outputs()
    n = config.limpar_outputs("estruturado_")   # apaga resultados antigos deste modulo
    if n:
        print(f"Limpeza: {n} arquivos antigos (estruturado_*) removidos.")
    df = dados.carregar_dados(args.csv)
    print("Formato:", df.shape)

    # graficos de exploracao (EDA)
    eda.gerar_graficos(df, out)

    X_train, X_test, y_train, y_test = dados.preparar_treino_teste(df)
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}\n")

    # 6 modelos de base + stacking
    modelos = modelos_estr.construir_modelos()
    modelos["Stacking"] = modelos_estr.construir_stacking(modelos_estr.construir_modelos())

    resultados = []
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        linha = {"Modelo": nome, **metrics.metricas(y_test, y_pred)}
        # validacao cruzada (5-fold) no treino: robustez do recall
        if not args.sem_cv:
            cv = cross_validate(modelo, X_train, y_train, cv=5,
                                scoring=["accuracy", "recall"], n_jobs=-1)
            linha["CV Acc (media)"] = cv["test_accuracy"].mean()
            linha["CV Recall (media)"] = cv["test_recall"].mean()
        resultados.append(linha)
        metrics.salvar_matriz_confusao(
            y_test, y_pred, out / f"estruturado_matriz_{nome.split()[0].lower()}.png",
            f"Matriz de confusao - {nome}")
        print(f"{nome}: treinado")

    metrics.salvar_roc(modelos, X_test, y_test, out / "estruturado_roc.png")

    res_df = pd.DataFrame(resultados).set_index("Modelo").round(4)
    res_df.to_csv(out / "estruturado_metricas.csv")
    print("\n=== Metricas (teste) + validacao cruzada ===")
    print(res_df.to_string())

    # Explicabilidade (nos modelos de base interpretaveis)
    explicabilidade.feature_importance_logistica(
        modelos["Regressao Logistica"], X_train.columns,
        out / "estruturado_feature_importance.png")
    try:
        explicabilidade.shap_arvore(
            modelos["Arvore de Decisao"], X_test, X_train.columns,
            out / "estruturado_shap.png")
    except Exception as e:
        print("Aviso: SHAP nao gerado:", e)

    # --- Analise de limiar orientada a recall (finalidade clinica) ---
    # Usa o Stacking (melhor modelo) e suas probabilidades no teste.
    prob = modelos["Stacking"].predict_proba(X_test)[:, 1]
    tab = limiar.tabela_limiares(y_test, prob)
    tab.to_csv(out / "estruturado_limiares.csv", index=False)
    t_recall = limiar.escolher_limiar_por_recall(y_test, prob, recall_alvo=1.0)
    limiar.plot_trade_off(tab, out / "estruturado_limiar_tradeoff.png", t_recall)
    limiar.comparar_matrizes(y_test, prob, 0.5, t_recall,
                             out / "estruturado_limiar_matrizes.png")

    def _fn_fp(t):
        p = (prob >= t).astype(int)
        from sklearn.metrics import confusion_matrix
        _, fp, fn, _ = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
        return int(fn), int(fp)
    fn05, fp05 = _fn_fp(0.5)
    fnr, fpr = _fn_fp(t_recall)
    print("\n=== Analise de limiar (Stacking) ===")
    print(f"Limiar padrao 0.50 -> falsos negativos: {fn05} | falsos positivos: {fp05}")
    print(f"Limiar {t_recall:.2f} (recall=100%) -> falsos negativos: {fnr} | falsos positivos: {fpr}")
    print("Custo de nao deixar passar nenhum cancer:", fpr - fp05, "falsos positivos a mais.")

    # Persistencia para uso real do modulo estruturado (inferencia CLI)
    dump(modelos["Stacking"], config.MODELO_ESTRUTURADO_PADRAO)
    dump(modelos["Regressao Logistica"], config.MODELO_ESTRUTURADO_LOGISTICA)
    meta = {
        "modelo_padrao": "Stacking",
        "artefato_modelo": str(config.MODELO_ESTRUTURADO_PADRAO),
        "artefato_modelo_interpretavel": str(config.MODELO_ESTRUTURADO_LOGISTICA),
        "threshold_padrao": 0.5,
        "threshold_priorizar_recall": round(float(t_recall), 6),
        "features": list(X_train.columns),
        "classe_positiva": {"codigo": 1, "rotulo_original": "M", "descricao": "Maligno"},
        "classe_negativa": {"codigo": 0, "rotulo_original": "B", "descricao": "Benigno"},
    }
    config.METADADOS_ESTRUTURADO.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Modelo salvo em: {config.MODELO_ESTRUTURADO_PADRAO}")
    print(f"Metadados salvos em: {config.METADADOS_ESTRUTURADO}")

    melhor = res_df["Recall (maligno)"].idxmax()
    print(f"\nMelhor modelo por recall (maligno): {melhor}")
    print(f"Resultados salvos em {out}/")


if __name__ == "__main__":
    main()
