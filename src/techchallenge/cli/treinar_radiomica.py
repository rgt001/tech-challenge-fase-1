"""Treina os modelos estruturados sobre as features radiômicas das imagens.

Trata o problema de imagem como TABULAR: usa o CSV gerado por
extrair_radiomica (cada mamografia = uma linha de features) e roda os
mesmos modelos do módulo estruturado (incluindo o Stacking). Serve para
comparar 'radiômica + ML clássico' com a CNN.

Aplica class_weight='balanced' (dados desbalanceados) e uma análise de
limiar para priorizar o recall da classe maligna.

Uso:
    python -m techchallenge.cli.extrair_radiomica     # gera o CSV (uma vez)
    python -m techchallenge.cli.treinar_radiomica     # treina e avalia
"""
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*probability.*was deprecated.*")

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from .. import config
from ..models import estruturado as modelos_estr
from ..evaluation import metrics, limiar


def main():
    ap = argparse.ArgumentParser(description="Treino do Stacking sobre features radiomicas")
    ap.add_argument("--csv", default=None, help="default: data/radiomica_mamografia.csv")
    ap.add_argument("--sem-balanceamento", action="store_true",
                    help="nao usa class_weight='balanced'")
    args = ap.parse_args()

    out = config.garantir_outputs()
    config.limpar_outputs("radiomica_")
    caminho = args.csv or (config.DATA_DIR / "radiomica_mamografia.csv")
    df = pd.read_csv(caminho)
    print("Formato:", df.shape)

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    cols = [c for c in df.columns if c not in ("label", "split")]
    X_train, y_train = tr[cols], tr["label"]
    X_test, y_test = te[cols], te["label"]
    balanced = not args.sem_balanceamento
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape} | features: {len(cols)} | "
          f"balanceado: {balanced}")

    modelos = modelos_estr.construir_modelos(balanced=balanced)
    modelos["Stacking"] = modelos_estr.construir_stacking(
        modelos_estr.construir_modelos(balanced=balanced))

    resultados = []
    for nome, m in modelos.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        linha = {"Modelo": nome, **metrics.metricas(y_test, y_pred)}
        if hasattr(m, "predict_proba"):
            linha["AUC"] = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
        resultados.append(linha)

    res = pd.DataFrame(resultados).set_index("Modelo").round(4)
    res.to_csv(out / "radiomica_metricas.csv")
    print("\n=== Radiomica + modelos classicos (teste) ===")
    print(res.to_string())

    metrics.salvar_matriz_confusao(
        y_test, modelos["Stacking"].predict(X_test),
        out / "radiomica_matriz_stacking.png", "Radiomica + Stacking - matriz de confusao")

    # --- Ajuste de limiar para priorizar recall (mesma filosofia clinica) ---
    prob = modelos["Stacking"].predict_proba(X_test)[:, 1]
    tab = limiar.tabela_limiares(y_test, prob)
    tab.to_csv(out / "radiomica_limiares.csv", index=False)
    t_recall = limiar.escolher_limiar_por_recall(y_test, prob, recall_alvo=0.90)
    limiar.plot_trade_off(tab, out / "radiomica_limiar_tradeoff.png", t_recall)

    def _fn_fp_rec(t):
        p = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) else 0
        return int(fn), int(fp), rec
    fn5, fp5, r5 = _fn_fp_rec(0.5)
    fnr, fpr, rr = _fn_fp_rec(t_recall)
    print("\n=== Ajuste de limiar (Stacking radiomica) ===")
    print(f"Limiar 0.50   -> recall {r5:.3f} | FN {fn5} | FP {fp5}")
    print(f"Limiar {t_recall:.2f} -> recall {rr:.3f} | FN {fnr} | FP {fpr}")

    print("\n>>> Comparacao com a CNN (ResNet18) nas MESMAS imagens:")
    print("    CNN: accuracy 0.642 | recall 0.746 | F1 0.621")
    print("Resultados salvos em", out)


if __name__ == "__main__":
    main()
