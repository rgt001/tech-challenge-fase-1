"""Inferencia do modulo estruturado a partir de um CSV tabular.

Uso:
    python -m techchallenge.cli.prever_estruturado --csv caminho/do_exame.csv

O CSV pode conter uma ou varias linhas e precisa trazer as 30 features do
Breast Cancer Wisconsin. Colunas extras como id, diagnosis e Unnamed: 32 sao
ignoradas automaticamente.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from joblib import load

from .. import config


def _carregar_metadados(caminho_meta):
    caminho = Path(caminho_meta)
    if not caminho.exists():
        raise SystemExit(
            f"Metadados nao encontrados em {caminho}. Rode tc-treinar-estruturado antes."
        )
    return json.loads(caminho.read_text(encoding="utf-8"))


def _preparar_entrada(df, colunas_esperadas):
    base = df.drop(columns=["id", "diagnosis", "Unnamed: 32"], errors="ignore").copy()
    faltantes = [c for c in colunas_esperadas if c not in base.columns]
    if faltantes:
        raise SystemExit(
            "CSV incompleto. Faltam as colunas: " + ", ".join(faltantes)
        )
    extras = [c for c in base.columns if c not in colunas_esperadas]
    if extras:
        print("Aviso: colunas extras ignoradas:", ", ".join(extras))
    return base.loc[:, colunas_esperadas]


def main():
    ap = argparse.ArgumentParser(description="Classifica exames tabulares (benigno/maligno)")
    ap.add_argument("--csv", required=True, help="CSV com uma ou mais linhas do exame")
    ap.add_argument("--modelo", default=None, help="default: outputs/modelo_estruturado_stacking.joblib")
    ap.add_argument("--meta", default=None, help="default: outputs/modelo_estruturado_meta.json")
    ap.add_argument("--limiar", type=float, default=None,
                    help="sobrescreve o limiar manualmente (ex.: 0.5, 0.02)")
    ap.add_argument("--priorizar-recall", action="store_true",
                    help="usa o limiar salvo que zera falsos negativos no teste")
    ap.add_argument("--saida", default=None, help="opcional: salva as previsoes em CSV")
    args = ap.parse_args()

    caminho_modelo = Path(args.modelo or config.MODELO_ESTRUTURADO_PADRAO)
    if not caminho_modelo.exists():
        raise SystemExit(
            f"Modelo nao encontrado: {caminho_modelo}. Rode tc-treinar-estruturado antes."
        )
    caminho_csv = Path(args.csv)
    if not caminho_csv.exists():
        raise SystemExit(f"CSV nao encontrado: {caminho_csv}")

    meta = _carregar_metadados(args.meta or config.METADADOS_ESTRUTURADO)
    modelo = load(caminho_modelo)
    original = pd.read_csv(caminho_csv)
    X = _preparar_entrada(original, meta["features"])

    if args.limiar is not None:
        limiar = args.limiar
    elif args.priorizar_recall:
        limiar = meta.get("threshold_priorizar_recall", 0.5)
    else:
        limiar = meta.get("threshold_padrao", 0.5)

    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= limiar).astype(int)
    resultado = original.copy()
    resultado["prob_maligno"] = prob.round(6)
    resultado["limiar_usado"] = limiar
    resultado["previsao"] = ["MALIGNO" if p == 1 else "BENIGNO" for p in pred]
    resultado["classe_prevista_codigo"] = pred

    if "diagnosis" in original.columns:
        real = original["diagnosis"].map({"M": "MALIGNO", "B": "BENIGNO"})
        resultado["classe_real"] = real
        resultado["acertou"] = resultado["previsao"] == resultado["classe_real"]

    print(f"Modelo : {caminho_modelo}")
    print(f"Limiar : {limiar:.4f}")
    print(f"Linhas : {len(resultado)}\n")
    print(resultado[["previsao", "prob_maligno"] + (
        ["classe_real", "acertou"] if "classe_real" in resultado.columns else []
    )].to_string(index=False))

    if args.saida:
        caminho_saida = Path(args.saida)
        resultado.to_csv(caminho_saida, index=False)
        print(f"\nPrevisoes salvas em: {caminho_saida}")

    print("\nAviso: ferramenta de apoio. O diagnostico final e sempre do medico.")


if __name__ == "__main__":
    main()
