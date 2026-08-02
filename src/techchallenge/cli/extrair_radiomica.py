"""Extrai features radiômicas das mamografias e salva um CSV tabular.

Cada imagem (recorte de lesão) vira uma linha de ~40 features numéricas + rótulo.
O resultado (data/radiomica_mamografia.csv) pode ser usado pelos modelos
estruturados (ex.: Stacking), tratando o problema de imagem como tabular.

Uso:
    python -m techchallenge.cli.extrair_radiomica            # todas as imagens
    python -m techchallenge.cli.extrair_radiomica --limite 100   # amostra rápida
"""
import argparse
import pandas as pd

from .. import config
from ..data import radiomica


def main():
    ap = argparse.ArgumentParser(description="Extrai features radiômicas das mamografias")
    ap.add_argument("--manifesto", default=None)
    ap.add_argument("--img-raiz", default=None)
    ap.add_argument("--saida", default=None, help="default: data/radiomica_mamografia.csv")
    ap.add_argument("--limite", type=int, default=0, help="N imagens por split (0=todas)")
    args = ap.parse_args()

    manifesto = args.manifesto or config.MANIFESTO_CSV
    img_raiz = args.img_raiz or config.IMAGENS_RAIZ
    saida = args.saida or (config.DATA_DIR / "radiomica_mamografia.csv")

    man = pd.read_csv(manifesto)
    if args.limite:
        man = pd.concat([g.sample(min(len(g), args.limite), random_state=config.RANDOM_STATE)
                         for _, g in man.groupby("split")]).reset_index(drop=True)

    print(f"Extraindo features de {len(man)} imagens...")
    df = radiomica.extrair_dataset(man, img_raiz)
    df.to_csv(saida, index=False)
    print("Salvo em:", saida)
    print("Formato:", df.shape, "| features:", df.shape[1]-2)
    print("Rótulos:", df["label"].value_counts().to_dict())
    print("Split:", df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
