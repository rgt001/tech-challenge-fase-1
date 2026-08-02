"""Gera o manifesto de imagens (imagem -> rotulo -> split) para a CNN.

Uso:  python -m techchallenge.cli.preparar_manifesto
"""
import argparse

from .. import config
from ..data import imagens


def main():
    ap = argparse.ArgumentParser(description="Gera data/cnn_manifest.csv")
    ap.add_argument("--raiz", default=None, help="pasta com csv/ e jpeg/ (default: nivel acima do projeto)")
    ap.add_argument("--saida", default=None, help="default: data/cnn_manifest.csv")
    args = ap.parse_args()

    linhas = imagens.construir_manifesto(args.raiz)
    saida, rotulos, splits = imagens.salvar_manifesto(linhas, args.saida)
    print("Manifesto salvo em:", saida)
    print("Total de imagens:", len(linhas))
    print("Rotulos:", rotulos)
    print("Split:", splits)


if __name__ == "__main__":
    main()
