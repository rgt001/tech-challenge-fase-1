"""Inferencia: classifica uma imagem de mamografia (ROI) com o modelo treinado.

Uso:  python -m techchallenge.cli.prever_cnn --imagem caminho/da/imagem.jpg
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from .. import config
from ..models import cnn


def main():
    ap = argparse.ArgumentParser(description="Classifica uma imagem (benigno/maligno)")
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--modelo", default=None, help="default: outputs/modelo_cnn_mobilenet.pt")
    ap.add_argument("--limiar", type=float, default=0.5)
    args = ap.parse_args()

    import os
    caminho_modelo = args.modelo or str(config.OUTPUTS_DIR / f"modelo_cnn_{config.MODELO_CNN_PADRAO}.pt")
    if not os.path.exists(caminho_modelo):
        raise SystemExit(f"Modelo nao encontrado: {caminho_modelo}. Treine antes.")
    if not os.path.exists(args.imagem):
        raise SystemExit(f"Imagem nao encontrada: {args.imagem}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(caminho_modelo, map_location=device)
    tipo, img_size = ckpt["modelo"], ckpt["img_size"]

    modelo = cnn.construir(tipo).to(device)
    modelo.load_state_dict(ckpt["state_dict"]); modelo.eval()

    t = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if tipo == "transfer":
        t.append(transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD))
    x = transforms.Compose(t)(Image.open(args.imagem).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(modelo(x)).item()
    rotulo = "MALIGNO" if prob >= args.limiar else "BENIGNO"
    print(f"Imagem  : {args.imagem}")
    print(f"Previsao: {rotulo}")
    print(f"Probabilidade de maligno: {prob:.1%}")
    print("\nAviso: ferramenta de apoio. O diagnostico final e sempre do medico.")


if __name__ == "__main__":
    main()
