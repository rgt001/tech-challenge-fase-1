"""Treina e avalia a CNN (PyTorch) para mamografias (CBIS-DDSM).

Usa a GPU NVIDIA se disponivel. Fine-tuning opcional em 2 fases (--finetune).

Uso tipico:
    python -m techchallenge.cli.treinar_cnn --modelo transfer --finetune --epocas 25
    python -m techchallenge.cli.treinar_cnn --limite 100 --epocas 2   # teste rapido
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import config
from ..data import imagens
from ..models import cnn
from ..evaluation import metrics


def _plot_historico(hist, caminho, tag):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="treino"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="treino"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    plt.suptitle(f"Historico de treino - CNN ({tag})")
    plt.tight_layout(); plt.savefig(caminho, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Treino da CNN (mamografias)")
    ap.add_argument("--manifesto", default=None)
    ap.add_argument("--img-raiz", default=None,
                    help="pasta com jpeg/ (default: nivel acima do projeto)")
    ap.add_argument("--modelo",
                    choices=["mobilenet", "densenet", "resnet18", "resnet50", "efficientnet", "custom"],
                    default="mobilenet")
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--limite", type=int, default=0, help="N imagens por split (0=todas)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--finetune", action="store_true",
                    help="2 fases: warmup + fine-tuning da base (so p/ transfer)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    args = ap.parse_args()

    out = config.garantir_outputs()
    config.limpar_outputs(f"cnn_{args.modelo}")   # apaga resultados antigos desta arquitetura
    manifesto = args.manifesto or config.MANIFESTO_CSV
    img_raiz = args.img_raiz or config.IMAGENS_RAIZ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device,
          "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma")

    # dados
    df = pd.read_csv(manifesto)
    df["abs"] = df["caminho"].apply(lambda p: os.path.join(str(img_raiz), p))
    df = df[df["abs"].apply(os.path.exists)].drop(columns=["abs"]).reset_index(drop=True)
    if args.limite:
        df = pd.concat([g.sample(min(len(g), args.limite), random_state=config.RANDOM_STATE)
                        for _, g in df.groupby("split")]).reset_index(drop=True)

    df_tv = df[df["split"] == "train"].sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    n_val = int(len(df_tv) * args.val_frac)
    df_val, df_train = df_tv.iloc[:n_val], df_tv.iloc[n_val:]
    print(f"Treino: {len(df_train)} | Val: {len(df_val)} | Teste: {len(df_test)}")

    transfer = (args.modelo != "custom")   # todas pre-treinadas usam normalizacao ImageNet
    Dataset, tf_tr, tf_ev = imagens._torch_bits()
    tfm_tr, tfm_ev = tf_tr(args.img_size, transfer), tf_ev(args.img_size, transfer)

    def loader(d, tfm, shuffle):
        return DataLoader(Dataset(d, img_raiz, tfm), batch_size=args.batch,
                          shuffle=shuffle, num_workers=args.workers)

    dl_train, dl_val, dl_test = loader(df_train, tfm_tr, True), loader(df_val, tfm_ev, False), loader(df_test, tfm_ev, False)

    n0 = int((df_train["label"] == 0).sum()); n1 = int((df_train["label"] == 1).sum())
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float32, device=device)
    print(f"Benigno/Maligno no treino: {n0}/{n1} | pos_weight: {pos_weight.item():.3f}")

    modelo = cnn.construir(args.modelo).to(device)
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    estado = {"ep": 0, "melhor_val": float("inf"), "espera": 0, "paciencia": 5, "parar": False}
    caminho_modelo = out / f"modelo_cnn_{args.modelo}.pt"

    def salvar():
        torch.save({"state_dict": modelo.state_dict(), "modelo": args.modelo,
                    "img_size": args.img_size}, caminho_modelo)

    if transfer and args.finetune:
        warmup = min(args.warmup, args.epocas)
        ft_epocas = max(args.epocas - warmup, 0)
        total = warmup + ft_epocas
        otim = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=args.lr)
        print(f"\n== Fase 1 (warmup) | {warmup} epocas | lr {args.lr} ==")
        cnn.loop_treino(modelo, dl_train, dl_val, criterio, otim, device, warmup, hist, estado, total, salvar)
        if ft_epocas > 0 and not estado["parar"]:
            cnn.descongelar(modelo)
            estado["espera"] = 0
            treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
            otim = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=args.lr_ft)
            print(f"\n== Fase 2 (fine-tune) | {ft_epocas} epocas | lr {args.lr_ft} | "
                  f"rede completa | params treinaveis: {treinaveis:,} ==")
            cnn.loop_treino(modelo, dl_train, dl_val, criterio, otim, device, ft_epocas, hist, estado, total, salvar)
    else:
        total = args.epocas
        otim = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=args.lr)
        cnn.loop_treino(modelo, dl_train, dl_val, criterio, otim, device, args.epocas, hist, estado, total, salvar)

    # avaliacao com o melhor modelo
    ckpt = torch.load(caminho_modelo, map_location=device)
    modelo.load_state_dict(ckpt["state_dict"]); modelo.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in dl_test:
            y_prob.extend(torch.sigmoid(modelo(x.to(device))).cpu().numpy().ravel())
            y_true.extend(y.numpy().ravel())
    y_true = np.array(y_true).astype(int); y_pred = (np.array(y_prob) >= 0.5).astype(int)

    _plot_historico(hist, out / f"cnn_{args.modelo}_historico.png", args.modelo)
    metrics.salvar_matriz_confusao(y_true, y_pred, out / f"cnn_{args.modelo}_matriz.png",
                                   f"Matriz de confusao - CNN ({args.modelo})")
    txt = metrics.relatorio_texto(y_true, y_pred, f"Modelo: {args.modelo}")
    print(txt)
    (out / f"cnn_{args.modelo}_metricas.txt").write_text(txt)
    print(f"\nConcluido. Modelo salvo em {caminho_modelo} e resultados em {out}/.")


if __name__ == "__main__":
    main()
