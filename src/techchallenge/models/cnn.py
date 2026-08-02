"""Arquiteturas e treino da CNN (PyTorch) para mamografias.

Suporta varias arquiteturas pre-treinadas (transfer learning) e uma CNN
propria, permitindo comparar qual funciona melhor no CBIS-DDSM.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (MobileNet_V2_Weights, DenseNet121_Weights,
                                ResNet18_Weights, ResNet50_Weights,
                                EfficientNet_B0_Weights)

# arquiteturas pre-treinadas disponiveis (+ 'custom')
ARQUITETURAS = ["mobilenet", "densenet", "resnet18", "resnet50", "efficientnet", "custom"]


def _cabeca(in_features):
    """Cabeca de classificacao comum (1 logit; sigmoid vem na loss)."""
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 64), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def _congelar(m):
    for p in m.parameters():
        p.requires_grad = False


def modelo_custom():
    """CNN convolucional simples treinada do zero."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )


def construir(tipo):
    """Cria o modelo pelo nome. Base pre-treinada congelada + nova cabeca.

    Tipos: mobilenet (alias 'transfer'), densenet, resnet18, resnet50,
    efficientnet, custom.
    """
    if tipo in ("transfer", "mobilenet"):
        m = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        _congelar(m); m.classifier = _cabeca(m.classifier[1].in_features); return m
    if tipo == "densenet":
        m = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        _congelar(m); m.classifier = _cabeca(m.classifier.in_features); return m
    if tipo == "resnet18":
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        _congelar(m); m.fc = _cabeca(m.fc.in_features); return m
    if tipo == "resnet50":
        m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        _congelar(m); m.fc = _cabeca(m.fc.in_features); return m
    if tipo == "efficientnet":
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        _congelar(m); m.classifier = _cabeca(m.classifier[1].in_features); return m
    if tipo == "custom":
        return modelo_custom()
    raise ValueError(f"arquitetura desconhecida: {tipo}")


def descongelar(modelo):
    """Libera toda a rede para o fine-tuning (usar com learning rate baixo)."""
    for p in modelo.parameters():
        p.requires_grad = True


def rodar_epoca(modelo, loader, criterio, device, otimizador=None):
    treino = otimizador is not None
    modelo.train() if treino else modelo.eval()
    total_loss, acertos, n = 0.0, 0, 0
    with torch.set_grad_enabled(treino):
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = modelo(x)
            loss = criterio(logits, y)
            if treino:
                otimizador.zero_grad()
                loss.backward()
                otimizador.step()
            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            acertos += (preds == y).sum().item()
            n += x.size(0)
    return total_loss / n, acertos / n


def loop_treino(modelo, dl_train, dl_val, criterio, otim, device, epocas,
                hist, estado, total, ao_salvar):
    """Roda 'epocas' epocas com early stopping e melhor-checkpoint (via ao_salvar)."""
    for _ in range(epocas):
        estado["ep"] += 1
        tl, ta = rodar_epoca(modelo, dl_train, criterio, device, otim)
        vl, va = rodar_epoca(modelo, dl_val, criterio, device)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["train_acc"].append(ta); hist["val_acc"].append(va)
        print(f"Epoca {estado['ep']:02d}/{total} - loss {tl:.4f}/{vl:.4f} - acc {ta:.3f}/{va:.3f}")
        if vl < estado["melhor_val"]:
            estado["melhor_val"], estado["espera"] = vl, 0
            ao_salvar()
        else:
            estado["espera"] += 1
            if estado["espera"] >= estado["paciencia"]:
                print(f"Early stopping na epoca {estado['ep']}.")
                estado["parar"] = True
                return
