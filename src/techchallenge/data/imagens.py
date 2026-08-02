"""Preparacao dos dados de imagem (CBIS-DDSM) para a CNN.

- construir_manifesto(): mapeia cada recorte de lesao (ROI "cropped") ao
  rotulo (BENIGN/MALIGNANT) e ao split oficial (train/test), via os
  metadados em csv/.
- MamografiaDataset / transforms: pipeline de imagem para o PyTorch.
"""
import os
import csv
from collections import Counter

from .. import config

DESCRICAO_CSVS = [
    ("mass_case_description_train_set.csv", "train"),
    ("mass_case_description_test_set.csv", "test"),
    ("calc_case_description_train_set.csv", "train"),
    ("calc_case_description_test_set.csv", "test"),
]


def _series_uid(path):
    partes = [s for s in path.split("/") if s.startswith("1.3.6")]
    return partes[-1] if partes else None


def _indice_dicom(raiz):
    indice = {}
    with open(os.path.join(raiz, "csv", "dicom_info.csv"), newline="") as f:
        for x in csv.DictReader(f):
            jpg = x["image_path"].replace("CBIS-DDSM/jpeg/", "jpeg/")
            indice.setdefault(x["SeriesInstanceUID"], []).append((jpg, x["SeriesDescription"]))
    return indice


def construir_manifesto(raiz=None):
    """Retorna lista de dicts (caminho, rotulo, label, split, tipo_lesao)."""
    raiz = str(raiz or config.IMAGENS_RAIZ)
    indice = _indice_dicom(raiz)

    def jpg_cropped(uid):
        for jpg, desc in indice.get(uid, []):
            if desc == "cropped images":
                return jpg
        return None

    linhas = []
    for arquivo, split in DESCRICAO_CSVS:
        caminho = os.path.join(raiz, "csv", arquivo)
        if not os.path.exists(caminho):
            continue
        with open(caminho, newline="") as f:
            for r in csv.DictReader(f):
                uid = _series_uid(r["cropped image file path"].strip())
                jpg = jpg_cropped(uid) if uid else None
                if not jpg or not os.path.exists(os.path.join(raiz, jpg)):
                    continue
                rotulo = "MALIGNANT" if r["pathology"] == "MALIGNANT" else "BENIGN"
                linhas.append({
                    "caminho": jpg.replace("\\", "/"),
                    "rotulo": rotulo,
                    "label": 1 if rotulo == "MALIGNANT" else 0,
                    "split": split,
                    "tipo_lesao": "mass" if "mass" in arquivo else "calc",
                })
    return linhas


def salvar_manifesto(linhas, saida=None):
    saida = str(saida or config.MANIFESTO_CSV)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["caminho", "rotulo", "label", "split", "tipo_lesao"])
        w.writeheader()
        w.writerows(linhas)
    return saida, dict(Counter(l["rotulo"] for l in linhas)), dict(Counter(l["split"] for l in linhas))


# --- Dataset PyTorch (import tardio de torch/torchvision) ---
def _torch_bits():
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    class MamografiaDataset(Dataset):
        def __init__(self, df, img_raiz, tfm):
            self.caminhos = [os.path.join(str(img_raiz), p) for p in df["caminho"].values]
            self.labels = df["label"].values.astype("float32")
            self.tfm = tfm

        def __len__(self):
            return len(self.caminhos)

        def __getitem__(self, i):
            img = Image.open(self.caminhos[i]).convert("RGB")
            return self.tfm(img), self.labels[i]

    def transforms_treino(img_size, transfer):
        t = [transforms.Resize((img_size, img_size)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(10),
             transforms.ToTensor()]
        if transfer:
            t.append(transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD))
        return transforms.Compose(t)

    def transforms_eval(img_size, transfer):
        t = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if transfer:
            t.append(transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD))
        return transforms.Compose(t)

    return MamografiaDataset, transforms_treino, transforms_eval
