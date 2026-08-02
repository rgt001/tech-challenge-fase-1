"""Configuracoes e caminhos centrais do projeto."""
from pathlib import Path

# projeto/  (dois niveis acima de src/techchallenge/config.py)
PROJETO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJETO_DIR / "data"
OUTPUTS_DIR = PROJETO_DIR / "outputs"

# Dados estruturados (Breast Cancer Wisconsin)
BREAST_CANCER_CSV = DATA_DIR / "breast_cancer.csv"
MODELO_ESTRUTURADO_PADRAO = OUTPUTS_DIR / "modelo_estruturado_stacking.joblib"
MODELO_ESTRUTURADO_LOGISTICA = OUTPUTS_DIR / "modelo_estruturado_logistica.joblib"
METADADOS_ESTRUTURADO = OUTPUTS_DIR / "modelo_estruturado_meta.json"

# Imagens CBIS-DDSM: as pastas jpeg/ e csv/ ficam dentro do projeto
IMAGENS_RAIZ = PROJETO_DIR
MANIFESTO_CSV = DATA_DIR / "cnn_manifest.csv"

RANDOM_STATE = 42

# CNN
MODELO_CNN_PADRAO = "resnet18"   # melhor recall na comparacao
IMG_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def garantir_outputs() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def limpar_outputs(prefixo):
    """Remove os arquivos gerados que comecam com <prefixo> (ex.: 'estruturado_').
    Nao mexe em outros prefixos (ex.: modelos .pt da CNN sao preservados)."""
    import glob, os
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    removidos = 0
    for caminho in glob.glob(str(OUTPUTS_DIR / (prefixo + "*"))):
        try:
            os.remove(caminho); removidos += 1
        except OSError:
            pass
    return removidos
