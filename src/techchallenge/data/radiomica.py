"""Extração de features radiômicas de imagens de mamografia (radiomica-lite).

Transforma cada recorte de lesão (ROI) em um vetor de números, montando um
dataset tabular a partir das imagens. Isso permite tratar o problema de imagem
com Machine Learning clássico (ex.: o Stacking), como alternativa/complemento
à CNN.

Grupos de features extraídas:
  - Primeira ordem (intensidade): média, desvio, min, max, mediana, percentis,
    assimetria (skew), curtose e entropia.
  - Textura GLCM (Haralick): contraste, dissimilaridade, homogeneidade, energia,
    correlação e ASM, em várias distâncias.
  - LBP (Local Binary Pattern): histograma de padrões locais de textura.
  - Gabor: energia de textura em várias orientações e frequências.
  - Gradiente/bordas: magnitude de Sobel e densidade de bordas (Canny).
  - Forma da lesão: via limiar de Otsu + regionprops (área, excentricidade,
    solidez, preenchimento) — mede o formato da região densa (candidata a lesão).
"""
import os
import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, canny
from skimage.filters import gabor, sobel, threshold_otsu
from skimage.measure import label, regionprops

IMG_SIZE = 128
GLCM_NIVEIS = 64          # quantização para o GLCM (mais níveis = mais detalhe)
GLCM_DIST = [1, 2, 3, 4, 5]
GLCM_ANG = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
LBP_P, LBP_R = 8, 1       # 8 vizinhos, raio 1
LBP_BINS = LBP_P + 2      # padrões uniformes
GABOR_FREQS = [0.15, 0.30]
GABOR_ANG = [0, np.pi/4, np.pi/2, 3*np.pi/4]


def _carregar_cinza(caminho):
    img = Image.open(caminho).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.uint8)


def features_primeira_ordem(g):
    x = g.astype(np.float64).ravel()
    p = np.percentile(x, [10, 25, 50, 75, 90])
    hist = np.bincount(g.ravel(), minlength=256) / x.size
    entropia = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return {
        "int_media": x.mean(), "int_desvio": x.std(),
        "int_min": x.min(), "int_max": x.max(),
        "int_p10": p[0], "int_p25": p[1], "int_mediana": p[2],
        "int_p75": p[3], "int_p90": p[4],
        "int_skew": float(skew(x)), "int_curtose": float(kurtosis(x)),
        "int_entropia": entropia,
    }


def features_glcm(g):
    q = (g / (256 / GLCM_NIVEIS)).astype(np.uint8)
    glcm = graycomatrix(q, distances=GLCM_DIST, angles=GLCM_ANG,
                        levels=GLCM_NIVEIS, symmetric=True, normed=True)
    out = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)          # shape (n_dist, n_ang)
        for i, d in enumerate(GLCM_DIST):
            out[f"glcm_{prop}_d{d}"] = float(vals[i].mean())   # media sobre angulos
    return out


def features_lbp(g):
    lbp = local_binary_pattern(g, LBP_P, LBP_R, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return {f"lbp_{i}": float(v) for i, v in enumerate(hist)}


def features_gabor(g):
    """Energia de textura (média e desvio da magnitude) em orientações/frequências."""
    gf = g.astype(np.float64) / 255.0
    out = {}
    for fr in GABOR_FREQS:
        for j, th in enumerate(GABOR_ANG):
            real, imag = gabor(gf, frequency=fr, theta=th)
            mag = np.sqrt(real**2 + imag**2)
            out[f"gabor_f{fr}_a{j}_media"] = float(mag.mean())
            out[f"gabor_f{fr}_a{j}_desvio"] = float(mag.std())
    return out


def features_gradiente(g):
    """Magnitude de bordas (Sobel) e densidade de bordas (Canny)."""
    gf = g.astype(np.float64) / 255.0
    sob = sobel(gf)
    bordas = canny(gf, sigma=1.0)
    return {
        "grad_sobel_media": float(sob.mean()),
        "grad_sobel_desvio": float(sob.std()),
        "grad_sobel_max": float(sob.max()),
        "borda_densidade": float(bordas.mean()),
    }


def features_forma(g):
    """Forma da região densa (candidata a lesão) via limiar de Otsu + regionprops."""
    default = {"forma_area_frac": 0.0, "forma_excentric": 0.0,
               "forma_solidez": 0.0, "forma_preench": 0.0}
    try:
        t = threshold_otsu(g)
        mascara = g > t
        rot = label(mascara)
        props = regionprops(rot)
        if not props:
            return default
        maior = max(props, key=lambda p: p.area)   # maior região densa
        return {
            "forma_area_frac": float(maior.area / g.size),
            "forma_excentric": float(maior.eccentricity),
            "forma_solidez": float(maior.solidity),
            "forma_preench": float(maior.extent),
        }
    except Exception:
        return default


def extrair_features(caminho):
    """Retorna dict com todas as features de uma imagem."""
    g = _carregar_cinza(caminho)
    feats = {}
    feats.update(features_primeira_ordem(g))
    feats.update(features_glcm(g))
    feats.update(features_lbp(g))
    feats.update(features_gabor(g))
    feats.update(features_gradiente(g))
    feats.update(features_forma(g))
    return feats


def extrair_dataset(df_manifesto, img_raiz, log_cada=200):
    """Extrai features de todas as imagens do manifesto.

    df_manifesto: DataFrame com colunas caminho, label, split.
    Retorna DataFrame com features + label + split.
    """
    import pandas as pd
    linhas = []
    total = len(df_manifesto)
    for n, (_, r) in enumerate(df_manifesto.iterrows(), 1):
        caminho = os.path.join(str(img_raiz), r["caminho"])
        if not os.path.exists(caminho):
            continue
        try:
            feats = extrair_features(caminho)
        except Exception as e:
            print("Falha em", caminho, ":", e); continue
        feats["label"] = int(r["label"]); feats["split"] = r["split"]
        linhas.append(feats)
        if log_cada and n % log_cada == 0:
            print(f"  {n}/{total} imagens processadas")
    return pd.DataFrame(linhas)
