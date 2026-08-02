"""Ponto de entrada unico do sistema (o "executavel" que roda tudo).

Executar:
    python -m techchallenge                 # testes + estruturado + previsao + radiomica
    python -m techchallenge --com-cnn       # tambem treina a CNN (requer torch/GPU)
    python -m techchallenge --comparar-cnn  # treina e compara varias CNNs
    tc                                      # idem, se instalado com 'pip install -e .'

Por padrao roda TUDO que nao precisa de GPU (testes, modulo estruturado,
previsao de demonstracao e radiomica). Use as flags abaixo para ligar a CNN
ou pular etapas.
"""
import argparse

from .pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser(
        prog="techchallenge",
        description="Sistema de apoio ao diagnostico em saude da mulher (Tech Challenge Fase 1).")

    # --- CNN (fase extra, PyTorch/GPU) ---
    ap.add_argument("--com-cnn", action="store_true",
                    help="tambem treina a CNN de mamografias (requer torch/GPU)")
    ap.add_argument("--comparar-cnn", action="store_true",
                    help="treina e compara varias arquiteturas de CNN")
    ap.add_argument("--preparar-manifesto", action="store_true",
                    help="regera data/cnn_manifest.csv antes de treinar a CNN")
    ap.add_argument("--modelo-cnn",
                    choices=["mobilenet", "densenet", "resnet18", "resnet50", "efficientnet", "custom"],
                    default="mobilenet")
    ap.add_argument("--epocas-cnn", type=int, default=25)
    ap.add_argument("--sem-finetune", action="store_true",
                    help="desativa o fine-tuning da CNN (usa base congelada)")

    # --- controle das etapas padrao ---
    ap.add_argument("--sem-testes", action="store_true",
                    help="pula a etapa de testes automatizados (pytest)")
    ap.add_argument("--sem-radiomica", action="store_true",
                    help="pula o treino de radiomica")
    ap.add_argument("--extrair-radiomica", action="store_true",
                    help="reextrai as features das imagens antes da radiomica (demora)")
    ap.add_argument("--rapido", action="store_true",
                    help="treino estruturado sem validacao cruzada (bem mais rapido)")

    args = ap.parse_args()

    Pipeline(
        com_cnn=args.com_cnn,
        comparar_cnn=args.comparar_cnn,
        preparar_manifesto=args.preparar_manifesto,
        modelo_cnn=args.modelo_cnn,
        epocas_cnn=args.epocas_cnn,
        finetune=not args.sem_finetune,
        com_testes=not args.sem_testes,
        com_radiomica=not args.sem_radiomica,
        extrair_radiomica=args.extrair_radiomica,
        rapido=args.rapido,
    ).executar()


if __name__ == "__main__":
    main()
