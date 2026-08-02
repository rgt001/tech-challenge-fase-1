"""Orquestrador do sistema: executa TUDO em ordem, do comeco ao fim.

Etapas padrao (nao exigem GPU):
  [0] Testes automatizados (pytest)  -> valida o codigo antes de treinar
  [1] Modulo estruturado (treino)    -> modelos scikit-learn + artefatos salvos
  [2] Previsao de demonstracao       -> usa o modelo salvo num CSV de exemplo
  [3] Radiomica (treino tabular)     -> ML classico sobre features das imagens

Etapas opcionais (mais pesadas, sob demanda):
  - Extracao radiomica (le TODAS as imagens; --extrair-radiomica)
  - CNN em PyTorch/GPU (--com-cnn / --comparar-cnn)

Os imports pesados (torch na CNN; extracao em lote na radiomica) sao tardios
(lazy), para o fluxo basico nao exigir torch nem ler as imagens.
"""
import os
import subprocess
import sys

from . import config
from .cli import (
    treinar_estruturado,
    prever_estruturado,
    treinar_radiomica,
    preparar_manifesto as cli_manifesto,
)


class Pipeline:
    """Executa as etapas do sistema de forma automatica, do inicio ao fim."""

    ARQUITETURAS_COMPARAR = ["mobilenet", "densenet", "resnet18", "efficientnet"]

    def __init__(self, com_cnn=False, comparar_cnn=False, preparar_manifesto=False,
                 modelo_cnn="mobilenet", epocas_cnn=25, finetune=True,
                 com_testes=True, com_radiomica=True, extrair_radiomica=False,
                 rapido=False):
        self.com_cnn = com_cnn
        self.comparar_cnn = comparar_cnn
        self.preparar_manifesto = preparar_manifesto
        self.modelo_cnn = modelo_cnn
        self.epocas_cnn = epocas_cnn
        self.finetune = finetune
        self.com_testes = com_testes
        self.com_radiomica = com_radiomica
        self.extrair_radiomica = extrair_radiomica
        self.rapido = rapido   # treino sem validacao cruzada (bem mais rapido)

    # ---- utilidades -------------------------------------------------------
    @staticmethod
    def _titulo(txt):
        print("\n" + "=" * 64 + f"\n{txt}\n" + "=" * 64)

    @staticmethod
    def _rodar(modulo, argv):
        """Executa o main() de um modulo CLI com os argumentos informados."""
        antigo = sys.argv
        sys.argv = argv
        try:
            modulo.main()
        finally:
            sys.argv = antigo

    # ---- [0] Testes -------------------------------------------------------
    def etapa_testes(self):
        self._titulo("[0] TESTES AUTOMATIZADOS (pytest)")
        testes_dir = config.PROJETO_DIR / "tests"
        if not testes_dir.exists():
            print("Pasta tests/ nao encontrada - etapa pulada.")
            return
        # roda pytest num subprocesso isolado (nao contamina o processo atual).
        # Garante que o pacote seja encontrado mesmo sem 'pip install -e .',
        # colocando o src/ no PYTHONPATH do subprocesso.
        env = os.environ.copy()
        src = str(config.PROJETO_DIR / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run(
            [sys.executable, "-m", "pytest", str(testes_dir)],
            cwd=str(config.PROJETO_DIR),
            env=env,
        )
        if res.returncode != 0:
            raise SystemExit(
                f"\nTestes falharam (codigo {res.returncode}). "
                "Corrija antes de treinar, ou rode com --sem-testes para pular."
            )
        print("Todos os testes passaram.")

    # ---- [1] Estruturado --------------------------------------------------
    def etapa_estruturado(self):
        self._titulo("[1] MODULO ESTRUTURADO - cancer de mama (scikit-learn)")
        argv = ["tc-treinar-estruturado"]
        if self.rapido:
            argv.append("--sem-cv")   # pula validacao cruzada (mais rapido)
        self._rodar(treinar_estruturado, argv)

    # ---- [2] Previsao de demonstracao ------------------------------------
    def etapa_previsao_demo(self):
        amostra = config.DATA_DIR / "amostra_previsao.csv"
        if not amostra.exists():
            print("(demo de previsao pulada: data/amostra_previsao.csv nao existe)")
            return
        self._titulo("[2] PREVISAO DE DEMONSTRACAO (usa o modelo salvo na etapa 1)")
        self._rodar(prever_estruturado,
                    ["tc-prever-estruturado", "--csv", str(amostra)])

    # ---- [3] Radiomica ----------------------------------------------------
    def etapa_radiomica_extracao(self):
        from .cli import extrair_radiomica  # import tardio (le todas as imagens)
        self._titulo("[*] EXTRACAO RADIOMICA (le as mamografias - pode demorar)")
        self._rodar(extrair_radiomica, ["tc-extrair-radiomica"])

    def etapa_radiomica(self):
        csv = config.DATA_DIR / "radiomica_mamografia.csv"
        if not csv.exists():
            print("(radiomica pulada: data/radiomica_mamografia.csv nao existe - "
                  "gere com --extrair-radiomica)")
            return
        self._titulo("[3] RADIOMICA - ML classico sobre features das imagens")
        self._rodar(treinar_radiomica, ["tc-treinar-radiomica"])

    # ---- CNN (opcional, PyTorch/GPU) -------------------------------------
    def etapa_manifesto(self):
        self._titulo("[*] Gerando manifesto de imagens (CBIS-DDSM)")
        self._rodar(cli_manifesto, ["tc-preparar-manifesto"])

    def etapa_cnn(self):
        from .cli import treinar_cnn   # import tardio: so exige torch aqui
        arquiteturas = self.ARQUITETURAS_COMPARAR if self.comparar_cnn else [self.modelo_cnn]
        for arq in arquiteturas:
            self._titulo(f"[4] MODULO CNN - {arq} (PyTorch)")
            argv = ["tc-treinar-cnn", "--modelo", arq, "--epocas", str(self.epocas_cnn)]
            if self.finetune:
                argv.append("--finetune")
            self._rodar(treinar_cnn, argv)

    # ---- Orquestracao -----------------------------------------------------
    def executar(self):
        """Roda o pipeline completo conforme a configuracao."""
        if self.com_testes:
            self.etapa_testes()

        self.etapa_estruturado()
        self.etapa_previsao_demo()

        if self.extrair_radiomica:
            self.etapa_radiomica_extracao()
        if self.com_radiomica:
            self.etapa_radiomica()

        if self.com_cnn or self.comparar_cnn:
            if self.preparar_manifesto:
                self.etapa_manifesto()
            self.etapa_cnn()

        self._titulo("PIPELINE CONCLUIDO. Resultados em outputs/.")
