# Tech Challenge — Fase 1 (IADT)
## Sistema de Apoio ao Diagnóstico em Saúde da Mulher

Sistema de Machine Learning para triagem em saúde da mulher, com dois módulos:

- **Estruturado** — classifica tumores de mama como maligno/benigno a partir de características de exames (dataset Breast Cancer Wisconsin), com scikit-learn.
- **Visão Computacional (extra)** — classifica recortes de mamografia (CBIS-DDSM) com uma CNN em PyTorch (usa GPU NVIDIA).

## Arquitetura

Projeto organizado como pacote Python (*src layout*):

```
projeto/
├── pyproject.toml            # metadados, dependências e comandos do sistema
├── requirements.txt          # deps do módulo estruturado
├── requirements-cnn.txt      # deps da CNN (PyTorch)
├── data/                     # breast_cancer.csv, cnn_manifest.csv
├── notebooks/                # 01_analise_breast_cancer.ipynb (demonstração visual)
├── outputs/                  # gráficos, métricas e modelos gerados
├── docs/                     # RELATORIO_TECNICO.md
└── src/techchallenge/        # o pacote do sistema
    ├── config.py             # caminhos e constantes centrais
    ├── data/                 # carga e preparação
    │   ├── estruturado.py    #   dados tabulares (breast cancer)
    │   └── imagens.py        #   manifesto + Dataset PyTorch (mamografias)
    ├── models/               # modelos
    │   ├── estruturado.py    #   pipelines scikit-learn
    │   └── cnn.py            #   arquiteturas e treino PyTorch
    ├── evaluation/           # avaliação
    │   ├── metrics.py        #   métricas e matriz de confusão
    │   └── explicabilidade.py#   feature importance + SHAP
    └── cli/                  # pontos de entrada (linha de comando)
        ├── treinar_estruturado.py
        ├── preparar_manifesto.py
        ├── treinar_cnn.py
        └── prever_cnn.py
```

## Instalação

Módulo estruturado (Python 3.10+):

```powershell
pip install -e .
```

Isso instala o pacote e cria os comandos `tc-treinar-estruturado`, `tc-prever-estruturado`, `tc-preparar-manifesto`, `tc-treinar-cnn` e `tc-prever`.

Módulo CNN (PyTorch com GPU NVIDIA, no Windows nativo — sem WSL2):

```powershell
# instale primeiro o torch pelo índice CUDA (confira a versão em pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[cnn]"
```

> Sem GPU: `pip install torch torchvision` (versão CPU).

## Como executar

### Execução automática (ponto de entrada único)

O sistema tem um orquestrador (classe `Pipeline`) que roda as etapas em ordem:

```powershell
python -m techchallenge                 # roda o módulo estruturado completo
python -m techchallenge --com-cnn       # também treina UMA CNN (requer torch/GPU)
python -m techchallenge --comparar-cnn  # treina e compara VÁRIAS arquiteturas de CNN
tc --com-cnn                            # idem, se instalado com pip install -e .
```

Opções úteis: `--modelo-cnn densenet` (escolhe a arquitetura), `--epocas-cnn 25`, `--sem-finetune`, `--preparar-manifesto`. Com `--comparar-cnn`, o sistema treina mobilenet, densenet, resnet18 e efficientnet em sequência, cada uma com seus próprios resultados em `outputs/`.

### Comandos individuais

Módulo estruturado (câncer de mama) — treino, avaliação e explicabilidade:

```powershell
tc-treinar-estruturado
# ou, sem instalar:  python -m techchallenge.cli.treinar_estruturado
```

Além dos gráficos e métricas, esse comando agora salva os artefatos de inferência do módulo estruturado em `outputs/`:

- `modelo_estruturado_stacking.joblib` — melhor modelo tabular para uso prático
- `modelo_estruturado_logistica.joblib` — baseline interpretável
- `modelo_estruturado_meta.json` — schema esperado e limiares sugeridos

Para classificar um novo exame tabular a partir de um CSV:

```powershell
tc-prever-estruturado --csv caminho/do_exame.csv

# se quiser usar o limiar clínico que zera falsos negativos no teste
tc-prever-estruturado --csv caminho/do_exame.csv --priorizar-recall
```

Também há o notebook `notebooks/01_analise_breast_cancer.ipynb` com a análise visual completa (ideal para demonstração).

Módulo CNN (mamografias):

```powershell
# 1. gerar o manifesto (já incluso em data/cnn_manifest.csv)
tc-preparar-manifesto

# 2. treinar (transfer learning + fine-tuning; usa a GPU automaticamente)
tc-treinar-cnn --modelo transfer --finetune --epocas 25

#    teste rápido do pipeline
tc-treinar-cnn --limite 100 --epocas 2

# 3. classificar uma nova imagem
tc-prever --imagem caminho/da/imagem.jpg
```

> As imagens (`jpeg/`) e os metadados (`csv/`) do CBIS-DDSM ficam **dentro da própria pasta do projeto** (`projeto/jpeg/` e `projeto/csv/`), tornando-o autossuficiente. Os comandos usam esse caminho por padrão (`config.IMAGENS_RAIZ`). Por serem ~6 GB, essas pastas não vão para o Git (ver `.gitignore`).

## Datasets

- **Breast Cancer Wisconsin** — 569 casos, alvo `diagnosis` (M/B), 30 features. Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **CBIS-DDSM** (extra) — recortes de lesão (ROI) de mamografias, 3.566 imagens rotuladas (benigno/maligno) com split oficial treino/teste.

## Resultados — módulo estruturado (teste)

Seis modelos de base (cobrindo linear, regras, distância, ensembles e margem) mais um **Stacking** que os combina com um meta-modelo:

| Modelo | Accuracy | Recall (maligno) | F1 (maligno) |
|---|---|---|---|
| **Stacking** | **0.983** | **0.952** | **0.976** |
| SVM | 0.974 | 0.929 | 0.963 |
| Random Forest | 0.974 | 0.929 | 0.963 |
| Regressão Logística | 0.965 | 0.929 | 0.951 |
| Gradient Boosting | 0.965 | 0.905 | 0.950 |
| KNN | 0.956 | 0.905 | 0.938 |
| Árvore de Decisão | 0.921 | 0.833 | 0.886 |

O **Stacking** teve o melhor desempenho (AUC 0.995; apenas 2 erros no teste), confirmado por validação cruzada 5-fold. Métrica prioritária: **recall da classe maligna**, pois o falso negativo é o erro mais grave. Para produção, a Regressão Logística continua atraente pela interpretabilidade.

**Ajuste de limiar (finalidade clínica).** A CLI também gera uma análise de limiar: no corte padrão (0.5) sobram 2 falsos negativos; baixando o limiar para priorizar recall, é possível **zerar os falsos negativos** ao custo de 18 falsos positivos — uma escolha consciente alinhada à finalidade (não deixar passar câncer). Gráficos em `outputs/estruturado_limiar_*.png`.

## Resultados — módulo CNN (extra, teste)

Transfer learning sobre recortes de mamografia do CBIS-DDSM. Estão disponíveis várias arquiteturas para comparação (`--modelo mobilenet|densenet|resnet18|resnet50|efficientnet|custom`):

| Métrica | MobileNetV2 base congelada | MobileNetV2 com fine-tuning |
|---|---|---|
| Accuracy | 0.580 | 0.644 |
| Recall (maligno) | 0.822 | 0.725 |
| F1 (maligno) | 0.605 | 0.614 |

O fine-tuning elevou a acurácia. Desempenho modesto é esperado nessa base difícil; o modelo mantém recall alto para a classe maligna. Detalhes em `docs/RELATORIO_TECNICO.md`.

## Radiômica — imagens viram tabela (ML clássico nas mamografias)

Uma alternativa/complemento à CNN: em vez de uma rede neural, **extraímos ~76 features numéricas de cada mamografia** (radiômica clássica — intensidade, textura GLCM/Haralick, LBP, Gabor, gradiente/bordas e forma da lesão), transformando cada imagem em uma linha de números. Aí rodamos os **mesmos modelos estruturados** (incluindo o Stacking), com `class_weight='balanced'` e ajuste de limiar para priorizar o recall. Isso não mistura com o Breast Cancer Wisconsin (são pacientes/escalas diferentes) — é um **banco tabular derivado das próprias imagens**.

```powershell
python -m techchallenge.cli.extrair_radiomica          # gera data/radiomica_mamografia.csv
python -m techchallenge.cli.treinar_radiomica          # treina o Stacking e compara com a CNN
```

> A extração lê todas as imagens uma vez (alguns minutos na primeira execução). Use `--limite N` para uma amostra rápida. Permite comparar diretamente **radiômica + ML clássico** vs. **CNN** na mesma tarefa.

## Sistema multimodal (roteador)

As duas bases são de **pacientes diferentes**, então não há fusão dos resultados num único número. Em vez disso, o notebook `notebooks/02_estudo_integrado.ipynb` demonstra um **roteador por modalidade** (`diagnosticar()`): recebe *ou* um exame tabular *ou* uma imagem, reconhece o tipo e encaminha para o modelo especialista, devolvendo um laudo padronizado. Traz também o estudo comparativo (métricas e curvas ROC das duas modalidades). A fusão multimodal real (uma IA que use exame e imagem juntos) fica como trabalho futuro, pois exigiria dados pareados por paciente.

## Uso na prática

Ferramenta de **apoio e triagem** — não substitui o médico. O modelo sinaliza risco e é auditável via explicabilidade (SHAP/feature importance), mas o diagnóstico e a conduta são sempre responsabilidade do profissional de saúde.

Detalhes completos no relatório técnico em `docs/RELATORIO_TECNICO.md`.
