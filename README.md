# Tech Challenge Fase 1

Projeto standalone em Python para apoio ao diagnóstico de câncer de mama, com duas frentes principais:

- dados estruturados com scikit-learn
- mamografias com CNN e radiômica

## Visão geral

O projeto foi organizado para permitir três caminhos de análise:

- módulo estruturado com o dataset Breast Cancer Wisconsin
- módulo de imagem com mamografias do CBIS-DDSM
- estudo integrado, comparando as modalidades sem misturar pacientes diferentes

Os notebooks principais do repositório são:

- `notebooks/01-3_analise_breast_cancer.ipynb` — versão principal da análise estruturada
- `notebooks/02_estudo_integrado.ipynb` — estudo multimodal e comparação entre modalidades

## Estrutura do projeto

```text
tech-challenge-fase-1/
├── data/
├── notebooks/
├── outputs/
├── src/techchallenge/
├── tests/
├── executar_projeto.py
├── pyproject.toml
├── requirements.txt
└── requirements-analiseimagem.txt
```

## Requisitos

- Python 3.10+
- para a parte de CNN, ambiente com PyTorch
- para treino de CNN em volume real, GPU NVIDIA é fortemente recomendada

## Instalação

Instalação base:

```powershell
pip install -e .
```

Instalação com suporte à parte de imagem:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[cnn]"
```

Se preferir, consulte também o arquivo `requirements-analiseimagem.txt`.

## Datasets usados

- Breast Cancer Wisconsin
- CBIS-DDSM

## Organização esperada das imagens

Para o fluxo de mamografias funcionar sem parâmetros extras, o projeto espera a estrutura do CBIS-DDSM com estas pastas na raiz do projeto:

```text
tech-challenge-fase-1/
├── csv/
├── jpeg/
└── ...
```

Essas pastas são usadas por:

- `tc-preparar-manifesto`
- `tc-treinar-cnn`
- `tc-extrair-radiomica`

Se as imagens estiverem fora da raiz do projeto, use os parâmetros:

- `--raiz` no manifesto
- `--img-raiz` na CNN e na extração radiômica

## Como executar

### Pipeline principal

```powershell
python -m techchallenge
```

Por padrão, esse comando executa:

- testes automatizados
- treino do módulo estruturado
- previsão de demonstração com `data/amostra_previsao.csv`
- treino da radiômica, se `data/radiomica_mamografia.csv` existir

### Pipeline com CNN

```powershell
python -m techchallenge --com-cnn
```

### Comparação entre arquiteturas CNN

```powershell
python -m techchallenge --comparar-cnn
```

### Flags úteis do pipeline

```powershell
python -m techchallenge --sem-testes
python -m techchallenge --sem-radiomica
python -m techchallenge --extrair-radiomica
python -m techchallenge --preparar-manifesto --com-cnn
python -m techchallenge --rapido
```

Resumo das flags:

- `--sem-testes`: pula o `pytest`
- `--sem-radiomica`: pula o treino radiômico
- `--extrair-radiomica`: reextrai as features das imagens antes do treino radiômico
- `--preparar-manifesto`: regenera `data/cnn_manifest.csv`
- `--rapido`: pula a validação cruzada do módulo estruturado

## Execução por etapa

### Treino estruturado

```powershell
python -m techchallenge.cli.treinar_estruturado
```

### Predição tabular

```powershell
python -m techchallenge.cli.prever_estruturado --csv data\amostra_previsao.csv
```

Se quiser usar o limiar salvo para priorizar recall:

```powershell
python -m techchallenge.cli.prever_estruturado --csv data\amostra_previsao.csv --priorizar-recall
```

### Gerar manifesto das imagens

```powershell
python -m techchallenge.cli.preparar_manifesto
```

Exemplo com base fora da raiz do projeto:

```powershell
python -m techchallenge.cli.preparar_manifesto --raiz D:\caminho\CBIS-DDSM
```

### Treino CNN

```powershell
python -m techchallenge.cli.treinar_cnn --modelo resnet18 --finetune --epocas 25
```

Exemplo usando base fora da raiz do projeto:

```powershell
python -m techchallenge.cli.treinar_cnn --img-raiz D:\caminho\CBIS-DDSM --modelo resnet18 --finetune --epocas 25
```

### Predição em imagem

```powershell
python -m techchallenge.cli.prever_cnn --imagem caminho\da\imagem.jpg
```

Observação importante: a predição em imagem exige que já exista um arquivo `outputs/modelo_cnn_*.pt`. Se ele não existir, é necessário treinar a CNN antes.

### Radiômica

Extração:

```powershell
python -m techchallenge.cli.extrair_radiomica
```

Treino:

```powershell
python -m techchallenge.cli.treinar_radiomica
```

Se o arquivo `data/radiomica_mamografia.csv` já existir, o treino radiômico pode ser executado diretamente sem reextração.

## Saídas geradas

Os principais artefatos ficam em `outputs/`, por exemplo:

- modelos treinados do módulo estruturado
- gráficos de EDA
- matrizes de confusão
- curva ROC
- análise de limiar
- gráficos de explicabilidade
- métricas da CNN
- métricas e gráficos da radiômica

## Observações importantes

- o projeto é uma ferramenta de apoio à decisão, não substitui avaliação médica
- o módulo estruturado funciona mesmo sem GPU
- a parte de CNN depende de PyTorch e normalmente faz mais sentido em máquina com GPU
- o estudo integrado compara modalidades, mas não faz fusão real entre exame e imagem da mesma paciente
