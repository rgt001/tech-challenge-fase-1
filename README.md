# Tech Challenge Fase 1

Projeto standalone em Python para apoio ao diagnóstico de câncer de mama, com duas frentes:

- dados estruturados com scikit-learn
- mamografias com CNN e radiômica

## Estrutura

```text
Projeto Tech Challenge 1/
├── data/
├── notebooks/
├── outputs/
├── src/techchallenge/
├── tests/
├── executar_projeto.py
├── pyproject.toml
├── requirements.txt
└── requirements-cnn.txt
```

## Requisitos

- Python 3.10+
- opcionalmente GPU NVIDIA para a parte de CNN

## Instalação

```powershell
pip install -e .
```

Para a parte de CNN:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[cnn]"
```

## Como executar

Pipeline principal:

```powershell
python -m techchallenge
```

Pipeline com CNN:

```powershell
python -m techchallenge --com-cnn
```

Comparação de arquiteturas CNN:

```powershell
python -m techchallenge --comparar-cnn
```

Treino estruturado:

```powershell
python -m techchallenge.cli.treinar_estruturado
```

Predição tabular:

```powershell
python -m techchallenge.cli.prever_estruturado --csv data\amostra_previsao.csv
```

Treino CNN:

```powershell
python -m techchallenge.cli.treinar_cnn --modelo resnet18 --finetune --epocas 25
```

Predição em imagem:

```powershell
python -m techchallenge.cli.prever_cnn --imagem caminho\da\imagem.jpg
```

Radiômica:

```powershell
python -m techchallenge.cli.extrair_radiomica
python -m techchallenge.cli.treinar_radiomica
```

## Datasets usados

- Breast Cancer Wisconsin
- CBIS-DDSM

## Observações

- arquivos pesados de imagem, pesos de modelo e artefatos grandes não são versionados
- a pasta `docs/` fica fora do repositório
- o projeto pode ser executado tanto pelos módulos Python quanto pelos scripts em `src/techchallenge/cli/`
