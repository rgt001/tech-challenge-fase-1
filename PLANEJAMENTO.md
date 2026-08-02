# Tech Challenge — Fase 1 (IADT)
## Planejamento Faseado do Projeto

**Tema:** Sistema de IA/Machine Learning para suporte ao diagnóstico em saúde da mulher
**Dataset principal:** Breast Cancer Wisconsin (`data.csv`) — classificação maligno (M) vs. benigno (B)
**Linguagem:** Python
**Extra (opcional):** Visão Computacional com CNN sobre mamografias (CBIS-DDSM)

---

## Visão geral

O desafio pede a base de um sistema de IA focado em Machine Learning, capaz de analisar dados médicos e identificar padrões de risco na saúde da mulher. Vamos construir um classificador de câncer de mama a partir de dados estruturados, cobrindo todo o fluxo: exploração → pré-processamento → modelagem → avaliação → explicabilidade → discussão crítica.

**Requisitos mínimos obrigatórios (extraídos do enunciado):**

- 1 ou mais datasets públicos de saúde/segurança da mulher, com discussão do problema.
- Exploração de dados (estatísticas descritivas, distribuições, padrões).
- Pré-processamento (limpeza, pipeline em Python, conversão de variáveis, correlação).
- **2 ou mais técnicas** de classificação, com separação clara treino/teste.
- Avaliação com accuracy, recall e F1, discutindo a escolha da métrica.
- Explicabilidade (feature importance e SHAP).
- Discussão crítica sobre uso na prática (o médico tem a palavra final).
- Código Python estruturado e documentado (Jupyter ou scripts).

**Entregáveis finais:** PDF com link do repositório Git, código-fonte, README, dataset (ou link), resultados, relatório técnico e vídeo de até 15 min.

---

## Por que este dataset e este caminho

O `data.csv` (Breast Cancer Wisconsin) é um dos datasets sugeridos no próprio enunciado. Está limpo, é totalmente numérico, tem alvo binário bem definido (357 benignos / 212 malignos) e 30 features. Isso permite focar energia na qualidade da análise, dos modelos e da explicabilidade — em vez de gastar tempo tratando dados sujos. Um único dataset estruturado bem executado já cumpre o requisito; a CNN entra só como extra para pontuação adicional.

---

## Fase 0 — Setup do projeto

**Objetivo:** preparar ambiente e estrutura do repositório.

- Criar repositório Git (privado ou público, conforme regras do grupo).
- Estrutura de pastas sugerida:

```
tech-challenge-fase1/
├── data/
│   └── data.csv
├── notebooks/
│   └── 01_analise_breast_cancer.ipynb
├── src/                  # scripts Python reutilizáveis (opcional)
├── outputs/              # gráficos e resultados exportados
├── requirements.txt
├── README.md
└── relatorio_tecnico.pdf
```

- Ambiente Python: criar `requirements.txt` com `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `shap`, `jupyter`.
- Definir se vamos usar Jupyter Notebook (recomendado para demonstração visual) ou scripts.

**Entrega da fase:** repositório inicializado + ambiente rodando.

---

## Fase 1 — Exploração de dados (EDA)

**Objetivo:** entender a base e identificar padrões.

- Carregar `data.csv` com pandas e inspecionar formato (`shape`, `info`, `head`).
- Remover coluna de `id` e a coluna vazia final, se houver.
- Estatísticas descritivas (`describe`) das 30 features.
- Distribuição da variável alvo `diagnosis` (M vs. B) — verificar desbalanceamento.
- Visualizações: histogramas das principais features, boxplots por diagnóstico, heatmap de correlação.
- Identificar padrões: quais medidas (raio, área, concavidade etc.) mais separam maligno de benigno.

**Entrega da fase:** notebook com gráficos e discussão escrita dos achados.

---

## Fase 2 — Pré-processamento

**Objetivo:** deixar os dados prontos para modelagem.

- Tratar valores ausentes/inconsistentes (verificar `isnull`; o dataset costuma ter uma coluna extra vazia a remover).
- Converter o alvo `diagnosis` (M/B) em numérico (ex.: M=1, B=0).
- Padronização/escalonamento das features numéricas (`StandardScaler`) — importante para KNN e Regressão Logística.
- Análise de correlação e decisão sobre features muito redundantes.
- Montar um **pipeline** de pré-processamento em Python (`sklearn.pipeline`).
- Separação treino/teste (ex.: 80/20) com `train_test_split` e `stratify` no alvo.

**Entrega da fase:** pipeline reproduzível + conjuntos de treino e teste definidos.

---

## Fase 3 — Modelagem

**Objetivo:** treinar 2 ou mais classificadores (requisito obrigatório).

Modelos propostos (mínimo dois, sugerimos três para comparação):

- **Regressão Logística** — baseline interpretável.
- **Árvore de Decisão** — regras claras, boa para explicabilidade.
- **K-Nearest Neighbors (KNN)** — comparação por similaridade.
- (Opcional) Random Forest — costuma dar melhor desempenho e casa bem com feature importance.

- Treinar cada modelo no conjunto de treino.
- Guardar os modelos para avaliação comparativa.

**Entrega da fase:** modelos treinados e prontos para avaliação.

---

## Fase 4 — Treinamento e avaliação

**Objetivo:** medir e comparar desempenho com métricas adequadas.

- Prever no conjunto de teste com cada modelo.
- Calcular **accuracy, recall e F1-score** (e matriz de confusão).
- **Discussão da métrica:** em diagnóstico de câncer, o **recall (sensibilidade)** para a classe "maligno" é o mais crítico — deixar de detectar um caso maligno (falso negativo) é muito mais grave do que um falso positivo. Vamos justificar isso no relatório.
- Tabela comparativa dos modelos.

**Entrega da fase:** tabela de métricas + matriz de confusão de cada modelo.

---

## Fase 5 — Explicabilidade e interpretação

**Objetivo:** explicar *por que* o modelo decide (requisito obrigatório).

- **Feature importance** (via Árvore/Random Forest ou coeficientes da Regressão Logística).
- **SHAP:** gráficos summary e de dependência para mostrar o impacto de cada feature nas previsões.
- Interpretar quais características medem maior risco de malignidade.
- **Discussão crítica:** o modelo pode ser usado na prática? Como? Reforçar que é uma ferramenta de **triagem e apoio à decisão** — o médico sempre tem a palavra final.

**Entrega da fase:** gráficos SHAP/feature importance + análise crítica escrita.

---

## Fase 6 — Extra opcional: CNN em mamografias

> Só será feita se houver tempo. Não é obrigatória, mas pode aumentar a nota.

**Objetivo:** classificar imagens de mamografia (CBIS-DDSM).

- Dados disponíveis: metadados em `csv/` (`calc_case_description_*`, `mass_case_description_*`, `dicom_info.csv`, `meta.csv`) + imagens na pasta `jpeg/`.
- Passos: entender os metadados, mapear imagens aos rótulos (benigno/maligno), pré-processar imagens (redimensionar/normalizar), montar uma CNN (ex.: com Keras/TensorFlow), treinar e avaliar.
- Avaliar com as mesmas métricas (accuracy, recall, F1).

**Entrega da fase:** notebook de CNN com resultados (se realizada).

---

## Fase 7 — Documentação e entregáveis finais

**Objetivo:** montar tudo para a entrega.

- **README.md** com instruções de execução (como rodar o notebook, instalar dependências).
- **Relatório técnico** explicando: discussões da EDA, estratégias de pré-processamento, modelos usados e porquê, resultados e interpretação.
- Exportar prints, gráficos e análises para o `outputs/`.
- **PDF final** com o link do repositório Git e o resumo dos resultados.
- **Vídeo** (até 15 min, YouTube/Vimeo não listado) demonstrando o sistema em execução.
- `Dockerfile` — apenas se decidirmos containerizar (opcional).

**Entrega da fase:** PDF + repositório completo + vídeo.

---

## Divisão sugerida (trabalho em grupo)

O enunciado indica que, em princípio, é atividade em grupo. Sugestão de frentes que podem correr em paralelo:

- **Frente A:** EDA + pré-processamento (Fases 1–2).
- **Frente B:** modelagem + avaliação (Fases 3–4).
- **Frente C:** explicabilidade + relatório técnico + vídeo (Fases 5, 7).
- **Frente D (opcional):** CNN (Fase 6).

---

## Checklist de conformidade com o enunciado

- [ ] Dataset público de saúde da mulher escolhido e problema discutido
- [ ] Exploração de dados com estatísticas e visualizações
- [ ] Limpeza e pipeline de pré-processamento em Python
- [ ] Conversão de variáveis categóricas/numéricas
- [ ] Análise de correlação
- [ ] 2+ técnicas de classificação
- [ ] Separação treino/teste
- [ ] Avaliação com accuracy, recall e F1 + discussão da métrica
- [ ] Explicabilidade (feature importance + SHAP)
- [ ] Discussão crítica sobre uso na prática
- [ ] Código Python estruturado e documentado
- [ ] README.md com instruções
- [ ] Relatório técnico
- [ ] PDF final com link do Git
- [ ] Vídeo de demonstração (até 15 min)
- [ ] (Extra) CNN em mamografias
