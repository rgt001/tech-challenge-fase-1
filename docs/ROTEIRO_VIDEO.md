# Roteiro do Vídeo de Demonstração — Tech Challenge Fase 1

**Duração alvo:** ~12 minutos (limite do desafio: 15 min)
**Formato:** gravação de tela com narração. Upload no YouTube/Vimeo (não listado).

---

## Antes de gravar (checklist)

- [ ] Ambiente ativado e dependências instaladas (`pip install -e .` e `.[cnn]`).
- [ ] Terminal com fonte grande e tema legível.
- [ ] Notebook `notebooks/01_analise_breast_cancer.ipynb` já executado (para não esperar durante a gravação).
- [ ] Modelo da CNN já treinado (`outputs/modelo_cnn_transfer.pt`) para demonstrar a inferência sem esperar o treino.
- [ ] Pasta `outputs/` com os gráficos gerados, aberta ao lado.
- [ ] Fechar notificações e abas desnecessárias.

---

## Timeline e falas

### Bloco 1 — Abertura e problema (0:00 – 1:30)

**Mostrar:** slide/tela inicial ou o README aberto.

**Falar:**
> "Olá, este é o nosso Tech Challenge da Fase 1. O desafio era criar a base de um sistema de IA, com Machine Learning, para apoiar o diagnóstico e a detecção de riscos na saúde da mulher. Escolhemos o problema de classificação de câncer de mama: a partir de características do exame, o modelo indica se o tumor é maligno ou benigno. E, como item extra, adicionamos um classificador de mamografias com redes neurais convolucionais."

### Bloco 2 — Arquitetura do projeto (1:30 – 3:00)

**Mostrar:** a árvore de pastas em `src/techchallenge/` (data, models, evaluation, cli), o `pyproject.toml`.

**Falar:**
> "Organizamos o projeto como um pacote Python profissional, no chamado src layout. Cada responsabilidade fica em um módulo: `data` para carga e preparação, `models` para os modelos, `evaluation` para métricas e explicabilidade, e `cli` para os comandos. Tem também um ponto de entrada único e uma classe orquestradora, o `Pipeline`, que executa todas as etapas em ordem."

### Bloco 3 — Execução automática do sistema (3:00 – 4:30)

**Mostrar:** rodar no terminal:
```
python -m techchallenge
```

**Falar:**
> "Com um único comando o sistema roda o pipeline inteiro: carrega os dados, faz o pré-processamento, treina os três modelos e avalia. Repare que ele imprime a tabela de métricas ao final. Se eu adicionar `--com-cnn`, ele também treina a rede neural das mamografias."

### Bloco 4 — Análise no notebook (4:30 – 7:30)

**Mostrar:** o notebook executado, rolando pelas seções.

**Falar (por seção):**
- **EDA:** "A base tem 569 casos, com um desbalanceamento moderado — 63% benignos. As features de tamanho e irregularidade do núcleo, como concave points e perimeter_worst, são as mais correlacionadas com malignidade."
- **Pré-processamento:** "Fazemos split estratificado 80/20 e padronização dentro de um Pipeline, o que evita vazamento de dados."
- **Modelagem:** "Treinamos três técnicas: Regressão Logística, Árvore de Decisão e KNN."
- **Avaliação:** "A métrica que mais importa aqui é o recall da classe maligna, porque o pior erro é deixar passar um tumor maligno — um falso negativo. A Regressão Logística foi a melhor, com recall de 93% e só 3 falsos negativos."
- **Explicabilidade:** "Usamos feature importance e SHAP. Os dois concordam sobre quais características pesam mais — e batem com o conhecimento clínico. Isso é essencial: o médico precisa entender o porquê da previsão."

### Bloco 5 — Fase extra: CNN em mamografias (7:30 – 10:00)

**Mostrar:** os gráficos `cnn_transfer_historico.png` e `cnn_transfer_matriz.png`; opcionalmente rodar a inferência:
```
tc-prever --imagem <uma_imagem>.jpg
```

**Falar:**
> "No item extra, usamos transfer learning com a MobileNetV2 e fizemos fine-tuning em duas fases. O resultado foi de 64% de acurácia, com recall de 72% para a classe maligna. É um resultado modesto perto dos dados estruturados, e isso é esperado: mamografia em recorte é uma tarefa difícil, e as features pré-treinadas vêm de fotos comuns, não de imagens médicas. Mesmo assim, o sistema já classifica uma nova imagem, como mostro aqui na inferência."

### Bloco 6 — Decisões técnicas (10:00 – 11:30)

**Mostrar:** esta seção do roteiro ou a tabela do README/relatório.

**Falar:** (ver a seção "Decisões técnicas" abaixo — este é o diferencial do vídeo).

### Bloco 7 — Uso na prática e encerramento (11:30 – 12:00)

**Falar:**
> "Por fim, é importante deixar claro: este sistema é uma ferramenta de apoio e triagem, não substitui o médico. Ele sinaliza risco e é auditável pela explicabilidade, mas a palavra final é sempre do profissional de saúde. Obrigado!"

---

## Decisões técnicas — tecnologias testadas e trocadas

> Este é um ótimo momento do vídeo para mostrar o raciocínio de engenharia: nem tudo deu certo de primeira, e cada troca teve um motivo.

**1. TensorFlow → PyTorch (para a CNN).**
Começamos a CNN em TensorFlow. Descobrimos dois problemas: o TensorFlow, a partir da versão 2.11, **não usa a GPU NVIDIA no Windows nativo** — exigiria rodar dentro do Linux via WSL2. Além disso, ele **não tinha versão compatível com o Python 3.14** que eu uso. Trocamos para o **PyTorch**, que usa a GPU (CUDA) direto no Windows e suporta o Python novo. Resultado: passamos a treinar na RTX 2060 sem instalar Linux.

**2. WSL2 / Ubuntu (Linux) → descartado.**
Cheguei a preparar o passo a passo para instalar o WSL2 (um Linux dentro do Windows) só para o TensorFlow enxergar a GPU. Como o PyTorch resolveu no Windows nativo, **abandonamos esse caminho** — evitando instalar um ambiente Linux extra na máquina.

**3. Ambiente Python 3.12 dedicado.**
Como o TensorFlow não rodava no Python 3.14, criamos um ambiente virtual isolado. Mesmo depois de migrar para PyTorch, mantivemos a boa prática do ambiente virtual separado para as dependências pesadas.

**4. CNN com base congelada → fine-tuning em duas fases.**
O primeiro treino, com a base da MobileNetV2 totalmente congelada, ficou em ~58% de acurácia. Adicionamos o **fine-tuning** (descongelar as últimas camadas com learning rate baixo), o que subiu para **~64%** e equilibrou o modelo.

**5. Código monolítico → pacote modular.**
No começo o código ficou concentrado (tudo junto no notebook e em scripts soltos). Refatoramos para um **pacote Python com módulos separados** por responsabilidade, e o notebook passou a **importar do pacote** em vez de duplicar código.

**6. Detalhe de arquitetura — backend de gráficos.**
As bibliotecas forçavam o backend `Agg` do matplotlib (bom para salvar imagens sem tela), mas isso **quebrava a exibição no notebook**. Movemos o `Agg` para os comandos de linha (modo headless) e deixamos a biblioteca neutra — assim o notebook mostra os gráficos inline e as CLIs continuam salvando os arquivos.

---

## Dicas de gravação

- Fale com calma; é melhor sobrar tempo do que passar de 15 min.
- Se algum comando demorar, corte na edição ou use resultados já prontos.
- Destaque sempre o "porquê" das escolhas — é o que diferencia o trabalho.
- Termine reforçando a mensagem ética: apoio ao médico, não substituição.
