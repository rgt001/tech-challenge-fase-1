"""EXECUTA O PROJETO INTEIRO - do comeco ao fim, num arquivo so.

Como usar (o mais simples):
    - No VSCode: abra este arquivo e clique em Run (o triangulo) / F5.
    - Ou no terminal, dentro da pasta do projeto:
          python executar_projeto.py

O que ele faz sozinho, em ordem:
    [0] Testes rapidos (pytest) - garante que o codigo esta ok.
    [1] Treina os modelos com PARTE do banco (split treino/teste) - estruturado.
    [2] Observacao / previsao   - usa o modelo treinado num CSV de exemplo.
    [3] Radiomica               - ML classico sobre as features das imagens.

Opcoes (nao obrigatorias):
    python executar_projeto.py --rapido         # treino sem validacao cruzada (bem mais rapido)
    python executar_projeto.py --sem-testes      # pula os testes
    python executar_projeto.py --sem-radiomica   # pula a radiomica
    python executar_projeto.py --com-cnn         # tambem treina a CNN (precisa de torch/GPU)
"""
import sys
from pathlib import Path

# Faz o pacote 'techchallenge' ser encontrado mesmo SEM 'pip install -e .'
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from techchallenge.__main__ import main


if __name__ == "__main__":
    main()
