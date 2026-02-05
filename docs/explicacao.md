# Explicação do código – Predição de Vendas

O script `src/model.py` segue os passos abaixo.

## 1. Leitura dos dados

O arquivo `data/advertising.csv` é carregado com Pandas. Colunas: **TV**, **Radio**, **Jornal** (investimento em cada mídia) e **Vendas** (alvo).

## 2. Análise exploratória

É gerado um **heatmap de correlação** entre as variáveis para ver quais investimentos mais se relacionam com as vendas. O gráfico é salvo em `output/correlacao.png`.

## 3. Preparação dos dados

- **Variáveis independentes (X):** TV, Radio, Jornal  
- **Variável dependente (Y):** Vendas  
- Os dados são divididos em **treino (80%)** e **teste (20%)** com `random_state=42` para reprodutibilidade.

## 4. Treinamento dos modelos

Dois modelos são treinados:

- **Regressão Linear**  
- **Random Forest** (ensemble de árvores de decisão)

## 5. Avaliação

As previsões no conjunto de teste são avaliadas com a métrica **R²** (coeficiente de determinação). O modelo com maior R² é escolhido como o melhor.

## 6. Visualização dos resultados

Um gráfico de linha compara os **valores reais** com as **previsões** de cada modelo. Salvo em `output/comparacao_modelos.png`.

## 7. Previsões futuras

O arquivo `data/novos.csv` é lido (apenas colunas TV, Radio, Jornal). O **melhor modelo** faz as previsões de vendas, que são exibidas no terminal e salvas em `output/previsoes.csv`.

---

Para mais detalhes, consulte o [README](../README.md) do repositório.
