# 📈 Predição de Vendas com Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Projeto de **Machine Learning** para prever vendas a partir do investimento em canais de propaganda (TV, Rádio e Jornal). Compara **Regressão Linear** e **Random Forest**, escolhe o melhor modelo e aplica previsões em novos cenários.

Ideal para estudos em ciência de dados, análise de impacto de mídia e introdução a pipelines de ML em Python.

---

## 🎯 O que o projeto faz

- **Análise exploratória**: heatmap de correlação entre investimentos e vendas  
- **Modelagem**: treinamento de Regressão Linear e Random Forest com divisão treino/teste  
- **Avaliação**: comparação por R² e seleção automática do melhor modelo  
- **Visualização**: gráficos salvos em `output/` (correlação e comparação de previsões)  
- **Previsão**: aplicação do modelo escolhido em novos dados e export para CSV  

---

## 📂 Estrutura do repositório

```
sales-prediction-ml/
├── data/
│   ├── advertising.csv   # Dados históricos (TV, Radio, Jornal, Vendas)
│   └── novos.csv         # Novos cenários para previsão
├── src/
│   └── model.py          # Pipeline completo: treino, avaliação e previsão
├── output/               # Gerado ao rodar (gráficos + previsões)
├── docs/
│   └── explicacao.md     # Passo a passo do código
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Como executar

**Pré-requisito:** Python 3.8 ou superior.

1. **Clonar o repositório**
   ```bash
   git clone https://github.com/JoseOtavioJunqueira/sales-prediction-ml.git
   cd sales-prediction-ml
   ```

2. **Criar ambiente virtual (recomendado)**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Instalar dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Rodar o pipeline**
   ```bash
   python src/model.py
   ```

Os gráficos e o CSV de previsões serão salvos na pasta `output/`.

---

## 📊 Tecnologias

| Ferramenta        | Uso                          |
|-------------------|------------------------------|
| **Python**        | Linguagem base               |
| **Pandas**        | Leitura e manipulação de dados |
| **Matplotlib**    | Geração de gráficos          |
| **Seaborn**       | Visualizações estatísticas   |
| **Scikit-learn**  | Modelos e métricas de ML     |

---

## 📜 Licença

Este projeto está sob a licença **MIT**. Ver [LICENSE](LICENSE).

---

## 👤 Autor

**José Otávio Junqueira Ramos**  
Projeto desenvolvido com fins didáticos em Ciência de Dados e Machine Learning.
