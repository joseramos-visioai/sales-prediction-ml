"""
Predição de Vendas com Machine Learning

Treina e compara Regressão Linear e Random Forest para prever vendas
com base em investimentos em TV, Rádio e Jornal. Seleciona o melhor
modelo e aplica previsões em novos dados.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Diretórios (funciona em qualquer sistema operacional)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # Carregar os dados
    tabela = pd.read_csv(DATA_DIR / "advertising.csv")

    # Visualização da correlação entre os investimentos
    plt.figure(figsize=(8, 6))
    sns.heatmap(tabela.corr(), cmap="Wistia", annot=True, fmt=".2f")
    plt.title("Correlação entre Investimentos e Vendas")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlacao.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Separação das variáveis independentes (X) e dependente (Y)
    x = tabela[["TV", "Radio", "Jornal"]]
    y = tabela["Vendas"]

    # Divisão dos dados em treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Instanciar e treinar os modelos
    modelo_rl = LinearRegression()
    modelo_rf = RandomForestRegressor(random_state=42)

    modelo_rl.fit(x_treino, y_treino)
    modelo_rf.fit(x_treino, y_treino)

    # Previsões e avaliação (métrica R²)
    previsao_rl = modelo_rl.predict(x_teste)
    previsao_rf = modelo_rf.predict(x_teste)

    r2_rl = metrics.r2_score(y_teste, previsao_rl)
    r2_rf = metrics.r2_score(y_teste, previsao_rf)

    print(f"R² Regressão Linear:    {r2_rl:.4f}")
    print(f"R² Random Forest:       {r2_rf:.4f}")

    # Escolher o melhor modelo
    melhor_modelo = modelo_rl if r2_rl > r2_rf else modelo_rf
    nome_melhor = "Regressão Linear" if r2_rl > r2_rf else "Random Forest"
    print(f"\nMelhor modelo: {nome_melhor}")

    # Gráfico comparativo (valores reais vs previsões)
    tabela_comparacao = pd.DataFrame({
        "Real": y_teste.values,
        "Previsão RL": previsao_rl,
        "Previsão RF": previsao_rf,
    })
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=tabela_comparacao)
    plt.title("Comparação: valores reais vs previsões dos modelos")
    plt.xlabel("Amostra")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparacao_modelos.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Previsões para novos dados
    tabela_nova = pd.read_csv(DATA_DIR / "novos.csv")
    previsao_futura = melhor_modelo.predict(tabela_nova)
    tabela_nova = tabela_nova.copy()
    tabela_nova["Previsão_Vendas"] = previsao_futura

    print("\n📊 Previsão para novos dados:\n")
    print(tabela_nova.to_string(index=False))
    tabela_nova.to_csv(OUTPUT_DIR / "previsoes.csv", index=False)
    print(f"\nGráficos e previsões salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
