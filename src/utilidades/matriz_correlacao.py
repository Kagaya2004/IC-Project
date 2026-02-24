import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def gerar_correlacao_pearson():
    # Carregar dados tratados
    caminho = r'C:\IC\Projeto_V2\data\tratados\dataset_PosFeaturesEngineered.csv'
    df = pd.read_csv(caminho)

    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    # Calcular a matriz de Pearson explicitamente
    # Pearson mede a correlação linear entre as variáveis (-1 a +1)
    corr = numeric_df.corr(method='pearson')

    # Criar máscara para o triângulo superior (limpa a visualização)
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, mask=0, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=0.5, annot_kws={"size": 8})
    
    plt.title('Matriz de Correlação de Pearson: Variáveis de Tráfego', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Salvar para o relatório
    plt.savefig(r'C:\IC\Projeto_V2\data\matriz_correlacao_pearson.png')
    print("✅ Matriz de Pearson salva em: data/matriz_correlacao_pearson.png")

if __name__ == "__main__":
    gerar_correlacao_pearson()