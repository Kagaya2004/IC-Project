import torch
import pandas as pd
import matplotlib.pyplot as plt

def visualizar_grafo_espacial(n_amostra=150):
    # Carregar Dados e o arquivo de arestas gerado
    df = pd.read_csv(r'C:\IC\Projeto_V2\data\tratados\dataset_PosFeaturesEngineered.csv')
    edge_index = torch.load(r'C:\IC\Projeto_V2\data\tratados\edge_index.pt', weights_only=True)

    plt.figure(figsize=(12, 8))

    # Filtrar arestas apenas para a amostra (evita sobrecarga visual)
    mask = (edge_index[0] < n_amostra) & (edge_index[1] < n_amostra)
    edges = edge_index[:, mask]

    # Desenhar conexões (arestas)
    for i in range(edges.shape[1]):
        u, v = edges[0, i].item(), edges[1, i].item()
        plt.plot([df.iloc[u]['Longitude'], df.iloc[v]['Longitude']],
                 [df.iloc[u]['Latitude'], df.iloc[v]['Latitude']],
                 color='black', alpha=0.2, linewidth=0.5, zorder=1)

    # Desenhar Sensores (nós) coloridos pela velocidade real
    scatter = plt.scatter(df.iloc[:n_amostra]['Longitude'], 
                          df.iloc[:n_amostra]['Latitude'], 
                          c=df.iloc[:n_amostra]['Traffic_Speed_kmh'], 
                          cmap='RdYlGn', s=40, edgecolors='black', zorder=2)

    plt.colorbar(scatter, label='Velocidade (Normalizada)')
    plt.title(f'Topologia do Grafo Urbano: Amostra de {n_amostra} Sensores', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.savefig(r'C:\IC\Projeto_V2\data\visualizacao_grafo_espacial.png')
    print("✅ Gráfico do Grafo salvo em: data/visualizacao_grafo_espacial.png")

if __name__ == "__main__":
    visualizar_grafo_espacial(n_amostra=200)