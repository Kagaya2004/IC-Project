import pandas as pd
import numpy as np
import torch
import os
from scipy.spatial.distance import cdist

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c 

def gerar_grafo_stgcn(threshold_km=1.5):
    base_dir = r'C:\IC\Projeto_V2'
    input_path = os.path.join(base_dir, 'data', 'tratados', 'dataset_PosFeaturesEngineered.csv')
    output_dir = os.path.join(base_dir, 'data', 'tratados')

    print(f"Carregando dados preparados de: {input_path}")
    df = pd.read_csv(input_path)
    
    # --- PASSO DE DIAGNÓSTICO ---
    print("\nVerificando e convertendo colunas:")
    
    # 1. Identifica o que não é número nem booleano (ex: strings residuais)
    cols_para_remover = df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    if cols_para_remover:
        print(f"⚠️ Removendo colunas incompatíveis: {cols_para_remover}")
        df = df.drop(columns=cols_para_remover)

    # 2. Garante que Latitude e Longitude não entrem nas features de treino
    x_columns = [col for col in df.columns if col not in ['Latitude', 'Longitude']]
    
    # 3. Converte tudo (incluindo bools do One-Hot) para float32
    x_data = df[x_columns].astype(np.float32).values
    x_tensor = torch.tensor(x_data, dtype=torch.float)

    # Extração das Coordenadas
    coords = df[['Longitude', 'Latitude']].values
    num_nodes = len(coords)

    # Construção da Matriz de Adjacência
    print(f"Calculando conexões (Raio: {threshold_km}km)...")
    dist_matrix = cdist(coords, coords, lambda u, v: haversine_np(u[0], u[1], v[0], v[1]))
    rows, cols = np.where((dist_matrix <= threshold_km) & (dist_matrix > 0))
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
    distances = dist_matrix[rows, cols]
    edge_attr = torch.tensor(1.0 / (distances + 1e-5), dtype=torch.float)

    # Preparação das Features de Nó (X)
    # Garantimos que Latitude e Longitude não entrem como features de treino
    x_columns = [col for col in df.columns if col not in ['Latitude', 'Longitude']]
    
    # CONVERSÃO FORÇADA: Converte para float32 antes de virar tensor do PyTorch
    x_data = df[x_columns].values.astype(np.float32) 
    x_tensor = torch.tensor(x_data, dtype=torch.float)

    # Salvamento
    torch.save(edge_index, os.path.join(output_dir, 'edge_index.pt'))
    torch.save(edge_attr, os.path.join(output_dir, 'edge_attr.pt'))
    torch.save(x_tensor, os.path.join(output_dir, 'x_features.pt'))

    print("-" * 40)
    print("✅ TENSORES GERADOS COM SUCESSO")
    print(f"Shape do Tensor X: {x_tensor.shape} (Nós, Features)")
    print(f"Arestas no Grafo: {edge_index.shape[1]}")
    print("-" * 40)

if __name__ == "__main__":
    gerar_grafo_stgcn()