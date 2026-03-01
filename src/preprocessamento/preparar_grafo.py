import pandas as pd
import pickle
import torch
import numpy as np
import os
import h5py

def preparar_tudo():
    base_dir = r'C:\IC\Projeto_V2'
    raw_dir = os.path.join(base_dir, 'data', 'brutos')
    processed_dir = os.path.join(base_dir, 'data', 'tratados')
    os.makedirs(processed_dir, exist_ok=True)

    # --- 1. Converter METR-LA.h5 para CSV ---
    print("⏳ Passo 1: Extraindo velocidades do METR-LA.h5...")
    h5_path = os.path.join(raw_dir, 'METR-LA.h5')
    
    if not os.path.exists(h5_path):
        print(f"❌ Erro: Arquivo {h5_path} não encontrado!")
        return

    with h5py.File(h5_path, 'r') as f:
        # Extração dos dados brutos
        data = f['df']['block0_values'][:]
        sensor_ids = f['df']['axis0'][:]
        sensor_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sensor_ids]
        timestamps = f['df']['axis1'][:]
        
        # Criando nomes amigáveis para as colunas
        friendly_sensor_names = [f"Sensor_{sid}" for sid in sensor_ids]
        
        # Verificação de orientação (Garante que Linhas = Tempo, Colunas = Sensores)
        if data.shape[0] == len(sensor_ids):
            df_vel = pd.DataFrame(data.T, columns=friendly_sensor_names)
        else:
            df_vel = pd.DataFrame(data, columns=friendly_sensor_names)
        
        # Inserção do Timestamp como primeira coluna
        df_vel.insert(0, 'Timestamp', pd.to_datetime(timestamps))
        
        # Salvando o CSV inicial (que servirá de entrada para o dataCleaning.py)
        csv_path = os.path.join(processed_dir, 'METR-LA_series.csv')
        df_vel.to_csv(csv_path, index=False)
    
    print(f"✅ CSV gerado: {csv_path} ({df_vel.shape[0]} registros)")

    # --- 2. Processar a Matriz de Adjacência (Grafo) ---
    print("⏳ Passo 2: Processando grafo (adj_METR-LA.pkl)...")
    pkl_path = os.path.join(raw_dir, 'adj_METR-LA.pkl')
    
    if not os.path.exists(pkl_path):
        print(f"❌ Erro: Arquivo {pkl_path} não encontrado!")
        return

    with open(pkl_path, 'rb') as f:
        # O padrão METR-LA retorna: [sensor_ids, id_to_ind, adj_mx]
        adj_data = pickle.load(f, encoding='latin1')
        adj_mx = adj_data[2] 
    
    # Normalização básica e extração de arestas
    adj_mx = np.nan_to_num(adj_mx.astype(np.float32)) # Remove NaNs ou Infs
    rows, cols = np.where(adj_mx > 0)
    
    # Conversão para tensores do PyTorch
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(adj_mx[rows, cols], dtype=torch.float)
    
    # Salvando os tensores para a ST-GCN
    torch.save(edge_index, os.path.join(processed_dir, 'edge_index.pt'))
    torch.save(edge_attr, os.path.join(processed_dir, 'edge_attr.pt'))
    
    print(f"✅ Grafo salvo em {processed_dir}")
    print(f"🔗 Total de conexões detectadas: {edge_index.shape[1]}")

if __name__ == "__main__":
    preparar_tudo()