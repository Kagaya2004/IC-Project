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

    # 1. Converter METR-LA.h5 para CSV com verificação de shape
    print("⏳ Convertendo METR-LA.h5...")
    h5_path = os.path.join(raw_dir, 'METR-LA.h5')
    
    with h5py.File(h5_path, 'r') as f:
        data = f['df']['block0_values'][:]
        sensor_ids = f['df']['axis0'][:]
        sensor_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sensor_ids]
        timestamps = f['df']['axis1'][:]
        
        # Criando nomes amigáveis: Sensor_773869, etc.
        friendly_sensor_names = [f"Sensor_{sid}" for sid in sensor_ids]
        
        if data.shape[0] == len(sensor_ids):
            df_vel = pd.DataFrame(data.T, columns=friendly_sensor_names)
        else:
            df_vel = pd.DataFrame(data, columns=friendly_sensor_names)
        
        # GARANTA QUE O NOME SEJA 'Timestamp'
        df_vel.insert(0, 'Timestamp', pd.to_datetime(timestamps))
        
        df_vel.to_csv(os.path.join(processed_dir, 'METR-LA_series.csv'), index=False)
    
    print(f"✅ CSV gerado com sucesso: {df_vel.shape[0]} linhas x {df_vel.shape[1]} colunas.")

    # 2. Processar a Matriz de Adjacência (Grafo)
    print("⏳ Processando matriz de adjacência...")
    with open(os.path.join(raw_dir, 'adj_METR-LA.pkl'), 'rb') as f:
        # O formato padrão do pkl do METR-LA retorna [ids, id_to_ind, adj_mx]
        adj_data = pickle.load(f, encoding='latin1')
        adj_mx = adj_data[2] # A matriz é o terceiro elemento
    
    adj_mx = adj_mx.astype(np.float32)
    rows, cols = np.where(adj_mx > 0)
    
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(adj_mx[rows, cols], dtype=torch.float)
    
    torch.save(edge_index, os.path.join(processed_dir, 'edge_index.pt'))
    torch.save(edge_attr, os.path.join(processed_dir, 'edge_attr.pt'))
    print(f"✅ Grafo preparado com {edge_index.shape[1]} conexões.")

if __name__ == "__main__":
    preparar_tudo()