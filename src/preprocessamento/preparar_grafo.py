import pandas as pd
import pickle
import torch
import numpy as np
import os
import h5py
import scipy.sparse as sp

def calculate_random_walk_matrix(adj_mx):
    """
    Calcula a matriz de transição para a DCRNN (Random Walk).
    P = D_out^-1 * W
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def preparar_tudo():
    base_dir = r'C:\IC\Projeto_V2'
    raw_dir = os.path.join(base_dir, 'data', 'brutos')
    processed_dir = os.path.join(base_dir, 'data', 'tratados')
    os.makedirs(processed_dir, exist_ok=True)

    # --- 1. Extração do CSV ---
    print("⏳ Passo 1: Extraindo velocidades...")
    h5_path = os.path.join(raw_dir, 'METR-LA.h5')
    with h5py.File(h5_path, 'r') as f:
        data = f['df']['block0_values'][:]
        sensor_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['df']['axis0'][:]]
        timestamps = f['df']['axis1'][:]
        df_vel = pd.DataFrame(data.T if data.shape[0] == len(sensor_ids) else data, 
                              columns=[f"Sensor_{sid}" for sid in sensor_ids])
        df_vel.insert(0, 'Timestamp', pd.to_datetime(timestamps))
        df_vel.to_csv(os.path.join(processed_dir, 'METR-LA_series.csv'), index=False)

    # --- 2. Processamento dos Grafos ---
    print("⏳ Passo 2: Processando matrizes para STGCN e DCRNN...")
    with open(os.path.join(raw_dir, 'adj_METR-LA.pkl'), 'rb') as f:
        adj_data = pickle.load(f, encoding='latin1')
        adj_mx = adj_data[2]

    # --- A. FORMATO STGCN  ---
    adj_stgcn = np.nan_to_num(adj_mx.astype(np.float32))
    rows, cols = np.where(adj_stgcn > 0)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(adj_stgcn[rows, cols], dtype=torch.float)
    
    torch.save(edge_index, os.path.join(processed_dir, 'edge_index_stgcn.pt'))
    torch.save(edge_attr, os.path.join(processed_dir, 'edge_attr_stgcn.pt'))

    # --- B. FORMATO DCRNN (Dual Random Walk) ---
    # DCRNN usa a difusão em dois sentidos: Direto (K-step forward) e Reverso (K-step backward)
    adj_dcrnn_list = []
    # Matriz Forward (Fluxo normal)
    adj_dcrnn_list.append(calculate_random_walk_matrix(adj_stgcn).toarray())
    # Matriz Backward (Fluxo inverso - ajuda a capturar influências de upstream)
    adj_dcrnn_list.append(calculate_random_walk_matrix(adj_stgcn.T).toarray())
    
    # Salvamos como um tensor único [2, Nodes, Nodes]
    adj_dcrnn_tensor = torch.tensor(np.stack(adj_dcrnn_list), dtype=torch.float)
    torch.save(adj_dcrnn_tensor, os.path.join(processed_dir, 'adj_mx_dcrnn.pt'))

    print(f"✅ Sucesso!")
    print(f"🔗 STGCN: {edge_index.shape[1]} arestas salvas.")
    print(f"🌊 DCRNN: Matriz de difusão {adj_dcrnn_tensor.shape} salva.")

if __name__ == "__main__":
    preparar_tudo()