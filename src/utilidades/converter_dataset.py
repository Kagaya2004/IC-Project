import pandas as pd
import numpy as np
import h5py
import pickle
import os

def converter_dataset():
    base_dir = r'C:\IC\Projeto_V2'
    raw_dir = os.path.join(base_dir, 'data', 'brutos')
    output_dir = os.path.join(base_dir, 'data', 'tratados')
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Iniciando exportação com nomes amigáveis (Sensor_X)...")

    # --- 1. Velocidades (METR-LA.h5) ---
    h5_path = os.path.join(raw_dir, 'METR-LA.h5')
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            data = f['df']['block0_values'][:]
            sensor_ids = f['df']['axis0'][:]
            timestamps = f['df']['axis1'][:]
            
            # Criar lista de nomes: [Sensor_773869, Sensor_767541, ...]
            sensor_ids_str = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sensor_ids]
            friendly_names = [f"Sensor_{sid}" for sid in sensor_ids_str]
            
            if data.shape[0] == len(sensor_ids):
                df_vel = pd.DataFrame(data.T, columns=friendly_names)
            else:
                df_vel = pd.DataFrame(data, columns=friendly_names)
            
            df_vel.insert(0, 'Data_Hora', pd.to_datetime(timestamps))
            
            out_vel = os.path.join(output_dir, 'Tabela_Velocidades_Sensores.csv')
            df_vel.to_csv(out_vel, index=False)
            print(f"✅ Tabela de Velocidades salva: {out_vel}")

    # --- 2. Conexões do Grafo (adj_METR-LA.pkl) ---
    pkl_path = os.path.join(raw_dir, 'adj_METR-LA.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            ids, _, adj_matrix = pickle.load(f, encoding='latin1')
            
        rows, cols = np.where(adj_matrix > 0)
        edges = []
        for r, c in zip(rows, cols):
            edges.append({
                'Origem': f"Sensor_{ids[r]}",
                'Destino': f"Sensor_{ids[c]}",
                'Peso_Conexao': adj_matrix[r, c]
            })
        
        df_edges = pd.DataFrame(edges)
        out_edges = os.path.join(output_dir, 'Tabela_Conexoes_Grafo.csv')
        df_edges.to_csv(out_edges, index=False)
        print(f"✅ Tabela de Conexões salva: {out_edges}")

if __name__ == "__main__":
    converter_dataset()