import pandas as pd
import h5py
import pickle
import os
import numpy as np

def exportar_para_excel():
    base_dir = r'C:\IC\Projeto_V2'
    raw_dir = os.path.join(base_dir, 'data', 'brutos')
    output_path = os.path.join(base_dir, 'data', 'tratados', 'Dataset_METR-LA_Completo.xlsx')

    # 1. Carregar Velocidades
    with h5py.File(os.path.join(raw_dir, 'METR-LA.h5'), 'r') as f:
        data = f['df']['block0_values'][:]
        sensor_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['df']['axis0'][:]]
        timestamps = pd.to_datetime(f['df']['axis1'][:])
        df_vel = pd.DataFrame(data.T if data.shape[0] == len(sensor_ids) else data, 
                              columns=[f"Sensor_{sid}" for sid in sensor_ids])
        df_vel.insert(0, 'Data_Hora', timestamps)

    # 2. Carregar Grafo
    with open(os.path.join(raw_dir, 'adj_METR-LA.pkl'), 'rb') as f:
        ids, _, adj_matrix = pickle.load(f, encoding='latin1')
    
    rows, cols = np.where(adj_matrix > 0)
    df_edges = pd.DataFrame({
        'Origem': [f"Sensor_{ids[r]}" for r in rows],
        'Destino': [f"Sensor_{ids[c]}" for c in cols],
        'Peso_Conexao': adj_matrix[rows, cols]
    })

    # 3. Salvar em Excel com abas
    with pd.ExcelWriter(output_path) as writer:
        df_vel.to_excel(writer, sheet_name='Séries_Temporais', index=False)
        df_edges.to_excel(writer, sheet_name='Conexões_Grafo', index=False)
    
    print(f"✅ Planilha completa gerada em: {output_path}")

if __name__ == "__main__":
    exportar_para_excel()