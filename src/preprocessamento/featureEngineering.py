import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def engineer_features():
    base_dir = r'C:\IC\Projeto_V2'
    input_path = os.path.join(base_dir, 'data', 'tratados', 'dataset_Limpo.csv')
    output_path = os.path.join(base_dir, 'data', 'tratados', 'dataset_PosFeaturesEngineered.csv')
    
    print(f"Processando Engenharia de Atributos em: {input_path}")
    df = pd.read_csv(input_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 1. Atributos Temporais Cíclicos (Preservando a periodicidade)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)
    df['Day_sin'] = np.sin(2 * np.pi * df['Timestamp'].dt.dayofweek / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Timestamp'].dt.dayofweek / 7)
    
    # 2. One-Hot Encoding (Clima e Semáforos)
    # Weather_Condition e Traffic_Light_State expandidos em colunas binárias
    cols_one_hot = ['Weather_Condition', 'Traffic_Light_State']
    df = pd.get_dummies(df, columns=cols_one_hot, prefix=cols_one_hot, dtype=int)
    print("  > One-Hot Encoding aplicado para Clima e Semáforos.")

    # 3. Label Encoding Manual (Condição de Tráfego - Ordinal)
    traffic_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    if 'Traffic_Condition' in df.columns:
        df['Traffic_Condition'] = df['Traffic_Condition'].map(traffic_mapping)
        print("  > Mapeamento manual (Label Encoding) aplicado em 'Traffic_Condition'.")

    # 4. Normalização de Escala (MinMaxScaler)
    # Aplicado em variáveis com diferentes ordens de grandeza
    cols_to_scale = ['Vehicle_Count', 'Emission_Levels_g_km', 'Traffic_Speed_kmh']
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"  > Normalização aplicada em: {cols_to_scale}")

    # Remover colunas que não serão usadas como features brutas
    df = df.drop(columns=['Timestamp'])
    
    # Salvar o dataset final para o grafo
    df.to_csv(output_path, index=False)
    print(f"✅ Sucesso! Arquivo gerado: {output_path}")

if __name__ == "__main__":
    engineer_features()