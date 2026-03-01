import pandas as pd
import numpy as np
import os

def engineer_features_metr_la(df):
    """
    df: DataFrame onde colunas são sensores e o índice é o Timestamp.
    Retorna o DataFrame com as colunas de tempo cíclico adicionadas.
    """
    # 1. Garantir que o índice seja datetime para extrair hora e dia
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 2. Criar Features Temporais Cíclicas (Hora do Dia: 0-23)
    # 
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    
    # 3. Criar Features Temporais Cíclicas (Dia da Semana: 0-6)
    # Fundamental para capturar o padrão de trânsito semanal (segunda vs domingo)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    
    return df

def executar_feature_engineering():
    base_dir = r'C:\IC\Projeto_V2'
    # Caminho do arquivo que foi LIMPO no dataCleaning.py
    input_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_cleaned.csv')
    output_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_featured.csv')

    if not os.path.exists(input_path):
        print(f"❌ Erro: O arquivo limpo não foi encontrado em {input_path}")
        return

    print(f"📂 Carregando base limpa: {input_path}")
    
    # --- CORREÇÃO AQUI: index_col=0 garante que o Timestamp volte a ser o índice ---
    df = pd.read_csv(input_path, index_col=0) 

    print("🛠️ Gerando features cíclicas (Hora e Dia da Semana)...")
    df_final = engineer_features_metr_la(df)

    # 4. Salvando o arquivo final com as novas colunas
    df_final.to_csv(output_path)
    print(f"✨ Sucesso! Base com Features salva em: {output_path}")
    print(f"📈 Dimensões com novas colunas: {df_final.shape}")

if __name__ == "__main__":
    executar_feature_engineering()