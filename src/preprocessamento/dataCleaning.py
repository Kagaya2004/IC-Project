import pandas as pd
import numpy as np
import os

def clean_metr_la(df):
    """
    Realiza a limpeza e tratamento da base METR-LA.
    """
    # 1. Garantir que o Timestamp seja o índice e esteja no formato correto
    # Verificamos se 'Timestamp' ou 'Data_Hora' existe nas colunas
    col_tempo = 'Timestamp' if 'Timestamp' in df.columns else ('Data_Hora' if 'Data_Hora' in df.columns else None)
    
    if col_tempo:
        df[col_tempo] = pd.to_datetime(df[col_tempo])
        df = df.set_index(col_tempo)
    
    # 2. Lidar com os Zeros (Sensores Falhos)
    # Transformamos 0 em NaN temporariamente para que a interpolação funcione
    df = df.replace(0, np.nan)
    
    # Interpolação linear: Preenche os buracos baseando-se nos vizinhos (antes e depois)
    # 
    df = df.interpolate(method='linear', axis=0)
    
    # ffill (forward fill) e bfill (backward fill) para erros nas extremidades do arquivo
    df = df.ffill().bfill()
    
    # 3. Converter para Float32 (Otimização de memória para a RTX 3050)
    df = df.astype(np.float32)
    
    # 4. Verificação final de sanidade
    if df.isnull().values.any():
        print("⚠️ Aviso: Ainda existem valores nulos após a limpeza!")
        df = df.dropna()
        
    return df

def executar_limpeza_completa():
    # Definição de caminhos
    base_dir = r'C:\IC\Projeto_V2'
    # Se o seu arquivo bruto for .h5, use pd.read_hdf. Se for .csv, use pd.read_csv.
    input_path = os.path.join(base_dir, 'data', 'brutos', 'METR-LA.h5') 
    output_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_cleaned.csv')

    # Criar a pasta 'tratados' caso ela não exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"📂 Carregando base de dados: {input_path}")
    
    try:
        # Carregando o arquivo (Ajuste para read_csv se o seu bruto for CSV)
        if input_path.endswith('.h5'):
            df_bruto = pd.read_hdf(input_path, key='df')
        else:
            df_bruto = pd.read_csv(input_path)

        # 5. Chamada da função de limpeza
        print("🧹 Iniciando processo de limpeza e interpolação...")
        df_limpo = clean_metr_la(df_bruto)

        # 6. Salvamento do novo arquivo
        df_limpo.to_csv(output_path, index=True)
        print(f"✨ Sucesso! Base limpa salva em: {output_path}")
        print(f"📊 Dimensões finais: {df_limpo.shape}")

    except Exception as e:
        print(f"❌ Erro ao processar os dados: {e}")

if __name__ == "__main__":
    executar_limpeza_completa()