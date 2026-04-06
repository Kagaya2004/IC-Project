import pandas as pd
import numpy as np
import os

def engineer_features_metr_la(df):
    # Verificar se o valor do índice não é do tipo DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        """ 
        Aqui é transformado o índice no tipo DateTimeIndex, facilitando 
        na extração de propriedades como hora e dia da semana.
        """
        df.index = pd.to_datetime(df.index)
    
    """
    Aplicação de Propriedades Temporais Cíclicas, através das funções
    Seno e Cosseno, na informações Hora, permitindo que os modelos 
    compreendam que horários como 23:59 e 00:00 estão próximos.
    """
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    
    """
    Aplicação de Propriedades Temporais Cíclicas, através das funções
    Seno e Cosseno, na informações Dua da Semana, permitindo que os modelos 
    compreendam que os dias como Domingo e Segunda estão próximos.
    """
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    
    return df

def executar_feature_engineering():
    # Caminho da raiz do projeto
    base_dir = r'C:\IC\Projeto_V2'
    # Caminho do arquivo no qual é o dataset após a limpeza realizado pelo dataCleaning.py
    input_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_cleaned.csv')
    # Caminho do arquivo final após a aplicação da Engenharia de Atriibutos/Features Engineering
    output_path = os.path.join(base_dir, 'data', 'tratados', 'METR-LA_featured.csv')

    # Verificação da existência do arquivo de entrada
    if not os.path.exists(input_path):
        print(f"Erro: O arquivo limpo não foi encontrado em {input_path}")
        return
    
    print(f"Carregando base limpa: {input_path}")
    
    """
    Ao carregar o dataset limpo, é indicado que a coluna 0 (a primeira),
    na qual contém o valor Data_Hora, é o índice do dataFrame.
    """
    df = pd.read_csv(input_path, index_col=0) 

    print("Convertendo os dadso de Hora e Dia da Semana em atributos ciclícos...")
    # Aplicação das transformações de Engenharia de Atributos no dataset limpo
    df_final = engineer_features_metr_la(df)

    # Salvando o arquivo final com as novas colunas, no formato .csv
    df_final.to_csv(output_path)
    print(f"Engenharia de Atributos aplicada com sucesso! Base salva em: {output_path}")
    print(f"Dimensões com novas colunas: {df_final.shape}")

if __name__ == "__main__":
    executar_feature_engineering()