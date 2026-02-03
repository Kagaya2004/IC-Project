import pandas as pd
import numpy as np
import os

def clean_data():
    base_dir = r'C:\IC\Projeto_V2'
    input_path = os.path.join(base_dir, 'data', 'brutos', 'smart_mobility_dataset.csv')
    output_path = os.path.join(base_dir, 'data', 'tratados', 'dataset_Limpo.csv')
    
    print(f"Iniciando limpeza: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Remover duplicatas
    df = df.drop_duplicates()
    
    # 2. Imputação Híbrida
    for col in df.columns:
        if df[col].isnull().any():
            if np.issubdtype(df[col].dtype, np.number):
                # Numéricos: Mediana (Robusta a outliers)
                valor_imputacao = df[col].median()
                df[col] = df[col].fillna(valor_imputacao)
                print(f"  > '{col}': preenchida com Mediana ({valor_imputacao})")
            else:
                # Categóricos: Moda (Valor mais frequente)
                valor_imputacao = df[col].mode()[0]
                df[col] = df[col].fillna(valor_imputacao)
                print(f"  > '{col}': preenchida com Moda ('{valor_imputacao}')")

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Sucesso! Gerado: {output_path}")

if __name__ == "__main__":
    clean_data()