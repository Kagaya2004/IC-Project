import pandas as pd
import numpy as np

def engineer_features_metr_la(df):
    """
    df: DataFrame onde colunas são sensores e o índice é o Timestamp
    """
    # 1. Garantir que o índice seja datetime
    df.index = pd.to_datetime(df.index)
    
    # 2. Criar Features Temporais Cíclicas
    hour_sin = np.sin(2 * np.pi * df.index.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * df.index.hour / 24.0)
    
    return df