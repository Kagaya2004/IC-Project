def clean_metr_la(df):
    # O METR-LA às vezes tem 0s que significam falta de dados
    # Podemos substituir 0s pela média para não enviesar o gráfico
    df = df.replace(0, df.mean())
    # Remover qualquer linha que ainda contenha NaN
    df = df.dropna()
    return df