import os
import torch
import pandas as pd
# Importamos as funções dos teus scripts (ajusta se os nomes das funções forem diferentes)
from src.preprocessamento.preparar_grafo import gerar_grafo_stgcn
from src.treinamento.treinamento_STGCN import treinar_modelo
from src.avaliar_modelo import avaliar 

def executar_experimentos():
    # Lista de raios que queres testar (em km)
    raios_para_teste = [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]
    resultados = []

    print("🚀 Iniciando Bateria de Testes Automatizados...")
    
    for raio in raios_para_teste:
        print(f"\n" + "="*40)
        print(f"🧪 TESTANDO RAIO: {raio}km")
        print("="*40)

        # 1. Gerar o Grafo com o novo raio
        # (Certifica-te que a função gerar_grafo_stgcn aceita o parâmetro threshold_km)
        num_arestas = gerar_grafo_stgcn(threshold_km=raio)

        # 2. Treinar o modelo do zero para este grafo
        print(f"⏱️ Treinando ST-GCN para raio de {raio}km...")
        treinar_modelo() # Esta função deve salvar o modelo em 'stgcn_treinada.pth'

        # 3. Avaliar e obter o MAE
        # (Ajusta a tua função avaliar() para retornar o valor do MAE)
        mae_resultado = avaliar()
        
        resultados.append({
            'Raio (km)': raio,
            'Arestas': num_arestas,
            'MAE Normalizado': mae_resultado
        })

    # 4. Exibir Tabela Final Comparativa
    df_resultados = pd.DataFrame(resultados)
    print("\n" + "📊 RESULTADOS FINAIS DOS EXPERIMENTOS")
    print("-" * 40)
    print(df_resultados.to_string(index=False))
    print("-" * 40)
    
    # Salvar em CSV para o teu relatório
    df_resultados.to_csv(r'C:\IC\Projeto_V2\data\comparativo_raios.csv', index=False)
    print("✅ Tabela salva em: data/comparativo_raios.csv")

if __name__ == "__main__":
    executar_experimentos()