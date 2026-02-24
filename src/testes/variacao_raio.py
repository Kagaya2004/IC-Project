import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.preprocessamento.preparar_grafo import gerar_grafo_stgcn
from src.treinamento.treinamento_STGCN import treinar_modelo
from src.utilidades.avaliar_modelo import avaliar 

def executar_experimentos():
    # 1. Configurações dos Testes
    raios_para_teste = [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]
    resultados = []
    
    # Valores para desnormalização (ajuste conforme o seu dataset real)
    V_MIN, V_MAX = 0.0, 100.0 

    print("🚀 Iniciando Bateria de Testes de Sensibilidade Espacial...")
    
    # Pasta para salvar os gráficos dos experimentos
    output_dir = r'C:\IC\Projeto_V2\data\experimentos'
    os.makedirs(output_dir, exist_ok=True)

    for raio in raios_para_teste:
        print(f"\n" + "="*50)
        print(f"🧪 EXPERIMENTO: Raio de {raio}km")
        print("="*50)

        # 2. Geração da Topologia (Espacial)
        print(f"🔗 Gerando grafo...")
        num_arestas = gerar_grafo_stgcn(threshold_km=raio)

        # 3. Treinamento (Temporal)
        print(f"⏱️ Treinando ST-GCN na RTX 3050...")
        treinar_modelo() 

        # 4. Avaliação Multimétrica
        # Ajustamos para retornar um dicionário de métricas
        metrics = avaliar(save_plot=True, suffix=f"_{raio}km")
        
        mae_norm = metrics['mae']
        mse_norm = metrics['mse']
        rmse_norm = np.sqrt(mse_norm)
        
        # 5. Desnormalização para km/h (Métrica de Negócio/Logística)
        mae_kmh = mae_norm * (V_MAX - V_MIN)

        resultados.append({
            'Raio_km': raio,
            'Arestas': num_arestas,
            'MAE_Normalizado': round(mae_norm, 4),
            'MSE_Normalizado': round(mse_norm, 4),
            'RMSE_Normalizado': round(rmse_norm, 4),
            'MAE_kmh': round(mae_kmh, 2)
        })
        
        print(f"✅ Resultado: MAE = {mae_norm:.4f} ({mae_kmh:.2f} km/h)")

    # 6. Consolidação dos Dados
    df_res = pd.DataFrame(resultados)
    
    # 7. Geração do Gráfico de Desempenho (MAE vs Raio)
    plt.figure(figsize=(10, 6))
    plt.plot(df_res['Raio_km'], df_res['MAE_Normalizado'], marker='o', linestyle='-', color='b')
    plt.title('Sensibilidade do Modelo ST-GCN ao Raio de Conectividade')
    plt.xlabel('Raio do Grafo (km)')
    plt.ylabel('Erro Médio Absoluto (MAE)')
    plt.grid(True, ls=':')
    plt.savefig(os.path.join(output_dir, 'curva_aprendizado_raio.png'))

    # 8. Exportação Final
    print("\n📊 TABELA COMPARATIVA FINAL")
    print(df_res.to_string(index=False))
    
    df_res.to_csv(r'C:\IC\Projeto_V2\data\relatorio_experimentos.csv', index=False)
    print(f"\n✅ Relatório e Gráficos salvos em: {output_dir}")

if __name__ == "__main__":
    executar_experimentos()