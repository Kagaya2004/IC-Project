import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import preparar_loaders

def avaliar():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧐 Avaliando modelo na {device}...")

    # 1. Carregar Dados
    # Usamos o test_loader para ver como o modelo se sai com dados que ele não "decorou"
    _, test_loader = preparar_loaders(batch_size=1)
    x_sample, y_real = next(iter(test_loader))
    num_features = x_sample.shape[-1]
    
    # 2. Instanciar e Carregar Pesos do Modelo
    model = STGCN_TrafficModel(num_features=num_features).to(device)
    model_path = r'C:\IC\Projeto_V2\src\modelo\stgcn_treinada.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval() # Modo de avaliação (desliga dropout)

    # 3. Carregar Grafo
    edge_index = torch.load(r'C:\IC\Projeto_V2\data\tratados\edge_index.pt', weights_only=True).to(device)

    # 4. Inferência (Previsão)
    with torch.no_grad():
        previsao = model(x_sample.to(device), edge_index)
    
    # Converter para CPU/Numpy para o gráfico
    previsao_np = previsao.cpu().numpy().flatten()
    real_np = y_real.cpu().numpy().flatten()

    # 5. Gerar Gráfico Comparativo
    plt.figure(figsize=(12, 6))
    
    # Vamos plotar os primeiros 100 sensores para o gráfico não ficar poluído
    plt.plot(real_np[:100], label='Velocidade Real', color='blue', alpha=0.7, linewidth=2)
    plt.plot(previsao_np[:100], label='Velocidade Prevista (ST-GCN)', color='red', linestyle='--', alpha=0.8)
    
    plt.title('Comparação: Velocidade Real vs. Prevista (Amostra de 100 Sensores)')
    plt.xlabel('ID do Sensor')
    plt.ylabel('Velocidade Normalizada (0-1)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Salvar o gráfico para o relatório
    output_img = r'C:\IC\Projeto_V2\data\grafico_resultado.png'
    plt.savefig(output_img)
    print(f"📊 Gráfico salvo com sucesso em: {output_img}")
    
    # Calcular Erro Médio Absoluto (MAE) simples
    mae = np.mean(np.abs(real_np - previsao_np))
    print(f"📈 Erro Médio Absoluto (MAE) na escala normalizada: {mae:.4f}")
    # No script de avaliação, após calcular o MAE:
    # Supondo que sua velocidade máxima era 120km/h e a mínima 0km/h
    v_min = 0.0  
    v_max = 100.0 # Verifique o valor real no seu dataset brutos

    mae_kmh = mae * (v_max - v_min)
    print(f"🚗 Erro Médio Real: {mae_kmh:.2f} km/h")

    return float(mae)

if __name__ == "__main__":
    avaliar()