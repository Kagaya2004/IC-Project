import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import METRLADataset, DataLoader

def avaliar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    # Dica: Verifique se o nome é 'stgcn_metrla.pth' ou 'stgcn_metrla_best.pth'
    model_path = os.path.join(base_dir, 'src', 'modelos', 'stgcn_metrla.pth')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log = os.path.join(log_dir, f"avaliacao_performance_{timestamp}.txt")

    # 1. Carregamento de Dados e Modelo
    dataset = METRLADataset()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = STGCN_TrafficModel(num_features=5).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Carregar Grafo para Inferência
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)

    preds, actuals = [], []
    latencias = [] 
    
    print(f"🧐 Iniciando Avaliação em {device}...")
    
    # 2. Loop de Inferência
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 2000: break # Avaliando ~7 dias de tráfego
            
            start_inf = time.time()
            out = model(x.to(device), edge_index, edge_weight=edge_attr)
            end_inf = time.time()
            
            latencias.append((end_inf - start_inf) * 1000) # ms
            preds.append(out.cpu().numpy())
            actuals.append(y.numpy())

    # 3. BLOCO DE CÁLCULO DE MÉTRICAS (Desnormalização e Masking)
    preds_kmh = (np.array(preds).reshape(-1, 207) * dataset.std) + dataset.mean
    actuals_kmh = (np.array(actuals).reshape(-1, 207) * dataset.std) + dataset.mean
    
    # Máscara: Avaliamos apenas onde a velocidade real > 0 (ignora falhas de sensor)
    mask = actuals_kmh > 0
    
    # Cálculo das métricas principais
    mae = np.abs(preds_kmh[mask] - actuals_kmh[mask]).mean()
    mse = ((preds_kmh[mask] - actuals_kmh[mask])**2).mean()
    rmse = np.sqrt(mse) # Raiz do Erro Quadrático Médio
    latencia_media = np.mean(latencias)

    # 4. Geração do Relatório
    res_msg = (
        f"{'='*40}\n"
        f"📊 RELATÓRIO DE PERFORMANCE FINAL\n"
        f"{'='*40}\n"
        f"Data: {datetime.datetime.now()}\n"
        f"Métrica MAE (Erro Médio): {mae:.4f} km/h\n"
        f"Métrica RMSE (Desvio): {rmse:.4f} km/h\n"
        f"Métrica MSE (Quadrático): {mse:.4f}\n"
        f"Latência Média: {latencia_media:.2f} ms por predição\n"
        f"{'='*40}"
    )
    
    print(res_msg)
    with open(eval_log, "w", encoding='utf-8') as f:
        f.write(res_msg)

    # 5. Visualização Gráfica
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_kmh[:144, 0], label='Real (km/h)', color='#1f77b4', linewidth=2)
    plt.plot(preds_kmh[:144, 0], label='Previsto (km/h)', color='#d62728', linestyle='--')
    plt.title('Validação do Modelo ST-GCN - Predição de 5 minutos')
    plt.ylabel('Velocidade')
    plt.xlabel('Tempo (Passos de 5 min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f"grafico_validacao_{timestamp}.png"))
    
    print(f"✅ Resultados e Gráfico salvos em: {log_dir}")

if __name__ == "__main__":
    avaliar_modelo()