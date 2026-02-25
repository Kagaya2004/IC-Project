import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import METRLADataset, DataLoader

def avaliar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_path = os.path.join(base_dir, 'src', 'modelos', 'stgcn_metrla.pth')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log = os.path.join(log_dir, f"avaliacao_final_{timestamp}.txt")

    dataset = METRLADataset()
    # Usamos batch_size=1 para avaliação detalhada
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = STGCN_TrafficModel(num_features=5).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)

    preds, actuals = [], []
    print("🧐 Realizando inferência em km/h...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 5000: break # Avalia 5000 amostras para velocidade
            out = model(x.to(device), edge_index, edge_weight=edge_attr)
            preds.append(out.cpu().numpy())
            actuals.append(y.numpy())

    # Converter para km/h (Desnormalização)
    preds_kmh = (np.array(preds).reshape(-1, 207) * dataset.std) + dataset.mean
    actuals_kmh = (np.array(actuals).reshape(-1, 207) * dataset.std) + dataset.mean

    # 4. Cálculo das Métricas Finais (Comparação MSE vs MAE)
    # Ignorando os zeros para métricas justas
    mask = actuals_kmh > 0
    mae_final = np.abs(preds_kmh[mask] - actuals_kmh[mask]).mean()
    mse_final = ((preds_kmh[mask] - actuals_kmh[mask])**2).mean()
    rmse_final = np.sqrt(mse_final)

    res_msg = (
        f"--- RELATÓRIO DE AVALIAÇÃO ---\n"
        f"Data/Hora: {datetime.datetime.now()}\n"
        f"Métrica MAE (Erro Médio): {mae_final:.4f} km/h\n"
        f"Métrica MSE (Erro Quadrático): {mse_final:.4f}\n"
        f"Métrica RMSE (Raiz do MSE): {rmse_final:.4f} km/h\n"
        f"------------------------------"
    )
    print(res_msg)
    with open(eval_log, "w", encoding='utf-8') as f:
        f.write(res_msg)

    # Gráfico de Comparação
    plt.figure(figsize=(12, 6))
    sensor_alvo = 0
    plt.plot(actuals_kmh[:120, sensor_alvo], label='Real (km/h)', color='#1f77b4', linewidth=2)
    plt.plot(preds_kmh[:120, sensor_alvo], label='Previsto (km/h)', color='#d62728', linestyle='--')
    plt.title(f'Desempenho do Modelo ST-GCN - Sensor {sensor_alvo}')
    plt.xlabel('Passos de Tempo (Próximas 10 horas)')
    plt.ylabel('Velocidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f"plot_final_{timestamp}.png"))
    print(f"📊 Gráfico e Log salvos em: {log_dir}")

if __name__ == "__main__":
    avaliar_modelo()