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
    model_path = os.path.join(base_dir, 'src', 'modelos', 'stgcn_metrla.pth')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log = os.path.join(log_dir, f"avaliacao_performance_{timestamp}.txt")

    dataset = METRLADataset()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = STGCN_TrafficModel(num_features=5).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)

    preds, actuals = [], []
    latencias = [] # Para medir milissegundos por inferência
    
    print("🧐 Iniciando Avaliação de Precisão e Latência...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 2000: break # Avaliando amostra significativa
            
            # --- CRONÔMETRO DE INFERÊNCIA ---
            start_inf = time.time()
            out = model(x.to(device), edge_index, edge_weight=edge_attr)
            end_inf = time.time()
            
            latencias.append((end_inf - start_inf) * 1000) # Converter para ms
            preds.append(out.cpu().numpy())
            actuals.append(y.numpy())

    # Processamento de Métricas
    preds_kmh = (np.array(preds).reshape(-1, 207) * dataset.std) + dataset.mean
    actuals_kmh = (np.array(actuals).reshape(-1, 207) * dataset.std) + dataset.mean
    
    mask = actuals_kmh > 0
    mae = np.abs(preds_kmh[mask] - actuals_kmh[mask]).mean()
    mse = ((preds_kmh[mask] - actuals_kmh[mask])**2).mean()
    latencia_media = np.mean(latencias)

    res_msg = (
        f"--- RELATÓRIO DE PERFORMANCE ---\n"
        f"Data: {datetime.datetime.now()}\n"
        f"MAE Final: {mae:.4f} km/h\n"
        f"MSE Final: {mse:.4f}\n"
        f"Latência Média: {latencia_media:.2f} ms por predição\n"
        f"--------------------------------"
    )
    
    print(res_msg)
    with open(eval_log, "w", encoding='utf-8') as f:
        f.write(res_msg)

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(actuals_kmh[:100, 0], label='Real', color='blue')
    plt.plot(preds_kmh[:100, 0], label='Previsto', color='red', linestyle='--')
    plt.title('Predição ST-GCN: Real vs Previsto')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"grafico_inf_{timestamp}.png"))

if __name__ == "__main__":
    avaliar_modelo()