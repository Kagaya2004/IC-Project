import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def analise_horizonte_completo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    dcrnn = DCRNNModel(adj_matrices=torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device), in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))

    mae_per_step_st = [[] for _ in range(12)]
    mae_per_step_dc = [[] for _ in range(12)]

    print("🚀 Calculando degradação temporal...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            out_dc = dcrnn(x)

            y_kmh = (y * dataset.std) + dataset.mean
            out_st_kmh = (out_st * dataset.std) + dataset.mean
            out_dc_kmh = (out_dc * dataset.std) + dataset.mean
            mask = y_kmh > 0

            for t in range(12):
                diff_st = torch.abs(out_st_kmh[:, t, :] - y_kmh[:, t, :])[mask[:, t, :]]
                diff_dc = torch.abs(out_dc_kmh[:, t, :] - y_kmh[:, t, :])[mask[:, t, :]]
                if diff_st.numel() > 0: mae_per_step_st[t].append(diff_st.mean().item())
                if diff_dc.numel() > 0: mae_per_step_dc[t].append(diff_dc.mean().item())

    minutos = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    plt.figure(figsize=(10, 6))
    
    # Estilo conforme referência: DCRNN (Red/Square), ST-GCN (Blue/Circle)
    plt.plot(minutos, [np.mean(s) for s in mae_per_step_dc], marker='s', label='DCRNN (SOTA)', color='#d62728', linewidth=2)
    plt.plot(minutos, [np.mean(s) for s in mae_per_step_st], marker='o', label='ST-GCN (Multi-step)', color='#1f77b4', linewidth=2)
    
    plt.title('Evolução do MAE (km/h) conforme o Horizonte de Previsão')
    plt.xlabel('Minutos à frente (Horizonte)'); plt.ylabel('MAE (km/h)')
    plt.xticks(minutos); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    
    save_path = os.path.join(log_dir, f"curva_degradacao_horizonte_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    pd.DataFrame({'Minutos': minutos, 'STGCN': [np.mean(s) for s in mae_per_step_st], 'DCRNN': [np.mean(s) for s in mae_per_step_dc]}).to_csv(os.path.join(log_dir, f"tabela_degradacao_horizonte_{timestamp}.csv"), index=False)
    print(f"✅ Gráfico de degradação salvo.")

if __name__ == "__main__":
    analise_horizonte_completo()