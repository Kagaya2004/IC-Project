import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def analise_espacial_completa():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("🤖 Carregando modelos...")
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    adj_dc = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dc, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    mae_st_sum, mae_dc_sum, counts = np.zeros(207), np.zeros(207), np.zeros(207)

    print("🚀 Processando erros por sensor...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            out_dc = dcrnn(x)

            y_kmh = (y * dataset.std) + dataset.mean
            p_st_kmh = (out_st * dataset.std) + dataset.mean
            p_dc_kmh = (out_dc * dataset.std) + dataset.mean
            mask = y_kmh > 0
            
            mae_st_sum += torch.sum(torch.abs(p_st_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            mae_dc_sum += torch.sum(torch.abs(p_dc_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            counts += torch.sum(mask, dim=(0, 1)).cpu().numpy()

    mae_st_final = mae_st_sum / (counts + 1e-8)
    mae_dc_final = mae_dc_sum / (counts + 1e-8)

    # Gráfico 1: Boxplot Estatístico
    plt.figure(figsize=(10, 6))
    plt.boxplot([mae_st_final, mae_dc_final], patch_artist=True, tick_labels=['ST-GCN', 'DCRNN'],
                boxprops=dict(facecolor='#D3D3D3', color='black'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Análise Estatística da Dispersão do Erro MAE entre os Sensores por Modelo')
    plt.ylabel('MAE Médio (km/h)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(log_dir, f"distribuicao_erro_boxplot_{timestamp}.png"), dpi=300)

    # Gráfico 2: Scatter Plot Espacial
    plt.figure(figsize=(12, 6))
    plt.scatter(range(207), mae_st_final, color='blue', alpha=0.6, s=20, label='ST-GCN')
    plt.scatter(range(207), mae_dc_final, color='red', alpha=0.6, s=20, label='DCRNN')
    plt.title('Comparativo do Erro MAE Individual entre os 207 Sensores por Modelo')
    plt.xlabel('Índice do Sensor'); plt.ylabel('MAE Médio (km/h)'); plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(log_dir, f"distribuicao_espacial_scatter_{timestamp}.png"), dpi=300)

    df_res = pd.DataFrame({'sensor_id': range(207), 'mae_stgcn': mae_st_final, 'mae_dcrnn': mae_dc_final})
    df_res.to_csv(os.path.join(log_dir, f"erro_detalhado_sensores_{timestamp}.csv"), index=False)
    print(f"✅ Análise espacial concluída.")

if __name__ == "__main__":
    analise_espacial_completa()