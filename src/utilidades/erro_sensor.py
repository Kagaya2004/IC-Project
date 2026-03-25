import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def analise_espacial_completa():
    # 1. Configurações de Caminhos e Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    # Dataset METR-LA: 207 sensores, janelas de 5 min (12 passos = 1 hora)
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Carregar Modelos e Dados do Grafo (Nomes Padronizados)
    print("🤖 Carregando modelos e matrizes de adjacência...")
    
    # ST-GCN Multi-step
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # DCRNN SOTA
    adj_dcrnn = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dcrnn, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # 3. Acumuladores para os 207 sensores
    mae_stgcn_sum = np.zeros(207)
    mae_dcrnn_sum = np.zeros(207)
    counts = np.zeros(207)

    print("🚀 Calculando erros espaciais (Média de 60 minutos)...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Predições completas [Batch, 12, 207]
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            out_dc = dcrnn(x)

            # Desnormalização para km/h
            y_kmh = (y * dataset.std) + dataset.mean
            p_st_kmh = (out_st * dataset.std) + dataset.mean
            p_dc_kmh = (out_dc * dataset.std) + dataset.mean

            mask = y_kmh > 0
            
            # Soma dos erros absolutos em todos os 12 passos para cada sensor
            mae_stgcn_sum += torch.sum(torch.abs(p_st_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            mae_dcrnn_sum += torch.sum(torch.abs(p_dc_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            counts += torch.sum(mask, dim=(0, 1)).cpu().numpy()

    # Médias finais por sensor
    mae_st_final = mae_stgcn_sum / (counts + 1e-8)
    mae_dc_final = mae_dcrnn_sum / (counts + 1e-8)

    # ---------------------------------------------------------
    # GRÁFICO 1: BOXPLOT (Distribuição Estatística)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.boxplot([mae_st_final, mae_dc_final], patch_artist=True, tick_labels=['ST-GCN', 'DCRNN'],
                boxprops=dict(facecolor='#D3D3D3', color='black'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Distribuição do MAE entre os 207 sensores da rede viária', fontsize=13)
    plt.ylabel('MAE (km/h)', fontsize=11)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(log_dir, "distribuicao_erro_boxplot.png"), dpi=300)
    print("✅ Gráfico Boxplot salvo.")

    # ---------------------------------------------------------
    # GRÁFICO 2: SCATTER PLOT (Performance Individual)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.scatter(range(207), mae_st_final, color='blue', alpha=0.5, s=15, label='ST-GCN')
    plt.scatter(range(207), mae_dc_final, color='red', alpha=0.5, s=15, label='DCRNN')
    plt.title('Distribuição Espacial do Erro: ST-GCN vs DCRNN (Média 60 min)', fontsize=13)
    plt.xlabel('Índice do Sensor (Rede de LA)', fontsize=11)
    plt.ylabel('MAE Médio (km/h)', fontsize=11)
    plt.legend(loc='upper center', frameon=True)
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.savefig(os.path.join(log_dir, "distribuicao_espacial_scatter.png"), dpi=300)
    print("✅ Gráfico Scatter Plot salvo.")

    # 4. Exportar CSV para Tabelas em LaTeX
    df_res = pd.DataFrame({'sensor_id': range(207), 'mae_stgcn': mae_st_final, 'mae_dcrnn': mae_dc_final})
    df_res.to_csv(os.path.join(log_dir, "erro_detalhado_sensores.csv"), index=False)
    print(f"✅ CSV gerado em: {log_dir}")

if __name__ == "__main__":
    analise_espacial_completa()