import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def analise_espacial_completa():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Carregamento dos dados de teste sequenciais (sem leakage)
    print("Preparando DataLoaders de Teste...")
    _, test_loader, _, _ = preparar_loaders(batch_size=32, window_size=12, horizon=12)

    # 2. Carrega parâmetros de normalização do treino
    scaler_path = os.path.join(data_dir, 'scaler_params.pt')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Erro: {scaler_path} não encontrado. Execute o treino primeiro!")
    
    scaler = torch.load(scaler_path, weights_only=True)
    mean_val = float(scaler['mean'])
    std_val = float(scaler['std'])

    print("Carregando modelos treinados...")
    # STGCN
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # DCRNN
    adj_dc = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dc, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # LSTM
    lstm = LSTM_TrafficModel(num_features=5).to(device)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_model.pth'), weights_only=True))
    lstm.eval()

    mae_st_sum = np.zeros(207)
    mae_dc_sum = np.zeros(207)
    mae_ls_sum = np.zeros(207)
    counts = np.zeros(207)

    print("Processando erros por sensor no dataset de teste...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Predições padronizadas: [Batch, Horizon (12), Nodes (207)]
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            out_dc = dcrnn(x)
            out_ls = lstm(x)

            # Desnormalização para km/h
            y_kmh = (y * std_val) + mean_val
            p_st_kmh = (out_st * std_val) + mean_val
            p_dc_kmh = (out_dc * std_val) + mean_val
            p_ls_kmh = (out_ls * std_val) + mean_val
            
            mask = y_kmh > 0
            
            # Acúmulo do erro absoluto por sensor ao longo do batch e horizonte
            mae_st_sum += torch.sum(torch.abs(p_st_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            mae_dc_sum += torch.sum(torch.abs(p_dc_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            mae_ls_sum += torch.sum(torch.abs(p_ls_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            
            counts += torch.sum(mask, dim=(0, 1)).cpu().numpy()

    # MAE médio por sensor
    mae_st_final = mae_st_sum / (counts + 1e-8)
    mae_dc_final = mae_dc_sum / (counts + 1e-8)
    mae_ls_final = mae_ls_sum / (counts + 1e-8)

    # 1. Gráfico Boxplot Estatístico
    plt.figure(figsize=(9, 6))
    plt.boxplot(
        [mae_st_final, mae_dc_final, mae_ls_final], 
        patch_artist=True, 
        tick_labels=['STGCN', 'DCRNN', 'LSTM'],
        boxprops=dict(facecolor='#D3D3D3', color='black'),
        medianprops=dict(color='red', linewidth=2)
    )
    plt.title('Distribuição do MAE entre os 207 Sensores da Rede Viária', fontsize=12, pad=10)
    plt.ylabel('MAE Médio (km/h)', fontsize=11)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"distribuicao_erro_boxplot_{timestamp}.png"), dpi=300)
    plt.close()

    # 2. Gráfico Scatter Espacial
    plt.figure(figsize=(14, 6))
    x_nodes = np.arange(207)
    plt.scatter(x_nodes, mae_st_final, color='blue', alpha=0.6, s=25, label='STGCN')
    plt.scatter(x_nodes, mae_dc_final, color='red', alpha=0.6, s=25, label='DCRNN', marker='x')
    plt.scatter(x_nodes, mae_ls_final, color='green', alpha=0.6, s=25, label='LSTM', marker='^')
    
    plt.title('Comparativo do Erro MAE Individual entre os 207 Sensores por Modelo', fontsize=12, pad=10)
    plt.xlabel('Índice do Sensor (Dataset METR-LA)', fontsize=11)
    plt.ylabel('MAE Médio (km/h)', fontsize=11)
    plt.xlim(left=0, right=206)  # Fixa o início no 0 encostado no eixo Y
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"distribuicao_espacial_scatter_{timestamp}.png"), dpi=300)
    plt.close()

    # Salvamento do CSV detalhado
    df_res = pd.DataFrame({
        'sensor_id': range(207), 
        'mae_stgcn': mae_st_final, 
        'mae_dcrnn': mae_dc_final,
        'mae_lstm': mae_ls_final
    })
    csv_path = os.path.join(log_dir, f"erro_detalhado_sensores_{timestamp}.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Análise espacial concluída! Resultados salvos em: {csv_path}")

if __name__ == "__main__":
    analise_espacial_completa()