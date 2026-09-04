import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def analise_espacial_completa():
    # Define o dispositivo (GPU se disponível, caso contrário CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Raiz do projeto
    base_dir = r'C:\IC\Projeto_V2'
    # Diretório de dados tratados
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    # Diretório de logs
    log_dir = os.path.join(base_dir, 'data', 'logs')
    # Diretório de modelos
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    # Identificação única do experimento para histórico dos logs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dataset de Teste (Horizonte 12 para 60 min)
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Carregando modelos...")
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

    # Inicialização dos acumuladores de erro e contagem por sensor
    mae_st_sum = np.zeros(207)
    mae_dc_sum = np.zeros(207)
    mae_ls_sum = np.zeros(207)
    counts = np.zeros(207)

    print("Processando erros por sensor...")
    # Desativa o cálculo de gradientes para avaliação, economizando memória e tempo
    with torch.no_grad():
        for x, y in test_loader:
            # Envia dados de entrada para o dispositivo (GPU ou CPU)
            x, y = x.to(device), y.to(device)
            
            # Predições do STGCN
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            
            # Predições do DCRNN
            out_dc = dcrnn(x)

            # Predições do LSTM (Adicionado com os ajustes de shape)
            b, t, n, f = x.shape
            x_lstm = x.permute(0, 2, 1, 3).reshape(-1, t, f)
            out_ls_raw = lstm(x_lstm) # Retorna [B*207, 12]
            out_ls = out_ls_raw.view(b, n, 12).permute(0, 2, 1) # Volta para [B, 12, 207]

            # Desnormalização para km/h
            y_kmh = (y * dataset.std) + dataset.mean
            # Desnormalização das predições para km/h do STGCN
            p_st_kmh = (out_st * dataset.std) + dataset.mean
            # Desnormalização das predições para km/h do DCRNN
            p_dc_kmh = (out_dc * dataset.std) + dataset.mean
            # Desnormalização das predições para km/h do LSTM
            p_ls_kmh = (out_ls * dataset.std) + dataset.mean
            
            # Máscara para considerar apenas os sensores ativos (velocidade > 0)
            mask = y_kmh > 0
            
            # MAE acumulado por sensor do STGCN
            mae_st_sum += torch.sum(torch.abs(p_st_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            # MAE acumulado por sensor do DCRNN
            mae_dc_sum += torch.sum(torch.abs(p_dc_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            # MAE acumulado por sensor do LSTM
            mae_ls_sum += torch.sum(torch.abs(p_ls_kmh - y_kmh) * mask, dim=(0, 1)).cpu().numpy()
            
            # Contagem de amostras válidas por sensor para média final
            counts += torch.sum(mask, dim=(0, 1)).cpu().numpy()

    # Cálculo do MAE final por sensor para cada modelo
    mae_st_final = mae_st_sum / (counts + 1e-8)
    mae_dc_final = mae_dc_sum / (counts + 1e-8)
    mae_ls_final = mae_ls_sum / (counts + 1e-8)

    # Gráfico Boxplot Estatístico
    plt.figure(figsize=(10, 6))
    plt.boxplot([mae_st_final, mae_dc_final, mae_ls_final], patch_artist=True, tick_labels=['STGCN', 'DCRNN', 'LSTM'],
                boxprops=dict(facecolor='#D3D3D3', color='black'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Análise Estatística da Dispersão do Erro MAE entre os Sensores por Modelo')
    plt.ylabel('MAE Médio (km/h)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(log_dir, f"distribuicao_erro_boxplot_{timestamp}.png"), dpi=300)

    # Gráfico Scatter Espacial
    plt.figure(figsize=(12, 6))
    plt.scatter(range(207), mae_st_final, color='blue', alpha=0.6, s=20, label='STGCN')
    plt.scatter(range(207), mae_dc_final, color='red', alpha=0.6, s=20, label='DCRNN')
    plt.scatter(range(207), mae_ls_final, color='green', alpha=0.6, s=20, label='LSTM')
    
    plt.title('Comparativo do Erro MAE Individual entre os 207 Sensores por Modelo')
    plt.xlabel('Índice do Sensor'); plt.ylabel('MAE Médio (km/h)'); plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(log_dir, f"distribuicao_espacial_scatter_{timestamp}.png"), dpi=300)

    df_res = pd.DataFrame({
        'sensor_id': range(207), 
        'mae_stgcn': mae_st_final, 
        'mae_dcrnn': mae_dc_final,
        'mae_lstm': mae_ls_final
    })
    df_res.to_csv(os.path.join(log_dir, f"erro_detalhado_sensores_{timestamp}.csv"), index=False)
    print(f"Análise espacial concluída.")

if __name__ == "__main__":
    analise_espacial_completa()