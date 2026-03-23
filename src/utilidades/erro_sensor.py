import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def analise_espacial_erro():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    
    # 1. Carregar Dados
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Instanciar e Carregar Modelos
    print("🤖 Carregando modelos para análise espacial...")
    
    # ST-GCN
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # DCRNN
    adj_dcrnn = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dcrnn, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # 3. Acumuladores de Erro por Sensor (207 sensores)
    mae_stgcn_sensors = np.zeros(207)
    mae_dcrnn_sensors = np.zeros(207)
    counts = np.zeros(207)

    print("🚀 Calculando distribuição de erro entre os 207 sensores...")
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Predições (focando no primeiro passo t+1 para comparação direta)
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            out_dc = dcrnn(x)[:, 0, :] # Pega o primeiro passo da DCRNN

            # Desnormalização
            y_kmh = (y[:, 0, :].cpu().numpy() * dataset.std) + dataset.mean
            out_st_kmh = (out_st.cpu().numpy() * dataset.std) + dataset.mean
            out_dc_kmh = (out_dc.cpu().numpy() * dataset.std) + dataset.mean

            # Máscara para evitar divisões por zero em sensores falhos
            mask = y_kmh > 0
            
            # Acumular erro absoluto por sensor
            mae_stgcn_sensors += np.sum(np.abs(out_st_kmh - y_kmh) * mask, axis=0)
            mae_dcrnn_sensors += np.sum(np.abs(out_dc_kmh - y_kmh) * mask, axis=0)
            counts += np.sum(mask, axis=0)

    # Calcular média final por sensor
    mae_stgcn_final = mae_stgcn_sensors / (counts + 1e-8)
    mae_dcrnn_final = mae_dcrnn_sensors / (counts + 1e-8)

    # 4. Plotagem: Distribuição de Erro (Histograma/Boxplot)
    plt.figure(figsize=(12, 6))
    data_to_plot = [mae_stgcn_final, mae_dcrnn_final]
    plt.boxplot(data_to_plot, patch_artist=True, labels=['ST-GCN', 'DCRNN'],
                boxprops=dict(facecolor='#D3D3D3', color='black'),
                medianprops=dict(color='red', linewidth=2))
    
    plt.title('Distribuição do MAE entre os 207 sensores da rede viária', fontsize=14)
    plt.ylabel('MAE (km/h)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    box_path = os.path.join(log_dir, "distribuicao_erro_sensores.png")
    plt.savefig(box_path)
    
    # 5. Plotagem: Comparação Individual (Scatter)
    plt.figure(figsize=(12, 6))
    plt.scatter(range(207), mae_stgcn_final, alpha=0.6, label='ST-GCN', color='blue', s=20)
    plt.scatter(range(207), mae_dcrnn_final, alpha=0.6, label='DCRNN', color='red', s=20)
    plt.title('Evolução do MAE (km/h) em relação ao horizonte de previsão', fontsize=14)
    plt.xlabel('ID do Sensor (Índice)', fontsize=12)
    plt.ylabel('MAE (km/h)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    scatter_path = os.path.join(log_dir, "mae_por_sensor_scatter.png")
    plt.savefig(scatter_path)

    print(f"✅ Análise concluída!")
    print(f"📊 Gráficos salvos em: {log_dir}")
    
    # Salvar dados para uso em tabelas de LATEX
    df_sensors = pd.DataFrame({
        'sensor_id': range(207),
        'mae_stgcn': mae_stgcn_final,
        'mae_dcrnn': mae_dcrnn_final
    })
    df_sensors.to_csv(os.path.join(log_dir, "erro_detalhado_por_sensor.csv"), index=False)

if __name__ == "__main__":
    analise_espacial_erro()