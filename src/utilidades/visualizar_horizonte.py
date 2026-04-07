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

    """
    Carregamentos dos Modelos e Grafos
    """
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    dcrnn = DCRNNModel(adj_matrices=torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device), in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))

    # Estruturas para acumular erros por passo de previsão do STGCN
    mae_per_step_st = [[] for _ in range(12)]
    # Estruturas para acumular erros por passo de previsão do DCRNN
    mae_per_step_dc = [[] for _ in range(12)]

    print("Calculando degradação temporal...")
    # Desativa o cálculo de gradientes para avaliação, economizando memória e tempo
    with torch.no_grad():
        for x, y in test_loader:
            # Envia dados de entrada para o dispositivo (GPU ou CPU)
            x, y = x.to(device), y.to(device)
            # Predições do STGCN
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            # Predições do DCRNN
            out_dc = dcrnn(x)

            # Desnormalização para km/h
            y_kmh = (y * dataset.std) + dataset.mean
            # Desnormalização das predições para km/h do STGCN
            out_st_kmh = (out_st * dataset.std) + dataset.mean
            # Desnormalização das predições para km/h do DCRNN
            out_dc_kmh = (out_dc * dataset.std) + dataset.mean
            # Máscara para considerar apenas os sensores ativos (velocidade > 0)
            mask = y_kmh > 0

            for t in range(12):
                # Cálculo do erro absoluto para o passo t do STGCN
                diff_st = torch.abs(out_st_kmh[:, t, :] - y_kmh[:, t, :])[mask[:, t, :]]
                # Cálculo do erro absoluto para o passo t do DCRNN
                diff_dc = torch.abs(out_dc_kmh[:, t, :] - y_kmh[:, t, :])[mask[:, t, :]]
                # Acumula o erro médio para o passo t do STGCN e DCRNN, se houver amostras válidas
                if diff_st.numel() > 0: mae_per_step_st[t].append(diff_st.mean().item())
                if diff_dc.numel() > 0: mae_per_step_dc[t].append(diff_dc.mean().item())

    # Lista dos passos de previsão (5, 10, ..., 60 minutos)
    minutos = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    plt.figure(figsize=(10, 6))
    
    # Estilo usado anteriormente: DCRNN (Red/Square), ST-GCN (Blue/Circle)
    plt.plot(minutos, [np.mean(s) for s in mae_per_step_dc], marker='s', label='DCRNN', color='#d62728', linewidth=2)
    plt.plot(minutos, [np.mean(s) for s in mae_per_step_st], marker='o', label='ST-GCN', color='#1f77b4', linewidth=2)
    
    plt.title('Evolução do MAE (km/h) conforme o Horizonte de Previsão')
    plt.xlabel('Minutos à frente (Horizonte)'); plt.ylabel('MAE (km/h)')
    plt.xticks(minutos); plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    
    save_path = os.path.join(log_dir, f"curva_degradacao_horizonte_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    pd.DataFrame({'Minutos': minutos, 'STGCN': [np.mean(s) for s in mae_per_step_st], 'DCRNN': [np.mean(s) for s in mae_per_step_dc]}).to_csv(os.path.join(log_dir, f"tabela_degradacao_horizonte_{timestamp}.csv"), index=False)
    print(f"Gráfico de degradação salvo.")

if __name__ == "__main__":
    analise_horizonte_completo()