import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def analise_horizonte_completo():
    # 1. Configurações de Ambiente e Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    
    # Dataset METR-LA: 207 sensores, janelas de 5 min (12 passos = 1 hora)
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Carregar Modelos Multi-step com nomes padronizados
    print("🧠 Carregando modelos Multi-step para análise de horizonte...")
    
    # DCRNN (Nativa do dataset METR-LA)
    adj_dcrnn = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dcrnn, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # ST-GCN (Arquitetura atualizada para 12 passos)
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    
    # Grafo específico para ST-GCN
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # 3. Listas para armazenar erro por passo (1 a 12)
    mae_per_step_dcrnn = [[] for _ in range(12)]
    mae_per_step_stgcn = [[] for _ in range(12)]

    print("🚀 Calculando degradação do erro (5 min -> 60 min)...")
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device) 
            
            # Predições: Ambas retornam [Batch, Horizon=12, Nodes=207]
            out_dc = dcrnn(x) 
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)

            # Desnormalização para escala real em km/h
            y_kmh = (y * dataset.std) + dataset.mean
            out_dc_kmh = (out_dc * dataset.std) + dataset.mean
            out_st_kmh = (out_st * dataset.std) + dataset.mean

            # Máscara para sensores com dados válidos
            mask = y_kmh > 0

            # Cálculo de MAE para cada um dos 12 passos temporais
            for t in range(12):
                # --- Erro DCRNN ---
                diff_dc = torch.abs(out_dc_kmh[:, t, :] - y_kmh[:, t, :])
                valid_dc = diff_dc[mask[:, t, :]]
                if valid_dc.numel() > 0:
                    mae_per_step_dcrnn[t].append(valid_dc.mean().item())
                
                # --- Erro ST-GCN (Isolando o passo t para evitar erro de dimensão) ---
                diff_st = torch.abs(out_st_kmh[:, t, :] - y_kmh[:, t, :])
                valid_st = diff_st[mask[:, t, :]]
                if valid_st.numel() > 0:
                    mae_per_step_stgcn[t].append(valid_st.mean().item())

    # 4. Consolidação das Médias
    final_mae_dcrnn = [np.mean(step) for step in mae_per_step_dcrnn]
    final_mae_stgcn = [np.mean(step) for step in mae_per_step_stgcn]

    # 5. Plotagem da Curva de Degradação do Horizonte
    minutos = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    plt.figure(figsize=(10, 6))
    plt.plot(minutos, final_mae_dcrnn, marker='s', label='DCRNN', color='red', linewidth=2)
    plt.plot(minutos, final_mae_stgcn, marker='o', label='ST-GCN', color='blue', linewidth=2)
    
    plt.title('Evolução do MAE (km/h) conforme o Horizonte de Previsão', fontsize=14)
    plt.xlabel('Minutos à frente (Horizonte)', fontsize=12)
    plt.ylabel('MAE (km/h)', fontsize=12)
    plt.xticks(minutos)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    save_path = os.path.join(log_dir, "analise_horizonte_completo.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Gráfico salvo em: {save_path}")
    
    # 6. Tabela de Resultados para o Relatório de IC
    df_res = pd.DataFrame({
        "Minutos": minutos,
        "DCRNN_MAE (km/h)": final_mae_dcrnn,
        "ST-GCN_MAE (km/h)": final_mae_stgcn
    })
    print("\n📊 Comparativo de Erro por Tempo:")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    analise_horizonte_completo()