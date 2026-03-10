import torch
import numpy as np
import matplotlib.pyplot as plt
import os
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
    
    # 1. Carregar Dados
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. Carregar Modelos
    print("🧠 Carregando modelos para análise de horizonte...")
    
    # DCRNN (Nativa multi-passo)
    adj_dcrnn = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dcrnn, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_best_comparativo.pth'), weights_only=True))
    dcrnn.eval()

    # ST-GCN (Como a nossa ST-GCN é single-step, usaremos ela como base para t+1)
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_best_comparativo.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # 3. Listas para armazenar erro por passo (1 a 12)
    # DCRNN terá erro para os 12 passos. ST-GCN compararemos no passo 1.
    mae_per_step_dcrnn = [[] for _ in range(12)]
    mae_step1_stgcn = []

    print("🚀 Calculando erros ao longo do horizonte (5 min -> 60 min)...")
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device) # Ambos na GPU
            
            # Predições
            out_dc = dcrnn(x) 
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)

            # Desnormalização (Operações feitas na GPU)
            y_kmh = (y * dataset.std) + dataset.mean
            out_dc_kmh = (out_dc * dataset.std) + dataset.mean
            out_st_kmh = (out_st * dataset.std) + dataset.mean

            # Criar máscara booleana (True para valores válidos)
            mask = y_kmh > 0

            # --- CÁLCULO CORRIGIDO USANDO TORCH ---
            # Erro DCRNN para cada um dos 12 passos
            for t in range(12):
                # Pegamos o erro absoluto apenas onde a máscara é True
                diff = torch.abs(out_dc_kmh[:, t, :] - y_kmh[:, t, :])
                valid_diff = diff[mask[:, t, :]]
                
                if valid_diff.numel() > 0:
                    step_mae = valid_diff.mean().item() # .item() traz o valor escalar para a CPU com segurança
                    mae_per_step_dcrnn[t].append(step_mae)
            
            # Erro ST-GCN (Passo 0)
            diff_st = torch.abs(out_st_kmh - y_kmh[:, 0, :])
            valid_diff_st = diff_st[mask[:, 0, :]]
            
            if valid_diff_st.numel() > 0:
                st_mae = valid_diff_st.mean().item()
                mae_step1_stgcn.append(st_mae)

    # 4. Consolidar Médias
    final_mae_dcrnn = [np.mean(step) for step in mae_per_step_dcrnn]
    final_mae_stgcn_t1 = np.mean(mae_step1_stgcn)

    # 5. Plotagem Científica
    minutos = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    plt.figure(figsize=(10, 6))
    plt.plot(minutos, final_mae_dcrnn, marker='o', label='DCRNN (Multi-step)', color='red', linewidth=2)
    plt.scatter([5], [final_mae_stgcn_t1], color='blue', s=100, label='ST-GCN (Single-step)', zorder=5)
    
    plt.title('Evolução do Erro (MAE) conforme o Horizonte de Predição', fontsize=14)
    plt.xlabel('Minutos à frente (Horizonte)', fontsize=12)
    plt.ylabel('MAE (km/h)', fontsize=12)
    plt.xticks(minutos)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    save_path = os.path.join(log_dir, "analise_horizonte_mae.png")
    plt.savefig(save_path)
    print(f"✅ Análise de horizonte concluída! Gráfico salvo em: {save_path}")
    
    # Exibir tabela rápida
    df_res = pd.DataFrame({
        "Minutos": minutos,
        "DCRNN_MAE": final_mae_dcrnn
    })
    print("\n📊 Tabela de Erro por Tempo (DCRNN):")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    analise_horizonte_completo()