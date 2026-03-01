import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
import pandas as pd
import numpy as np
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import preparar_loaders

def treinar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_completo_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # 1. Carregamento de Dados e Modelo
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    train_loader, test_loader = preparar_loaders(batch_size=batch_size)
    
    model = STGCN_TrafficModel(num_features=5).to(device)
    
    # Verificação de carregamento do grafo
    try:
        edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
        edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)
        log_print(f"✅ Grafo carregado com sucesso! ({edge_index.shape[1]} conexões)")
    except Exception as e:
        log_print(f"❌ Erro ao carregar arquivos do grafo: {e}")
        return

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mae_criterion = nn.L1Loss(reduction='none')
    mse_criterion = nn.MSELoss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"🚀 Treino Iniciado: {datetime.datetime.now()}")
    log_print(f"💻 Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # --- CRONÔMETRO E TRACKING ---
    start_train = time.time()
    best_val_mae = float('inf')
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_mae_accum = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, edge_index, edge_weight=edge_attr)
            
            # Máscara para ignorar zeros (falhas de sensor)
            mask = (y > 0).float()
            loss = (mae_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            train_mae_accum += loss.item()

        # Validação
        model.eval()
        val_mae_accum, val_mse_accum = 0, 0
        with torch.no_grad():
            for xv, yv in test_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, edge_index, edge_weight=edge_attr)
                
                mask_v = (yv > 0).float()
                # Calculamos MAE e MSE separadamente para o log
                v_mae = (mae_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                v_mse = (mse_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                
                val_mae_accum += v_mae.item()
                val_mse_accum += v_mse.item()

        # Médias da Época
        avg_train_mae = train_mae_accum / len(train_loader)
        avg_val_mae = val_mae_accum / len(test_loader)
        avg_val_mse = val_mse_accum / len(test_loader)
        
        scheduler.step(avg_val_mae)
        
        # 2. Lógica de "Best Model" (Salva se for o melhor resultado até agora)
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(model_dir, 'stgcn_metrla_best.pth'))
            status_best = "⭐ [Novo Melhor!]"
        else:
            status_best = ""

        msg = (f"Ep [{epoch+1:02d}/{epochs}] | "
               f"Train MAE: {avg_train_mae:.4f} | "
               f"Val MAE: {avg_val_mae:.4f} | "
               f"Val MSE: {avg_val_mse:.4f} | "
               f"LR: {optimizer.param_groups[0]['lr']:.6f} {status_best}")
        log_print(msg)
        
        history.append({
            "epoch": epoch+1, 
            "train_mae": avg_train_mae, 
            "val_mae": avg_val_mae, 
            "val_mse": avg_val_mse
        })

    # --- FINALIZAÇÃO ---
    end_train = time.time()
    total_time = str(datetime.timedelta(seconds=int(end_train - start_train)))
    
    log_print("\n" + "="*50)
    log_print(f"🏆 TREINO CONCLUÍDO!")
    log_print(f"⏱️ Tempo Total: {total_time}")
    log_print(f"📉 Melhor Val MAE: {best_val_mae:.4f}")
    log_print("="*50)

    # Salva o histórico em CSV para gerar gráficos depois
    pd.DataFrame(history).to_csv(os.path.join(log_dir, f"history_{timestamp}.csv"), index=False)
    log_print(f"✅ Arquivo de histórico salvo: history_{timestamp}.csv")

if __name__ == "__main__":
    treinar_modelo()