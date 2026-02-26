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
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_completo_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # Configurações
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    train_loader, test_loader = preparar_loaders(batch_size=batch_size)
    model = STGCN_TrafficModel(num_features=5).to(device)
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mae_criterion = nn.L1Loss(reduction='none')
    mse_criterion = nn.MSELoss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"🚀 Treino Iniciado: {datetime.datetime.now()}")
    log_print(f"💻 GPU Detectada: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # --- CRONÔMETRO DE TREINO ---
    start_train = time.time()
    
    history = []
    for epoch in range(epochs):
        model.train()
        train_mae = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, edge_index, edge_weight=edge_attr)
            
            mask = (y > 0).float()
            loss = (mae_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            train_mae += loss.item()

        # Validação
        model.eval()
        val_mae, val_mse = 0, 0
        with torch.no_grad():
            for xv, yv in test_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, edge_index, edge_weight=edge_attr)
                mask_v = (yv > 0).float()
                v_mae = (mae_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                v_mse = (mse_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                val_mae += v_mae.item()
                val_mse += v_mse.item()

        avg_v_mae = val_mae/len(test_loader)
        scheduler.step(avg_v_mae)
        
        log_print(f"Ep [{epoch+1}/{epochs}] | Val MAE: {avg_v_mae:.4f} | LR: {optimizer.param_groups[0]['lr']}")
        history.append({"epoch": epoch+1, "val_mae": avg_v_mae})

    # --- FIM DO CRONÔMETRO ---
    end_train = time.time()
    total_time = str(datetime.timedelta(seconds=int(end_train - start_train)))
    
    log_print("\n" + "="*40)
    log_print(f"🏆 TREINO CONCLUÍDO!")
    log_print(f"⏳ Tempo Total: {total_time}")
    log_print("="*40)

    torch.save(model.state_dict(), os.path.join(base_dir, 'src', 'modelos', 'stgcn_metrla.pth'))
    pd.DataFrame(history).to_csv(os.path.join(log_dir, f"history_{timestamp}.csv"), index=False)

if __name__ == "__main__":
    treinar_modelo()