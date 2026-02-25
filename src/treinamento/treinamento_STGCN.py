import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
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
    log_file = os.path.join(log_dir, f"treino_comparativo_{timestamp}.txt")
    csv_history = os.path.join(log_dir, f"historico_detalhado_{timestamp}.csv")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # Hiperparâmetros
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    train_loader, test_loader = preparar_loaders(batch_size=batch_size)
    model = STGCN_TrafficModel(num_features=5).to(device)
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)

    # 1. Definindo Critérios (MAE e MSE)
    mae_criterion = nn.L1Loss(reduction='none')
    mse_criterion = nn.MSELoss(reduction='none')
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Scheduler: reduz a LR se a perda de validação parar de cair
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"🚀 Treino Iniciado: {timestamp} | Otimizando com Masked MAE")
    
    history = []
    for epoch in range(epochs):
        model.train()
        train_mae, train_mse = 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x, edge_index, edge_weight=edge_attr)
            
            # 2. Aplicando a Máscara (Ignora valores reais = 0)
            mask = (y > 0).float()
            m_mae = (mae_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)
            m_mse = (mse_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)
            
            # Otimizamos pelo MAE (mais estável)
            m_mae.backward()
            optimizer.step()
            
            train_mae += m_mae.item()
            train_mse += m_mse.item()

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

        avg_t_mae = train_mae/len(train_loader)
        avg_v_mae = val_mae/len(test_loader)
        avg_v_mse = val_mse/len(test_loader)
        
        # Atualiza o Scheduler baseado no MAE de Validação
        scheduler.step(avg_v_mae)
        
        msg = f"Ep [{epoch+1}/{epochs}] | Train MAE: {avg_t_mae:.4f} | Val MAE: {avg_v_mae:.4f} | Val MSE: {avg_v_mse:.4f}"
        log_print(msg)
        
        history.append({
            "epoch": epoch+1, 
            "train_mae": avg_t_mae, 
            "val_mae": avg_v_mae, 
            "val_mse": avg_v_mse,
            "lr": optimizer.param_groups[0]['lr']
        })

    # 3. Salvamento
    pd.DataFrame(history).to_csv(csv_history, index=False)
    torch.save(model.state_dict(), os.path.join(base_dir, 'src', 'modelos', 'stgcn_metrla.pth'))
    log_print(f"✅ Treino finalizado. Logs em: {log_dir}")

if __name__ == "__main__":
    treinar_modelo()