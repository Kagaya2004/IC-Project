import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
import pandas as pd

# Importa a nova versão do modelo desenhada para prever os 12 passos (1 hora)
from src.modelos.modelo_STGCN_12h import STGCN_TrafficModel_12h
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def treinar_stgcn_12h():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_stgcn_12h_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # 1. Configurações do Experimento
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    # O DataLoader já entrega [Batch, 12, Nodes]
    train_loader, test_loader = preparar_loaders(batch_size=batch_size, window_size=12, horizon=12)
    
    # Instancia o novo modelo preparado para o horizonte completo
    model = STGCN_TrafficModel_12h(num_features=5, horizon=12).to(device)
    
    # Carregamento do Grafo
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mae_criterion = nn.L1Loss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"📊 [ST-GCN 12 Passos] Início do Experimento de Longo Prazo")
    log_print(f"💻 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    best_val_mae = float('inf')
    history = []
    start_train = time.time()

    # 2. Loop de Treinamento
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # ---------------------------------------------------------
            # A GRANDE MUDANÇA: Agora pegamos o horizonte completo!
            # y tem o formato [Batch, Horizon=12, Nodes]
            # ---------------------------------------------------------
            target = y 
            
            optimizer.zero_grad()
            
            # O output do novo modelo também tem o formato [Batch, 12, Nodes]
            output = model(x, edge_index, edge_weight=edge_attr)
            
            # Masked Loss: Calcula o erro médio ao longo de toda a 1 hora projetada
            mask = (target > 0).float()
            loss = (mae_criterion(output, target) * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 3. Validação
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for xv, yv in test_loader:
                xv, yv = xv.to(device), yv.to(device)
                
                # Valida em cima de todos os 12 passos
                target_v = yv 
                pred = model(xv, edge_index, edge_weight=edge_attr)
                
                mask_v = (target_v > 0).float()
                v_mae = (mae_criterion(pred, target_v) * mask_v).sum() / (mask_v.sum() + 1e-8)
                val_mae += v_mae.item()

        avg_train_mae = train_loss / len(train_loader)
        avg_val_mae = val_mae / len(test_loader)
        scheduler.step(avg_val_mae)
        
        # Checkpoint do melhor modelo
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            # Salvando com um nome novo para não sobrescrever o experimento antigo!
            torch.save(model.state_dict(), os.path.join(model_dir, 'stgcn_model_12h.pth'))
            status = "⭐ Melhor Modelo!"
        else:
            status = ""

        log_print(f"Ep [{epoch+1:02d}/{epochs}] | Train: {avg_train_mae:.4f} | Val: {avg_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} {status}")
        
        history.append({"epoch": epoch+1, "train_mae": avg_train_mae, "val_mae": avg_val_mae})

    end_train = time.time()
    total_time = str(datetime.timedelta(seconds=int(end_train - start_train)))
    
    log_print("\n" + "="*50)
    log_print(f"🏆 Treino ST-GCN (12 Passos) Finalizado em {total_time}")
    log_print(f"📉 Melhor MAE alcançado (Média de 1h): {best_val_mae:.4f}")
    log_print("="*50)

    # Salva histórico com nome específico
    pd.DataFrame(history).to_csv(os.path.join(log_dir, f"history_stgcn_12h_{timestamp}.csv"), index=False)

if __name__ == "__main__":
    treinar_stgcn_12h()