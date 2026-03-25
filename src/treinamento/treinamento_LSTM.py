import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
import pandas as pd
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def treinar_lstm_novo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_lstm_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # 1. Configurações (Mantendo a paridade com ST-GCN e DCRNN)
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    # O loader universal entrega [Batch, 12, 207, 5]
    train_loader, test_loader = preparar_loaders(batch_size=batch_size, window_size=12, horizon=12)
    
    # Instancia a nova arquitetura (com Embedding e LayerNorm)
    model = LSTM_TrafficModel(num_features=5).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mae_criterion = nn.L1Loss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"📈 [LSTM NOVO] Iniciando Treinamento da Arquitetura Atualizada")
    log_print(f"💻 Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    best_val_mae = float('inf')
    history = []
    start_train = time.time()

    # 2. Loop de Treinamento
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # y: [Batch, 12, 207] -> Pegamos t+1: [Batch, 207]
            target = y[:, 0, :]
            
            # Ajuste de Shape para a LSTM:
            # x está como [B, 12, 207, 5]. Precisamos de [B*207, 12, 5]
            batch_curr, seq_len, num_nodes, num_feats = x.shape
            x_lstm = x.permute(0, 2, 1, 3).reshape(-1, seq_len, num_feats)
            target_lstm = target.reshape(-1, 1).squeeze() # [B*207]

            optimizer.zero_grad()
            output = model(x_lstm) # Saída: [B*207, 1]
            output = output.squeeze()

            # Masked Loss
            mask = (target_lstm > 0).float()
            loss = (mae_criterion(output, target_lstm) * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 3. Validação
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for xv, yv in test_loader:
                xv, yv = xv.to(device), yv.to(device)
                target_v = yv[:, 0, :]
                
                bv, sv, nv, fv = xv.shape
                xv_lstm = xv.permute(0, 2, 1, 3).reshape(-1, sv, fv)
                target_v_lstm = target_v.reshape(-1, 1).squeeze()

                pred = model(xv_lstm).squeeze()
                mask_v = (target_v_lstm > 0).float()
                v_mae = (mae_criterion(pred, target_v_lstm) * mask_v).sum() / (mask_v.sum() + 1e-8)
                val_mae += v_mae.item()

        avg_train_mae = train_loss / len(train_loader)
        avg_val_mae = val_mae / len(test_loader)
        scheduler.step(avg_val_mae)
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            # Salvando com o novo nome para não confundir com o baseline
            torch.save(model.state_dict(), os.path.join(model_dir, 'lstm_best_novo.pth'))
            status = "⭐ Melhor Modelo!"
        else:
            status = ""

        log_print(f"Ep [{epoch+1:02d}/{epochs}] | Train: {avg_train_mae:.4f} | Val: {avg_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} {status}")
        history.append({"epoch": epoch+1, "train_mae": avg_train_mae, "val_mae": avg_val_mae})

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_train)))
    log_print(f"\n🏆 Treino Finalizado em {total_time} | Melhor Val MAE: {best_val_mae:.4f}")
    pd.DataFrame(history).to_csv(os.path.join(log_dir, f"history_lstm_novo_{timestamp}.csv"), index=False)

if __name__ == "__main__":
    treinar_lstm_novo()