import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
import pandas as pd

from src.preprocessamento.dataLoader_LSTM import preparar_loaders_lstm_local

# ==========================================
# 1. Definição do Modelo LSTM
# ==========================================
class LSTM_TrafficModel(nn.Module):
    def __init__(self, num_features=5, hidden_size=64, num_layers=2):
        super(LSTM_TrafficModel, self).__init__()
        # batch_first=True significa que a entrada será [Batch, Sequência, Features]
        self.lstm = nn.LSTM(input_size=num_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True)
        # Camada final para prever a velocidade (1 único valor)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [Batch, Window_Size, Features]
        out, (hn, cn) = self.lstm(x)
        
        # Pegamos apenas a saída do último instante de tempo da sequência
        # out[:, -1, :] tem o formato [Batch, Hidden_Size]
        last_time_step = out[:, -1, :]
        
        # Passamos pela camada linear -> [Batch, 1]
        pred = self.fc(last_time_step)
        
        # Removemos a dimensão extra para ficar apenas [Batch]
        return pred.squeeze(-1)

# ==========================================
# 2. Loop de Treinamento
# ==========================================
def treinar_lstm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_lstm_baseline_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # Aumentamos o batch_size significativamente porque as amostras são muito menores (1 nó por vez)
    batch_size = 512 
    epochs = 50
    lr = 0.001
    
    train_loader, test_loader = preparar_loaders_lstm_local(batch_size=batch_size, window_size=12, horizon=12)
    
    # Instancia o modelo
    model = LSTM_TrafficModel(num_features=5, hidden_size=64).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mae_criterion = nn.L1Loss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"📊 [LSTM Baseline] Início do Experimento")
    log_print(f"💻 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    best_val_mae = float('inf')
    history = []
    start_train = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # y original do loader_local: [Batch, Horizon]
            # Selecionamos apenas o primeiro passo (t+1) para comparar com a ST-GCN
            target = y[:, 0] 
            
            optimizer.zero_grad()
            
            # Repare que não passamos mais os grafos (edge_index) aqui
            output = model(x)
            
            # Masked Loss (ignora zeros)
            mask = (target > 0).float()
            loss = (mae_criterion(output, target) * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validação
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for xv, yv in test_loader:
                xv, yv = xv.to(device), yv.to(device)
                target_v = yv[:, 0]
                
                pred = model(xv)
                
                mask_v = (target_v > 0).float()
                v_mae = (mae_criterion(pred, target_v) * mask_v).sum() / (mask_v.sum() + 1e-8)
                val_mae += v_mae.item()

        avg_train_mae = train_loss / len(train_loader)
        avg_val_mae = val_mae / len(test_loader)
        scheduler.step(avg_val_mae)
        
        # Checkpoint do melhor modelo
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(model_dir, 'lstm_best_baseline.pth'))
            status = "⭐ Melhor Modelo!"
        else:
            status = ""

        log_print(f"Ep [{epoch+1:02d}/{epochs}] | Train: {avg_train_mae:.4f} | Val: {avg_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} {status}")
        
        history.append({"epoch": epoch+1, "train_mae": avg_train_mae, "val_mae": avg_val_mae})

    end_train = time.time()
    total_time = str(datetime.timedelta(seconds=int(end_train - start_train)))
    
    log_print("\n" + "="*50)
    log_print(f"🏆 Treino LSTM Finalizado em {total_time}")
    log_print(f"📉 Melhor MAE alcançado: {best_val_mae:.4f}")
    log_print("="*50)

    # Salva histórico para gráficos comparativos
    pd.DataFrame(history).to_csv(os.path.join(log_dir, f"history_lstm_{timestamp}.csv"), index=False)

if __name__ == "__main__":
    treinar_lstm()