import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
import pandas as pd
from src.modelos.modelo_DCRNN import DCRNNModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def treinar_dcrnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    
    # Garante que as pastas existam
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_dcrnn_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # 1. Configurações de Paridade (Horizonte 12: 60 min)
    # Com 207 sensores e 5 features, a DCRNN consome muita VRAM na RTX 3050.
    # Se der 'Out of Memory', mude para 16.
    batch_size = 32 
    epochs = 50
    lr = 0.001
    
    train_loader, test_loader = preparar_loaders(batch_size=batch_size, window_size=12, horizon=12)
    
    # Carregamento das Matrizes de Difusão geradas no preparar_grafo.py
    try:
        adj_matrices = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
        log_print(f"✅ Matrizes de Difusão DCRNN carregadas: {adj_matrices.shape}")
    except Exception as e:
        log_print(f"❌ Erro crítico: Matrizes DCRNN não encontradas. {e}")
        return

    # O hidden_dim=64 mantém a paridade com os canais da ST-GCN
    model = DCRNNModel(adj_matrices=adj_matrices, in_channels=5, hidden_dim=64).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mae_criterion = nn.L1Loss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"🌊 [DCRNN] Iniciando Treinamento")
    log_print(f"💻 GPU Detectada: {torch.cuda.get_device_name(0)}")
    
    best_val_mae = float('inf')
    history = []
    start_train = time.time()

    # 2. Loop de Treinamento
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # y: [Batch, 12, 207]
            
            optimizer.zero_grad()
            # Predição completa de 12 passos
            output = model(x) 
            
            # Masked MAE sobre todo o horizonte temporal
            mask = (y > 0).float()
            loss = (mae_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 3. Validação (Test Set)
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for xv, yv in test_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv)
                
                mask_v = (yv > 0).float()
                v_mae = (mae_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                val_mae += v_mae.item()

        avg_train_mae = train_loss / len(train_loader)
        avg_val_mae = val_mae / len(test_loader)
        scheduler.step(avg_val_mae)
        
        # Salvamento de Checkpoint
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(model_dir, 'dcrnn_model.pth'))
            status = "⭐ Melhor Modelo!"
        else:
            status = ""

        log_print(f"Ep [{epoch+1:02d}/{epochs}] | Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} {status}")
        
        history.append({"epoch": epoch+1, "train_mae": avg_train_mae, "val_mae": avg_val_mae})

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_train)))
    
    log_print("\n" + "="*50)
    log_print(f"🏆 Treino DCRNN Finalizado em {total_time}")
    log_print(f"📉 Melhor MAE no Teste (Horizonte 12): {best_val_mae:.4f}")
    log_print("="*50)

    # Exportação para análise visual posterior
    history_path = os.path.join(log_dir, f"history_dcrnn_{timestamp}.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    log_print(f"📄 Histórico salvo em: {history_path}")

if __name__ == "__main__":
    treinar_dcrnn()