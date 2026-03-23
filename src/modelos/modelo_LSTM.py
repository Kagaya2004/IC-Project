import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_TrafficModel(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        """
        Arquitetura LSTM Baseline (Modelo Local / Por Nó).
        num_features: 5 (Velocidade + 4 Features Temporais)
        """
        super(LSTM_TrafficModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Embedding + Normalização
        # Ajuda a rede a balancear a velocidade e os senos/cossenos temporais
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.ln_embed = nn.LayerNorm(hidden_dim)
        
        # 2. Camada Temporal: LSTM
        # batch_first=True faz com que a entrada seja [Batch, Time, Features]
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        
        self.ln_lstm = nn.LayerNorm(hidden_dim)
        
        # 3. Regressão Final (Idêntica à da ST-GCN para comparação justa)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x shape esperado pelo DataLoader Local: [Batch, Time, Features]
        
        # --- PASSO 1: Embedding & Normalização ---
        x = self.embedding(x)
        x = self.ln_embed(x)
        x = F.relu(x)
        
        # --- PASSO 2: Bloco Temporal (LSTM) ---
        # out: contém as saídas de todos os passos de tempo [Batch, Time, Hidden]
        # hn, cn: contêm os estados ocultos finais
        out, (hn, cn) = self.lstm(x)
        
        # A sacada: Queremos prever o futuro com base no ÚLTIMO passo de tempo (t=12)
        # Então extraímos apenas a última fatia da sequência temporal
        last_out = out[:, -1, :] # Shape: [Batch, Hidden]
        last_out = self.ln_lstm(last_out)
        
        # --- PASSO 3: Regressão Final ---
        pred = self.regressor(last_out) # Shape: [Batch, 1]
        
        # Removemos a última dimensão vazia para retornar um vetor simples [Batch]
        return pred.squeeze(-1)

if __name__ == "__main__":
    # Teste de Integridade da LSTM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulação: Como estamos no modelo Local (Por Nó), o DataLoader entrega
    # batches muito maiores. Ex: 512 amostras independentes, 12 passos de tempo, 5 features.
    model = LSTM_TrafficModel(num_features=5).to(device)
    dummy_x = torch.randn(512, 12, 5).to(device)
    
    print(f"🚀 Testando modelo LSTM no dispositivo: {device}")
    
    # Note que a LSTM não recebe 'edge_index' (o grafo), apenas os dados sequenciais 'x'
    output = model(dummy_x)
    
    print(f"✅ Sucesso! Shape da saída: {output.shape}") # Esperado: torch.Size([512])
    print(f"📊 Parâmetros totais: {sum(p.numel() for p in model.parameters()):,}")