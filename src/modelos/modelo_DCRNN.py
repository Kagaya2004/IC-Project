import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_order=2):
        """
        Implementa a Convolução de Difusão.
        k_order: Ordem da difusão (quantos 'saltos' no grafo o modelo olha).
        """
        super(DiffusionGraphConv, self).__init__()
        self.k_order = k_order
        # DCRNN olha para Forward e Backward, então temos (2 * k_order + 1) estados
        self.num_matrices = 2 * k_order + 1
        self.weight = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.zeros_(self.bias)

    def forward(self, x, adj_matrices):
        """
        x: [Batch, Nodes, In_Channels]
        adj_matrices: [2, Nodes, Nodes] (Forward e Backward)
        """
        batch_size, num_nodes, in_channels = x.shape
        # Estado inicial (identidade)
        out = [x]

        # Difusão para cada matriz (Forward e Backward)
        for i in range(adj_matrices.shape[0]):
            x_k = x
            for _ in range(self.k_order):
                # Multiplicação esparsa/densa da matriz de adjacência
                x_k = torch.einsum('nm,bmc->bnc', adj_matrices[i], x_k)
                out.append(x_k)

        # Concatena todos os estados de difusão e aplica os pesos
        out = torch.cat(out, dim=-1) # [B, N, In * Num_Matrices]
        out = torch.matmul(out, self.weight) + self.bias
        return out

class DCGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, adj_matrices, k_order=2):
        super(DCGRUCell, self).__init__()
        self.adj_matrices = adj_matrices
        # Portões Reset e Update usam Diffusion Conv
        self.gate_conv = DiffusionGraphConv(in_channels + hidden_dim, 2 * hidden_dim, k_order)
        # Candidato a estado oculto
        self.candidate_conv = DiffusionGraphConv(in_channels + hidden_dim, hidden_dim, k_order)

    def forward(self, x, h):
        """
        x: [B, N, In]
        h: [B, N, Hidden]
        """
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(combined, self.adj_matrices))
        
        # Divide em Reset e Update gates
        reset_gate, update_gate = torch.split(gates, h.shape[-1], dim=-1)
        
        combined_candidate = torch.cat([x, reset_gate * h], dim=-1)
        candidate = torch.tanh(self.candidate_conv(combined_candidate, self.adj_matrices))
        
        h_next = update_gate * h + (1 - update_gate) * candidate
        return h_next

class DCRNNModel(nn.Module):
    def __init__(self, adj_matrices, in_channels=5, hidden_dim=64, out_channels=1, seq_len=12, horizon=12):
        """
        Estrutura Encoder-Decoder da DCRNN.
        """
        super(DCRNNModel, self).__init__()
        self.adj_matrices = adj_matrices
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Encoder DCGRU
        self.encoder_cell = DCGRUCell(in_channels, hidden_dim, adj_matrices)
        # Decoder DCGRU
        self.decoder_cell = DCGRUCell(out_channels, hidden_dim, adj_matrices)
        
        # Projeção final do Decoder para a velocidade
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        # x: [Batch, Time, Nodes, Features]
        batch_size, seq_len, num_nodes, _ = x.shape
        device = x.device
        
        # 1. ENCODER
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim).to(device)
        for t in range(seq_len):
            h = self.encoder_cell(x[:, t, :, :], h)
        
        # 2. DECODER (Sequence-to-Sequence)
        # O decoder começa com o último estado do encoder
        decoder_input = torch.zeros(batch_size, num_nodes, 1).to(device) # Começa com zeros ou último real
        outputs = []
        
        for t in range(self.horizon):
            h = self.decoder_cell(decoder_input, h)
            out = self.out_proj(h) # [B, N, 1]
            outputs.append(out)
            # No teste/validação, usamos a própria predição como entrada do próximo passo
            decoder_input = out 
            
        # Concatena o horizonte: [B, Horizon, N]
        return torch.stack(outputs, dim=1).squeeze(-1)

if __name__ == "__main__":
    # Teste rápido
    adj = torch.randn(2, 207, 207)
    model = DCRNNModel(adj_matrices=adj)
    dummy_in = torch.randn(16, 12, 207, 5) # Batch 16, 1h passado, 207 nós, 5 feat
    output = model(dummy_in)
    print(f"DCRNN Saída: {output.shape}") # Esperado: [16, 12, 207] (Horizonte de 1 hora)