import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_order=2):
        super(DiffusionGraphConv, self).__init__()
        self.k_order = k_order
        self.num_matrices = 2 * k_order + 1
        self.weight = nn.Parameter(torch.empty(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj_matrices):
        # x: [Batch, Nodes, In_Channels]
        # adj_matrices: [2, Nodes, Nodes]
        out = [x]

        for i in range(adj_matrices.shape[0]):
            x_k = x
            for _ in range(self.k_order):
                x_k = torch.einsum('nm,bmc->bnc', adj_matrices[i], x_k)
                out.append(x_k)

        out = torch.cat(out, dim=-1)  # [B, N, In * Num_Matrices]
        return torch.matmul(out, self.weight) + self.bias


class DCGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, k_order=2):
        super(DCGRUCell, self).__init__()
        self.gate_conv = DiffusionGraphConv(in_channels + hidden_dim, 2 * hidden_dim, k_order)
        self.candidate_conv = DiffusionGraphConv(in_channels + hidden_dim, hidden_dim, k_order)

    def forward(self, x, h, adj_matrices):
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_conv(combined, adj_matrices))
        
        reset_gate, update_gate = torch.split(gates, h.shape[-1], dim=-1)
        combined_candidate = torch.cat([x, reset_gate * h], dim=-1)
        candidate = torch.tanh(self.candidate_conv(combined_candidate, adj_matrices))
        
        h_next = update_gate * h + (1 - update_gate) * candidate
        return h_next


class DCRNNModel(nn.Module):
    def __init__(self, adj_matrices, in_channels=5, hidden_dim=64, out_channels=1, seq_len=12, horizon=12):
        super(DCRNNModel, self).__init__()
        # Registra matrizes no buffer para acompanhar .to(device) e os checkpoints
        self.register_buffer('adj_matrices', adj_matrices)
        
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        self.encoder_cell = DCGRUCell(in_channels, hidden_dim)
        self.decoder_cell = DCGRUCell(out_channels, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        # x: [Batch, Time (12), Nodes (207), Features (5)]
        batch_size, seq_len, num_nodes, _ = x.shape
        device = x.device
        
        # 1. ENCODER
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
        for t in range(seq_len):
            h = self.encoder_cell(x[:, t, :, :], h, self.adj_matrices)
        
        # 2. DECODER
        # Inicializa autoregressão com a última velocidade observada (canal 0 em t = seq_len - 1)
        decoder_input = x[:, -1, :, 0:1]  # [B, N, 1]
        outputs = []
        
        for t in range(self.horizon):
            h = self.decoder_cell(decoder_input, h, self.adj_matrices)
            out = self.out_proj(h)  # [B, N, 1]
            outputs.append(out)
            decoder_input = out     # Predição alimenta o próximo passo
            
        # Saída: [Batch, Horizon (12), Nodes (207)]
        return torch.stack(outputs, dim=1).squeeze(-1)


if __name__ == "__main__":
    adj = torch.randn(2, 207, 207)
    model = DCRNNModel(adj_matrices=adj)
    dummy_in = torch.randn(16, 12, 207, 5)
    output = model(dummy_in)
    print(f"DCRNN Saída: {output.shape}")  # [16, 12, 207]