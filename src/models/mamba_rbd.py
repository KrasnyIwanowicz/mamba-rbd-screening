import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Projekcje parametrów zależnych od wejścia: B, C, Delta (Selective Mechanism)
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Inicjalizacja macierzy A (HiPPO)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        A = -torch.exp(self.A_log.float())  # [D, N]

        # Wyznaczenie parametrów zależnych od sekwencji
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        delta, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        delta = F.softplus(self.dt_proj(delta))  # [B, L, D]

        # Dyskretny model SSM (ZOH / Euler scan)
        y = torch.zeros_like(x)
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)

        for t in range(seq_len):
            dt = delta[:, t, :].unsqueeze(-1)          # [B, D, 1]
            b_t = B[:, t, :].unsqueeze(1)               # [B, 1, N]
            c_t = C[:, t, :].unsqueeze(-1)              # [B, N, 1]
            u_t = x[:, t, :].unsqueeze(-1)              # [B, D, 1]

            dA = torch.exp(dt * A.unsqueeze(0))         # [B, D, N]
            dB = dt * b_t                               # [B, D, N]

            h = h * dA + dB * u_t                       # [B, D, N]
            y_t = torch.matmul(h, c_t).squeeze(-1)      # [B, D]
            y[:, t, :] = y_t + self.D * x[:, t, :]

        return y


class BiMambaBlock(nn.Module):
    """
    Dwukierunkowy blok Mamba (przetwarzający kontekst czasowy w przód i w tył).
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Konwolucje 1D modelujące lokalne zależności (aktywność fazową)
        self.conv_fwd = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4, padding=3, groups=self.d_inner)
        self.conv_bwd = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4, padding=3, groups=self.d_inner)

        self.ssm_fwd = SelectiveSSM(self.d_inner, d_state=d_state)
        self.ssm_bwd = SelectiveSSM(self.d_inner, d_state=d_state)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Forward path
        x_fwd = x_proj.transpose(1, 2)
        x_fwd = F.silu(self.conv_fwd(x_fwd)[:, :, :L]).transpose(1, 2)
        y_fwd = self.ssm_fwd(x_fwd)

        # Backward path
        x_bwd = torch.flip(x_proj, dims=[1]).transpose(1, 2)
        x_bwd = F.silu(self.conv_bwd(x_bwd)[:, :, :L]).transpose(1, 2)
        y_bwd = torch.flip(self.ssm_bwd(x_bwd), dims=[1])

        y = (y_fwd + y_bwd) * F.silu(z)
        out = self.out_proj(y)
        return out + res


class MambaRBDClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,       # Chin EMG + Leg EMG
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        num_classes: int = 2
    ):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=50, stride=25, padding=25),  # [B, 32, 241]
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                              # [B, 32, 120]
            nn.Conv1d(32, d_model, kernel_size=7, stride=2, padding=3),        # [B, d_model, 60]
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

        # Bloki Bi-Mamba
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 1. Ekstrakcja cech lokalnych
        x = self.patch_embed(x)      # [B, d_model, L]
        x = x.transpose(1, 2)        # [B, L, d_model]

        # 2. Przetwarzanie sekwencji przez bloki Mamba
        for layer in self.layers:
            x = layer(x)

        # 3. Global Pooling i Klasyfikacja
        feat = x.mean(dim=1)         # [B, d_model]
        logits = self.head(feat)     # [B, num_classes]
        return logits