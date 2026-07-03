import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLocalEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            kernel_sizes: list = None,
            d_model: int = 64,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [16, 8, 4]

        self.num_scales = len(kernel_sizes)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, d_model,
                          kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])

        self.scale_weights = nn.Parameter(torch.zeros(self.num_scales))
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]

        outs = [branch(x)[..., :T] for branch in self.branches]

        outs_stacked = torch.stack(outs, dim=0)

        weights = F.softmax(self.scale_weights, dim=0)

        weights = weights.view(self.num_scales, 1, 1, 1)

        out = torch.sum(outs_stacked * weights, dim=0)

        out = self.proj(out)

        return out


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model:    int   = 64,
        nhead:      int   = 4,
        num_layers: int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return x


class BandEncoder(nn.Module):
    def __init__(
        self,
        n_freq:            int = 10,
        d_model:           int = 64,
        temporal_channels: int = 32,
    ):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, temporal_channels,
                      kernel_size=(1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(temporal_channels),
            nn.GELU(),
        )

        self.spectral_conv = nn.Sequential(
            nn.Conv2d(temporal_channels, temporal_channels,
                      kernel_size=(5, 1), padding=0, bias=False),
            nn.BatchNorm2d(temporal_channels),
            nn.GELU(),
        )

        self.proj = nn.Conv1d(temporal_channels, d_model, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1]
        x = self.temporal_conv(x)
        x = x[..., :T]

        x = self.spectral_conv(x)
        x = x.mean(dim=2)
        x = self.proj(x)
        return x


class BandAttentiveFusion(nn.Module):
    def __init__(
        self,
        num_bands: int = 5,
        d_model:   int = 64,
        nhead:     int = 4,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.d_model   = d_model

        self.scorer = nn.Sequential(
            nn.Linear(num_bands * d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_bands),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )

        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, band_feats: list, tau: float = 1.0):
        B = band_feats[0].shape[0]
        T = band_feats[0].shape[2]

        pooled = [f.mean(dim=-1) for f in band_feats]
        scores = self.scorer(torch.cat(pooled, dim=-1))

        if self.training:
            gumbel_w = F.gumbel_softmax(scores, tau=tau, hard=True)
        else:
            idx      = scores.argmax(dim=-1)
            gumbel_w = F.one_hot(idx, num_classes=self.num_bands).float()

        band_seq = torch.stack(
            [f.permute(0, 2, 1) for f in band_feats], dim=1
        )

        query = (gumbel_w[:, :, None, None] * band_seq).sum(dim=1)
        kv    = band_seq.reshape(B, self.num_bands * T, self.d_model)

        attn_out, _ = self.cross_attn(
            self.norm_q(query),
            self.norm_kv(kv),
            self.norm_kv(kv),
        )
        z_f = attn_out + query

        return z_f, gumbel_w


class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.attn_f2t = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.attn_t2f = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_f2t = nn.LayerNorm(d_model)
        self.norm_t2f = nn.LayerNorm(d_model)

    def forward(self, z_t: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        out_f2t, _ = self.attn_f2t(z_f, z_t, z_t)
        out_t2f, _ = self.attn_t2f(z_t, z_f, z_f)

        branch_f2t = self.norm_f2t(out_f2t + z_f)
        branch_t2f = self.norm_t2f(out_t2f + z_t)

        fused = torch.cat([branch_f2t, branch_t2f], dim=-1)
        return fused


class MSTA(nn.Module):
    def __init__(
        self,
        num_class:              int   = 3,
        n_freq:                 int   = 10,
        num_bands:              int   = 5,
        d_model:                int   = 64,
        nhead:                  int   = 4,
        num_transformer_layers: int   = 2,
        kernel_sizes:           list  = None,
        out_channels_per_scale: int   = 16,
        gumbel_tau:             float = 1.0,
        dropout:                float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [16, 8, 4]

        self.num_bands  = num_bands
        self.gumbel_tau = gumbel_tau

        self.temporal_encoder = MultiScaleLocalEncoder(
            in_channels            = 1,
            kernel_sizes           = kernel_sizes,
            d_model                = d_model,
        )
        self.transformer = TemporalTransformer(
            d_model    = d_model,
            nhead      = nhead,
            num_layers = num_transformer_layers,
            dropout    = dropout,
        )

        self.band_encoders = nn.ModuleList([
            BandEncoder(n_freq=n_freq, d_model=d_model)
            for _ in range(num_bands)
        ])
        self.band_fusion = BandAttentiveFusion(
            num_bands = num_bands,
            d_model   = d_model,
            nhead     = nhead,
        )

        self.cross_fusion = CrossModalFusion(d_model=d_model, nhead=nhead)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_class),
        )

    def forward(
        self,
        x:         torch.Tensor,
        band_list: list,
    ):
        local_feat = self.temporal_encoder(x)
        z_t        = self.transformer(local_feat)

        band_feats = [
            self.band_encoders[i](band_list[i])
            for i in range(self.num_bands)
        ]

        z_f, band_weights = self.band_fusion(
            band_feats, tau=self.gumbel_tau
        )

        fused  = self.cross_fusion(z_t, z_f)
        fused  = fused.mean(dim=1)
        logits = self.classifier(fused)

        return logits, band_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    B      = 4
    T      = 400
    n_freq = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSTA(
        num_class = 3,
        n_freq    = n_freq,
        d_model   = 64,
        nhead     = 4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f} M")

    x         = torch.randn(B, 1, T).to(device)
    band_list = [torch.randn(B, 1, n_freq, T).to(device) for _ in range(5)]

    model.train()
    logits, bw = model(x, band_list)
    print(f"[训练] logits:       {logits.shape}")
    print(f"[训练] band_weights: {bw.shape}")
    print(f"[训练] weights sum:  {bw.sum(dim=-1)}")

    model.eval()
    with torch.no_grad():
        logits, bw = model(x, band_list)
    print(f"[推理] logits:       {logits.shape}")
    print(f"[推理] band_weights:\n{bw}")