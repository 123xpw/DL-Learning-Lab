import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block: two 3×3 convs with skip connection."""
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    """
    1-level encoder.
    Input → [level0 → skip] → down → [bottleneck]
    Returns: (bottleneck_feat, skip_feat)
    """
    def __init__(self, in_ch, n_feat):
        super().__init__()
        self.level0 = nn.Sequential(
            nn.Conv2d(in_ch, n_feat, 3, 1, 1),
            nn.PReLU(),
            ResBlock(n_feat),
        )
        self.down   = nn.Conv2d(n_feat, n_feat * 2, kernel_size=2, stride=2)
        self.bottle = nn.Sequential(
            ResBlock(n_feat * 2),
            ResBlock(n_feat * 2),
        )

    def forward(self, x):
        skip = self.level0(x)               # [B, n,  H,   W  ]
        bn   = self.bottle(self.down(skip)) # [B, 2n, H/2, W/2]
        return bn, skip


class Decoder(nn.Module):
    """
    1-level decoder: upsample + fuse skip + refine.
    """
    def __init__(self, n_feat):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1),
        )
        self.body = ResBlock(n_feat)

    def forward(self, bn, skip):
        return self.body(self.up(bn) + skip)    # [B, n, H, W]


class SAM(nn.Module):
    """
    Supervised Attention Module (between stages).
    - Predicts rain streak → stage prediction via residual subtraction
    - Generates attention gate → passes gated features to next stage
    """
    def __init__(self, n_feat):
        super().__init__()
        self.rain_head = nn.Conv2d(n_feat, 3, 1)
        self.gate      = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat, rainy):
        rain_hat  = self.rain_head(feat)
        pred      = (rainy - rain_hat).clamp(0, 1)   # stage 1 output
        attn_feat = feat * self.gate(feat)            # gated features → stage 2
        return pred, attn_feat


class SimpleMPRNet(nn.Module):
    """
    2-stage MPRNet (simplified) for image deraining.

    Stage 1: Encoder-Decoder → SAM → (pred1, attention features)
    Stage 2: Encoder-Decoder(rainy + attn) → residual rain pred → pred2

    Both stages supervised with Charbonnier loss during training.
    """
    def __init__(self, n_feat=16):
        super().__init__()
        # Stage 1
        self.enc1 = Encoder(3,          n_feat)
        self.dec1 = Decoder(n_feat)
        self.sam1 = SAM(n_feat)

        # Stage 2 — takes rainy concat with SAM attention features
        self.enc2 = Encoder(3 + n_feat, n_feat)
        self.dec2 = Decoder(n_feat)
        self.out2 = nn.Conv2d(n_feat, 3, 3, 1, 1)

    def forward(self, rainy):
        # ── Stage 1 ──────────────────────────────────────────────
        bn1, sk1          = self.enc1(rainy)
        d1                = self.dec1(bn1, sk1)
        pred1, attn_feat1 = self.sam1(d1, rainy)

        # ── Stage 2 ──────────────────────────────────────────────
        bn2, sk2 = self.enc2(torch.cat([rainy, attn_feat1], dim=1))
        d2       = self.dec2(bn2, sk2)
        rain2    = self.out2(d2)
        pred2    = (rainy - rain2).clamp(0, 1)

        return pred1, pred2
