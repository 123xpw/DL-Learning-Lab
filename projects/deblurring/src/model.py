import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    """
    3-level UNet for image deblurring.
    Direct prediction: outputs the deblurred image directly (not a residual).
    Skip connections pass encoder feature maps to decoder to recover fine detail.

    Shape trace (base=16, H=128):
      enc1:        [B,  16, 128, 128]
      enc2:        [B,  32,  64,  64]
      enc3:        [B,  64,  32,  32]
      bottleneck:  [B, 128,  16,  16]
      dec3 (cat):  [B,  64,  32,  32]
      dec2 (cat):  [B,  32,  64,  64]
      dec1 (cat):  [B,  16, 128, 128]
      out:         [B,   3, 128, 128]
    """
    def __init__(self, in_channels=3, base=16):
        super().__init__()
        b = base
        # ── Encoder ────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, b)      # 3  → 16
        self.enc2 = ConvBlock(b,           b * 2)  # 16 → 32
        self.enc3 = ConvBlock(b * 2,       b * 4)  # 32 → 64
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ─────────────────────────────────────────────
        self.bottleneck = ConvBlock(b * 4, b * 8)  # 64 → 128

        # ── Decoder ────────────────────────────────────────────────
        # up3: 128 → 64 (H/8 → H/4), then cat with enc3 (64) → 128 → dec3 → 64
        self.up3  = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)   # 128 → 64

        # up2: 64 → 32 (H/4 → H/2), then cat with enc2 (32) → 64 → dec2 → 32
        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)   # 64 → 32

        # up1: 32 → 16 (H/2 → H), then cat with enc1 (16) → 32 → dec1 → 16
        self.up1  = nn.ConvTranspose2d(b * 2, b,     2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)        # 32 → 16

        self.out = nn.Conv2d(b, in_channels, 1)  # 16 → 3

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               # [B, 16, H,   W  ]
        e2 = self.enc2(self.pool(e1))   # [B, 32, H/2, W/2]
        e3 = self.enc3(self.pool(e2))   # [B, 64, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # [B, 128, H/8, W/8]

        # Decoder + skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))  # [B, 64, H/4, W/4]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, 32, H/2, W/2]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 16, H,   W  ]

        return torch.sigmoid(self.out(d1))  # [B, 3, H, W] in [0, 1]
