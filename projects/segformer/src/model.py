import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Overlapping Patch Embedding ────────────────────────────────────────
# 和 ViT 不同：用有重叠的卷积提取 patch，保留边缘连续性
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, stride):
        super().__init__()
        # padding = patch_size // 2 保证输出尺寸 = H / stride
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                                       # [B, C, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()         # [B, H'*W', C]
        x = self.norm(x)
        return x, H, W


# ── 2. Efficient Self-Attention（序列降采样）────────────────────────────
# 核心创新：K 和 V 先用卷积缩小空间分辨率，再做注意力
# 计算量从 O(N²) 降到 O(N * N/sr²)
class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q    = nn.Linear(dim, dim)
        self.kv   = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 把序列还原成特征图，用卷积降采样，再展平
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Q 保持完整序列长度
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        # q: [B, heads, N, head_dim]

        # K, V 使用降采样后的短序列
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).contiguous().reshape(B, C, H, W)  # [B, C, H, W]
            x_ = self.sr(x_)                                           # [B, C, H/sr, W/sr]
            x_ = x_.flatten(2).transpose(1, 2).contiguous()           # [B, N/sr², C]
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        N_kv = kv.shape[1]
        kv = kv.reshape(B, N_kv, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0].permute(0, 2, 1, 3).contiguous()  # [B, heads, N_kv, head_dim]
        v = kv[:, :, 1].permute(0, 2, 1, 3).contiguous()  # [B, heads, N_kv, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale      # [B, heads, N, N_kv]
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)  # [B, N, C]
        out = self.proj(out)
        return out


# ── 3. Mix-FFN（局部卷积混合）──────────────────────────────────────────
# 用 3×3 深度可分离卷积替代位置编码，隐式编码位置信息
class MixFFN(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        hidden = dim * expand
        self.fc1     = nn.Linear(dim, hidden)
        self.dw_conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act     = nn.GELU()
        self.fc2     = nn.Linear(hidden, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)                                                    # [B, N, hidden]
        hidden = x.shape[-1]
        x = x.transpose(1, 2).contiguous().reshape(B, hidden, H, W)       # [B, hidden, H, W]
        x = self.dw_conv(x)                                                # [B, hidden, H, W]
        x = x.reshape(B, hidden, N).transpose(1, 2).contiguous()          # [B, N, hidden]
        x = self.act(x)
        x = self.fc2(x)                                                    # [B, N, dim]
        return x


# ── 4. Transformer Block ──────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = MixFFN(dim, expand=mlp_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


# ── 5. 单个 MiT Stage ─────────────────────────────────────────────────────
class MiTStage(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size, stride,
                 depth, num_heads, sr_ratio, mlp_ratio=4):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_ch, embed_dim, patch_size, stride)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, sr_ratio, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, H, W = self.patch_embed(x)           # [B, H'*W', C]
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, C, H', W']
        return x


# ── 6. MiT-B0 编码器（4 阶段，输出多尺度特征）───────────────────────────
#
# MiT-B0 配置（论文 Table 1）：
#   embed_dims = [32, 64, 160, 256]
#   num_heads  = [1,  2,  5,   8]
#   sr_ratios  = [8,  4,  2,   1]
#   depths     = [2,  2,  2,   2]
#
# 输入 [B, 3, 512, 512] 的输出尺寸：
#   stage1 → [B, 32,  128, 128]   (stride=4)
#   stage2 → [B, 64,  64,  64 ]   (stride=2)
#   stage3 → [B, 160, 32,  32 ]   (stride=2)
#   stage4 → [B, 256, 16,  16 ]   (stride=2)
class MixTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # stage1: 7×7 patch，stride=4（较大感受野，大步长）
        self.stage1 = MiTStage(3,   32,  patch_size=7, stride=4,
                                depth=2, num_heads=1, sr_ratio=8)
        # stage2–4: 3×3 patch，stride=2（逐步降分辨率）
        self.stage2 = MiTStage(32,  64,  patch_size=3, stride=2,
                                depth=2, num_heads=2, sr_ratio=4)
        self.stage3 = MiTStage(64,  160, patch_size=3, stride=2,
                                depth=2, num_heads=5, sr_ratio=2)
        self.stage4 = MiTStage(160, 256, patch_size=3, stride=2,
                                depth=2, num_heads=8, sr_ratio=1)

    def forward(self, x):
        c1 = self.stage1(x)   # [B, 32,  H/4,  W/4 ]
        c2 = self.stage2(c1)  # [B, 64,  H/8,  W/8 ]
        c3 = self.stage3(c2)  # [B, 160, H/16, W/16]
        c4 = self.stage4(c3)  # [B, 256, H/32, W/32]
        return c1, c2, c3, c4


# ── 7. All-MLP Decoder ────────────────────────────────────────────────────
# 论文核心：极简解码器——4 个线性层统一通道 → 上采样到 H/4 → 融合 → 分类
# 没有任何卷积，比 FPN / ASPP 都简单，但效果更好（因为 MiT 编码器已经有全局感受野）
class AllMLPDecoder(nn.Module):
    def __init__(self, in_channels=(32, 64, 160, 256), embed_dim=256, num_classes=21):
        super().__init__()
        # 每个 stage 的特征投影到统一 embed_dim
        self.proj = nn.ModuleList([
            nn.Linear(c, embed_dim) for c in in_channels
        ])
        # 拼接后融合
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.drop       = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def _project_and_upsample(self, feat, linear, target_H, target_W):
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2).contiguous()   # [B, H*W, C]
        x = linear(x)                                       # [B, H*W, embed_dim]
        x = x.transpose(1, 2).contiguous().reshape(B, -1, H, W)
        # 上采样到 c1 的空间尺寸（H/4, W/4）
        x = F.interpolate(x, size=(target_H, target_W),
                          mode='bilinear', align_corners=False)
        return x

    def forward(self, c1, c2, c3, c4):
        tH, tW = c1.shape[2], c1.shape[3]    # 目标尺寸 = c1 的空间大小

        _c1 = self._project_and_upsample(c1, self.proj[0], tH, tW)
        _c2 = self._project_and_upsample(c2, self.proj[1], tH, tW)
        _c3 = self._project_and_upsample(c3, self.proj[2], tH, tW)
        _c4 = self._project_and_upsample(c4, self.proj[3], tH, tW)

        x = torch.cat([_c4, _c3, _c2, _c1], dim=1)  # [B, 4*embed_dim, tH, tW]
        x = self.fuse(x)                              # [B, embed_dim, tH, tW]
        x = self.drop(x)
        x = self.classifier(x)                        # [B, num_classes, tH, tW]
        return x


# ── 8. 完整 SegFormer ─────────────────────────────────────────────────────
class SegFormer(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.encoder = MixTransformer()
        self.decoder = AllMLPDecoder(
            in_channels=(32, 64, 160, 256),
            embed_dim=256,
            num_classes=num_classes,
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        c1, c2, c3, c4 = self.encoder(x)
        logits = self.decoder(c1, c2, c3, c4)           # [B, C, H/4, W/4]
        logits = F.interpolate(logits, size=(H, W),
                               mode='bilinear', align_corners=False)
        return logits
