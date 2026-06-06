import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SelfAttention(nn.Module):
    """
    在 Bottleneck 对特征图做自注意力
    让模型在去噪时能「看到全局结构」，而不只是局部卷积感受野
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.norm     = nn.GroupNorm(1, dim)
        self.qkv      = nn.Conv2d(dim, dim * 3, 1)
        self.proj     = nn.Conv2d(dim, dim, 1)
        self.scale    = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        # 生成 Q K V，再拆分成多头
        qkv = self.qkv(h).reshape(B, 3, self.heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)                        # 各自 [B, heads, head_dim, N]

        # 注意力权重：[B, heads, N, N]
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # 加权聚合：[B, heads, head_dim, N]
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)   # 残差连接


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1     = nn.GroupNorm(4, out_ch)
        self.norm2     = nn.GroupNorm(4, out_ch)
        self.act       = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """
    输入：有噪声的图 [B,1,28,28] + 时间步 t [B] + 可选类别标签 label [B]
    输出：预测的噪声 [B,1,28,28]

    改动（v2）：
    1. Bottleneck 加入 SelfAttention → 全局感受野，生成质量提升
    2. 支持类别条件输入（0-9）→ 指定生成哪个数字
    """
    def __init__(self, in_ch=1, base_ch=32, time_dim=128, num_classes=10):
        super().__init__()

        # 时间步编码
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # 类别编码：把数字 0-9 各自映射成一个 128 维向量
        # 推理时和时间步向量相加，让网络知道"要生成哪个数字"
        self.class_emb = nn.Embedding(num_classes, time_dim)

        ch = base_ch

        # 编码器
        self.down1 = ResBlock(in_ch, ch, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(ch, ch * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈：ResBlock + SelfAttention
        # 此时特征图是 7×7 = 49 个 token，注意力计算量很小
        self.middle   = ResBlock(ch * 2, ch * 2, time_dim)
        self.mid_attn = SelfAttention(ch * 2)              # ← 新增

        # 解码器
        self.up1  = nn.Upsample(scale_factor=2)
        self.dec1 = ResBlock(ch * 2 + ch * 2, ch, time_dim)
        self.up2  = nn.Upsample(scale_factor=2)
        self.dec2 = ResBlock(ch + ch, ch, time_dim)

        self.out = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x, t, label=None):
        t_emb = self.time_mlp(t)                           # [B, 128]

        if label is not None:
            t_emb = t_emb + self.class_emb(label)          # 类别信息叠加到时间向量上

        s1 = self.down1(x, t_emb)
        s2 = self.down2(self.pool1(s1), t_emb)

        m  = self.middle(self.pool2(s2), t_emb)
        m  = self.mid_attn(m)                              # ← 新增：全局注意力

        d1 = self.dec1(torch.cat([self.up1(m), s2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), s1], dim=1), t_emb)

        return self.out(d2)
