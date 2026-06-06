---
tags: [stable-diffusion, LoRA, diffusion, fine-tuning, generative-model]
created: 2026-06-05
project: diffusion
---

# Stable Diffusion + LoRA 入门

上级笔记：[[CV学习总路线图]]
相关笔记：[[DDPM-扩散模型入门]] · [[迁移学习与微调]]

---

## Stable Diffusion 的组成

Stable Diffusion（SD）是 LDM（Latent Diffusion Model）的开源实现，由四个模块组成：

```
文字提示词
    ↓
[CLIP Text Encoder]  ← 文字 → 向量（77×768）
    ↓ cross-attention
[UNet2DConditionModel]  ← 在潜空间去噪，接收文字引导
    ↑
[VAE Encoder]       ← 把原始图片（512×512×3）压缩到潜空间（64×64×4）
    ↓
[VAE Decoder]       ← 把去噪后的潜变量解码回像素空间
    ↓
生成图片（512×512×3）
```

| 模块 | 类型 | 作用 |
|------|------|------|
| VAE | AutoencoderKL | 图像 ↔ 潜空间（压缩比 8×，即 512→64）|
| CLIP | CLIPTextModel | 文字 → 语义向量 |
| UNet | UNet2DConditionModel | 潜空间去噪 + 接收文字 Cross-Attention |
| Scheduler | PNDMScheduler | 控制采样步骤（推理时决定如何从噪声逐步去噪）|

### VAE：信息损失了多少？

VAE 把图片从 512×512×3（786,432 个数值）压缩到 64×64×4（16,384 个数值），压缩比约 48×。

实验：把原图编码再解码，肉眼几乎看不出差异。说明 VAE 学到了极高效的视觉表示——大量像素级细节可以从低维隐变量重建。

---

## LoRA：为什么需要它？

### 微调的问题

要让 SD 生成特定风格（比如宝可梦风格），最直观的做法是用风格图片微调模型。

**问题**：SD 1.5 有 **8.6 亿参数**——
- 全量微调：显存需求极大（>16GB），训练时间极长
- 每个风格都要存一个完整模型副本：860MB × N 个风格

### LoRA 的核心思想

**低秩分解（Low-Rank Adaptation）**：不修改原始权重，而是在旁边加一个"小矩阵旁路"。

```
原始层（冻结）：W  ∈ R^{out × in}

LoRA 旁路：
  A ∈ R^{rank × in}    （随机初始化）
  B ∈ R^{out × rank}   （零初始化）

前向传播：
  output = W·x + (B·A·x) × (alpha/rank)
           ↑          ↑
        原始输出   LoRA 增量（初始为 0）
```

关键点：
- rank 通常为 4-16，远小于 in/out（通常 768 或 1024）
- B 初始化为 0 → 训练开始时 LoRA 增量为 0，不破坏原始模型
- 只训练 A 和 B，原始 W 完全冻结

### 参数量对比

SD 1.5 UNet：8.6 亿参数  
注入 96 个 LoRA 层（Q/K/V 各 32 层），rank=4：

```
每个 LoRA 层参数 = rank × in + out × rank = 4×768 + 768×4 = 6,144
96 层总计 = 6,144 × 96 ≈ 597,504 ≈ 60万
占比 = 60万 / 8.6亿 ≈ 0.069%
```

**只需训练 0.07% 的参数**，就能学到特定风格。

---

## 手写 LoRA 注入（不依赖 PEFT）

代码路径：[`colab/`](../../colab/)（Colab notebook）

```python
class LoRALinear(nn.Module):
    def __init__(self, original, rank=4, alpha=4):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)          # 冻结原始层
        dev = original.weight.device
        # A：随机初始化（小值），B：零初始化（确保初始增量=0）
        self.lora_A = nn.Parameter(torch.randn(rank, original.in_features, device=dev) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank, device=dev))
        self.scale  = alpha / rank

    def forward(self, x):
        # 用 float32 计算增量防止 float16 溢出导致 NaN
        delta = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scale
        return self.original(x) + delta.to(x.dtype)
```

注入函数（把 UNet 的 to_q / to_k / to_v 替换为 LoRALinear）：

```python
def inject_lora(unet, targets=("to_k", "to_q", "to_v"), rank=4):
    injected = []
    for name, module in list(unet.named_modules()):
        if not (isinstance(module, nn.Linear) and name.split(".")[-1] in targets):
            continue
        parts = name.split(".")
        parent = unet
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        lora = LoRALinear(module, rank=rank)
        setattr(parent, parts[-1], lora)
        injected.append(lora)
    return injected
```

---

## 实验记录

### Exp1：Pokemon 风格 LoRA（Colab A100）

**目标**：让 SD 1.5 生成宝可梦风格图片

**配置**：
- 基础模型：`runwayml/stable-diffusion-v1-5`
- 数据集：HuggingFace `lambdalabs/pokemon-blip-captions`（833 张）
- LoRA：rank=4，注入 to_q/to_k/to_v，共 96 层，597,504 参数（0.069%）
- 训练：3 epochs，lr=1e-4，batch_size=1，float16

| 指标 | 值 |
|------|---|
| 训练时间 | ~10 分钟（A100）|
| 可训练参数 | 597,504（0.069%）|
| 最终 loss | ~0.054 |

**结果**：生成图片具有明显的宝可梦风格——色彩饱和、卡通轮廓、简洁背景。

**踩坑**：
- `torchao` 版本不兼容导致 PEFT 无法使用 → 手写 LoRALinear
- float16 矩阵乘法溢出 → NaN loss → 改用 `x.float()` + cast back
- diffusers 接近 v1.0 后 `LoRAAttnProcessor` 签名改变 → 同样用手写注入

---

## 与你已学知识的连接

| 已学 | LoRA 中的体现 |
|------|-------------|
| 线性层（全连接）| LoRA 就是在 Linear 旁边加小矩阵 |
| 迁移学习（SegFormer 微调）| LoRA 是更极端的参数高效微调——只改 0.07% |
| Attention（SegFormer/DDPM）| LoRA 注入 Cross-Attention 的 Q/K/V |
| MSE Loss（SRCNN/DDPM）| SD LoRA 训练也用 MSE 预测噪声 |

---

## LoRA 的局限

- **rank 越小，表达能力越弱**：rank=4 适合风格迁移，复杂任务需要更大 rank
- **只改 Attention 层**：LoRA 通常只注入注意力层（Q/K/V），卷积层改动效果不如全量
- **多风格叠加**：多个 LoRA 权重可以线性叠加（`alpha` 控制各自强度），但可能互相干扰
