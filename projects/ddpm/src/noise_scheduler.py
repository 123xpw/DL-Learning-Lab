import torch
import numpy as np


class NoiseScheduler:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T

        # β_t：每一步加多少噪声，从很小线性增大到较大
        self.betas = torch.linspace(beta_start, beta_end, T)

        # α_t = 1 - β_t：每一步保留多少原始信号
        self.alphas = 1.0 - self.betas

        # ᾱ_t = α_1 × α_2 × ... × α_t：从 x_0 直接跳到 x_t 时保留多少原始信号
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        # 前向过程公式：x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε
        ab = self.alpha_bars.to(t.device)[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    def step(self, xt, t, predicted_noise):
        # 反向过程：给定 x_t 和预测的噪声，计算 x_{t-1}
        beta = self.betas.to(t.device)[t].view(-1, 1, 1, 1)
        alpha = self.alphas.to(t.device)[t].view(-1, 1, 1, 1)
        ab = self.alpha_bars.to(t.device)[t].view(-1, 1, 1, 1)

        # 用预测的噪声还原出 x_0 的估计，再往前推一步
        x0_pred = (xt - torch.sqrt(1 - ab) * predicted_noise) / torch.sqrt(ab)
        x0_pred = x0_pred.clamp(-1, 1)

        mean = (torch.sqrt(alpha) * (1 - ab / alpha) * xt +
                torch.sqrt(ab / alpha) * beta * x0_pred) / (1 - ab)

        if t[0] == 0:
            return mean
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(beta) * noise
