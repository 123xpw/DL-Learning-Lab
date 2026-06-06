import torch
import torchvision
from src.noise_scheduler import NoiseScheduler
from src.unet import UNet


DEVICE    = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
T         = 1000
CFG_SCALE = 3.0   # guidance scale：越大越符合条件，越小越随机（通常 2-5 即可）


def sample(target_digit=None, n=16):
    """
    target_digit: 想生成的数字（0-9），None 表示随机生成
    n: 生成几张图
    """
    scheduler = NoiseScheduler(T=T)
    model = UNet(in_ch=1, base_ch=32, time_dim=128, num_classes=11).to(DEVICE)
    model.load_state_dict(torch.load("ddpm_mnist.pth", map_location=DEVICE))
    model.eval()

    xt = torch.randn(n, 1, 28, 28).to(DEVICE)

    # 有条件标签：全部设为 target_digit；无条件标签：全部设为 10（占位符）
    if target_digit is not None:
        label_cond   = torch.full((n,), target_digit, dtype=torch.long, device=DEVICE)
    else:
        label_cond   = torch.randint(0, 10, (n,), device=DEVICE)
    label_uncond = torch.full((n,), 10, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for t_val in reversed(range(T)):
            t = torch.full((n,), t_val, dtype=torch.long, device=DEVICE)

            # CFG：有条件预测 + 无条件预测，加权组合
            noise_cond   = model(xt, t, label_cond)
            noise_uncond = model(xt, t, label_uncond)
            noise_pred   = noise_uncond + CFG_SCALE * (noise_cond - noise_uncond)

            xt = scheduler.step(xt, t, noise_pred)

    imgs = (xt.clamp(-1, 1) + 1) / 2

    fname = f"generated_digit{target_digit}.png" if target_digit is not None else "generated.png"
    torchvision.utils.save_image(imgs, fname, nrow=4)
    print(f"生成完成：{fname}")


if __name__ == "__main__":
    # 生成全部数字 0-9，每个数字 16 张
    for d in range(10):
        sample(target_digit=d, n=16)
