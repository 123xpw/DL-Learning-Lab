import math
import torch
import matplotlib.pyplot as plt

from src.dataset import NoisyDataset
from src.model   import DnCNN

DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "denoising.pth"
N_FEAT     = 64
DEPTH      = 9
N_SAMPLES  = 3


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


def to_img(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


model = DnCNN(n_feat=N_FEAT, depth=DEPTH).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

val_ds = NoisyDataset(
    "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages",
    patch_size=128, split="val"
)

fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(10, 3.5 * N_SAMPLES))
for i in range(N_SAMPLES):
    noisy, clean = val_ds[i * 30]
    noisy_t = noisy.unsqueeze(0).to(DEVICE)
    clean_t = clean.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_noise = model(noisy_t)
        denoised_t = (noisy_t - pred_noise).clamp(0, 1)

    p_noisy    = psnr(noisy_t, clean_t)
    p_denoised = psnr(denoised_t, clean_t)

    axes[i][0].imshow(to_img(noisy_t));    axes[i][0].set_title(f"Noisy\n{p_noisy:.2f} dB")
    axes[i][1].imshow(to_img(denoised_t)); axes[i][1].set_title(f"Denoised\n{p_denoised:.2f} dB")
    axes[i][2].imshow(to_img(clean_t));    axes[i][2].set_title("Clean (GT)")
    for ax in axes[i]: ax.axis("off")

plt.tight_layout()
plt.savefig("visualize_output.png", dpi=150)
print("已保存：visualize_output.png")
