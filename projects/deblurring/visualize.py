import math
import torch
import matplotlib.pyplot as plt

from src.dataset import BlurDataset
from src.model   import SimpleUNet

DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "deblurring.pth"
BASE       = 16
N_SAMPLES  = 3


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


def to_img(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


model = SimpleUNet(base=BASE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

val_ds = BlurDataset(
    "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages",
    patch_size=128, split="val"
)

fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(10, 3.5 * N_SAMPLES))
for i in range(N_SAMPLES):
    blurry, clean = val_ds[i * 30]
    blurry_t = blurry.unsqueeze(0).to(DEVICE)
    clean_t  = clean.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        deblur_t = model(blurry_t)

    p_blurry = psnr(blurry_t, clean_t)
    p_deblur = psnr(deblur_t, clean_t)

    axes[i][0].imshow(to_img(blurry_t)); axes[i][0].set_title(f"Blurry\n{p_blurry:.2f} dB")
    axes[i][1].imshow(to_img(deblur_t)); axes[i][1].set_title(f"Deblurred\n{p_deblur:.2f} dB")
    axes[i][2].imshow(to_img(clean_t));  axes[i][2].set_title("Clean (GT)")
    for ax in axes[i]: ax.axis("off")

plt.tight_layout()
plt.savefig("visualize_output.png", dpi=150)
print("已保存：visualize_output.png")
