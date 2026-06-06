import math
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.dataset import LowLightDataset
from src.model import DCENet

VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
PATCH_SIZE = 256
N_SAMPLES  = 6
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


def to_img(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


if __name__ == "__main__":
    model = DCENet().to(DEVICE)
    model.load_state_dict(torch.load("zerodce.pth", map_location=DEVICE))
    model.eval()

    val_ds  = LowLightDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")
    indices = list(range(0, len(val_ds), max(1, len(val_ds) // N_SAMPLES)))[:N_SAMPLES]

    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(10, N_SAMPLES * 3))
    axes[0, 0].set_title("Dark Input",    fontsize=12)
    axes[0, 1].set_title("Enhanced",      fontsize=12)
    axes[0, 2].set_title("Clean GT",      fontsize=12)

    psnr_dark_list = []
    psnr_enh_list  = []

    with torch.no_grad():
        for row, idx in enumerate(indices):
            dark, clean = val_ds[idx]
            dark  = dark.unsqueeze(0).to(DEVICE)
            clean = clean.unsqueeze(0).to(DEVICE)

            enhanced, _ = model(dark)

            p_dark = psnr(dark, clean)
            p_enh  = psnr(enhanced, clean)
            psnr_dark_list.append(p_dark)
            psnr_enh_list.append(p_enh)

            axes[row, 0].imshow(to_img(dark));     axes[row, 0].axis("off")
            axes[row, 1].imshow(to_img(enhanced)); axes[row, 1].axis("off")
            axes[row, 2].imshow(to_img(clean));    axes[row, 2].axis("off")
            axes[row, 1].set_xlabel(
                f"dark: {p_dark:.1f} dB  enhanced: {p_enh:.1f} dB", fontsize=8)

    avg_dark = sum(psnr_dark_list) / len(psnr_dark_list)
    avg_enh  = sum(psnr_enh_list)  / len(psnr_enh_list)
    diff = avg_enh - avg_dark
    print(f"Avg PSNR  Dark: {avg_dark:.2f} dB  |  Enhanced: {avg_enh:.2f} dB  |  Delta: {diff:+.2f} dB")

    plt.suptitle(f"Zero-DCE  dark→enhanced PSNR delta: {diff:+.2f} dB", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualize_output.png", dpi=150, bbox_inches="tight")
    print("Saved: visualize_output.png")
    plt.show()
