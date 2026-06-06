import math
import torch
import matplotlib.pyplot as plt

from src.dataset import RainDataset
from src.model   import SimpleMPRNet

VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
PATCH_SIZE = 128
N_SAMPLES  = 6
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


def to_img(t):
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


if __name__ == "__main__":
    model = SimpleMPRNet(n_feat=16).to(DEVICE)
    model.load_state_dict(torch.load("deraining.pth", map_location=DEVICE))
    model.eval()

    val_ds  = RainDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")
    indices = list(range(0, len(val_ds), max(1, len(val_ds) // N_SAMPLES)))[:N_SAMPLES]

    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(9, N_SAMPLES * 3))
    axes[0, 0].set_title("Rainy Input")
    axes[0, 1].set_title("Derained (Stage 2)")
    axes[0, 2].set_title("Clean GT")

    psnr_rainy_list, psnr_dn_list = [], []

    with torch.no_grad():
        for row, idx in enumerate(indices):
            rainy, clean = val_ds[idx]
            rainy = rainy.unsqueeze(0).to(DEVICE)
            clean = clean.unsqueeze(0).to(DEVICE)
            _, derained = model(rainy)

            p_in  = psnr(rainy,    clean)
            p_out = psnr(derained, clean)
            psnr_rainy_list.append(p_in)
            psnr_dn_list.append(p_out)

            axes[row, 0].imshow(to_img(rainy));    axes[row, 0].axis("off")
            axes[row, 1].imshow(to_img(derained)); axes[row, 1].axis("off")
            axes[row, 2].imshow(to_img(clean));    axes[row, 2].axis("off")
            axes[row, 1].set_xlabel(
                f"rainy: {p_in:.1f} dB  →  derained: {p_out:.1f} dB", fontsize=8)

    avg_in  = sum(psnr_rainy_list) / len(psnr_rainy_list)
    avg_out = sum(psnr_dn_list)    / len(psnr_dn_list)
    delta   = avg_out - avg_in
    print(f"Avg PSNR  Rainy: {avg_in:.2f} dB  |  Derained: {avg_out:.2f} dB  |  Delta: {delta:+.2f} dB")
    plt.suptitle(f"SimpleMPRNet  rainy→derained PSNR delta: {delta:+.2f} dB", fontsize=11)
    plt.tight_layout()
    plt.savefig("visualize_output.png", dpi=150, bbox_inches="tight")
    print("Saved: visualize_output.png")
    plt.show()
