import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.dataset import HazeDataset, synthesize_haze
from src.model import AODNet
import math

VOC_IMGS   = "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages"
PATCH_SIZE = 256
N_SAMPLES  = 6
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20 * math.log10(1.0 / math.sqrt(mse.item())) if mse > 0 else float("inf")


def to_img(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


if __name__ == "__main__":
    model = AODNet().to(DEVICE)
    model.load_state_dict(torch.load("aodnet.pth", map_location=DEVICE))
    model.eval()

    val_dataset = HazeDataset(VOC_IMGS, patch_size=PATCH_SIZE, split="val")
    indices = list(range(0, len(val_dataset), max(1, len(val_dataset) // N_SAMPLES)))[:N_SAMPLES]

    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(10, N_SAMPLES * 3))
    axes[0, 0].set_title("Hazy Input",    fontsize=12)
    axes[0, 1].set_title("AOD-Net Output", fontsize=12)
    axes[0, 2].set_title("Clean GT",       fontsize=12)

    psnr_hazy_list   = []
    psnr_dehaze_list = []

    with torch.no_grad():
        for row, idx in enumerate(indices):
            hazy, clean = val_dataset[idx]
            hazy_t  = hazy.unsqueeze(0).to(DEVICE)
            clean_t = clean.unsqueeze(0).to(DEVICE)

            dehazed = model(hazy_t)

            p_hazy   = psnr(hazy_t,   clean_t)
            p_dehaze = psnr(dehazed, clean_t)
            psnr_hazy_list.append(p_hazy)
            psnr_dehaze_list.append(p_dehaze)

            axes[row, 0].imshow(to_img(hazy_t));   axes[row, 0].axis("off")
            axes[row, 1].imshow(to_img(dehazed));  axes[row, 1].axis("off")
            axes[row, 2].imshow(to_img(clean_t));  axes[row, 2].axis("off")
            axes[row, 0].set_ylabel(f"#{idx}", fontsize=9)
            axes[row, 1].set_xlabel(f"PSNR: {p_dehaze:.2f} dB  (hazy: {p_hazy:.2f} dB)", fontsize=8)

    avg_hazy   = sum(psnr_hazy_list)   / len(psnr_hazy_list)
    avg_dehaze = sum(psnr_dehaze_list) / len(psnr_dehaze_list)
    print(f"Avg PSNR — Hazy: {avg_hazy:.2f} dB  |  Dehazed: {avg_dehaze:.2f} dB  |  Gain: +{avg_dehaze - avg_hazy:.2f} dB")

    plt.suptitle(f"AOD-Net Dehazing  (avg PSNR: {avg_dehaze:.2f} dB)", fontsize=13)
    plt.tight_layout()
    plt.savefig("visualize_output.png", dpi=150, bbox_inches="tight")
    print("已保存：visualize_output.png")
    plt.show()
