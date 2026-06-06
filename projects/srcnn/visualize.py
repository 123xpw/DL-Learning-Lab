import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.dataset import SRDataset
from src.model import SRCNN
from src.evaluate import psnr

def visualize(num_samples=4, save_path="sr_results.png"):
    model = SRCNN()
    model.load_state_dict(torch.load("srcnn.pth", map_location="cpu"))
    model.eval()

    dataset = SRDataset(
        "../segformer-voc/data/VOCdevkit/VOC2012/JPEGImages",
        scale=2, patch_size=128, split="val"
    )

    indices = [0, 5, 15, 30][:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3.5))
    fig.suptitle("SRCNN Super-Resolution (2x)", fontsize=13)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            lr, hr = dataset[idx]

            sr = model(lr.unsqueeze(0)).squeeze(0).clamp(0, 1)

            bicubic_psnr = psnr(lr, hr)
            srcnn_psnr   = psnr(sr,  hr)

            # 转成可显示格式 (H, W, 3)
            lr_np = lr.permute(1, 2, 0).numpy()
            sr_np = sr.permute(1, 2, 0).numpy()
            hr_np = hr.permute(1, 2, 0).numpy()

            axes[row, 0].imshow(lr_np)
            axes[row, 0].set_title(f"Bicubic input\nPSNR: {bicubic_psnr:.2f} dB", fontsize=9)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(sr_np)
            axes[row, 1].set_title(f"SRCNN output\nPSNR: {srcnn_psnr:.2f} dB", fontsize=9)
            axes[row, 1].axis("off")

            axes[row, 2].imshow(hr_np)
            axes[row, 2].set_title("Ground Truth HR", fontsize=9)
            axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize(num_samples=4)
