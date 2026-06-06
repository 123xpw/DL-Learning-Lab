import sys
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from src.model import AODNet

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_image(path):
    img = Image.open(path).convert("RGB")
    # 保持原始尺寸，只转 tensor
    return TF.to_tensor(img).unsqueeze(0)


def to_pil(tensor):
    return TF.to_pil_image(tensor.squeeze(0).clamp(0, 1).cpu())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python infer.py <image1.jpg> [image2.jpg ...]")
        sys.exit(1)

    model = AODNet().to(DEVICE)
    model.load_state_dict(torch.load("aodnet.pth", map_location=DEVICE))
    model.eval()

    paths = sys.argv[1:]
    fig, axes = plt.subplots(len(paths), 2, figsize=(10, 4 * len(paths)))
    if len(paths) == 1:
        axes = [axes]

    with torch.no_grad():
        for i, path in enumerate(paths):
            inp = load_image(path).to(DEVICE)
            out = model(inp)

            axes[i][0].imshow(to_pil(inp)); axes[i][0].set_title("Real Hazy Input");  axes[i][0].axis("off")
            axes[i][1].imshow(to_pil(out)); axes[i][1].set_title("AOD-Net Output");   axes[i][1].axis("off")
            axes[i][0].set_ylabel(path.split("/")[-1], fontsize=8)

    plt.tight_layout()
    out_path = "infer_output.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存：{out_path}")
    plt.show()
