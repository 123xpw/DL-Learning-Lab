import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSES  = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '轮船', '卡车']

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

# ── 加载模型 ──────────────────────────────────────────────────────────────
from src.model import ResNet18
model = ResNet18(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load("resnet18.pth", map_location=DEVICE))
model.eval()

# ── 加载 val 集 ───────────────────────────────────────────────────────────
val_ds = datasets.CIFAR10("./data", train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]))

def denorm(t):
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1)

# ── 推理：取前 24 张图 ────────────────────────────────────────────────────
imgs   = torch.stack([val_ds[i][0] for i in range(24)])
labels = torch.tensor([val_ds[i][1] for i in range(24)])
with torch.no_grad():
    preds = model(imgs.to(DEVICE)).argmax(1).cpu()

# ── 绘图 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 6, figsize=(12, 9))
fig.suptitle("ResNet-18 预测结果（绿色=正确 / 红色=错误）", fontsize=14)

for i, ax in enumerate(axes.flat):
    img_show = denorm(imgs[i]).permute(1, 2, 0).numpy()
    ax.imshow(img_show, interpolation='bilinear')
    ax.axis('off')

    true_c = labels[i].item()
    pred_c = preds[i].item()
    correct = (true_c == pred_c)

    if correct:
        title = f"✓ {CLASSES[pred_c]}"
        color = 'green'
    else:
        title = f"✗ 真:{CLASSES[true_c]}\n  预:{CLASSES[pred_c]}"
        color = 'red'

    ax.set_title(title, fontsize=9, color=color, pad=3)

plt.tight_layout()
plt.savefig("visualize_output.png", dpi=150, bbox_inches='tight')
print("已保存：visualize_output.png")
