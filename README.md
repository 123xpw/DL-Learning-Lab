# DL Learning Lab

Deep learning study notes and project code — learning bottom-up through hands-on experiments.

## Structure

```
DL-Learning-Lab/
├── _DL_Notes/          # Obsidian notes (theory + experiment analysis)
│   └── _02_Architectures/
│       ├── DDPM-扩散模型入门.md
│       ├── SegFormer-语义分割入门.md
│       ├── Stable-Diffusion-LoRA入门.md
│       └── ...
├── projects/           # Source code for each project
│   ├── ddpm/           # DDPM conditional generation (MNIST + CFG)
│   ├── segformer/      # SegFormer from-scratch (MiT-B0, VOC2012)
│   ├── segformer-voc/  # SegFormer fine-tuning (HuggingFace, VOC2012)
│   ├── srcnn/          # SRCNN super-resolution (VOC2012)
│   ├── zero-dce/       # Zero-DCE low-light enhancement
│   ├── deraining/      # SimpleMPRNet image deraining
│   ├── aodnet/         # AOD-Net image dehazing
│   ├── resnet/         # ResNet classification
│   ├── deblurring/     # Image deblurring
│   ├── denoising/      # Image denoising
│   └── yolo-football/  # YOLOv11s football detection
└── colab/              # Colab notebooks (SD LoRA, etc.)
```

## Projects

| Project | Task | Dataset | Key Result |
|---------|------|---------|------------|
| [ddpm](projects/ddpm/) | Conditional image generation | MNIST | CFG digit generation (0-9) |
| [segformer-voc](projects/segformer-voc/) | Semantic segmentation (fine-tune) | VOC2012 | mIoU = 0.6658 |
| [segformer](projects/segformer/) | Semantic segmentation (from scratch) | VOC2012 | mIoU = 0.1367 (epoch 40) |
| [srcnn](projects/srcnn/) | Super-resolution | VOC2012 | PSNR = 26.32 dB |
| [zero-dce](projects/zero-dce/) | Low-light enhancement | VOC2012 | PSNR delta +0.80 dB |
| [deraining](projects/deraining/) | Image deraining | VOC2012 | PSNR = 30.60 dB |
| [aodnet](projects/aodnet/) | Image dehazing | VOC2012 | — |
| [yolo-football](projects/yolo-football/) | Object detection | Football dataset | mAP50 = 0.816 |

## Notes

Study notes are written in Obsidian and cover architecture theory, experiment logs, and latest research.  
Each note links to the corresponding code in `projects/`.

Model weights (`.pth`, `.pt`) and datasets are excluded from this repo (too large).
