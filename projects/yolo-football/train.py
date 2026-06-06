from ultralytics import YOLO

# ── exp1：基线 ──────────────────────────────────────────────────
# model = YOLO('yolo11s.pt')
# model.train(
#     data='/Users/quokka/Academic/projects/yolo-football/configs/data.yaml',
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     lr0=0.01,
#     project='/Users/quokka/Academic/projects/yolo-football/runs',
#     name='exp1',
#     device='cpu',
# )

# ── exp2（实际目录 exp2-2）：copy_paste 增强 ──────────────────────
# copy_paste=0.3 在 640px 下无明显效果，mAP50-95 与 exp1 持平
# model = YOLO('yolo11s.pt')
# model.train(
#     data='/Users/quokka/Academic/projects/yolo-football/configs/data.yaml',
#     epochs=50,
#     imgsz=640,
#     batch=16,
#     lr0=0.01,
#     copy_paste=0.3,
#     project='/Users/quokka/Academic/projects/yolo-football/runs',
#     name='exp2',
#     device='cpu',
# )

# ── exp3：高分辨率 + MPS 加速 ────────────────────────────────────
# imgsz=1280 使训练 loss 下降，但 val loss 略升，mAP50-95 依然 ~0.58
# 结论：瓶颈是验证集太小（28 个球），不是分辨率
# model = YOLO('yolo11s.pt')
# model.train(
#     data='/Users/quokka/Academic/projects/yolo-football/configs/data.yaml',
#     epochs=50,
#     imgsz=1280,
#     batch=8,
#     lr0=0.01,
#     copy_paste=0.3,
#     project='/Users/quokka/Academic/projects/yolo-football/runs',
#     name='exp3',
#     device='mps',
# )

# ── 当前运行：修改下方配置后取消注释 ────────────────────────────────
model = YOLO('yolo11s.pt')

results = model.train(
    data='/Users/quokka/Academic/projects/yolo-football/configs/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    project='/Users/quokka/Academic/projects/yolo-football/runs',
    name='exp1',
    device='cpu',
)

metrics = model.val()
print(metrics)
