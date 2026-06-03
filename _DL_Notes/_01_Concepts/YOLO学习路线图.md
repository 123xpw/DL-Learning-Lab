---
tags: [deep-learning, yolo, object-detection, roadmap]
created: 2026-05-28
source: https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb
---

# YOLO 学习路线图

> 从最有趣的项目出发，向下挖掘每一个基础概念。
> 这是整个知识图谱的 **根节点**。

---

## 为什么从 YOLO 开始？

YOLO（You Only Look Once）是目标检测领域最经典的工程实践：
- **一次前向传播**就能输出所有目标的位置和类别
- 从 v1（2016）到 v11（2024），每一次迭代都踩在一个核心问题上
- Ultralytics 的训练框架把复杂的训练流程包成了几行命令，**适合从实践反推原理**

---

## 一次完整的 YOLO 训练流程（来自 Colab 项目）

```mermaid
flowchart TD
    A[准备数据集 ≥200张] --> B[标注边界框 生成.txt标签]
    B --> C[划分 Train/Val 90%/10%]
    C --> D[编写 data.yaml 定义类别+路径]
    D --> E[选择预训练模型 yolo11s.pt]
    E --> F[执行训练 epochs=60 imgsz=640]
    F --> G[监控曲线 mAP / Loss / Precision]
    G --> H{模型达标?}
    H -- 是 --> I[导出 best.pt 部署]
    H -- 否 --> J[调整超参数或扩充数据]
    J --> F
```

---

## 三个必须理解的核心问题

### 1. 怎么判断模型训练好了？

| 指标 | 含义 | 目标 |
|------|------|------|
| **mAP@0.5** | IoU>0.5 时的平均精度 | >80% 合格，>90% 优秀 |
| **Precision** | 预测为正中真正是正的比例 | 越高误报越少 |
| **Recall** | 所有真实目标中被检测到的比例 | 越高漏报越少 |
| **Train Loss** | 训练集误差 | 应持续下降并收敛 |
| **Val Loss** | 验证集误差 | 应与 Train Loss 同步，否则过拟合 |

> **关键信号**：Val Loss 开始上升而 Train Loss 还在降 → [[过拟合与正则化]] 发生了

### 2. 超参数调整的逻辑是什么？

```mermaid
flowchart LR
    P[观察到的问题] --> A[Loss不收敛]
    P --> B[过拟合]
    P --> C[mAP低但Loss正常]
    A --> A1[降低学习率 检查标注质量]
    B --> B1[增加数据增强 减少epochs]
    C --> C1[扩充数据集 增加epochs]
```

详细调参逻辑见 [[超参数调整指南]]

### 3. YOLO 各版本更新了什么？

```mermaid
timeline
    title YOLO 版本迭代核心突破
    2016 : YOLOv1 : one-stage检测 Grid cell直接回归
    2017 : YOLOv2 : 引入Anchor Box + BN层
    2018 : YOLOv3 : 多尺度检测 Darknet-53
    2020 : YOLOv4/v5 : CSP结构 Mosaic增强
    2022 : YOLOv8 : Anchor-free 统一框架
    2024 : YOLO11 : C3k2模块 更低参数量
```

每个版本背后的技术细节：
- [[Anchor Box 机制]] — v2 引入，v8 彻底抛弃，这是理解 YOLO 演进的主线
- [[多尺度特征金字塔 FPN]] — v3 开始支持小目标检测
- [[Mosaic 数据增强]] — v4/v5 引入，大幅提升小目标检测
- [[Anchor-free 检测头]] — v8 的核心改变

---

## 学习路径建议

```mermaid
flowchart TD
    Start([从这里开始]) --> T1[理解训练流程]
    T1 --> T2[深入单个组件]
    T1 --> T3[动手调参实验]
    T2 --> T4[卷积神经网络基础]
    T2 --> T5[Anchor Box 机制]
    T2 --> T6[检测头 Detection Head]
    T3 --> T7[超参数调整指南]
    T4 --> T8[YOLOv3 完整架构]
    T5 --> T8
    T6 --> T8
    T8 --> T9[对比 YOLOv8 理解演进]
    T9 --> Goal([目标: mAP 大于 90%])
```

对应笔记链接：
- [[卷积神经网络基础]] — stride, padding, receptive field
- [[Anchor Box 机制]] — 为什么需要它，v8 为什么去掉它
- [[损失函数 YOLO Loss]] — 分类损失 + 定位损失 + 置信度损失
- [[超参数调整指南]] — 学习率、batch size、epochs 的系统调法
- [[过拟合与正则化]] — Dropout、数据增强、早停

---

*下一步*：[[Anchor Box 机制]]
