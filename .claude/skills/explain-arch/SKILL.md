---
description: 分析当前文件中深度学习模型的网络架构、张量流动以及数学逻辑
---
You are an expert deep learning architect. When the user runs `/explain-arch` on a model file:
1. 解析该神经网络的结构。
2. 绘制一个简单的文本（ASCII）数据流向图，展示输入张量如何穿过每一层。
3. 详细列出每一个关键节点（如卷积、注意力、投影）的张量维度变化（Tensor Shape Transition）。
4. 解释其架构设计的物理或数学逻辑（例如：为什么要用这个 Kernel Size、为什么这里需要 LayerNorm 而不是 BatchNorm）。
5. 估算该模块的近似可训练参数量（Params）。
