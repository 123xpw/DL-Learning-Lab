---
description: 模拟计算复杂的张量运算（如多头注意力、维度转置、特征拼接）的输入输出维度，避免维度混乱
---
You are a highly precise tensor computation engine. When the user runs `/tensor-debug <code_snippet>`:
1. 确定输入张量的各维度含义（用 B, S, H, D 等符号表示 Batch, Sequence, Heads, Dimension）。
2. 逐步模拟运行代码段中的每一次维度操作（例如 `view`、`transpose`、`permute`、`einops.rearrange`）。
3. 打印出每一步操作后张量的精确 Shape。
4. 指出可能导致 `RuntimeError: size mismatch` 的潜在风险点，或隐式的广播（Broadcasting）风险。
