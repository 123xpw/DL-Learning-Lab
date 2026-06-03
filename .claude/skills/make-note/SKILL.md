---
description: 根据指定的模型文件，在 Obsidian 中生成包含 Mermaid 架构图的详细阅读笔记
---
You are an expert technical writer. When the user runs `/make-note <model_file_path>`:
1. 深入分析该 Python 模型文件。
2. 在 `📝_DL_Notes/_02_Architectures/` 下生成一个与模型同名的 `.md` 文件。
3. 笔记内容必须包括：
   - 顶部 YAML 标签和元数据。
   - 用 `mermaid` 绘制的、在 Obsidian 中可直接渲染的模块流向图。
   - 对核心张量形状变换的表格化整理。
   - 关键代码片段的逐行中文注释。
