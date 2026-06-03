# DL Learning Lab & Obsidian 规范

## 运行环境约束
- 我们不在本地运行代码。
- 所有的笔记输出和代码修改必须保持同步。

## Obsidian 笔记规范
- 所有的知识性输出必须保存在 `📝_DL_Notes/` 下的相应子目录中，且必须为 `.md` 格式。
- **YAML Frontmatter**：每篇生成的笔记顶部必须包含标准的前言，如：
  ---
  tags: [deep-learning, pytorch, concept]
  created: 2026-05-26
  ---
- **双向链接**：在解释概念时，尽可能使用 Obsidian 的双向链接语法（例如 `[[梯度消失]]` 或 `[[Transformer]]`），方便在 Obsidian 中形成知识图谱。
- **Mermaid 支持**：在解释网络架构或数据流时，必须使用 ```mermaid 渲染出可视化的流程图，以便在 Obsidian 中直接预览。
