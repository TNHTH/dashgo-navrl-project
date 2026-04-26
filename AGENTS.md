# AGENTS.md

## Scope
- 本文件适用于 `/home/gwh/dashgo_navrl_project`，优先于上级目录中的 `/home/gwh/AGENTS.md`。
- 本仓库的项目级长期规则、记录位置和执行约束以本文件为准。

## 输出与记录
- 默认使用中文回复，默认使用中文代码注释。
- 先给结论，再给执行细节。
- 长任务、训练、调试、评测、部署、复盘、阶段总结等正式项目记录，默认写入 Obsidian 项目库：
  - `/home/gwh/文档/Obsidian Vault/03_项目记录/DashGo/`
- 对需要记录的长任务、多步骤任务、训练、调试、评测、部署与复盘任务，默认执行“步骤同步记录”，不要等用户再次提醒“记录一下”才补写。
- 仓库内 `docs/` 默认只放以下内容：
  - 与代码版本强绑定的技术设计文档
  - 仓库内使用说明、接口契约、对上游差异说明
  - 用户明确要求保留在仓库内的文档
- 如果一份内容属于“正式项目记录”，先写 Obsidian，再决定仓库里是否需要保留简短索引；不要先把完整正式记录写进仓库 `docs/`。

## 执行前分类
- 创建任何新的 Markdown 文档前，先在心里完成一次分类：
  - `formal_project_record`
  - `repo_doc`
- 若属于 `formal_project_record`，优先动作是：
  1. 确认 Obsidian 存放路径
  2. 在 Obsidian 建立或复用记录
  3. 再开始实质性执行或补充仓库内索引
- 若属于 `repo_doc`，才直接在仓库中创建。

## 项目默认
- 本仓库的训练、评测、自治、benchmark、长跑与失败复盘，都按 `formal_project_record` 处理。
- `task_plan.md / findings.md / progress.md` 继续保留在仓库内，作为工作台账；但正式复盘、正式执行记录和用户要求长期保留的项目记录，应优先落到 Obsidian。
- 若仓库内与 Obsidian 同时出现同类正式记录，以 Obsidian 为主版本，仓库副本默认视为待清理候选。

## 通用复盘沉淀规则
- 只把跨项目复用的流程规则写入 `AGENTS.md` 或 shared skill；项目事实、实验结论、仓库特定路径和一次性判断写入项目文档、工作台账或 Obsidian。
- 涉及相似仓库名、跨仓引用、路径迁移，或用户强调“不是 X 是 Y”时，首次编辑前必须核对 `pwd`、`git rev-parse --show-toplevel` 和目标文件绝对路径；`apply_patch` 优先使用目标仓库绝对路径。
- 宣称 smoke、训练、评测、构建或导出成功前，必须同时核对退出码和至少一个产物证据，例如 manifest、checkpoint、报告 JSON、日志哨兵或 schema 校验。
- 做 schema、合同、模型、配置、评测口径等破坏性改动时，必须把 contract/version metadata 写入产物，并在 compare/load 阶段做 fail-fast 校验。
- 若发现误写非目标仓库，先停止本轮编辑，列出误写路径，只清理本轮新建或修改的文件，不触碰该仓库其他脏改动。
- 使用 subagent 后，完成或报错即关闭；若 subagent 结论冲突，以用户当前目标、仓库事实和可运行验证为准，并在复盘中记录取舍。
- `.codex/tmp/**/RETRO.md` 与 `.codex/tmp/**/ERROR_TRACE.md` 只视为当前窗口临时草稿；跨窗口后不得当作长期记忆。若用户要求清扫，先列出精确候选路径，再只删除这些临时文件。
