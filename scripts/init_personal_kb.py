from __future__ import annotations

import sys
from pathlib import Path


ROOT_AGENTS_SECTION = """## personal-kb

本项目有独立知识库：personal-kb/。

规则：
- 回答项目相关问题前，先读 personal-kb/index.md。
- 值得沉淀的问答追加到 personal-kb/raw/YYYY-MM-DD.md。
- 稳定结论整理到 personal-kb/wiki/*.md。
- 所有原始问答和整理后的知识点默认使用中文。
- 更新 Wiki 页面后，同步更新 personal-kb/index.md 和 personal-kb/log.md。
"""


KB_AGENTS = """# 个人知识库维护协议

这个目录是本项目的独立个人知识库。未来所有维护动作都遵守以下规则。

## 语言

- 默认使用中文书写。
- `raw/` 中的原始问答使用中文保存。
- `wiki/` 中整理后的知识、概念、流程、决策和总结使用中文保存。
- 英文来源、英文术语和专有名词可以保留英文原文，但应在必要时补充中文解释。

## 导航

- 回答项目相关问题前，先读 `index.md`。
- 需要上下文时，再读相关 `wiki/*.md` 页面。
- `raw/` 是出处和历史，不应该作为首选导航入口。

## 归档

- 值得沉淀的问答，要追加到当天的 `raw/YYYY-MM-DD.md`。
- `raw/` 文件只追加，不改写历史问答。
- 每条原始记录应包含日期、主题、用户问题、助手回答、整理状态和相关 Wiki 页面。

## 整理

- 可复用的结论、概念、流程、经验和决策，要整理进 `wiki/*.md`。
- `wiki/` 页面是当前理解，可以重写、合并、拆分和交叉链接。
- 不要把 Wiki 写成聊天记录；Wiki 应该写成以后可以直接使用的知识。

## 索引和日志

- 新增、重命名或大幅修改 Wiki 页面时，更新 `index.md`。
- 每次捕获、整理、重构或留下待处理事项时，更新 `log.md`。
- 如果原始问答已保存但还未整理进 Wiki，在 `log.md` 标记为待处理。
"""


INDEX = """# 项目知识库索引

这是本项目独立知识库的中文导航入口。查询项目知识时先读这里，再进入相关页面。

## Wiki 页面

- [项目概览](wiki/project.md)：项目目标、上下文、入口和重要背景。
- [设计决策](wiki/decisions.md)：重要决策、取舍和原因。
- [工作流程](wiki/workflows.md)：常用命令、检查流程和操作经验。

## 原始问答

- 暂无。

## 待处理

- 暂无。
"""


LOG = """# 项目知识库日志

## 初始化

- 创建项目独立 `personal-kb/` 目录结构。
- 写入中文维护协议、索引、种子 Wiki 页面和模板。
- 更新项目根目录 `AGENTS.md` 的 `## personal-kb` 规则。
"""


PROJECT_PAGE = """---
title: 项目概览
tags: [项目]
updated:
---

# 项目概览

## 摘要

记录本项目的目标、上下文、重要入口和长期背景。

## 关键入口

- 项目根目录：
- 主要文档：
- 常用命令：

## 相关页面

- [设计决策](decisions.md)
- [工作流程](workflows.md)
"""


DECISIONS_PAGE = """---
title: 设计决策
tags: [决策]
updated:
---

# 设计决策

## 摘要

记录本项目中值得长期保留的重要决策、取舍和原因。

## 决策记录

- 暂无。

## 相关页面

- [项目概览](project.md)
- [工作流程](workflows.md)
"""


WORKFLOWS_PAGE = """---
title: 工作流程
tags: [流程]
updated:
---

# 工作流程

## 摘要

记录本项目常用的命令、检查方法、调试流程和操作经验。

## 常用流程

- 暂无。

## 相关页面

- [项目概览](project.md)
- [设计决策](decisions.md)
"""


RAW_TEMPLATE = """# YYYY-MM-DD 原始问答

## 条目：主题

- 时间：YYYY-MM-DD HH:MM
- 状态：已整理 / 待整理
- 相关 Wiki：[`页面标题`](../wiki/page.md)

### 用户问题

在这里保存用户原始问题。

### 助手回答

在这里保存助手原始回答或尽量完整的回答记录。

### 整理摘要

在这里写入本条问答已经沉淀出的知识点。
"""


WIKI_TEMPLATE = """---
title: 页面标题
tags: [主题]
updated: YYYY-MM-DD
---

# 页面标题

## 摘要

用一小段中文说明这个页面解决什么问题。

## 关键观点

- 观点一。
- 观点二。

## 细节

把可复用的知识、流程、决策和背景放在这里。

## 相关页面

- [相关页面](related-page.md)
"""


SEED_FILES = {
    "personal-kb/index.md": INDEX,
    "personal-kb/log.md": LOG,
    "personal-kb/wiki/project.md": PROJECT_PAGE,
    "personal-kb/wiki/decisions.md": DECISIONS_PAGE,
    "personal-kb/wiki/workflows.md": WORKFLOWS_PAGE,
    "personal-kb/templates/raw-entry.md": RAW_TEMPLATE,
    "personal-kb/templates/wiki-page.md": WIKI_TEMPLATE,
    "personal-kb/inbox/.gitkeep": "",
}


def remove_default_kb_agents(project: Path) -> bool:
    agents = project / "personal-kb" / "AGENTS.md"
    if not agents.exists() or not agents.is_file():
        return False
    if agents.read_text(encoding="utf-8") != KB_AGENTS:
        return False
    agents.unlink()
    return True


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def replace_or_append_section(content: str, marker: str, section: str) -> str:
    section = section.strip() + "\n"
    if marker not in content:
        if content.strip():
            return content.rstrip() + "\n\n" + section
        return section

    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if start is None:
        return content.rstrip() + "\n\n" + section

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break

    before = "\n".join(lines[:start]).rstrip()
    after = "\n".join(lines[end:]).lstrip()
    parts = [p for p in (before, section.strip(), after) if p]
    return "\n\n".join(parts) + "\n"


def install(project: Path) -> list[str]:
    created: list[str] = []
    for rel, content in SEED_FILES.items():
        if write_if_missing(project / rel, content):
            created.append(rel)

    if remove_default_kb_agents(project):
        created.append("removed personal-kb/AGENTS.md")

    raw_dir = project / "personal-kb" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    agents = project / "AGENTS.md"
    existing = agents.read_text(encoding="utf-8") if agents.exists() else ""
    updated = replace_or_append_section(existing, "## personal-kb", ROOT_AGENTS_SECTION)
    if updated != existing:
        agents.write_text(updated, encoding="utf-8")
        created.append("AGENTS.md")
    return created


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("Usage: python scripts/init_personal_kb.py <project-dir>", file=sys.stderr)
        return 2

    project = Path(args[0]).expanduser().resolve()
    if not project.exists():
        print(f"error: path not found: {project}", file=sys.stderr)
        return 1
    if not project.is_dir():
        print(f"error: project path must be a directory: {project}", file=sys.stderr)
        return 1

    created = install(project)
    print(f"personal-kb initialized at {project / 'personal-kb'}")
    if created:
        print(f"created/updated {len(created)} item(s)")
    else:
        print("already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
