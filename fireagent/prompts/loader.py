"""Prompt 模板加载与渲染工具。"""

from __future__ import annotations

import re
from pathlib import Path
from string import Formatter
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from fireagent.utils.config import FireAgentConfig, PROJECT_ROOT, get_config


class PromptTemplateError(RuntimeError):
    """Prompt 模板加载或渲染失败时抛出的异常。"""


class PromptTemplate(BaseModel):
    """已加载的 Prompt 模板。"""

    model_config = ConfigDict(extra="ignore")

    name: str
    path: str
    template: str
    variables: list[str] = Field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        """使用关键字参数渲染模板。"""
        missing = [variable for variable in self.variables if variable not in kwargs]
        if missing:
            raise PromptTemplateError(f"渲染模板 {self.name} 缺少变量：{', '.join(missing)}")
        safe_kwargs = {key: "" if value is None else value for key, value in kwargs.items()}
        rendered = self.template
        for variable in self.variables:
            rendered = rendered.replace("{" + variable + "}", str(safe_kwargs[variable]))
        return rendered


class PromptTemplateLoader:
    """按配置加载 FireAgent prompt 模板。"""

    def __init__(
        self,
        config: Optional[FireAgentConfig] = None,
        project_root: Optional[str | Path] = None,
    ) -> None:
        self.config = config or get_config()
        self.project_root = Path(project_root).resolve() if project_root else PROJECT_ROOT
        self._cache: dict[str, PromptTemplate] = {}

    def load(self, name: str) -> PromptTemplate:
        """加载单个 prompt 模板。"""
        if name in self._cache:
            return self._cache[name]

        prompt_path = self._path_for_name(name)
        template_text = self._read_template(prompt_path)
        prompt = PromptTemplate(
            name=name,
            path=str(prompt_path),
            template=template_text,
            variables=extract_template_variables(template_text),
        )
        self._cache[name] = prompt
        return prompt

    def render(self, name: str, **kwargs: Any) -> str:
        """加载并渲染指定 prompt。"""
        return self.load(name).render(**kwargs)

    def load_all(self) -> dict[str, PromptTemplate]:
        """加载配置中注册的全部 prompt。"""
        return {name: self.load(name) for name in self.names()}

    def names(self) -> list[str]:
        """返回配置中注册的 prompt 名称。"""
        return [
            "intent_router",
            "query_rewrite",
            "sufficiency_check",
            "answer_generation",
            "hallucination_check",
        ]

    def _path_for_name(self, name: str) -> Path:
        """根据 prompt 名称解析模板路径。"""
        if name not in self.names():
            raise PromptTemplateError(f"未知 prompt 名称：{name}")
        raw_path = getattr(self.config.prompts, name, "")
        if not raw_path:
            raise PromptTemplateError(f"prompt 未配置路径：{name}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    @staticmethod
    def _read_template(path: Path) -> str:
        """读取模板文件。"""
        if not path.exists():
            raise PromptTemplateError(f"prompt 模板文件不存在：{path}")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise PromptTemplateError(f"prompt 模板为空：{path}")
        return text


def extract_template_variables(template: str) -> list[str]:
    """提取模板中的 ``{variable}`` 变量名。

    Markdown 示例中的 JSON 花括号不应被当作变量，因此本函数会先移除 fenced
    code block，再交给 Python Formatter 解析。
    """
    cleaned = remove_fenced_code_blocks(template)
    variables: list[str] = []
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(cleaned):
        if not field_name:
            continue
        variable = field_name.split(".", 1)[0].split("[", 1)[0]
        if variable and variable not in variables:
            variables.append(variable)
    return variables


def remove_fenced_code_blocks(text: str) -> str:
    """移除 Markdown fenced code block，避免 JSON 示例干扰变量提取。"""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def get_prompt_loader() -> PromptTemplateLoader:
    """返回默认 PromptTemplateLoader。"""
    return PromptTemplateLoader()
