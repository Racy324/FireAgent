"""FireAgent Prompt 模板包。"""

from fireagent.prompts.loader import (
    PromptTemplate,
    PromptTemplateError,
    PromptTemplateLoader,
    extract_template_variables,
    get_prompt_loader,
)

__all__ = [
    "PromptTemplate",
    "PromptTemplateError",
    "PromptTemplateLoader",
    "extract_template_variables",
    "get_prompt_loader",
]

