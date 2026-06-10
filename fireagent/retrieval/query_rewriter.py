"""查询改写模块。"""

from __future__ import annotations

import re

from fireagent.retrieval.schema import QueryRewriteResult
from fireagent.utils.config import FireAgentConfig, get_config


FIRE_SYNONYMS: dict[str, list[str]] = {
    "火灾": ["燃烧", "起火", "火情"],
    "烟气": ["烟雾", "有毒烟气", "烟气蔓延"],
    "烟雾": ["烟气", "火灾烟雾"],
    "疏散": ["逃生", "人员疏散", "应急疏散"],
    "逃生": ["疏散", "应急疏散"],
    "风险": ["风险评估", "安全风险"],
    "预警": ["监测", "检测", "早期预警"],
    "检测": ["识别", "监测", "目标检测"],
    "森林": ["林火", "森林火灾"],
    "隧道": ["公路隧道", "地铁隧道", "隧道火灾"],
    "排烟": ["烟气控制", "通风控制"],
    "标准": ["规范", "法规", "技术标准"],
}


class QueryRewriter:
    """规则化查询改写器。

    MVP 阶段先使用可解释的规则改写；后续 LangGraph 节点可以替换为 LLM 改写，
    但输出仍保持 QueryRewriteResult 结构不变。
    """

    def __init__(self, max_expanded_queries: int | None = None, config: FireAgentConfig | None = None) -> None:
        self.config = config or get_config()
        self.max_expanded_queries = max_expanded_queries or self.config.retrieval.max_rewrite_queries

    def rewrite(self, query: str) -> QueryRewriteResult:
        """将用户问题改写为主查询和若干扩展查询。"""
        original_query = query.strip()
        main_query = self._normalize_query(original_query)
        expanded_queries = self._expand_query(main_query)
        return QueryRewriteResult(
            original_query=original_query,
            main_query=main_query,
            expanded_queries=expanded_queries,
            metadata={"strategy": "rule_based_fire_domain"},
        )

    def _normalize_query(self, query: str) -> str:
        """清理口语化前后缀，保留核心检索意图。"""
        query = re.sub(r"\s+", " ", query).strip()
        query = re.sub(r"^(请问|请|帮我|麻烦|能否|可以)?\s*", "", query)
        query = re.sub(r"(吗|呢|么|？|\?)$", "", query).strip()
        return query or "火灾领域知识问答"

    def _expand_query(self, query: str) -> list[str]:
        """基于火灾领域同义词生成扩展查询。"""
        expansions: list[str] = []
        for term, synonyms in FIRE_SYNONYMS.items():
            if term not in query:
                continue
            for synonym in synonyms:
                expanded = query.replace(term, synonym)
                if expanded != query and expanded not in expansions:
                    expansions.append(expanded)
                if len(expansions) >= self.max_expanded_queries:
                    return expansions

        # 如果没有命中同义词，补一个火灾领域约束，降低跨领域召回噪声。
        if not expansions and "火灾" not in query:
            expansions.append(f"{query} 火灾")
        return expansions[: self.max_expanded_queries]


def rewrite_query(query: str) -> QueryRewriteResult:
    """便捷函数：执行默认查询改写。"""
    return QueryRewriter().rewrite(query)

