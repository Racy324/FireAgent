"""MinerU 增强解析适配器测试。"""

from __future__ import annotations

import json

from fireagent.ingestion.mineru_parser import MinerUPDFParser
from fireagent.ingestion.section_splitter import SectionSplitter


def test_mineru_parser_loads_content_list_v2_without_running_cli(tmp_path) -> None:
    """已有 MinerU 输出时，应直接读取 content_list_v2 并映射为 ParsedDocument。"""
    pdf_path = tmp_path / "fire-paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "mineru"
    job_dir = output_dir / "fire-paper"
    job_dir.mkdir(parents=True)
    content_file = job_dir / "fire-paper_content_list_v2.json"
    content_file.write_text(
        json.dumps(
            [
                [
                    {
                        "type": "title",
                        "page_idx": 0,
                        "text_level": 1,
                        "title_content": [{"type": "text", "text": "1 引言"}],
                    },
                    {
                        "type": "text",
                        "page_idx": 0,
                        "paragraph_content": [{"type": "text", "text": "火灾烟气控制研究内容。"}],
                    },
                    {
                        "type": "table",
                        "page_idx": 0,
                        "table_caption": [{"type": "text", "text": "表1 温度参数"}],
                        "table_body": (
                            "<table><tr><th>参数</th><th>值</th></tr>"
                            "<tr><td>温度</td><td>600</td></tr></table>"
                        ),
                    },
                ],
                [
                    {
                        "type": "title",
                        "page_idx": 1,
                        "text_level": 2,
                        "title_content": [{"type": "text", "text": "1.1 实验设置"}],
                    },
                    {
                        "type": "text",
                        "page_idx": 1,
                        "text": "实验记录了不同通风条件下的烟气蔓延。",
                    },
                    {
                        "type": "image",
                        "page_idx": 1,
                        "image_caption": [{"type": "text", "text": "图1 烟气蔓延路径"}],
                    },
                ],
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    parser = MinerUPDFParser(output_dir=output_dir, reuse_output=True, max_pages=None)
    document = parser.parse(pdf_path)

    assert document.metadata["parser"] == "mineru"
    assert document.metadata["parser_variant"] == "content_list_v2"
    assert len(document.pages) == 2
    assert "火灾烟气控制研究内容" in document.pages[0].text
    assert document.pages[0].metadata["mineru_blocks"][0]["text_level"] == 1
    assert len(document.tables) == 1
    assert document.tables[0].rows == [["参数", "值"], ["温度", "600"]]
    assert len(document.figure_captions) == 1
    assert document.figure_captions[0].text == "图1 烟气蔓延路径"


def test_section_splitter_uses_mineru_title_levels(tmp_path) -> None:
    """章节切分应优先使用 MinerU 的 title/text_level。"""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "mineru"
    job_dir = output_dir / "paper"
    job_dir.mkdir(parents=True)
    (job_dir / "paper_content_list_v2.json").write_text(
        json.dumps(
            [
                [
                    {"type": "title", "page_idx": 0, "text_level": 1, "text": "摘要"},
                    {"type": "text", "page_idx": 0, "text": "本文研究火灾探测。"},
                    {"type": "title", "page_idx": 0, "text_level": 1, "text": "1 方法"},
                    {"type": "text", "page_idx": 0, "text": "方法部分描述传感器布置。"},
                    {"type": "title", "page_idx": 0, "text_level": 2, "text": "1.1 数据采集"},
                    {"type": "text", "page_idx": 0, "text": "采集温度和烟雾浓度。"},
                ]
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    document = MinerUPDFParser(output_dir=output_dir).parse(pdf_path)
    sections = SectionSplitter().split(document)

    assert [section.title for section in sections] == ["摘要", "1 方法", "1.1 数据采集"]
    assert sections[0].chunk_type == "abstract"
    assert sections[-1].path == ["1 方法", "1.1 数据采集"]
    assert "烟雾浓度" in sections[-1].text
