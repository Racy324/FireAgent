"""文档级命中指标测试。"""

from __future__ import annotations

from fireagent.evaluation.doc_hit_metrics import (
    DocHitGold,
    DocHitScore,
    _normalize_doc_name,
    extract_doc_hit_gold_from_metadata,
    score_doc_hit,
)
from fireagent.evaluation.retrieval_metrics import RetrievedChunk


def _chunk(rank: int, doc_id: str = "", paper_title: str = "", text: str = "") -> RetrievedChunk:
    return RetrievedChunk(rank=rank, chunk_id=f"c{rank}", text=text, doc_id=doc_id, paper_title=paper_title)


class TestNormalizeDocName:
    def test_pdf_path(self) -> None:
        assert _normalize_doc_name("data/raw_pdfs/隧道火灾研究_张三.pdf") == "隧道火灾研究_张三"

    def test_stem_only(self) -> None:
        assert _normalize_doc_name("隧道火灾研究") == "隧道火灾研究"

    def test_empty(self) -> None:
        assert _normalize_doc_name("") == ""

    def test_backslash_path(self) -> None:
        assert _normalize_doc_name("data\\raw_pdfs\\test.pdf") == "test"


class TestScoreDocHit:
    def test_hit_at_k(self) -> None:
        gold = DocHitGold(source_files=["data/raw_pdfs/隧道火灾研究.pdf"])
        chunks = [_chunk(1, doc_id="doc1", paper_title="隧道火灾研究综述", text="")]
        score = score_doc_hit(gold, chunks, ks=(1, 5))
        assert score.gold_doc_count == 1
        assert score.hit_at_k["doc_hit@1"] == 1.0
        assert score.recall_at_k["doc_recall@1"] == 1.0

    def test_miss(self) -> None:
        gold = DocHitGold(source_files=["data/raw_pdfs/目标论文.pdf"])
        chunks = [_chunk(1, doc_id="doc1", paper_title="其他论文", text="")]
        score = score_doc_hit(gold, chunks, ks=(5,))
        assert score.hit_at_k["doc_hit@5"] == 0.0
        assert score.recall_at_k["doc_recall@5"] == 0.0

    def test_multi_doc_recall(self) -> None:
        gold = DocHitGold(source_documents=["论文A", "论文B"])
        chunks = [
            _chunk(1, paper_title="论文A研究"),
            _chunk(2, paper_title="无关"),
            _chunk(3, paper_title="论文B分析"),
        ]
        score = score_doc_hit(gold, chunks, ks=(3,))
        assert score.hit_at_k["doc_hit@3"] == 1.0
        assert score.recall_at_k["doc_recall@3"] == 1.0

    def test_partial_recall(self) -> None:
        gold = DocHitGold(source_documents=["论文A", "论文B", "论文C"])
        chunks = [_chunk(1, paper_title="论文A")]
        score = score_doc_hit(gold, chunks, ks=(1,))
        assert score.hit_at_k["doc_hit@1"] == 1.0
        assert score.recall_at_k["doc_recall@1"] == round(1 / 3, 4)

    def test_unique_doc_count(self) -> None:
        gold = DocHitGold(source_documents=["论文A"])
        chunks = [
            _chunk(1, doc_id="d1", paper_title="论文A"),
            _chunk(2, doc_id="d1", paper_title="论文A"),
            _chunk(3, doc_id="d2", paper_title="其他"),
        ]
        score = score_doc_hit(gold, chunks, ks=(3,))
        assert score.unique_doc_count_at_k["unique_doc_count@3"] == 2

    def test_empty_gold(self) -> None:
        gold = DocHitGold()
        chunks = [_chunk(1)]
        score = score_doc_hit(gold, chunks, ks=(5,))
        assert score.gold_doc_count == 0
        assert score.hit_at_k["doc_hit@5"] is None

    def test_to_dict(self) -> None:
        gold = DocHitGold(source_documents=["论文A"])
        score = score_doc_hit(gold, [_chunk(1, paper_title="论文A")], ks=(1,))
        d = score.to_dict()
        assert "hit_at_k" in d
        assert "recall_at_k" in d
        assert "unique_doc_count_at_k" in d


class TestExtractDocHitGold:
    def test_from_source_files(self) -> None:
        gold = extract_doc_hit_gold_from_metadata({"source_files": ["data/raw_pdfs/a.pdf"]})
        assert gold.gold_doc_names == ["a"]

    def test_from_source_documents(self) -> None:
        gold = extract_doc_hit_gold_from_metadata({"source_documents": ["论文B"]})
        assert gold.gold_doc_names == ["论文b"]

    def test_empty_metadata(self) -> None:
        gold = extract_doc_hit_gold_from_metadata({})
        assert gold.gold_doc_names == []
