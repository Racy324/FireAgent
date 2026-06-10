"""分析已入库 chunks 的文本长度分布，辅助调优 chunk_size 参数。"""

import statistics
from collections import Counter

from fireagent.utils.config import get_config
from fireagent.vectorstore.qdrant_client import FireAgentQdrantClient


def main():
    cfg = get_config()
    vs = FireAgentQdrantClient(config=cfg)
    client = vs._create_client()
    collection = cfg.qdrant.collection

    # 取所有 chunks
    all_points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    lengths = [len(p.payload.get("text", "")) for p in all_points]
    doc_chunks: Counter = Counter()
    section_chunks: Counter = Counter()

    for p in all_points:
        payload = p.payload or {}
        doc_chunks[payload.get("doc_id", "")] += 1
        section_chunks[f'{payload.get("doc_id", "")[:8]}|{payload.get("section_title", "")}'] += 1

    # ── Chunk 长度统计 ──
    print(f"{'='*50}")
    print(f"Chunk 文本长度统计 (共 {len(lengths)} 个)")
    print(f"{'='*50}")
    print(f"平均:   {statistics.mean(lengths):>8.0f} 字")
    print(f"中位数: {statistics.median(lengths):>8.0f} 字")
    print(f"最短:   {min(lengths):>8} 字")
    print(f"最长:   {max(lengths):>8} 字")
    print(f"标准差: {statistics.stdev(lengths):>8.0f} 字")
    sorted_len = sorted(lengths)
    print(f"P10:    {sorted_len[len(sorted_len)//10]:>8} 字")
    print(f"P25:    {sorted_len[len(sorted_len)//4]:>8} 字")
    print(f"P75:    {sorted_len[3*len(sorted_len)//4]:>8} 字")
    print(f"P90:    {sorted_len[9*len(sorted_len)//10]:>8} 字")

    # ── 长度分布 ──
    print(f"\n{'='*50}")
    print("长度分布")
    print(f"{'='*50}")
    bins = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 800),
            (800, 1000), (1000, 1200), (1200, 1500), (1500, 2000), (2000, 99999)]
    for lo, hi in bins:
        count = sum(1 for l in lengths if lo <= l < hi)
        pct = count / len(lengths) * 100
        bar = "█" * int(pct / 2)
        print(f"{lo:>6}-{hi:<6}: {count:>5} ({pct:5.1f}%) {bar}")

    # ── 每篇论文的 chunk 数 ──
    print(f"\n{'='*50}")
    print(f"每篇论文 chunk 数 (共 {len(doc_chunks)} 篇)")
    print(f"{'='*50}")
    chunk_counts = list(doc_chunks.values())
    print(f"平均:   {statistics.mean(chunk_counts):>8.0f}")
    print(f"中位数: {statistics.median(chunk_counts):>8.0f}")
    print(f"最少:   {min(chunk_counts):>8}")
    print(f"最多:   {max(chunk_counts):>8}")

    # ── 按 chunk_type 统计 ──
    type_counter: Counter = Counter()
    type_lengths: dict = {}
    for p in all_points:
        payload = p.payload or {}
        ct = payload.get("chunk_type", "unknown")
        type_counter[ct] += 1
        type_lengths.setdefault(ct, []).append(len(payload.get("text", "")))

    print(f"\n{'='*50}")
    print("按 chunk_type 统计")
    print(f"{'='*50}")
    for ct, count in type_counter.most_common():
        avg_len = statistics.mean(type_lengths[ct])
        print(f"{ct:<15}: {count:>5} 个, 平均 {avg_len:.0f} 字")

    # ── Section 级别统计 ──
    section_total_texts = {}
    for p in all_points:
        payload = p.payload or {}
        section = payload.get("section_title", "unknown")
        if section not in section_total_texts:
            section_total_texts[section] = 0
        section_total_texts[section] += len(payload.get("text", ""))

    print(f"\n{'='*50}")
    print("Section 总文本长度 top 15")
    print(f"{'='*50}")
    for section, total in sorted(section_total_texts.items(), key=lambda x: -x[1])[:15]:
        chunk_count = section_chunks.get(section, "?")
        print(f"{section[:35]:<35}: {total:>6} 字, {chunk_count} chunks")


if __name__ == "__main__":
    main()
