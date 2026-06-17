"""迁移脚本：给 Qdrant 中无 status 字段的旧数据补上 status: active。

增量索引功能需要所有 chunks 都有 status 字段。对于在该功能上线前入库的旧数据，
其 payload 中没有 status 字段。虽然 Qdrant 的 must_not + MatchValue 对缺失字段
不会排除（即旧数据仍可正常召回），但为了一致性，建议运行此脚本统一补上。

用法：
    python scripts/migrate_status_field.py [--batch-size 500] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))


logger = logging.getLogger(__name__)


def migrate(batch_size: int = 500, dry_run: bool = False) -> dict:
    """给无 status 字段的 chunks 补上 status: active。"""
    from fireagent.utils.config import get_config
    from fireagent.vectorstore import FireAgentQdrantClient

    cfg = get_config()
    client = FireAgentQdrantClient(config=cfg)
    qdrant = client.client
    collection = cfg.qdrant.collection

    # 统计
    total_scanned = 0
    needs_migration = 0
    migrated = 0

    offset = None
    batch_ids: list = []

    logger.info("开始扫描 collection=%s ...", collection)

    while True:
        result = qdrant.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result

        for point in points:
            total_scanned += 1
            payload = point.payload or {}
            if "status" not in payload:
                needs_migration += 1
                batch_ids.append(point.id)

        # 批量更新
        if batch_ids and not dry_run:
            qdrant.set_payload(
                collection_name=collection,
                payload={"status": "active"},
                points=batch_ids,
            )
            migrated += len(batch_ids)
            logger.info("已迁移 %d 个 points (累计)", migrated)
        elif batch_ids and dry_run:
            logger.info("[dry-run] 将迁移 %d 个 points", len(batch_ids))

        batch_ids = []

        if next_offset is None:
            break
        offset = next_offset

    summary = {
        "total_scanned": total_scanned,
        "needs_migration": needs_migration,
        "migrated": migrated,
        "dry_run": dry_run,
    }
    logger.info("迁移完成: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="给旧数据补 status: active 字段。")
    parser.add_argument("--batch-size", type=int, default=500, help="每批处理的 points 数量。")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入。")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = migrate(batch_size=args.batch_size, dry_run=args.dry_run)
    print(f"\n扫描: {result['total_scanned']} | 需迁移: {result['needs_migration']} | 已迁移: {result['migrated']}")
    if result["dry_run"]:
        print("(dry-run 模式，未实际写入)")


if __name__ == "__main__":
    main()
