"""PDF 上传与增量索引 API 端点。"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import Field

from fireagent.api.schemas import APIModel
from fireagent.ingestion.index_builder import PDFIndexBuilder
from fireagent.utils.config import PROJECT_ROOT, get_config
from fireagent.vectorstore import FireAgentQdrantClient


logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploaded_pdfs"


class UploadIngestResponse(APIModel):
    """上传入库响应。"""

    status: str = Field(description="indexed | skipped | error")
    doc_id: str = ""
    filename: str = ""
    content_hash: str = ""
    chunks: int = 0
    message: str = ""
    old_chunks_deleted: int = 0


def _make_doc_id_from_filename(filename: str) -> str:
    """基于文件名生成稳定的 doc_id（不含路径和大小）。"""
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]


@router.post("/ingest/upload", response_model=UploadIngestResponse)
async def upload_and_ingest(
    file: UploadFile = File(...),
    max_pages: Optional[int] = Form(default=None),
) -> UploadIngestResponse:
    """上传 PDF 文件并增量写入知识库。

    流程：
    1. 校验文件类型和大小
    2. 读取文件内容，计算 content_hash
    3. 生成 doc_id（基于文件名）
    4. 查询 Qdrant 是否已有该 doc_id
       - 已有且 content_hash 相同 → 跳过（返回 skipped）
       - 已有且 content_hash 不同 → 软删除旧 chunks → 重新入库
       - 没有 → 直接入库
    5. 保存文件到 data/uploaded_pdfs/
    6. 走 ingestion pipeline → upsert 到 Qdrant
    """
    # 1. 校验文件类型
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 2. 读取文件内容
    file_bytes = await file.read()
    cfg = get_config()
    max_mb = getattr(cfg.ingestion, "max_upload_mb", 50)
    if len(file_bytes) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {max_mb}MB 限制")

    # 3. 计算 content_hash
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # 4. 生成 doc_id
    doc_id = _make_doc_id_from_filename(file.filename)

    # 5. 查询已有状态
    vectorstore = FireAgentQdrantClient(config=cfg, allow_hash_dense_fallback=True)
    vectorstore.create_collection(recreate=False)

    existing_hash = vectorstore.get_content_hash_by_doc_id(doc_id)

    if existing_hash == content_hash:
        return UploadIngestResponse(
            status="skipped",
            doc_id=doc_id,
            filename=file.filename,
            content_hash=content_hash,
            message="文件内容未变化，跳过索引",
        )

    # 6. 内容变化 → 软删除旧 chunks
    old_deleted = 0
    if existing_hash is not None:
        old_deleted = vectorstore.soft_delete_by_doc_id(doc_id)
        logger.info("软删除 doc_id=%s 的旧 chunks: %d 个", doc_id, old_deleted)

    # 7. 保存文件到磁盘（pipeline 需要文件路径）
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(file_bytes)

    # 8. 走 ingestion pipeline
    try:
        builder = PDFIndexBuilder.from_config(cfg, max_pages=max_pages)
        chunks = builder.build_from_pdf(save_path)

        # 注入 content_hash 和 status 到 chunk metadata
        for chunk in chunks:
            chunk.metadata["content_hash"] = content_hash
            chunk.metadata["status"] = "active"

        vectorstore.upsert_chunks(chunks)

        return UploadIngestResponse(
            status="indexed",
            doc_id=doc_id,
            filename=file.filename,
            content_hash=content_hash,
            chunks=len(chunks),
            old_chunks_deleted=old_deleted,
            message=f"成功索引 {len(chunks)} 个 chunks"
            + (f"，替换了 {old_deleted} 个旧 chunks" if old_deleted else ""),
        )
    except Exception as exc:
        logger.exception("PDF ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"索引失败: {exc}") from exc
