#!/usr/bin/env python3
"""Pipeline nạp dữ liệu UEH lên Pinecone Cloud Vector Database và truy vấn tìm kiếm."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(override=True)

from pinecone import Pinecone
from src.chunking import FixedSizeChunker
from src.embeddings import OpenAIEmbedder


def main() -> None:
    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_HOST")
    index_name = os.getenv("PINECONE_INDEX_NAME", "lab08")

    if not api_key:
        print("Lỗi: Chưa cấu hình PINECONE_API_KEY trong .env")
        return

    print("=== 1. Kết nối Pinecone Cloud Database ===")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(host=host) if host else pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Kết nối thành công! Index: {index_name} | Dimension: {stats.dimension} | Tổng vector hiện có: {stats.total_vector_count}")

    print("\n=== 2. Cắt nhỏ văn bản UEH bằng FixedSizeChunker ===")
    data_dir = Path("data/quy-dinh-dai-hoc")
    md_files = sorted(data_dir.glob("*.md"))
    chunker = FixedSizeChunker(chunk_size=500, overlap=50)

    chunks_to_embed: list[dict] = []
    
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        # Tách frontmatter đơn giản
        frontmatter = {}
        content_body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                content_body = parts[2].strip()
                for line in parts[1].splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        frontmatter[k.strip()] = v.strip().strip('"').strip("'")

        doc_chunks = chunker.chunk(content_body)
        for i, chunk_text in enumerate(doc_chunks):
            chunk_id = f"{md_file.stem}-chunk-{i}"
            chunks_to_embed.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": md_file.name,
                    "doc_id": frontmatter.get("doc_id", md_file.stem),
                    "title": frontmatter.get("title", md_file.stem),
                    "audience": frontmatter.get("audience", "student"),
                    "department": frontmatter.get("department", "general"),
                    "text": chunk_text[:1000],  # Lưu text vào metadata để khi search hiển thị được
                }
            })

    print(f"Tổng số chunks đã tạo từ {len(md_files)} tài liệu: {len(chunks_to_embed)} chunks.")

    embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL"))
    print(f"\n=== 3. Tạo Embeddings qua OpenAI ({embedder.model_name} {embedder.dimensions or 2048}-dim) ===")
    print(f"Mô hình nhúng: {embedder.model_name} | Số chiều: {embedder.dimensions or 2048}")

    def embed_with_retry(text: str, retries: int = 3) -> list[float]:
        for attempt in range(retries):
            try:
                return embedder(text)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                print(f"\n  [Lỗi tạm thời] Thử lại ({attempt + 1}/{retries})... {e}")
                time.sleep(2)
        return embedder(text)

    vectors_to_upsert = []
    for idx, item in enumerate(chunks_to_embed, start=1):
        print(f"  [{idx}/{len(chunks_to_embed)}] Đang embed: {item['id']}...", end="\r")
        vec = embed_with_retry(item["text"])
        vectors_to_upsert.append({
            "id": item["id"],
            "values": vec,
            "metadata": item["metadata"]
        })
        time.sleep(0.1)  # OpenAI xử lý rất nhanh, chỉ cần giãn cách nhẹ 100ms

    print(f"\nEmbed hoàn tất {len(vectors_to_upsert)} vector!")

    print("\n=== 4. Đẩy (Upsert) Vectors lên Pinecone Cloud Database ===")
    # Upsert theo batch 20
    batch_size = 20
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Đã đẩy batch {i // batch_size + 1} ({len(batch)} vectors) lên Pinecone.")

    time.sleep(2)
    new_stats = index.describe_index_stats()
    print(f"\n===> THÀNH CÔNG! Tổng số vector trên Pinecone hiện tại: {new_stats.total_vector_count}")

    print("\n=== 5. Thử nghiệm Truy vấn Semantic Search trên Pinecone ===")
    test_queries = [
        "Sinh viên nộp học phí qua ngân hàng nào và số tài khoản bao nhiêu?",
        "Thời gian mở cửa Thư viện UEH Smart Library tại cơ sở Nguyễn Tri Phương?",
        "Quy trình nộp đơn phúc khảo điểm thi kết thúc học phần?"
    ]

    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        q_vec = embedder(q)
        res = index.query(vector=q_vec, top_k=2, include_metadata=True)
        for rank, match in enumerate(res.matches, start=1):
            src = match.metadata.get("source", "N/A")
            txt_preview = match.metadata.get("text", "").replace("\n", " ")[:120]
            print(f"  Top {rank} (Score: {match.score:.4f}) | File: {src}")
            print(f"       Nội dung: {txt_preview}...")


if __name__ == "__main__":
    main()
