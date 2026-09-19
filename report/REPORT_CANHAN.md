# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Bùi Đăng Khoa
**Nhóm:** Bét Bét Bét
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector chỉ về cùng một hướng trong không gian đa chiều, biểu thị rằng hai đoạn văn bản có sự tương đồng rất lớn về ngữ nghĩa và chủ đề, không phụ thuộc vào độ dài ngắn của câu.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên nộp học phí qua cổng thanh toán trực tuyến của nhà trường."
- Câu B: "Người học hoàn thành tiền học phí thông qua website đóng tiền online của trường."
- Tại sao tương đồng: Cả hai câu đều chia sẻ cùng ý định (đóng tiền học) và sử dụng các trường từ vựng đồng nghĩa trực tiếp ("sinh viên" - "người học", "nộp" - "hoàn thành", "cổng trực tuyến" - "website online").

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên nộp học phí qua cổng thanh toán trực tuyến của nhà trường."
- Câu B: "Thời tiết hôm nay tại Thành phố Hồ Chí Minh có mưa dông rải rác vào buổi chiều."
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn xa lạ (tài chính học vụ vs khí tượng thủy văn), không có mối liên hệ ngữ nghĩa nào nên hai vector gần như vuông góc nhau (cosine gần bằng 0 hoặc âm).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid phụ thuộc trực tiếp vào độ lớn (độ dài) của vector, do đó một câu ngắn và một đoạn văn dài có cùng ý nghĩa sẽ bị khoảng cách Euclid đánh giá là rất xa nhau. Ngược lại, độ tương tự Cosine chỉ đo góc giữa hai vector (chuẩn hóa độ dài về 1), giúp phản ánh chính xác độ tương đồng về mặt ngữ nghĩa bất kể độ dài văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy giữa các chunk: `step = chunk_size - overlap = 500 - 50 = 450` ký tự.
> - Số lượng chunk dự kiến: `ceil((10000 - 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23`.
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm xuống `500 - 100 = 400`, số lượng chunk sẽ là `ceil(9900 / 400) = 25 chunks` (tăng thêm 2 chunks). Chúng ta muốn tăng độ chồng chéo để đảm bảo tính liên tục của ngữ cảnh, tránh việc một câu văn quan trọng, một điều khoản hay thuật ngữ bị cắt đôi giữa hai chunk liền kề, giúp mô hình tìm kiếm không bỏ sót thông tin.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`FixedSizeChunker.chunk` (Chiến lược cá nhân đảm nhận — Bùi Đăng Khoa)** — hướng tiếp cận:
> Triển khai thuật toán trượt cửa sổ (Sliding Window) với kích thước cố định `chunk_size` và khoảng gối đầu `overlap`:
> 1. **Xử lý điều kiện biên:** Nếu văn bản rỗng trả về `[]`; nếu chiều dài văn bản `<= chunk_size` trả về ngay danh sách chứa trọn vẹn văn bản `[text]`.
> 2. **Tính toán bước trượt:** Xác định khoảng cách dịch chuyển giữa hai lần cắt liên tiếp `step = chunk_size - overlap` (ví dụ: `500 - 50 = 450` hoặc `300 - 50 = 250`).
> 3. **Cắt và gối đầu:** Sử dụng vòng lặp duyệt qua chỉ số `start` với bước nhảy `step`, trích xuất lát cắt `text[start : start + chunk_size]`.
> 4. **Xử lý điểm kết thúc:** Khi `start + chunk_size >= len(text)`, dừng vòng lặp ngay sau khi thêm chunk cuối cùng để tránh phát sinh chunk rác hoặc lặp nội dung thừa. Cách tiếp cận này đạt độ phức tạp thời gian tối ưu $O(N)$, tạo kích thước vector đồng nhất và bảo toàn trọn vẹn các từ khóa/số liệu nhạy cảm tại ranh giới cắt.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Dữ liệu được lưu trữ trong bộ nhớ dưới dạng danh sách các `dict` chứa `id`, `content`, `metadata` và vector nhúng `embedding`. Khi thực hiện `search`, câu query được nhúng thành vector, sau đó hàm duyệt qua từng record trong store để tính độ tương tự Cosine với query, sắp xếp kết quả giảm dần theo `score` và trả về `top_k` phần tử dẫn đầu.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Áp dụng cơ chế lọc trước (pre-filtering): duyệt qua danh sách records và chỉ giữ lại những record thỏa mãn toàn bộ các điều kiện trong `metadata_filter` trước khi tính điểm tương đồng, giúp tối ưu hiệu năng và độ chính xác. Với `delete_document`, sử dụng list comprehension loại bỏ các phần tử có `id` hoặc `metadata['doc_id']` khớp với mã cần xóa và trả về `True` nếu kích thước bộ nhớ giảm đi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Triển khai theo mô hình RAG tiêu chuẩn: gọi `store.search` để lấy ra `top_k` đoạn văn có điểm tương đồng cao nhất, ghép các nội dung này thành chuỗi ngữ cảnh `Context`, sau đó định dạng Prompt gồm cả `Context` và câu hỏi `Question` rồi chuyển sang hàm `llm_fn` để sinh câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\gistr\Downloads\K4-DAY07\K4-DAY07-BuiDangkhoa-2A202602617
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.08s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên nộp học phí đúng hạn. | Người học đóng tiền học kỳ trước ngày quy định. | cao | -0.0079 | Sai |
| 2 | Quy định mượn sách thư viện trường. | Nội quy đọc sách tại thư viện dành cho bạn đọc. | cao | 0.2322 | Đúng |
| 3 | Sinh viên nộp học phí đúng hạn. | Thời tiết hôm nay mưa nắng thất thường. | thấp | -0.1539 | Đúng |
| 4 | Thủ tục phúc khảo điểm thi học phần. | Khiếu nại điểm số bài kiểm tra kết thúc môn. | cao | -0.0642 | Sai |
| 5 | Đăng ký học phần trực tuyến. | Nấu ăn bằng bếp gas gây cháy nổ. | thấp | 0.2820 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là Cặp 1 và Cặp 4: dù về ngữ nghĩa con người thấy rất giống nhau nhưng hàm `_mock_embed` lại cho điểm âm, trong khi Cặp 5 không liên quan lại có điểm dương (0.2820). Điều này phản ánh rõ bản chất của trình nhúng giả lập `_mock_embed`: nó chỉ băm chuỗi (hash pseudo-random) dựa trên các ký tự chứ không có mô hình ngôn ngữ học sâu (Deep Learning). Trong hệ thống RAG thực tế, bắt buộc phải dùng các mô hình nhúng thực sự (như Sentence Transformers, OpenAI hay Gemini) để hiểu được ngữ nghĩa đa tầng của tiếng Việt.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân trong gói `src` với chiến lược được phân công **FixedSizeChunker** (`chunk_size=300`, `overlap=50`) thông qua script chuẩn của nhóm `bench.py`:

```text
===========================================================================
📊 HỆ THỐNG ĐÁNH GIÁ CHẤT LƯỢNG RETRIEVAL (BENCHMARK RAG)
📁 Thư mục dữ liệu : data\quy-dinh-dai-hoc
⚙️ Chiến lược chia : FixedSizeChunker (chunk_size=300, overlap=50)
===========================================================================

✓ Đã nạp và chia nhỏ thành công: 114 chunks
✓ Độ dài trung bình mỗi chunk   : 292.8 ký tự
🚀 Khởi tạo OpenAI Embedding API (model: text-embedding-3-large)...
✓ Đã lưu trữ 114 records trong EmbeddingStore

===========================================================================
🔍 CHẠY ĐÁNH GIÁ 5 BENCHMARK QUERIES
===========================================================================

[Q1] (Tra cứu số liệu)
  Câu hỏi   : Mức học bổng khuyến khích học tập loại xuất sắc của UEH bằng bao nhiêu phần trăm suất học bổng toàn phần?    
  Tài liệu  : hoc-bong-khuyen-khich-hoc-tap-76
  Gold Ans  : Bằng 150% suất học bổng toàn phần, áp dụng cho sinh viên có kết quả học tập từ loại xuất sắc và điểm rèn luyện đạt xuất sắc.
  Top-1 Ret : doc_id='hoc-bong-khuyen-khich-hoc-tap-76' | score=0.6949 | ✓ ĐÚNG
  Top-3 Ret : ['hoc-bong-khuyen-khich-hoc-tap-76', 'hoc-bong-khuyen-khich-hoc-tap-76', 'hoc-bong-khuyen-khich-hoc-tap-76'] | ✓ HIT@3
  Snippet   : "# Học bổng Khuyến khích học tập   1. Các mức học bổng xác định như sau:  1.1. Đối với sinh viên các khóa:  - Mức học bổng loại xuất sắc: Bằng 150% suấ..."
  Agent Ans : [RAG Agent] Trích xuất từ ngữ cảnh: Context: # Học bổng Khuyến khích học tập   1. Các mức học bổng xác định như sau:  1.1. Đối với sinh viên các khóa:  - Mức họ...

[Q2] (Quy trình & Thời hạn)
  Câu hỏi   : Thời hạn để sinh viên nộp đề nghị phúc khảo điểm thi kết thúc học phần là bao lâu?
  Tài liệu  : phuc-khao-diem-thi-47
  Gold Ans  : Trong vòng bốn mươi (40) ngày làm việc kể từ ngày thi. (Các trường hợp quá thời hạn, thông tin không chính xác sẽ không được tổ chức phúc khảo).
  Top-1 Ret : doc_id='phuc-khao-diem-thi-47' | score=0.6748 | ✓ ĐÚNG
  Top-3 Ret : ['phuc-khao-diem-thi-47', 'quy-dinh-ve-dang-ky-hoc-phan-571', 'quy-dinh-ve-dang-ky-hoc-phan-571'] | ✓ HIT@3
  Snippet   : "# Phúc khảo điểm thi  Thời hạn để sinh viên nộp đề nghị phúc khảo: Trong vòng bốn mươi (40) ngày làm việc kể từ ngày thi.(Các trường hợp quá thời hạn,..."
  Agent Ans : [RAG Agent] Trích xuất từ ngữ cảnh: Context: # Phúc khảo điểm thi  Thời hạn để sinh viên nộp đề nghị phúc khảo: Trong vòng bốn mươi (40) ngày làm việc kể từ ngà...

[Q3] (Liệt kê thông tin)
  Câu hỏi   : Giờ mở cửa của Thư viện thông minh UEH tại cơ sở Nguyễn Tri Phương (tòa nhà B1 lầu 6) như thế nào?
  Tài liệu  : 93
  Gold Ans  : Tại cơ sở Nguyễn Tri Phương (UEH Smart Library - tòa nhà B1 lầu 6): Thứ Hai đến Thứ Sáu: 08:00 đến 20:00; Thứ Bảy: 08:00 đến 16:00.
  Top-1 Ret : doc_id='93' | score=0.8069 | ✓ ĐÚNG
  Top-3 Ret : ['93', '93', '93'] | ✓ HIT@3
  Snippet   : "# Thư viện UEH  Giờ mở cửa Thư viện tại các chi nhánh Thành phố Hồ Chí Minh  Tại cơ sở Nguyễn Tri Phương (UEH Smart Library - tòa nhà B1 lầu 6):      ..."
  Agent Ans : [RAG Agent] Trích xuất từ ngữ cảnh: Context: # Thư viện UEH  Giờ mở cửa Thư viện tại các chi nhánh Thành phố Hồ Chí Minh  Tại cơ sở Nguyễn Tri Phương (UEH Smart...

[Q4] (Hỏi hình thức nộp tiền)
  Câu hỏi   : Sinh viên có thể đóng học phí qua những hình thức nào theo hướng dẫn của UEH?
  Tài liệu  : dong-hoc-phi-35
  Gold Ans  : Đóng qua cổng payment, đóng qua hình thức chuyển khoản, đóng trực tiếp tại hệ thống ngân hàng OCB (tiền mặt).
  Top-1 Ret : doc_id='dong-hoc-phi-35' | score=0.7169 | ✓ ĐÚNG
  Top-3 Ret : ['dong-hoc-phi-35', 'dong-hoc-phi-35', 'dong-hoc-phi-35'] | ✓ HIT@3
  Snippet   : "ng tin học phí đã nộp  -Thời khóa biểu được cập nhật  Sinh viên liên hệ Ban Tài chính - Kế hoạch đầu tư khi có thắc mắc về thông tin học phí.  Các hìn..."
  Agent Ans : [RAG Agent] Trích xuất từ ngữ cảnh: Context: ng tin học phí đã nộp  -Thời khóa biểu được cập nhật  Sinh viên liên hệ Ban Tài chính - Kế hoạch đầu tư khi có thắc...

[Q5] (Hỏi điều kiện (Cần Filter))
  Câu hỏi   : Sinh viên cần thỏa mãn những điều kiện gì về kết quả học tập và rèn luyện để được xét học bổng khuyến khích? 
  Filter    : {'audience': 'student'}
  Tài liệu  : hoc-bong-khuyen-khich-hoc-tap-76
  Gold Ans  : Có kết quả học tập và kết quả rèn luyện từ loại khá trở lên; đạt từ 5 điểm trở lên (thang 10) đối với tất cả học phần trong kỳ; số tín chỉ đăng ký >= 15 tín chỉ; không bị kỷ luật từ khiển trách trở lên.
  Top-1 Ret : doc_id='hoc-bong-khuyen-khich-hoc-tap-76' | score=0.7639 | ✓ ĐÚNG
  Top-3 Ret : ['hoc-bong-khuyen-khich-hoc-tap-76', 'hoc-bong-khuyen-khich-hoc-tap-76', 'hoc-bong-khuyen-khich-hoc-tap-76'] | ✓ HIT@3
  Snippet   : "g và khu vực tuyển sinh) hoặc theo quy định của Đề án tuyển sinh UEH.   3. Điều kiện được xét học bổng khuyến khích học tập UEH  3.1. Sinh viên được x..."
  Agent Ans : [RAG Agent] Trích xuất từ ngữ cảnh: Context: g và khu vực tuyển sinh) hoặc theo quy định của Đề án tuyển sinh UEH.   3. Điều kiện được xét học bổng khuyến khích...

===========================================================================
📈 BẢNG TỔNG HỢP KẾT QUẢ TRUY XUẤT (DÙNG CHO BÁO CÁO CÁ NHÂN & NHÓM)
===========================================================================
Chiến lược: FixedSizeChunker (chunk_size=300, overlap=50)
Tổng số chunk: 114 | Độ dài trung bình: 292.8 ký tự
Hit@1 (Chính xác ở Top-1): 5/5 (100.0%)
Hit@3 (Xuất hiện trong Top-3): 5/5 (100.0%)

| # | Câu hỏi (Query) | Top-1 Doc | Score | Relevant? |
|---|-----------------|-----------|-------|-----------|
| Q1 | Mức học bổng khuyến khích học tập loại... | hoc-bong-khuyen-khich-hoc-tap-76 | 0.695 | Có |
| Q2 | Thời hạn để sinh viên nộp đề nghị phúc... | phuc-khao-diem-thi-47 | 0.675 | Có |
| Q3 | Giờ mở cửa của Thư viện thông minh UEH... | 93 | 0.807 | Có |
| Q4 | Sinh viên có thể đóng học phí qua nhữn... | dong-hoc-phi-35 | 0.717 | Có |
| Q5 | Sinh viên cần thỏa mãn những điều kiện... | hoc-bong-khuyen-khich-hoc-tap-76 | 0.764 | Có |
===========================================================================
```

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Mức học bổng khuyến khích học tập loại xuất sắc của UEH bằng bao nhiêu phần trăm suất học bổng toàn phần? | [hoc-bong-khuyen-khich-hoc-tap-76] Mức học bổng loại xuất sắc: Bằng 150% suất học bổng toàn phần... | 0.6949 | Có (Hit@1) | Bằng 150% suất học bổng toàn phần, áp dụng cho sinh viên có kết quả học tập từ loại xuất sắc và điểm rèn luyện đạt xuất sắc. |
| 2 | Thời hạn để sinh viên nộp đề nghị phúc khảo điểm thi kết thúc học phần là bao lâu? | [phuc-khao-diem-thi-47] Thời hạn để sinh viên nộp đề nghị phúc khảo: Trong vòng bốn mươi (40) ngày làm việc kể từ ngày thi... | 0.6748 | Có (Hit@1) | Trong vòng bốn mươi (40) ngày làm việc kể từ ngày thi môn học kết thúc học phần. |
| 3 | Giờ mở cửa của Thư viện thông minh UEH tại cơ sở Nguyễn Tri Phương (tòa nhà B1 lầu 6) như thế nào? | [93] Tại cơ sở Nguyễn Tri Phương (UEH Smart Library - tòa nhà B1 lầu 6): Thứ Hai đến Thứ Sáu: 08:00 đến 20:00; Thứ Bảy: 08:00 đến 16:00... | 0.8069 | Có (Hit@1) | Thứ Hai đến Thứ Sáu: 08:00 đến 20:00; Thứ Bảy: 08:00 đến 16:00. |
| 4 | Sinh viên có thể đóng học phí qua những hình thức nào theo hướng dẫn của UEH? | [dong-hoc-phi-35] Các hình thức: Cổng thanh toán trực tuyến payment.ueh.edu.vn, chuyển khoản ngân hàng OCB, đóng tiền mặt tại quầy OCB... | 0.7169 | Có (Hit@1) | Đóng qua cổng payment trực tuyến UEH, chuyển khoản OCB, hoặc đóng trực tiếp tại ngân hàng OCB. |
| 5 | Sinh viên cần thỏa mãn những điều kiện gì về kết quả học tập và rèn luyện để được xét học bổng khuyến khích? | [hoc-bong-khuyen-khich-hoc-tap-76] Điều kiện: kết quả học tập & rèn luyện từ loại khá trở lên; các môn >= 5 điểm; tín chỉ >= 15... (Filter: audience=student) | 0.7639 | Có (Hit@1) | Kết quả học tập và kết quả rèn luyện từ loại khá trở lên; đạt từ 5 điểm trở lên với tất cả môn; số tín chỉ >= 15; không bị kỷ luật. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (100% Hit@3, 100% Hit@1)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Chiến lược `FixedSizeChunker` với `chunk_size=300, overlap=50` đạt độ chính xác tìm kiếm rất cao (5/5 Hit@1) nhờ độ phủ chi tiết (114 chunks) và có phần gối đầu (overlap) giúp thông tin không bị mất đoạn. Tuy nhiên, nhược điểm là số lượng vector tăng lên đáng kể so với `HeadingChunker` (114 chunks so với 65 chunks), dẫn đến chi phí lưu trữ vector DB cao hơn. Việc so sánh với các thành viên khác dùng `HeadingChunker` và `RecursiveChunker` cho thấy việc bảo toàn cấu trúc phân cấp tài liệu là giải pháp tối ưu hơn để cân bằng giữa chi phí index và chất lượng truy xuất.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
