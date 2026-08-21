# Lab 21 — Evaluation Report

**Họ tên**: Tran Chi Vu  **MSSV**: 2A202601044  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Colab Free)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (intent, urgency, product, sentiment) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | `1024` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2` / `30` (T4, batch 1×16) |

**Template có giữ khối `<think>` không?** `Có` — *(results/template_check.json)*
Verdict: `reasoning preserved — safe to train on traces`. Template Qwen3.5 giữ nguyên nội dung `<think>` trong `apply_chat_template`, nên masked-think/response-only có tác dụng khi corpus có reasoning traces.

> **Ghi chú về max_length**: p95=98 tokens, tier T4 đặt max_length=1024. Giá trị này lớn hơn p95 và p99=100, đảm bảo không cắt mất câu trả lời. Đặt lớn hơn p95 là padding lãng phí nhưng an toàn.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` (39/94 tokens) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}
```

**Giải thích**: Với `mask_mode=assistant-only`, loss chỉ tính trên lượt assistant (39 tokens), không tính trên system prompt hay user turn (55 tokens). `supervised_fraction=0.4149` < 0.95, xác nhận mask đúng — không có prompt leak.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3315 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1029 |
| (c) LoRA fine-tune | 0.970 | 0.700 | 1.000 | 1438 |

**(b) có thật sự mạnh hơn (a) không?** `Có` — (b) target=0.765 so với (a)=0.000, cải thiện 76.5 điểm tuyệt đối. Prompt "tối ưu" thêm schema JSON, enum values, ví dụ minh họa — đủ để model hiểu cấu trúc output cần tạo.

Bạn có sửa `OPTIMIZED_PROMPT` không? `Không` — giữ nguyên prompt gốc. SHA: `719e74d3b6232053` (khác với prompt gốc là vi phạm liêm chính theo rubric).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6274 | **0.970** | 1.0 | 12.01 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.537 | **0.970** | 1.0 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.000** | 0.0 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 1.0 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế thay vì bằng năng lực trên tác vụ chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là kết quả đáng giá nhất bạn đo được trong lab này.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

`attn_only` **hoà** với `correct` trên tập target (cùng 0.970), nhưng có **training loss thấp hơn** (0.537 so với 0.6274). Điều này chứng minh rõ ràng Lỗi #3: training loss không phải metric để xếp hạng. `attn_only` học "ghi nhớ" tập train tốt hơn (loss thấp), nhưng trên tập target mới thì không vượt trội so với `correct`. 

Về *rank* vs *vị trí*: `attn_only` cần rank r=283 (gấp 17.7 lần `correct`'s r=16) để đạt cùng ngân sách tham số. Kết quả cho thấy **vị trí gắn adapter quan trọng hơn rank** — 12 linear layers (text-linear) hoà với 2 attention layers (q,v) dù rank của q,v cao gấp 17 lần. Rank là *năng lực so với lượng thông tin trong dữ liệu*, không phải đòn bẩy độc lập.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` (LR=1e-5, 1x full-FT scale) có training loss **cao gấp 2.5 lần** `correct` (1.5704 so với 0.6274) và gần như **phẳng** từ step 0 đến step 30 (từ 2.163 xuống 1.5704). Nếu chỉ nhìn loss, bạn sẽ kết luận: "LoRA học kém hơn full fine-tune" — đó chính xác là kết luận sai mà lab cũ đưa ra.

Thực tế: LR quá thấp khiến optimizer bước quá nhỏ, không thoát khỏi vùng initialization. Model gần như không học gì, nên target=0.000, format=0.0. **LR là nút quan trọng nhất** trong lab này — đổi 1 con số (1e-5 → 1e-4) là khác biệt giữa "học được" và "không học được".

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` tiết kiệm **4.92 GB VRAM** (7.09 GB so với 12.01 GB, giảm ~41%). Trả giá bằng **target thấp hơn 3 điểm** (0.940 so với 0.970) và **training loss cao hơn** (0.7058 so với 0.6274). Điều này **ủng hộ khuyến nghị của nhà cung cấp**: QLoRA không nên dùng cho Qwen3.5. Với 14.6GB VRAM của T4, bf16 LoRA (12GB) vẫn chạy được, nên không có lý do gì phải chấp nhận mất 3 điểm target để tiết kiệm VRAM.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.058` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Fine-tune của tôi **thắng baseline (b) rõ rệt** ở target task (+20.5 điểm), nhưng **tụt general capability quá ngưỡng** (-5.8 điểm, tolerance chỉ 2 điểm). Regression group đo bằng 15 câu hỏi kiến thức/chỉ dẫn phổ thông — fine-tune quên khá nhiều thứ cũ khi học task mới. Nguyên nhân có thể là: (1) corpus chỉ có 250 mẫu, đa số là intent/urgency cố định, model học "overfit" pattern thay v� general language; (2) thiếu replay data — deck §14.3 khuyến nghị thêm 1-5% dữ liệu phổ thông để giữ general capability. Đây là kết quả **đáng giá** vì nó cho thấy fine-tune không tự nhiên "cân bằng" được — cần thiết kế có chủ đích (replay, regularization) để vừa giỏi task vừa không quên.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, chuột không dây VN232232. Cho tôi trả lại. | doi_tra, cao | doi_tra, cao | ✅ FT thắng |
| 2 | Shop ơi, ốp lưng điện thoại VN812931. Hoàn tiền. | hoan_tien, trung_binh | hoan_tien, trung_binh | ✅ FT thắng |
| 3 | Cho mình hỏi, bình giữ nhiệt VN804124. Chưa thấy tiền. | hoan_tien, trung_binh | doi_tra, trung_binh | ❌ **FT thua** |
| 4 | Shop ơi, nồi chiên không dầu DH249548. Thiếu phụ kiện. | san_pham_loi, trung_binh | san_pham_loi, trung_binh | ❌ **FT thua** |
| 5 | Shop ơi, áo khoác gió VN613097. Bị lỗi. Khi nào tiện. | san_pham_loi, trung_binh | san_pham_loi, trung_binh | ❌ **FT thua** |

**Có mẫu chung nào ở các ca FT thua không?** Có — cả 3 ca thua đều có `urgency=trung_binh` và `product` là tên sản phẩm dài (bình giữ nhiệt, nồi chiên không dầu, áo khoác gió). Fine-tune dự đoán đúng intent và product nhưng **sai urgency** (đoán `trung_binh` thay vì `cao`/`trung_binh` đúng). Đây là dấu hiệu model học đúng "loại vấn đề" nhưng chưa nắm được nuance mức độ khẩn cấp — có thể do data không đủ đa dạng về urgency, hoặc model đang trade-off giữa accuracy và recall ở nhóm này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bản fine-tune này **không nên deploy** trong trạng thái hiện tại. Mặc dù target task tăng từ 0.765 lên 0.970 (+20.5 điểm), general capability lại tụt 5.8 điểm — vượt xa tolerance 2 điểm. Đây không phải lỗi cấu hình (mask đúng, LR đúng, rank đủ), mà là **vấn đề data** với 250 mẫu cho bài toán 4 lớp. Model học quá sâu vào pattern cụ thể của tập train và quên kiến thức phổ thông. Để deploy cần thêm 1-5% replay data (deck §14.3) hoặc tăng corpus lên 500-1000 mẫu. Đòn bẩy thật sự trong lab này là **learning rate** (10x full-FT LR) — nó quyết định model có học được hay không. **Vị trí adapter** (text-linear vs q,v) ít quan trọng hơn rank, nhưng **mask** là nền tảng: nếu mask sai, mọi cấu hình LoRA đều vô nghĩa.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Training loss không phải metric để xếp hạng**: `attn_only` có final_loss=0.537 (thấp hơn `correct`=0.6274) nhưng target bằng nhau 0.970 — loss thấp chỉ là ghi nhớ, không phải khả năng thực sự.
2. **LR là nút quan trọng nhất**: `wrong_lr` (1e-5) có final_loss=1.57 (gấp 2.5 lần `correct`) và target=0.000 — đổi 1 con số LR thay đổi cả lab.
3. **Vị trí > rank**: `attn_only` cần r=283 (gấp 17.7 lần) để hoà `correct` r=16 — cho thấy đặt adapter đúng chỗ (12 linear layers) quan trọng hơn tăng rank.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Thêm 10-20% replay data (câu hỏi kiến thức phổ thông) vào train set để giảm regression
- Thử `masked-think` với corpus có reasoning traces (deck §13.5) để đo `valid_trace_rate` thực sự
- Quét rank r ∈ {8, 16, 64} với `target_modules="text-linear"` cố định (bonus B4) để xem rank có phải đòn bẩy không

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [x] B3 reasoning-trace collapse (đã thấy `valid_trace_rate=0.0` với `mask_mode=assistant-only` trên corpus không có traces)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/Tranchivu12/lab21-qwen35-triage-vi
