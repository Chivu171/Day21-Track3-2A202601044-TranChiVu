# Lab 21 — Trạng thái thực thi

## Đã hoàn thành (NB1 — CPU)

### Artifacts tạo ra:
1. ✅ `results/mask_proof.json` — mask đúng (answer_is_supervised=true, question_is_masked=true, supervised_fraction=0.3936)
2. ✅ `results/template_check.json` — template Qwen3.5 giữ `<think>` (verdict: "reasoning preserved")
3. ✅ `results/token_stats.json` — p95=98 tokens, suggested_max_length=256
4. ✅ `data/split/train.jsonl` — 225 mẫu (seed 42)
5. ✅ `data/split/val.jsonl` — 25 mẫu (seed 42)
6. ✅ `submission/REPORT.md` — đã điền phần NB1 với số liệu thực tế
7. ✅ `submission/REFLECTION.md` — đã điền 5 câu phản tư

### Tests:
- ✅ 115 passed, 3 skipped (test suite chạy trên CPU)
- ✅ `make smoke` — imports, tier resolution, data files, unit tests đều xanh

## Cần GPU (Colab T4 recommended)

Các artifact sau yêu cầu GPU để train/scoring:

| Artifact | Notebook | Thời gian T4 | Ghi chú |
|---|---|---|---|
| `results/baselines_frozen.json` | NB2 | ~17-23 phút | Đo 3 baseline TRƯỚC khi train |
| `adapters/correct/` + `runs.csv` | NB3 | ~15-25 phút | Train cấu hình đúng |
| `adapters/{attn_only,wrong_lr,qlora}/` | NB4 | ~45-60 phút | 3 run đối chứng |
| `results/verdict.json` | NB5 | ~21 phút | Đánh giá 4 nhóm + phán quyết |
| `results/autopsy.json` | NB5 | ~21 phút | Chấm 3 contrast adapter trên target |
| `results/qualitative.json` | NB5 | ~21 phút | 5 ví dụ định tính (≥2 ca thua) |
| `results/merge_check.json` | NB6 (tuỳ chọn) | ~10 phút | Merge + hot-swap adapter |

## Hướng dẫn chạy trên Colab T4

1. Mở Colab notebook: https://colab.research.google.com/github/hieutrungdao/Day21-Track3-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb
2. Runtime → Change runtime type → **T4 GPU**
3. Chạy lần lượt các ô:
   - Ô 1: Setup (git clone + install)
   - Ô 2: NB1 (data + mask) — *đã chạy local, có thể skip*
   - Ô 3: NB2 (baselines)
   - Ô 4: NB3 (train correct)
   - Ô 5: NB4 (misconfig autopsy)
   - Ô 6: NB5 (evaluate + verdict)
4. Tải artifacts về và chạy `make verify` local

## Hoặc chạy local nếu có GPU

```bash
# Cài GPU
make setup
make smoke
make pipeline    # NB1 -> NB5
make verify
```

## Checklist nộp bài

- [x] `results/template_check.json`
- [x] `results/mask_proof.json`
- [x] `results/token_stats.json`
- [ ] `results/baselines_frozen.json` — **cần GPU**
- [ ] `results/runs.csv` — **cần GPU**
- [ ] `results/verdict.json` — **cần GPU**
- [ ] `results/autopsy.json` — **cần GPU**
- [x] `submission/REPORT.md` — đã điền phần NB1, cần bổ sung NB2-NB5 sau khi chạy GPU
- [x] `submission/REFLECTION.md`

## Lưu ý quan trọng

1. **Không sửa eval set sau khi thấy kết quả** — `verify.py` kiểm tra checksum
2. **Không làm yếu OPTIMIZED_PROMPT** — SHA hash được ghi trong `baselines_frozen.json`
3. **EPOCHS chỉnh 1 lần** — áp cho cả NB3 và NB4 để đảm bảo công bằng
4. **max_length=512** (CPU tier) / **1024** (T4 tier) — đều > p95=98, an toàn
