# Pha 2 — Dev Validation trên Batch nhỏ

- **Mục tiêu:** Chứng minh từng phương pháp thực sự cải thiện MAE / RMSE / tốc độ và tinh chỉnh ngưỡng **trước khi** đốt giờ GPU cho toàn bộ dataset.
- **Nguyên tắc vàng:** **CẤM tune ngưỡng trên `FSC-147 test` (1,190 ảnh)** để tránh rò rỉ tập test. Chỉ dùng `MBM` (44 ảnh) và `FSC-147 val` (subset `n=58`, stratified 2 ảnh/class × 29 classes).
- **Trạng thái:** `PENDING` (chờ Pha 1)

---

## Ma trận chạy thử nhanh (Paired — cùng `sample_seed=42`)

Mỗi dòng dưới mất **~12 phút (`MBM`)** hoặc **~16 phút (`FSC-147 val n=58`)**:

- [ ] **2.1. Baseline chuẩn hóa trên tập Dev**
  - `B0`: `raw` (gốc) trên `MBM` (44) và `FSC-147 val` (`n=58`).
  - `B1`: `+ maskclip + clip_no_crop` trên `MBM` (44) và `FSC-147 val` (`n=58`).
- [ ] **2.2. Kiểm chứng độc lập từng đề xuất (cộng dồn trên `B1`)**
  - `D1` (`B1 + --sdes`): xem `stage_samples` xem 3 exemplar có hết dính vào 1 vật thể không, kiểm tra `s/img`.
  - `D2` (`B1 + --md_fce`): xem hiệu quả khi ảnh có nền phức tạp (chọn tham số `alpha` tốt nhất trên `val`/`MBM`).
  - `D3` (`B1 + --smdf`): xem có giảm các case nổ count / over-count nặng không.
- [ ] **2.3. Kiểm chứng tổ hợp (`B1 + SDES + MD-FCE + SMDF`)**
  - Chạy tổ hợp đầy đủ trên `MBM` (44) và `FSC-147 val` (`n=58`), kiểm tra `predictions.csv` từng ảnh xem cái nào tốt lên / cái nào tệ đi.

---

## Lệnh mẫu Dev trên `FSC-147 val` (`n=58`)

```bash
# Baseline + MaskCLIP trên val (n=58)
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --clip_sim_mode maskclip --clip_no_crop --stage_samples 10 --visualize \
  --outdir dev_fscval58_maskclip

# Full proposals trên cùng 58 ảnh val
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --clip_sim_mode maskclip --clip_no_crop --sdes --md_fce --smdf \
  --stage_samples 10 --visualize \
  --outdir dev_fscval58_full
```

---

## Cổng quyết định (Go / No-Go Gate trước Pha 3)
- **GIỮ** đề xuất nếu cải thiện MAE hoặc RMSE (hoặc giữ nguyên độ chính xác nhưng giảm đáng kể thời gian chạy) trên cả 2 domain (`FSC-147 val` và `MBM`) mà không gây outlier mới.
- **ĐIỀU CHỈNH / LOẠI BỎ** ngay trong pha này nếu một đề xuất làm tụt metric đều cả 2 tập, không tốn GPU chạy full ở Pha 3.
- **Đóng băng toàn bộ hyperparameter** sau khi xong Pha 2.
