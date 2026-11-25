# Pha 2 — Dev Validation trên Batch nhỏ (Tách bạch Độc lập → Cộng dồn)

- **Mục tiêu:** Chứng minh minh bạch từng cải tiến khi đứng **độc lập** so với Paper Baseline `B0`, sau đó kiểm tra **cộng dồn từng nấc 1 thay đổi** trước khi chạy 100% dataset ở Pha 3.
- **Nguyên tắc vàng:** **CẤM tune trên `FSC-147 test` (1,190 ảnh).** Chỉ chạy trên `FSC-147 val` (`n=58`, stratified 2 ảnh/class × 29 classes, `seed=42`).
- **Trạng thái:** `DONE` *(Đã chốt Go/No-Go: giữ **`CLIP Fix + SDES`** trên **`paper`**, giữ `MD-FCE g=0` kiểm chứng thêm trên tế bào `VGG`/`MBM`, loại **`le1`** và **`SMDF`**)*

---

## Kết quả Bước 2A — Độc lập từng thành phần trên `B0` (`FSC-147 val`, `n=58`, `seed=42`)

| Cấu hình | MAE ↓ | RMSE ↓ | `s/img` | Cơ chế nổi bật (`proposal_diagnostics`) |
| :--- | :---: | :---: | :---: | :--- |
| **`B0`** (`paper`, `raw`) | 19.5172 | 63.7565 | 10.334s | *(mốc Paper Baseline; các run `raw` cho `clip_contrast=-0.1502`)* |
| **`I1: + CLIP Fix`** (`MaskCLIP + No-Crop`) | **16.6552** | **61.9828** | 10.482s | **`clip_fg_bg_contrast`: `-0.1502 → +0.2081`**; Thắng `24` / Hòa `17` / Thua `17` |
| `B0 + le1` (`0 < c <= 1.5`) | 18.5517 | 63.8533 | 13.529s | `stage2_pre`: `16.24 → 35.12` (+31% thời gian khi chưa SDES) |
| **`I2: + SDES`** | **19.2586** | **63.4316** | **9.051s** | `stage2`: `16.24→8.41`; `dup_rate_iou50`: `31%→15.5%`; `scale_spread`: `1.19→1.40` |
| `I3: + MD-FCE` (`gamma=1.0`) | 19.7414 | 63.9236 | 10.369s | `stage3_fg_ratio=0.6508`; `fce_centroid_sep`: `0.379→0.4134` |
| `I4: + SMDF` (`priors=0.20/0.25/0.55`) | 20.0345 | 63.1427 | 10.374s | `smdf_active=36/58`; cứu outlier nhưng phá nhiều ảnh thường (`948.jpg`, `7155.jpg`) |

---

## Kết quả Bước 2B — Cộng dồn từng nấc 1 thay đổi (`FSC-147 val`, `n=58`, `seed=42`)

### Nhánh 1 — Chuẩn `paper` (ĐÚNG THEO BÀI BÁO — Nhánh thắng chính)
| Nấc | Cấu hình | MAE ↓ | RMSE ↓ | `s/img` | Ghi chú đối chiếu từng ảnh |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `B0` | Paper Baseline (`paper`, `raw`) | 19.5172 | 63.7565 | 10.334s | Mốc gốc |
| `I1` | `+ CLIP Fix` (`MaskCLIP + No-Crop`) | 16.6552 | 61.9828 | 10.482s | `-2.86 MAE`; ca `1902.jpg`: `40 → 91 / GT=94` |
| **`P1`** | **`+ SDES`** | **15.0690** | **61.2122** | **8.915s** | **Tốt nhất:** `-4.45 MAE (-22.8%)`, nhanh `-13.7%`, `dup_rate: 37.9%→20.7%` |
| `P2` | `+ MD-FCE (gamma=0.0)` | 15.1207 | 61.2315 | 8.938s | Hòa `53/58` ảnh vs `P1` (Thắng `3` / Thua `2`); `centroid_sep: 0.397→0.401` |
| `P3` | `+ SMDF (0.05/0.10/0.85)` | 17.6379 | 62.3487 | 8.945s | Thua `20/58` ảnh (`5748` `62→18`, `948` `23→1`, `7155` `23→2`) do `KDE` thấp |

### Nhánh 2 — Thử nghiệm với `le1` (Kết luận: loại `le1`)
- `le1 + CLIP Fix`: MAE `18.9483` | RMSE `63.5882` | `13.786s` *(kém `paper + CLIP Fix` tới `+2.29 MAE` do hộp mảnh nhiễu tràn vào Stage 2: `36.43` hộp)*.
- `+ SDES` (`18.1379`, `9.803s`, `dup_rate=3.45%`) → `+ MD-FCE g=0` (`18.0000`) → `+ SMDF 0.05/0.10/0.85` (`16.9483`, `63.0868`). Tuy cộng hưởng đều trên nền `le1`, toàn chuỗi vẫn thua mốc **`15.0690`** của `paper + CLIP Fix + SDES`.

---

## Quyết định Cổng Go / No-Go sau Pha 2
- [x] **GIỮ LÀM LÕI (`CountZES+ Core`):** **`CLIP Fix (MaskCLIP + No-Crop)` + `SDES`** trên **`single_instance_mode=paper`**.
- [x] **GIỮ ĐỂ KIỂM CHỨNG THÊM TRÊN TẾ BÀO (Pha 3):** **`MD-FCE (gamma=0.0)`** trên `VGG` (200 ảnh) & `MBM` (44 ảnh) — nơi vật thể tròn chiếm ưu thế và nền 4 góc hộp nhiều hơn `FSC-147`.
- [x] **LOẠI BỎ `le1`:** Gây nhiễu hộp nhỏ khi `CLIP Fix` đã làm sạch Stage 1 (`18.95` vs `16.66` MAE).
- [x] **LOẠI BỎ `SMDF`:** Kể cả khi hạ tiên nghiệm xuống `0.05/0.10/0.85`, trọng số `R_kde` vẫn đánh sập dự đoán 3-shot đúng khi `KDE` bị tụt nhỏ (`948.jpg`, `5748.jpg`, `7155.jpg`). Giữ cờ `--smdf` trong code cho mục đích tái lập kết quả âm tính trong báo cáo, **không** bật trong cấu hình `CountZES+` chính thức.
