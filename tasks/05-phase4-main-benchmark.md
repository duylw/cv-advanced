# Pha 4 — Main Benchmark đa miền (3 Domain)

- **Mục tiêu:** Đánh giá sức mạnh tổng quát (generalization) của **CountZES (Reproduced Baseline)** vs **CountZES + Ours (Full)** trên đầy đủ các domain của paper (tuyệt đối không tune lại tham số ở pha này).
- **Trạng thái:** `PENDING` (chờ Pha 3)

---

## 3 Domain chính

| Domain | Dataset (Split) | Số ảnh | Script | Trạng thái |
| :--- | :--- | :---: | :--- | :--- |
| **1. Generic / Natural** | `FSC-147` (`test`) | 1,190 | `countzes_fsc147.py` | ☐ Baseline / ☐ Ours |
| **2. Aerial / Vehicle** | `CARPK` (`test`) | 459 | `countzes_carpk.py` | ☐ Baseline / ☐ Ours |
| **3. Microscopy / Cell** | `VGG` + `MBM` | 200 + 44 | `countzes_vgg.py`, `countzes_mbm.py` | ☐ Baseline / ☐ Ours *(kế thừa từ Pha 3)* |
| *(Tùy chọn)* Dense/Human | `PerSense-D` | — | `countzes_persense_D.py` | ☐ Nếu kịp tải dataset & checkpoint |

---

## Checklist thực thi

- [ ] **4.1. Hoàn tất `CARPK test` (459 ảnh, ~2.1h/run)**
  - Chạy `Baseline (raw)` và `Ours (Full)` trên cùng `countzes_carpk.py` (`--fsoc DSALVANet`, `--stage_samples 15`).
  - Kiểm tra lại mapping checkpoint `DSALVANet` / `CounTR` cho CARPK theo `countzes_carpk.py`.
- [ ] **4.2. Hoàn tất `FSC-147 test` (1,190 ảnh, ~5.5h/run)**
  - Chạy trong `tmux` trên EC2 (`--stage_samples 20 --visualize` để lấy minh họa đại diện 29 classes).
  - Kiểm tra `Succeeded=1190/1190 | Failed=0`.
- [ ] **4.3. *(Tùy chọn)* Đối chiếu với `--fsoc countr`**
  - Nếu có sẵn `CounTR/output_allnew_dir/FSC147.pth`, chạy thêm `Ours` với `--fsoc countr` trên `MBM` hoặc `FSC-147 test` để chứng minh cải tiến **độc lập với backbone counter**.
- [ ] **4.4. Kiểm tra tính nhất quán số liệu**
  - Đảm bảo bảng kết quả tách rõ 3 cột: **Paper Reported** | **Reproduced Baseline** | **Ours (Full)** — kèm delta MAE, RMSE và thời gian `s/img`.
