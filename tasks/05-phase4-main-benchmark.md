# Pha 4 — Main Benchmark (3 Track đang chốt)

- **Mục tiêu:** Đánh giá sức mạnh **Baseline vs Ours (Full)** trên 3 track mục tiêu (tuyệt đối không tune lại tham số ở pha này).
- **Bộ nền cố định (chốt `2026-11-26`):** Cả `Baseline` lẫn `Ours` dùng chung bộ mặc định GitHub gốc: `--box_threshold 0.20 --text_threshold 0.15 --sses_start_percentile 80 --github_sses_inversion --single_instance_mode paper` *(Ours bật thêm các module cải tiến thắng cuộc từ Pha 3)*.
- **Trạng thái:** `PENDING` (chờ Pha 3)

---

## Các Track đánh giá

| Track / Domain | Dataset (Split) | Số ảnh | Script & Counter (`--fsoc`) | Trạng thái |
| :--- | :--- | :---: | :--- | :--- |
| **1. Generic In-domain** | `FSC-147` (`test`) | 1,190 | `countzes_fsc147.py --fsoc DSALVANet` (`checkpoint_200.pth`) | ☐ Baseline / ☐ Ours |
| **2. Generic Cross-domain** | `FSC-147` (`test`) | 1,190 | `countzes_fsc147.py --fsoc countr --countr_ckpt ./CounTR/output_allnew_dir/CARPK.pth` | ☐ Baseline / ☐ Ours |
| **3. Medical / Cell Cross-domain** | `MBM` | 44 | `countzes_mbm.py --fsoc countr --countr_ckpt ./CounTR/output_allnew_dir/FSC147.pth` | ✅ Baseline (`19.5864/37.1069`) / ☐ Ours |
| *(Tạm gác)* Aerial / Cell / Dense | `CARPK`, `VGG`, `PerSense-D` | — | *Hoãn khỏi phạm vi đợt này* | `DEFERRED` |

---

## Checklist thực thi

- [ ] **4.1. Hoàn tất `FSC-147 test` In-domain (`DSALVANet`, 1,190 ảnh)**
  - Chạy `Baseline` và `Ours` trong `tmux` (`--stage_samples 20 --visualize`), kiểm tra `Succeeded=1190/1190 | Failed=0`.
- [ ] **4.2. Hoàn tất `FSC-147 test` Cross-domain (`countr + CARPK.pth`, 1,190 ảnh)**
  - Chạy `Baseline` và `Ours` với `--countr_ckpt ./CounTR/output_allnew_dir/CARPK.pth`, kiểm tra `Succeeded=1190/1190 | Failed=0`.
- [ ] **4.3. Hoàn tất `MBM` (`countr + FSC147.pth`, 44 ảnh)**
  - Đã có `Baseline` (`19.5864 / 37.1069`); chạy `Ours` (`--stage_samples -1 --visualize`) và kiểm tra `Succeeded=44/44 | Failed=0`.
- [ ] **4.4. Kiểm tra tính nhất quán số liệu**
  - Đảm bảo bảng kết quả tách rõ 3 cột: **Paper Reported** | **GitHub Baseline** | **Ours (Full)** — kèm delta MAE, RMSE và thời gian `s/img`.
