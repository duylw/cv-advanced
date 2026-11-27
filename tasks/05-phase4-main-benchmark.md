# Pha 4 — Main Benchmark (2 Track đang chốt)

- **Mục tiêu:** Đánh giá sức mạnh **Baseline vs Ours (Full)** trên 2 track mục tiêu chính (tuyệt đối không tune lại tham số ở pha này).
- **Bộ nền cố định (chốt `2026-11-26`):** Cả `Baseline` lẫn `Ours` dùng chung bộ mặc định GitHub gốc: `--box_threshold 0.20 --text_threshold 0.15 --sses_start_percentile 80 --github_sses_inversion --single_instance_mode paper` *(khi `Ours` bật `CLIP Fix`, luôn đi chung bộ `--clip_sim_mode maskclip --clip_no_crop --fix_sses_inversion` + các module thắng cuộc từ Pha 3)*.
- **Trạng thái:** `DONE` *(Số liệu Benchmark chính lấy trực tiếp trên cùng bộ nền GitHub gốc `2026-11-26` từ Pha 3)*

---

## Các Track đánh giá

| Track / Domain | Dataset (Split) | Số ảnh | Script & Counter (`--fsoc`) | Trạng thái |
| :--- | :--- | :---: | :--- | :--- |
| **1. Generic In-domain** | `FSC-147` (`test`) | 1,190 | `countzes_fsc147.py --fsoc DSALVANet` (`checkpoint_200.pth`) | ✅ Baseline (`16.7000/90.6302`) / ✅ `+ SDES` (**`16.3496/87.8786`**) / `CLIP Fix + SDES` (`17.6168/110.1464`) |
| **2. Medical / Cell Cross-domain** | `MBM` | 44 | `countzes_mbm.py --fsoc countr --countr_ckpt ./CounTR/output_allnew_dir/FSC147.pth` | ✅ Baseline (**`19.5864/37.1069`**) / ✅ `+ SDES` (`24.4593/39.2429`) |
| *(Chỉ giữ Baseline / Bỏ chạy Ours)* | `FSC-147` (`test`) Cross-domain | 1,190 | `countzes_fsc147.py --fsoc countr` (`CARPK.pth`) | ✅ Baseline (`21.8700/112.2941`) / `DROPPED` |
| *(Tạm gác)* Aerial / Cell / Dense | `CARPK`, `VGG`, `PerSense-D` | — | *Hoãn khỏi phạm vi đợt này* | `DEFERRED` |

---

## Checklist thực thi

- [x] **4.1. Hoàn tất `FSC-147 test` In-domain (`DSALVANet`, 1,190 ảnh)**
  - Đã có đủ Baseline (`16.7000 / 90.6302`) và các biến thể (`+ SDES` đạt **`16.3496 / 87.8786`**, `10.152s/img`; `Succeeded=1190/1190 | Failed=0`).
- [x] **4.2. Hoàn tất `MBM` (`countr + FSC147.pth`, 44 ảnh)**
  - Đã có đủ Baseline (`19.5864 / 37.1069`) và cả 4 biến thể (`+ SDES`, `+ CLIP Fix`, `+ CLIP Fix + SDES`, `+ CLIP Fix + SDES + MD-FCE`; `Succeeded=44/44 | Failed=0`).
- [ ] **4.3. Kiểm tra tính nhất quán số liệu**
  - Đảm bảo bảng kết quả tách rõ 3 cột: **Paper Reported** | **GitHub Baseline** | **Ours** — kèm delta MAE, RMSE và thời gian `s/img` (chờ kéo local `ablation_fsc147_clipfix_sdes`).
