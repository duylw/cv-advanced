# Pha 3 — Ablation Study (100% Dataset)

- **Mục tiêu:** Tạo bảng Ablation khoa học **100% toàn bộ tập dữ liệu (không lấy mẫu con)** trên **bộ nền mặc định GitHub gốc đã chốt (`2026-11-26`)**, tập trung trên **2 track**: **`MBM` (44, `countr + FSC147.pth`)** và **`FSC-147 test` In-domain (1,190, `DSALVANet`)** *(bỏ chạy tiếp `FSC-147` Cross-domain; tạm gác `VGG`, `CARPK`)*.
- **Bộ cờ nền cố định cho mọi hàng `A0 → A3`:**
  `--box_threshold 0.20 --text_threshold 0.15 --sses_start_percentile 80 --github_sses_inversion --single_instance_mode paper --clip_sim_mode raw`
- **Quy ước `CLIP Fix` luôn đi chung:** Mỗi khi bật `CLIP Fix` (`A1`, `A2`, `A3`), luôn dùng đồng bộ **`--clip_sim_mode maskclip --clip_no_crop --fix_sses_inversion`** (không tách riêng `MaskCLIP` bỏ `No-Crop`).
- **Chiến lược tiết kiệm GPU (`gpu-server` ~9–12.9s/img):**
  1. Chạy trước trên **`MBM` (44 ảnh, ~7–9p/run)** để chốt tổ hợp thắng cuộc (`raw + 1-sim + SDES` vs `CLIP Fix [MaskCLIP+No-Crop+fix] + SDES`).
  2. Chạy trên **`FSC-147 test` In-domain (`DSALVANet`, 1,190 ảnh, ~3.5–4h/run)**: đã có `A0` (`16.7000 / 90.6302`), chạy tiếp các hàng thắng cuộc từ bước kiểm chứng.
- **Trạng thái:** `DONE` *(Đã hoàn tất 100% dataset trên cả `MBM` 44/44 và ma trận 2×2 `FSC-147 test` 1,190/1,190)*

---

## Ma trận Ablation chuẩn (Nền cố định: GitHub Default `0.20 / 0.15 / 80 / --github_sses_inversion / paper`, bỏ `le1` & `SMDF`)

> Bảng chi tiết các lượt đo (gồm `FSC-147 n=24`, `MBM 44/44`, `FSC-147 1190/1190` kèm tốc độ, Stage 2 & CLIP contrast) xem tại **[ablation-tables.md](ablation-tables.md)**.

| ID | `CLIP Fix` | `SDES` | `MD-FCE` (`g=0.0`) | `MBM` (44, `countr`) | `FSC-147` In-domain (`DSALVANet`, 1,190) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `A0` *(Baseline)* | | | | **`19.5864 / 37.1069`** *(Colab)* | `16.7000 / 90.6302` *(EC2)* |
| `A0+S` | | ✓ | | `24.4593 / 39.2429` *(EC2)* | **`16.3496 / 87.8786`** *(EC2 `1190/1190`)* |
| `A1` | ✓ | | | `27.4356 / 44.1526` *(EC2)* | `17.5866 / 113.1749` *(EC2 `1190/1190`)* |
| `A0+M` | | | ✓ | `19.7784 / 37.2569` *(Colab)* | — |
| `A2` | ✓ | ✓ | | `33.0592 / 54.6344` *(EC2)* | `17.6168 / 110.1464` *(EC2 `1190/1190`)* |
| `A3` | ✓ | ✓ | ✓ | `35.2233 / 56.5361` *(EC2)* | ☐ *(tùy chọn)* |

---

## Công việc chi tiết

- [x] **3.1. Chuẩn bị `MBM` (44) & `CounTR/FSC147.pth` trên `gpu-server`**
  - Đã tải `mbm`, `CounTR/output_allnew_dir/FSC147.pth` và pass smoke test `--max_images 1` (`Failed=0`).
- [x] **3.2. Hoàn tất Baseline `A0`**
  - `MBM` (`19.5864 / 37.1069`) và `FSC-147 test` In-domain (`16.7000 / 90.6302`) đều hoàn tất 100% (`Failed=0`). *(Đã có thêm mốc `FSC-147` Cross-domain `21.8700 / 112.2941`, không chạy thêm nhánh này.)*
- [x] **3.3. Chạy Ablation trên `MBM` (`44/44` Full Data)**
  - Đã hoàn tất đủ cả 4 biến thể (`A0+S`, `A1`, `A2`, `A3`) với `--stage_samples -1 --visualize` (`Failed=0`). Kết quả cho thấy trên tập tế bào nhỏ đồng nhất (`dup=0.00%`), `Baseline` (`19.5864 / 37.1069`) vẫn tốt hơn khi không ép `scale_spread` (`SDES`), lưới CLIP `7×7` (`CLIP Fix`), hay lọc mask thuần (`MD-FCE g=0.0`).
- [x] **3.4. Chạy Ablation đã chốt trên `FSC-147 test` In-domain (`DSALVANet`, 1,190 ảnh)**
  - Đã hoàn tất đủ bảng `2×2` (`Baseline`, `+ SDES`, `+ CLIP Fix`, `+ CLIP Fix + SDES`, tất cả `1190/1190 | Failed=0`). Kết quả cho thấy **`+ SDES`** thắng toàn diện trên Baseline (`16.7000/90.6302 → 16.3496/87.8786`, nhanh hơn `21.5%`: `12.92s → 10.15s/img`), còn `CLIP Fix` gặp ngoại lai nặng ở `markers` / `legos` / `nail polish` trên full test.
- [ ] **3.5. Tổng hợp `summary.json`, `proposal_diagnostics`, `predictions.csv`, `logs/resource_usage.log`**
  - Đặt tên `--outdir` nhất quán: `ablation_<track>_<A0|A1|A2|A3>`.
