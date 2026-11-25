# Pha 3 — Ablation Study (100% Dataset)

- **Mục tiêu:** Tạo bảng Ablation khoa học **100% toàn bộ tập dữ liệu (không lấy mẫu con)** trên **bộ nền mặc định GitHub gốc đã chốt (`2026-11-26`)**, tập trung trên 3 track hiện tại: **`MBM` (44, `countr + FSC147.pth`)**, **`FSC-147 test` In-domain (1,190, `DSALVANet`)**, và **`FSC-147 test` Cross-domain (1,190, `countr + CARPK.pth`)** *(tạm gác `VGG`, `CARPK`)*.
- **Bộ cờ nền cố định cho mọi hàng `A0 → A3`:**
  `--box_threshold 0.20 --text_threshold 0.15 --sses_start_percentile 80 --github_sses_inversion --single_instance_mode paper --clip_sim_mode raw`
- **Chiến lược tiết kiệm GPU (`gpu-server` ~9–10.5s/img):**
  1. Chạy trước trên **`MBM` (44 ảnh, ~7p/run)** — tổng ~28 phút cho 4 hàng, kiểm chứng nhanh trước.
  2. Chạy trên **`FSC-147 test` (1,190 ảnh, ~3–3.4h/run)**: chạy In-domain (`DSALVANet`) và Cross-domain (`countr --countr_ckpt ./CounTR/output_allnew_dir/CARPK.pth`) cho `A0` → `A2` (thêm `A3` nếu `MD-FCE` thắng trên `MBM`).
- **Trạng thái:** `NEXT`

---

## Ma trận Ablation chuẩn (Nền cố định: GitHub Default `0.20 / 0.15 / 80 / --github_sses_inversion / paper`, bỏ `le1` & `SMDF`)

| ID | Cấu hình | `CLIP Fix` (`MaskCLIP + No-Crop`) | `SDES` | `MD-FCE` (`gamma=0.0`) | `MBM` (44, `countr`) | `FSC-147` In-domain (`DSALVANet`) | `FSC-147` Cross-domain (`countr + CARPK.pth`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `A0` | GitHub Baseline (`raw`, `0.20/80/github_inv`) | ✗ | ✗ | ✗ | ✅ `19.5864 / 37.1069` *(Colab `baseline_mbm_github`)* | ☐ | ☐ |
| `A1` | `+ CLIP Fix` *(Stage 1)* | ✓ | ✗ | ✗ | ☐ | ☐ | ☐ |
| `A2` | **`+ SDES` (`CountZES+ Core`)** *(Stage 2–3)* | ✓ | ✓ | ✗ | ☐ | ☐ | ☐ |
| `A3` | `+ MD-FCE (g=0.0)` *(Stage 3 mask)* | ✓ | ✓ | ✓ | ☐ | ☐ *(nếu thắng trên `MBM`)* | ☐ *(nếu thắng trên `MBM`)* |

---

## Công việc chi tiết

- [ ] **3.1. Chuẩn bị `MBM` (44) & `CounTR/{FSC147,CARPK}.pth` trên `gpu-server`**
  - Tải `mbm` (`data/mbm/Images/blood cells/`), `CounTR/output_allnew_dir/FSC147.pth`, `CounTR/output_allnew_dir/CARPK.pth` và pass smoke test `--max_images 1` cho cả 3 track.
- [ ] **3.2. Hoàn tất Baseline `A0` cho 3 track**
  - `MBM` đã xong trên Colab (`19.5864 / 37.1069`); chạy `A0` trên `FSC-147 test` In-domain (`DSALVANet`) và Cross-domain (`countr + CARPK.pth`).
- [ ] **3.3. Chạy Ablation `A1 → A3` trên `MBM` (44 ảnh × 3 runs ≈ ~21 phút)**
  - Dùng `--stage_samples -1 --visualize` để lưu toàn bộ 44 ảnh phân tích.
- [ ] **3.4. Chạy Ablation `A1 → A2` (+ `A3` nếu thắng) trên `FSC-147 test` (In-domain & Cross-domain)**
  - Chạy qua `tmux` + `bash scripts/ec2_idle_stop.sh --timeout 1800` với `--stage_samples 20`.
- [ ] **3.5. Tổng hợp `summary.json`, `proposal_diagnostics`, `predictions.csv`, `logs/resource_usage.log`**
  - Đặt tên `--outdir` nhất quán: `ablation_<track>_<A0|A1|A2|A3>`.
