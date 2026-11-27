# Bảng kết quả Ablation Study — CountZES (GitHub Base `2026-11-26`)

> **Nền Baseline cố định (khi chưa tick đề xuất nào):** `--box_threshold 0.20 --text_threshold 0.15 --sses_start_percentile 80 --github_sses_inversion --single_instance_mode paper --clip_sim_mode raw`
>
> **Ý nghĩa từng cột đề xuất (bật cột nào thì đánh `✓`, không bật để trống):**
> - **`CLIP Fix`:** `--clip_sim_mode maskclip --clip_no_crop --fix_sses_inversion` *(thay cho `raw + 1-sim` ở Stage 1)*
> - **`SDES`:** `--sdes` *(Spatial-Scale Diverse Exemplar Selection ở Stage 2–3)*
> - **`MD-FCE`:** `--md_fce --md_fce_gamma 0.0` *(Mask- & Density-Attentive Feature Pooling ở Stage 3)*

---

## 1. Bảng Ablation Full-Data Tổng hợp (`FSC-147 test` 1,190 ảnh + `MBM` 44 ảnh)

| Cấu hình | `CLIP Fix` | `SDES` | `MD-FCE` | `FSC-147` MAE ↓ | `FSC-147` RMSE ↓ | `FSC-147` Tốc độ | `MBM` MAE ↓ | `MBM` RMSE ↓ | `MBM` Tốc độ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | | | | `16.7000` | `90.6302` | `12.92s/img` | **`19.5864`** *(Colab)* | **`37.1069`** *(Colab)* | `15.61s/img` |
| + `SDES` | | ✓ | | **`16.3496`** | **`87.8786`** | `10.15s/img` | `24.4593` | `39.2429` | `12.29s/img` |
| + `CLIP Fix` | ✓ | | | `17.5866` | `113.1749` | `13.53s/img` | `27.4356` | `44.1526` | `13.30s/img` |
| + `MD-FCE` | | | ✓ | — | — | — | `19.7784` *(Colab)* | `37.2569` *(Colab)* | `17.10s/img` |
| + `CLIP Fix` + `SDES` | ✓ | ✓ | | `17.6168` | `110.1464` | **`9.62s/img`** | `33.0592` | `54.6344` | `12.22s/img` |
| + `CLIP Fix` + `SDES` + `MD-FCE` | ✓ | ✓ | ✓ | — | — | — | `35.2233` | `56.5361` | **`11.77s/img`** |

*(Ghi chú thiết lập counter từng tập: `FSC-147 test` chạy `1,190/1,190` ảnh In-domain với `--fsoc DSALVANet`; `MBM` chạy `44/44` ảnh Cross-domain với `--fsoc countr --countr_ckpt ./CounTR/output_allnew_dir/FSC147.pth`.)*

---

## 2. Chỉ số chẩn đoán cơ chế (Diagnostics) trên Full Data

| Cấu hình | `CLIP Fix` | `SDES` | `MD-FCE` | `FSC-147` Box S2 | `FSC-147` Trùng (`IoU≥0.5`) | `FSC-147` Contrast | `MBM` Box S2 | `MBM` Trùng (`IoU≥0.5`) | `MBM` Contrast |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | | | | `31.11` | `18.66%` | `-0.1657` | `27.86` | `0.00%` | `-0.1923` |
| + `SDES` | | ✓ | | **`31.11 → 9.99`** | **`6.64%`** | `-0.1657` | `27.86 → 13.48` | `0.00%` | `-0.1923` |
| + `CLIP Fix` | ✓ | | | `29.42` | `18.32%` | **`+0.2365`** | `26.80` | `0.00%` | **`+0.1416`** |
| + `MD-FCE` | | | ✓ | — | — | — | `27.86` | `0.00%` | `-0.1923` |
| + `CLIP Fix` + `SDES` | ✓ | ✓ | | **`29.42 → 9.88`** | `7.82%` | **`+0.2365`** | `26.80 → 12.50` | `0.00%` | **`+0.1416`** |
| + `CLIP Fix` + `SDES` + `MD-FCE` | ✓ | ✓ | ✓ | — | — | — | **`26.80 → 12.50`** | `0.00%` | **`+0.1416`** |

*(Folder outputs tương ứng: `FSC-147` = `baseline_fsc147_test_indomain_github`, `ablation_fsc147_sdes`, `ablation_fsc147_clipfix`, `ablation_fsc147_clipfix_sdes`; `MBM` = `baseline_mbm_github`, `mbm_gh_raw_1sim_sdes`, `mbm_gh_clipfix`, `mbm_gh_raw_1sim_mdfce0`, `mbm_gh_clipfix_sdes`, `mbm_gh_clipfix_sdes_mdfce0`.)*

---

## 3. Phụ lục: Dev Ablation trên `FSC-147 test` (`n=24` stratified, `seed=42`, `DSALVANet`)

| Cấu hình | `CLIP Fix` | `SDES` | `MD-FCE` | MAE ↓ | RMSE ↓ | Tốc độ | Box S2 (`pre → post`) | Trùng (`IoU≥0.5`) | CLIP Contrast | Folder |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline | | | | `12.2917` | `36.6794` | `13.87s/img` | `28.96` | `20.83%` | `-0.1547` | `ec2_gh_test24_raw_1sim` |
| + `SDES` | | ✓ | | **`11.1250`** | `34.2144` | `10.65s/img` | `28.96 → 11.08` | **`4.17%`** | `-0.1547` | `ec2_gh_test24_raw_1sim_sdes` |
| + `CLIP Fix` | ✓ | | | `12.0000` | **`32.5947`** | `12.26s/img` | `19.67` | `16.67%` | **`+0.2208`** | `ec2_gh_test24_maskclip_nocrop_fix` |
| + `CLIP Fix` + `SDES` | ✓ | ✓ | | `11.4167` | `33.5373` | **`10.55s/img`** | **`19.67 → 10.08`** | `8.33%` | **`+0.2208`** | `ec2_gh_test24_maskclip_nocrop_sdes` |
