# Hướng Dẫn Sử Dụng & Danh Mục Option — CountZES

Tài liệu tổng hợp toàn bộ cách chạy, các Colab notebooks, và **đầy đủ CLI options đang có sẵn** trong mã nguồn CountZES.

> **Cập nhật gần nhất:**
> - Thêm `--clip_sim_mode {raw,maskclip}` và `--clip_no_crop` trong `countzes_core.py` để sửa lỗi nền của CLIP tầng cuối và lỗi center-crop ảnh chữ nhật.
> - Thêm `--split {test,val}` trong `countzes_fsc147.py` cùng `--sample_mode {first,stratified,random}`, `--sample_seed`, `--stage_samples -1`, và `per_class_metrics` trong `summary.json`.
> - Thêm notebook riêng `colab_run_fsc147.ipynb` để chạy và phân tích đa lớp trên FSC-147.

---

## 1. Các Dataset Entrypoints & Colab Notebooks

### A. Script dòng lệnh (`countzes_*.py`)
| Script | Dataset | Counter mặc định (`--fsoc`) | Thư mục output mặc định (`--outdir`) | Checkpoint CounTR mặc định | Option riêng |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `countzes_vgg.py` | **VGG** (200 ảnh tế bào) | `DSALVANet` | `countzes_vgg` | `./CounTR/output_allnew_dir/FSC147.pth` | — |
| `countzes_mbm.py` | **MBM** (44 ảnh nhân tế bào) | `countr` | `countzes_mbm` | `./CounTR/output_allnew_dir/FSC147.pth` | — |
| `countzes_carpk.py` | **CARPK** (459 ảnh bãi xe) | `DSALVANet` | `countzes_carpk` | `./CounTR/output_allnew_dir/CARPK.pth` | — |
| `countzes_persense_D.py` | **PerSense-D** (717 ảnh đa lớp) | `countr` | `countzes_persenseD` | `./CounTR/output_allnew_dir/FSC147.pth` | — |
| `countzes_fsc147.py` | **FSC-147** (`test`: 1,190 / `val`: 1,286) | `DSALVANet` | `countzes_fsc147` | `./CounTR/output_allnew_dir/FSC147.pth` | `--split {test,val}` |

### B. Notebooks Google Colab (`.ipynb`)
- **`colab_spike_test.ipynb`**: Thử nghiệm tự do (*spike test*) trên ảnh lẻ (URL, Drive, hoặc upload) với prompt class bất kỳ để quan sát nhanh kết quả.
- **`colab_run_pipeline.ipynb`**: Chạy pipeline đánh giá tổng quát (mặc định đang cấu hình cho `VGG`, hỗ trợ đổi sang `countzes_mbm.py`, `countzes_carpk.py`, `countzes_persense_D.py`, `countzes_fsc147.py`), tự động mount Drive và hiển thị bằng chứng.
- **`colab_run_fsc147.ipynb`**: Notebook chuyên biệt cho **FSC-147** (hỗ trợ tải `images_384_VarV2`, chọn `MAX_IMAGES`, lấy mẫu `stratified` qua 29 classes, bật tắt `CLIP_SIM_MODE` / `CLIP_NO_CROP`, và thống kê MAE/RMSE theo từng class).

---

## 2. Toàn Bộ CLI Options Đang Có Sẵn

### A. Nhóm Dữ liệu, Output & Lấy mẫu
| Flag | Mặc định | Giá trị hợp lệ | Chức năng |
| :--- | :--- | :--- | :--- |
| `--data` | `./data` | Đường dẫn thư mục | Thư mục gốc chứa các dataset (`vgg`, `mbm`, `FSC-147`, `CARPK`, `persenseD`). |
| `--outdir` | Theo từng script | Tên thư mục | Thư mục con lưu kết quả dưới `./outputs/<outdir>/`. |
| `--max_images` | `0` | Số nguyên `>= 0` | Giới hạn số mẫu đánh giá (`0` = chạy toàn bộ split/dataset). |
| `--sample_mode` | `first` | `first`, `stratified`, `random` | Cách lấy mẫu khi `--max_images > 0`. `stratified` chọn rải đều vòng tròn qua mọi class. |
| `--sample_seed` | `42` | Số nguyên | Seed cố định để tái lập chính xác cùng tập mẫu khi dùng `stratified` hoặc `random`. |
| `--split` *(chỉ `countzes_fsc147.py`)* | `test` | `test`, `val` | Chọn split của FSC-147 (`test` = 1,190 ảnh; `val` = 1,286 ảnh). |

### B. Nhóm Model Backbone & Checkpoint
| Flag | Mặc định | Giá trị hợp lệ | Chức năng |
| :--- | :--- | :--- | :--- |
| `--sam_type` | `vit_h` | `vit_h`, `vit_l`, `vit_b`, `vit_t` | Chọn biến thể SAM (`vit_t` dùng cho MobileSAM). |
| `--ckpt` | `None` | Đường dẫn `.pth`/`.pt` | Ghi đè đường dẫn checkpoint SAM (mặc định `vit_h` -> `data/sam_vit_h_4b8939.pth`). |
| `--fsoc` | Theo từng script | `DSALVANet`, `countr` | Chọn mô hình Few-Shot Object Counter backend. |
| `-w`, `--weight` | `./DSALVANet/checkpoints/checkpoint_200.pth` | Đường dẫn `.pth` | Đường dẫn checkpoint của DSALVANet. |
| `--countr_ckpt` | Theo từng script | Đường dẫn `.pth` | Đường dẫn checkpoint của CounTR (`FSC147.pth` hoặc `CARPK.pth`). |
| `--ref_idx` | `00` | Chuỗi | Tham số legacy giữ tương thích giao diện dòng lệnh cũ. |

### C. Nhóm Siêu tham số Pipeline (Stage 1–3 & CLIP)
| Flag | Mặc định | Giá trị hợp lệ | Chức năng |
| :--- | :--- | :--- | :--- |
| `--box_threshold` | `0.15` | Số thực `(0, 1]` | Ngưỡng box của GroundingDINO (paper Sec. 4 = `0.15`; repo GitHub gốc = `0.20`). |
| `--text_threshold` | `0.15` | Số thực `(0, 1]` | Ngưỡng text của GroundingDINO. |
| `--sses_start_percentile` | `90` | Số nguyên | Phần trăm khởi đầu trong SSES để tìm đỉnh CLIP/SAM (paper = `90`; repo GitHub = `80`). |
| `--fix_sses_inversion` | `True` | Cờ boolean | Dùng điểm similarity trực tiếp theo đúng Eq. 2–3 của paper (không đảo `1 - sim_clip`). |
| `--github_sses_inversion` | — | Cờ hành động | Tắt `--fix_sses_inversion`, bật lại phép đảo `sim_clip = 1 - sim_clip` như repo GitHub gốc. |
| `--single_instance_mode` | `paper` | `paper`, `unit`, `le1` | Ngưỡng lọc hộp đơn cá thể theo mật độ RoI: `paper` (`1 < c < 2`), `unit` (`0.5 <= c <= 1.5`), `le1` (`0 < c <= 1.5`). |
| `--clip_sim_mode` | `raw` | `raw`, `maskclip` | `raw`: lấy `last_hidden_state[:, 1:, :]` gốc; `maskclip`: trích Value projection tầng cuối + `post_layernorm` (tránh trộn nền $Q\text{-}K$). |
| `--clip_no_crop` | `False` | Cờ boolean | Resize toàn ảnh về kích thước input của CLIP (`224x224`) mà không center-crop mất 2 rìa ảnh chữ nhật. |

### D. Nhóm Lưu Log, Hình ảnh & Bằng chứng
| Flag | Mặc định | Giá trị hợp lệ | Chức năng |
| :--- | :--- | :--- | :--- |
| `--visualize` | `False` | Cờ boolean | Lưu ảnh density map kết quả cuối (`refined_*.png` / `<id>.png`) vào `outputs/<outdir>/`. |
| `--stage_samples` | `5` | Số nguyên | Số ảnh lưu lưới phân tích 8 ô `2x4` (`S1a -> S3b -> Final`) vào `stage_samples/`. Đặt `-1` để lưu **tất cả** mẫu đã chạy; `0` để tắt. |

---

## 3. Các Preset Lệnh Hay Dùng

### Preset 1 — Smoke test nhanh trên FSC-147 (2–30 mẫu chia đều theo class)
```bash
python countzes_fsc147.py \
  --data ./data --split test --fsoc DSALVANet \
  --outdir fsc147_smoke_n30 \
  --max_images 30 --sample_mode stratified --sample_seed 42 \
  --visualize --stage_samples -1
```

### Preset 2 — Baseline chuẩn theo Paper (mặc định hiện tại)
```bash
python countzes_vgg.py \
  --data ./data --fsoc DSALVANet \
  --outdir baseline_vgg_paper_exact \
  --box_threshold 0.15 --text_threshold 0.15 \
  --sses_start_percentile 90 --single_instance_mode paper \
  --visualize --stage_samples 6
```

### Preset 3 — Tái lập hành vi GitHub Repo gốc (có `1 - sim_clip`, ngưỡng `0.20`, percentile `80`)
```bash
python countzes_vgg.py \
  --data ./data --fsoc DSALVANet \
  --outdir baseline_vgg_github \
  --github_sses_inversion --box_threshold 0.20 --sses_start_percentile 80 \
  --visualize --stage_samples 6
```

### Preset 4 — Thử nghiệm sửa lỗi nền CLIP với MaskCLIP + No-Crop
```bash
python countzes_fsc147.py \
  --data ./data --split test --fsoc DSALVANet \
  --outdir fsc147_test_maskclip_nocrop_n30 \
  --max_images 30 --sample_mode stratified --sample_seed 42 \
  --clip_sim_mode maskclip --clip_no_crop \
  --visualize --stage_samples -1
```

### Preset 5 — Đổi sang tập Validation để tune / kiểm tra chéo (tránh rò rỉ tập Test)
```bash
python countzes_fsc147.py \
  --data ./data --split val --fsoc DSALVANet \
  --outdir fsc147_val_stratified_n58 \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --visualize --stage_samples 15
```

---

## 4. Tùy Chọn Nhanh Trong Colab Notebook (`colab_run_fsc147.ipynb`)

Trong **[CELL-7]** của `colab_run_fsc147.ipynb`, chỉ cần sửa các biến cấu hình ở đầu cell:
- `SPLIT`: `"test"` (1,190 ảnh) hoặc `"val"` (1,286 ảnh).
- `MAX_IMAGES`: `0` (chạy hết) hoặc số nguyên `> 0` (vd `2`, `29`, `58`, `290`).
- `SAMPLE_MODE` & `SAMPLE_SEED`: `"stratified"` (nên dùng cho FSC-147), `"random"`, hoặc `"first"`.
- `STAGE_SAMPLES`: số ảnh lưu lưới `2x4` (`-1` để lưu hết mọi mẫu trong lượt chạy).
- `SAVE_FINAL_VIS`: `True` / `False`.
- `SSES_MODE`: `"paper"` hoặc `"github"`.
- `CLIP_SIM_MODE`: `"raw"` hoặc `"maskclip"`.
- `CLIP_NO_CROP`: `True` hoặc `False`.

> **Lưu ý khi chỉnh sửa code/notebook cho Colab:**
> 1. Sau khi chỉnh `.py`/`.ipynb` ở máy local, nhớ `git commit` và `git push origin main` rồi chạy lại **CELL-2** (`git pull`) trên Colab.
> 2. Nếu file `.ipynb` đang mở sẵn trên IDE vừa được cập nhật trên ổ đĩa, bấm **File → Revert File** trước khi chạy/lưu để tránh ghi đè buffer cũ.

---

## 5. Cấu Trúc Output & Bằng Chứng Lưu Lại (`outputs/<outdir>/`)

Mỗi lần chạy tự động lưu đầy đủ bằng chứng vào `outputs/<outdir>/` (trên Colab symlink thẳng tới `MyDrive/CountZES/outputs/<outdir>/`):
- `run_config.json`: Toàn bộ tham số CLI, thông số phần cứng (CPU/RAM/GPU), commit Git, thời gian chạy.
- `summary.json`: Chỉ số tổng `MAE`, `RMSE`, số ảnh thành công/lỗi, thời gian trung bình/ảnh, tài nguyên tiêu thụ, và **`per_class_metrics`** (MAE/RMSE từng class).
- `predictions.csv` & `predictions.jsonl`: Số liệu chi tiết từng ảnh qua từng stage (`gt_count`, `pred_count`, `abs_error`, `gdino_boxes_n`, `dae`, `count_stage2`, `pseudo_gt_kde`, `dge`, `count_stage3`, `single2_n`, `fce`, `duration_sec`, `status`).
- `logs/resource_usage.log` & `logs/resource_usage.jsonl`: CPU avg/peak, System RAM, Process RSS, và CUDA VRAM peak theo từng giai đoạn (`load_models`, `eval_*`). *(Lưu ý: nếu chạy trùng lại cùng một `OUTDIR`, log tài nguyên được ghi nối thêm vào cuối file; các file khác được ghi mới).*
- `stage_samples/*.png`: Lưới ảnh `2x4` từng bước (`S1a. CLIP` -> `S1b. GDINO+SSES->DAE` -> `S1c. SAM Sim` -> `S2a. Density+P2P` -> `S2b. GGES/KDE->DGE` -> `S3a. Density+P2P` -> `S3b. FRES 2-Means->FCE` -> `Final`).
- `<DATASET>/`: Ảnh kết quả cuối cùng với 3 bounding box exemplar và density overlay (khi bật `--visualize`).
