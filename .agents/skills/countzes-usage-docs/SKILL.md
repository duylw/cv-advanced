---
name: countzes-usage-docs
description: >-
  Use this skill whenever adding or modifying features, CLI flags, model variants,
  sampling options, or Colab notebook parameters in CountZES to generate or update
  the Markdown usage guide (USAGE.md) with all currently available options.
---

# CountZES Usage Documentation Skill

Skill này hướng dẫn quy trình tạo và duy trì tài liệu Markdown hướng dẫn sử dụng (`USAGE.md` ở root repo, hoặc tài liệu hướng dẫn chuyên biệt khi được yêu cầu) **mỗi khi dự án thêm tính năng, tùy chọn CLI, preset hoặc thay đổi luồng chạy mới**.

---

## 1. Khi nào kích hoạt Skill
- Sau khi thêm/sửa tham số trong `build_argument_parser()` tại `countzes_core.py` hoặc parser riêng trong `countzes_*.py`.
- Sau khi thêm phương pháp cải tiến mới (ví dụ `--clip_sim_mode`, `--clip_no_crop`, `--sdes`, `--md_fce`, `--smdf`, ...) hoặc chế độ lấy mẫu/đánh giá mới (`--sample_mode`, `--split`, ...).
- Sau khi thêm hoặc sửa biến cấu hình trong các notebook Colab (`colab_run_pipeline.ipynb`, `colab_run_fsc147.ipynb`, `colab_spike_test.ipynb`).
- Khi người dùng yêu cầu viết/cập nhật file Markdown hướng dẫn dùng hoặc tổng hợp các tùy chọn đang có sẵn.

---

## 2. Quy trình bắt buộc (Source-of-Truth First)

1. **Đọc trực tiếp mã nguồn hiện tại (không viết từ trí nhớ):**
   - Đọc `build_argument_parser()` trong `countzes_core.py` để lấy danh sách đầy đủ mọi flag, kiểu dữ liệu, `default`, `choices`, và `help`.
   - Đọc các entrypoint `countzes_vgg.py`, `countzes_mbm.py`, `countzes_persense_D.py`, `countzes_carpk.py`, `countzes_fsc147.py` để lấy `default_outdir`, `default_fsoc`, `default_countr_ckpt` và flag riêng (ví dụ `--split` ở `countzes_fsc147.py`).
   - Đọc cell cấu hình (`[CELL-7]`, `[CELL-6]`) trong `colab_run_pipeline.ipynb`, `colab_run_fsc147.ipynb` và `colab_spike_test.ipynb` để cập nhật đúng tên biến notebook.
   - Đọc `run_recorded_evaluation()` trong `countzes_core.py` nếu có thay đổi về file log/evidence sinh ra dưới `outputs/<outdir>/`.
2. **Tạo / Cập nhật `USAGE.md` theo cấu trúc chuẩn** (Mục 3 bên dưới):
   - Làm nổi bật rõ **những option vừa mới thêm** hoặc thay đổi gần nhất.
   - Giữ bảng tham chiếu **tất cả option đang có sẵn** được phân nhóm rõ ràng.
   - Cung cấp sẵn các **Preset lệnh mẫu copy-paste được** cho Baseline, Smoke test, Paper-exact, MaskCLIP, và Ablation.
3. **Đồng bộ lại chính Skill này (nếu có flag mới):**
   - Nếu vừa thêm CLI flag mới vào repo, cập nhật thêm dòng tương ứng vào Mục 4 của chính file `SKILL.md` này để danh mục option tham chiếu luôn khớp code.
4. **Kiểm tra chéo (Verification):**
   - Đối chiếu 1-1 mọi tên flag (`--...`), giá trị `default`, và `choices` giữa `USAGE.md` với `countzes_core.py` và `countzes_*.py`.
   - Đảm bảo không đề cập tới file, cờ lệnh hoặc kết quả không có thật trong mã nguồn.

---

## 3. Cấu trúc chuẩn khi xuất Markdown hướng dẫn (`USAGE.md`)

Tài liệu Markdown tạo ra nên có đủ 6 phần sau:
1. **Tính năng / Cập nhật mới nhất:** Tóm tắt ngắn gọn option mới thêm để làm gì, flag/biến tương ứng.
2. **Các Entrypoint & Colab Notebooks hiện có:** Bảng liệt kê từng dataset script (`default_fsoc`, `default_outdir`, flag riêng) và vai trò từng notebook `.ipynb`.
3. **Toàn bộ CLI Options đang có sẵn:** Bảng đầy đủ gồm 5 cột: `Flag` | `Mặc định` | `Giá trị hợp lệ` | `Nhóm` | `Chức năng`.
4. **Các Preset / Kịch bản chạy hay dùng:** Lệnh CLI đầy đủ cho từng mục tiêu (Smoke test nhanh, Paper-exact baseline, GitHub-legacy baseline, MaskCLIP + No-crop, Đổi Counter/SAM).
5. **Tùy chọn nhanh trên Colab Notebooks:** Giải thích các biến chỉnh tay trong `[CELL-7]` và lưu ý *Revert File* + `git push` -> `git pull`.
6. **Cấu trúc Output & Evidence lưu trên Drive:** Danh sách file sinh ra trong `outputs/<outdir>/` (`run_config.json`, `summary.json`, `predictions.csv`, `predictions.jsonl`, `logs/`, `stage_samples/`).

---

## 4. Danh mục đầy đủ các Option đang có sẵn (Baseline Schema)

> **Lưu ý:** Khi dùng skill này sau một thay đổi mới, hãy đọc lại `countzes_core.py` và `countzes_*.py` để bổ sung/điều chỉnh nếu có flag mới hơn bảng này.

### A. Entrypoints theo Dataset
| Script | Dataset | `default_fsoc` | `default_outdir` | `default_countr_ckpt` | Flag riêng |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `countzes_vgg.py` | VGG (200 ảnh) | `DSALVANet` | `countzes_vgg` | `./CounTR/output_allnew_dir/FSC147.pth` | — |
| `countzes_mbm.py` | MBM (44 ảnh) | `countr` | `countzes_mbm` | `./CounTR/output_allnew_dir/FSC147.pth` | — |
| `countzes_carpk.py` | CARPK (459 ảnh test) | `DSALVANet` | `countzes_carpk` | `./CounTR/output_allnew_dir/CARPK.pth` | — |
| `countzes_persense_D.py` | PerSense-D (717 ảnh) | `countr` | `countzes_persenseD` | `./CounTR/output_allnew_dir/FSC147.pth` | — |
| `countzes_fsc147.py` | FSC-147 (`test` 1,190 / `val` 1,286) | `DSALVANet` | `countzes_fsc147` | `./CounTR/output_allnew_dir/FSC147.pth` | `--split {test,val}` (default `test`) |

### B. Shared CLI Options (`countzes_core.py`)
- **Dữ liệu, Output & Sampling:**
  - `--data` (default: `./data`): Thư mục gốc chứa các dataset.
  - `--outdir` (default theo script): Tên thư mục con dưới `./outputs/`.
  - `--max_images` (default: `0`): Số ảnh tối đa để chạy (`0` = chạy toàn bộ dataset).
  - `--sample_mode` (default: `first`, choices: `first`, `stratified`, `random`): Cách lấy mẫu khi `--max_images > 0` (`stratified` chia đều qua các class).
  - `--sample_seed` (default: `42`): Seed cố định cho `stratified` / `random`.
- **Backbone & Weights:**
  - `--sam_type` (default: `vit_h`, choices: `vit_h`, `vit_l`, `vit_b`, `vit_t`): Biến thể SAM / MobileSAM.
  - `--ckpt` (default: `None`): Ghi đè đường dẫn checkpoint SAM (`vit_h` mặc định `data/sam_vit_h_4b8939.pth`).
  - `--fsoc` (default theo script, choices: `countr`, `DSALVANet`): Few-shot counter backend.
  - `-w`, `--weight` (default: `./DSALVANet/checkpoints/checkpoint_200.pth`): Checkpoint DSALVANet.
  - `--countr_ckpt` (default theo script): Checkpoint CounTR.
  - `--ref_idx` (default: `00`): Chỉ số tham chiếu legacy của parser.
- **Stage 1–3 Pipeline & CLIP Options:**
  - `--box_threshold` (default: `0.15`): Ngưỡng box GroundingDINO (`0.15` theo paper Sec. 4; `0.20` ở repo GitHub gốc).
  - `--text_threshold` (default: `0.15`): Ngưỡng text GroundingDINO.
  - `--sses_start_percentile` (default: `90`): Phần trăm khởi đầu trong SSES (`90` theo paper Sec. 3.1; `80` ở repo GitHub gốc).
  - `--fix_sses_inversion` (default: `True`): Giữ nguyên `sim_clip` theo công thức Eq. 2–3 của paper (không đảo `1 - sim_clip`).
  - `--github_sses_inversion`: Tắt `--fix_sses_inversion` để tái lập hành vi `sim_clip = 1 - sim_clip` của repo GitHub gốc.
  - `--single_instance_mode` (default: `paper`, choices: `paper`, `unit`, `le1`): Bộ lọc RoI single-instance (`paper`: `1 < c < 2`, `unit`: `0.5 <= c <= 1.5`, `le1`: `0 < c <= 1.5`).
  - `--clip_sim_mode` (default: `raw`, choices: `raw`, `maskclip`): Chế độ trích bản đồ CLIP (`raw`: `last_hidden_state[:, 1:, :]`, `maskclip`: Value projection bỏ qua Q–K tầng cuối + `post_layernorm`).
  - `--clip_no_crop` (default: `False`): Resize toàn bộ ảnh về kích thước input của CLIP mà không center-crop ảnh chữ nhật.
- **Visualization & Evidence:**
  - `--visualize` (default: `False`): Lưu ảnh density output cuối vào `outputs/<outdir>/<DATASET>/`.
  - `--stage_samples` (default: `5`): Số ảnh lưu lưới `2x4` từng bước vào `outputs/<outdir>/stage_samples/` (`-1` = lưu toàn bộ mẫu đã chạy, `0` = tắt).
