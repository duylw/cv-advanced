# Pha 1 — Hiện thực các đề xuất cải tiến trong Code

- **Mục tiêu:** Đưa 3 cải tiến training-free từ `IMPROVEMENTS.md` vào [countzes_core.py](../countzes_core.py) dưới dạng cờ CLI opt-in (không phá mặc định baseline).
- **Trạng thái:** `NEXT`

---

## Danh sách công việc

- [ ] **1.1. Hiện thực `SDES` (`--sdes`, `--sdes_iou_thresh`, `--sdes_max_exemplars`)**
  - Spatial Deduplication for Exemplar Selection trong `SSES`: thay vì giữ top-k điểm trùng trên cùng 1 object, chạy Greedy Spatial Suppression (NMS theo khoảng cách chuẩn hóa / IoU mask SAM hoặc radius) để ép đa dạng tối đa 3 exemplars cho DGE.
  - Kỳ vọng phụ: giảm gọi SAM trùng lặp ở Stage 2 → tăng tốc `s/img`.
- [ ] **1.2. Hiện thực `MD-FCE` (`--md_fce`, `--md_fce_alpha`)**
  - Mask-Driven Feature Contrastive Enhancement ở Stage 3 (FCE): dùng `best_mask` của SAM phân tách Foreground vs Background trong `hr_features` `(256, 256)` rồi khử/giảm thành phần nền trước khi đưa vector exemplar vào `DSALVANet`/`CounTR`.
  - Kỳ vọng: giảm over-counting khi nền rối hoặc exemplar có background lớn.
- [ ] **1.3. Hiện thực `SMDF` (`--smdf`, `--smdf_weights`)**
  - Semantic-Morphological Density Fusion: thay `max(D_dae, D_dge, D_fce)` hoặc cộng thô bằng fusion có trọng số thích ứng dựa trên độ tin cậy CLIP / phân tán mật độ (tránh đỉnh nhiễu cục bộ gây sai lệch tổng đếm).
- [ ] **1.4. Kiểm tra kỹ thuật & Smoke Test**
  - `python -m py_compile countzes_core.py countzes_*.py` pass không lỗi.
  - Viết unit test nhanh (CPU/mock tensor) kiểm tra `baseline` (`--sdes=False`, `--md_fce=False`, `--smdf=False`) cho ra kết quả giống hệt trước khi sửa.
  - Chạy EC2 smoke `n=5` với tất cả cờ mới bật (`--clip_sim_mode maskclip --clip_no_crop --sdes --md_fce --smdf`).
- [ ] **1.5. Cập nhật tài liệu & Commit**
  - Chạy skill `countzes-usage-docs` để cập nhật [USAGE.md](../USAGE.md) và `colab_run_fsc147.ipynb` (nếu cần).
  - Commit code sạch qua `/commit`.

---

## Lệnh kiểm tra nhanh trên EC2 sau khi code xong

```bash
source /opt/pytorch/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m py_compile countzes_core.py countzes_*.py

python countzes_fsc147.py --data ./data --split test --fsoc DSALVANet \
  --max_images 5 --sample_mode stratified --sample_seed 42 \
  --clip_sim_mode maskclip --clip_no_crop \
  --sdes --md_fce --smdf --stage_samples -1 --visualize \
  --outdir ec2_smoke_all_proposals_n5
```

## Tiêu chí qua Pha 1
- Toàn bộ cờ hoạt động không crash, VRAM vẫn < 8 GB trên T4 (`15 GB`).
- Có log xác nhận `SDES` giảm trùng box trong `stage_samples/*.png`.
