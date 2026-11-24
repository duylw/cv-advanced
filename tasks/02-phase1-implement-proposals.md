# Pha 1 — Hiện thực các đề xuất cải tiến trong Code

- **Mục tiêu:** Đưa 3 cải tiến training-free từ `IMPROVEMENTS.md` vào [countzes_core.py](../countzes_core.py) dưới dạng cờ CLI opt-in (không phá mặc định baseline).
- **Trạng thái:** `READY_FOR_EC2_SMOKE`

---

## Danh sách công việc

- [x] **1.1. Hiện thực `SDES` (`--sdes`, `--sdes_iou_thr`, `--sdes_max_keep`, `--sdes_lambda_iou`, `--sdes_lambda_scale`)**
  - Unit-prioritized Spatial NMS (`_sdes_select_indices`) trên `single_instance_boxes1` trước vòng `process_box` & KDE ở Stage 2, kèm `_sdes_diversity_factor` phạt trùng không gian (`DGE`, `FCE`) và thưởng bổ trợ tỷ lệ log-scale (`FCE`).
  - Ghi nhận số box trước/sau SDES vào `predictions.csv` (`single1_pre_sdes_n -> single1_n`) và tiêu đề ô `S2b` trên `stage_samples`.
- [x] **1.2. Hiện thực `MD-FCE` (`--md_fce`, `--md_fce_gamma`)**
  - Giữ lại mask SAM ở Stage 2 (`dge_mask`) và Stage 3 (`masks_sam_new`, kèm `dae_mask`) thay vì bỏ vào `_`, kết hợp trọng số tâm mật độ `1 + gamma * norm_density` để pooling đặc trưng AnyUp sạch nền trong `box_to_featvec`.
- [x] **1.3. Hiện thực `SMDF` (`--smdf`, `--smdf_sigma_u`, `--smdf_priors`)**
  - Giữ đủ `density_raw_s2`, `density_raw_s3`, `density_raw_final` và hợp nhất cả tổng đếm lẫn bản đồ mật độ hiển thị qua `_smdf_fuse_stage_densities` theo $R_{\text{unit}}^{(s)} \cdot R_{\text{kde}}^{(s)}$.
- [x] **1.4. Kiểm tra kỹ thuật (Local) & [ ] Smoke Test EC2**
  - [x] `python -m py_compile` pass toàn bộ 7 file `.py` + unit test logic (`_sdes_select_indices`, `_sdes_diversity_factor`, `_smdf_fuse_stage_densities`, default parser flags = `False`).
  - [ ] Chạy EC2 smoke `n=5` với tất cả cờ mới bật (`--single_instance_mode le1 --clip_sim_mode maskclip --clip_no_crop --sdes --md_fce --smdf`).
- [x] **1.5. Cập nhật tài liệu (`USAGE.md` & skill schema) & [ ] Commit**
  - [x] Đã cập nhật [USAGE.md](../USAGE.md) và `.agents/skills/countzes-usage-docs/SKILL.md` theo đúng parser mới.
  - [ ] Commit code sạch qua `/commit`.

---

## Lệnh kiểm tra nhanh trên EC2 sau khi push & pull code

```bash
cd ~/cv-advanced && git pull
source /opt/pytorch/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m py_compile countzes_core.py countzes_*.py

python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 5 --sample_mode stratified --sample_seed 42 \
  --single_instance_mode le1 --clip_sim_mode maskclip --clip_no_crop \
  --sdes --md_fce --smdf --stage_samples -1 --visualize \
  --outdir ec2_smoke_all_proposals_n5
```

## Tiêu chí qua Pha 1
- Toàn bộ cờ hoạt động không crash, VRAM vẫn < 8 GB trên T4 (`15 GB`).
- Có log xác nhận `SDES` giảm trùng box trong `stage_samples/*.png`.
