# Pha 2 — Dev Validation trên Batch nhỏ (Tách bạch Độc lập → Cộng dồn)

- **Mục tiêu:** Chứng minh minh bạch từng cải tiến khi đứng **độc lập** so với Paper Baseline `B0`, sau đó kiểm tra **cộng dồn từng nấc 1 thay đổi** trước khi chạy 100% dataset ở Pha 3.
- **Nguyên tắc vàng:** **CẤM tune trên `FSC-147 test` (1,190 ảnh).** Chỉ chạy trên `FSC-147 val` (`n=58`, stratified 2 ảnh/class × 29 classes, `seed=42`) và `MBM` (44 ảnh).
- **Trạng thái:** `NEXT`

---

## Phân vai minh bạch 3 lớp (không gộp chung)

- **Lớp 0 — Paper Baseline (`B0`):** `fix_sses_inversion=True` + `single_instance_mode=paper` + `clip_sim_mode=raw` (`clip_no_crop=False`).
- **Lớp Hiệu chỉnh cài đặt (từ P2):** `--single_instance_mode le1` (`0 < c <= 1.5`).
- **Lớp 4 Module Thuật toán (mỗi module giải quyết 1 stage):**
  1. **CLIP Fix (Stage 1):** `--clip_sim_mode maskclip --clip_no_crop`
  2. **SDES (Stage 2–3):** `--sdes`
  3. **MD-FCE (Stage 3):** `--md_fce`
  4. **SMDF (Final):** `--smdf`

---

## Bước 2A — Kiểm chứng ĐỘC LẬP từng thành phần trên nền `B0`

Chạy **từng thành phần một mình** trên đúng `B0` (`FSC-147 val n=58`, ~10 phút/run → ~60 phút tổng):

- [ ] `B0`: `paper` + `raw` *(mốc Paper Baseline)*
- [ ] `B0_le1`: `B0 + --single_instance_mode le1` *(đo riêng hiệu ứng đổi ngưỡng đơn vật thể)*
- [ ] `I1_clip`: `B0 + --clip_sim_mode maskclip --clip_no_crop` *(đo riêng cải tiến CLIP Stage 1)*
- [ ] `I2_sdes`: `B0 + --sdes` *(đo riêng SDES khi không có MaskCLIP/le1)*
- [ ] `I3_mdfce`: `B0 + --md_fce` *(đo riêng MD-FCE)*
- [ ] `I4_smdf`: `B0 + --smdf` *(đo riêng SMDF)*
- [ ] *(Bổ sung khi có `MBM` trên EC2)* Chạy lặp 6 mốc trên `MBM` (44 ảnh, ~8–9 phút/run).

```bash
cd ~/cv-advanced
source /opt/pytorch/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# B0. Paper Baseline (paper + raw)
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --stage_samples 10 --outdir dev_2a_val58_B0

# B0 + le1 (hiệu chỉnh ngưỡng P2)
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --single_instance_mode le1 --stage_samples 10 --outdir dev_2a_val58_le1

# I1. + CLIP Fix (MaskCLIP + No-Crop)
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --clip_sim_mode maskclip --clip_no_crop --stage_samples 10 --outdir dev_2a_val58_clip

# I2. + SDES (độc lập)
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --sdes --stage_samples 10 --outdir dev_2a_val58_sdes

# I3. + MD-FCE (độc lập)
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --md_fce --stage_samples 10 --outdir dev_2a_val58_mdfce

# I4. + SMDF (độc lập)
python countzes_fsc147.py --data ./data --split val --fsoc DSALVANet \
  --max_images 58 --sample_mode stratified --sample_seed 42 \
  --smdf --stage_samples 10 --outdir dev_2a_val58_smdf

# Tổng hợp kết quả Bước 2A
python -c "import glob, json; [print(p, '->', {k: d.get(k) for k in ('MAE','RMSE','avg_sec_per_image','single_instance_mode','clip_sim_mode','clip_no_crop','sdes','md_fce','smdf')}) for p in sorted(glob.glob('outputs/dev_2a_val58_*/summary.json')) for d in [json.load(open(p))]]"
```

---

## Bước 2B — Kiểm chứng CỘNG DỒN từng nấc 1 thay đổi (sau khi chốt 2A)

Chỉ giữ các thành phần vượt qua cổng 2A rồi ghép tuần tự theo đúng luồng pipeline (mỗi bước thêm đúng **1 thay đổi duy nhất**):

- [ ] `S1`: `B0 + le1` *(lấy lại số từ Bước 2A)*
- [ ] `S2`: `S1 + CLIP Fix` (`--clip_sim_mode maskclip --clip_no_crop`)
- [ ] `S3`: `S2 + SDES` (`--sdes`)
- [ ] `S4`: `S3 + MD-FCE` (`--md_fce`)
- [ ] `S5`: `S4 + SMDF` (`--smdf`) → **CountZES+ Full**
- [ ] Kiểm tra `predictions.csv` từng ảnh xem bước nào thắng / thua outlier.

---

## Cổng quyết định (Go / No-Go Gate trước Pha 3)
- **GIỮ** thành phần nếu ở Bước 2A nó cải thiện MAE / RMSE (hoặc giảm mạnh `s/img` không hại MAE) trên tập Dev mà không sinh outlier nặng.
- **TINH CHỈNH / LOẠI BỎ** ngay nếu tụt metric khi đứng một mình hoặc khi cộng dồn ở 2B.
- **Đóng băng toàn bộ hyperparameter** trước khi sang Pha 3.
