# Task 01: Sửa các bug trong pipeline CountZES & DSALVANet

- **Trạng thái:** `READY_FOR_GPU_TEST` (Đã hoàn tất toàn bộ sửa code P0/P1 & `py_compile`; chờ test runtime trên máy có GPU + weights/data)
- **Mục tiêu:** Khắc phục các lỗi crash, tràn VRAM, sai tọa độ feature, sai checkpoint và lỗi tổng hợp metric trên toàn bộ 5 script đánh giá (`countzes_fsc147.py`, `countzes_carpk.py`, `countzes_persense_D.py`, `countzes_mbm.py`, `countzes_vgg.py`) và các module phụ trợ trong `DSALVANet/`, `CounTR/`.

---

## P0 — Bug nghiêm trọng (Crash / Sai kết quả / Tràn VRAM)

- [x] **1. Sửa xung đột `argparse` khi chạy `--fsoc DSALVANet`**
  - Tách parser dùng chung `build_argument_parser` trong `countzes_core.py` (gộp `--weight`, `--countr_ckpt`, `--ckpt`, `--visualize`...).

- [x] **2. Bật `model.eval()`, `map_location`, và `torch.no_grad()` cho `DSALVANet`**
  - `DSALVANet/utils/model_helper.py`: thêm `map_location=device` và `model.eval()`.
  - `DSALVANet/models/DSALVANet.py`: sửa `LVAM.reparameterize` bỏ `.cuda()` cứng; sửa `MDM.reset_parameters` bọc `self.bias` thành `nn.Parameter`.
  - `countzes_core.py`: bọc `with torch.no_grad():` quanh toàn bộ `_run_counter` (cả CounTR & DSALVANet).

- [x] **3. Sửa lỗi tọa độ & kích thước lưới trong `box_to_featvec` (Stage 3 — FCE)**
  - `countzes_core.py`: clamp box theo `(W, H)` ảnh gốc và map sang `(w_hr, h_hr)` của `hr_features` `(256, 256)`.

- [x] **4. Sửa checkpoint CounTR trên FSC-147 & lỗi CounTR đọc `bbox.txt`**
  - `countzes_fsc147.py`: đổi default `--countr_ckpt` sang `./CounTR/output_allnew_dir/FSC147.pth`.
  - `CounTR/models_mae_cross.py`: bỏ đọc file cứng `./DSALVANet/test_data/bbox.txt`, suy ra `shot_num = min(shot_num, y_.shape[1], 5)` để Final Counting dùng đủ 3 exemplars `[dae, dge, fce]`.
  - `CounTR/demo.py`: clamp tọa độ crop exemplar tránh slice rỗng khi box sát mép.

- [x] **5. Sửa lỗi không cộng dồn metric `count_item`, `mae`, `mse` giữa các class**
  - `countzes_persense_D.py`, `countzes_mbm.py`, `countzes_vgg.py`: trả về và cập nhật `(count_item, mae, mse)` qua tất cả thư mục class.

- [x] **6. Bỏ `ThreadPoolExecutor` cho GPU inference ở Stage 2**
  - `countzes_core.py`: chạy tuần tự trong `with torch.no_grad():`.

---

## P1 — Bug phụ, Tương thích & Tối ưu cơ bản

- [x] **7. Sửa `--visualize`, `--ckpt` và đường dẫn lưu output**
  - Đổi `--visualize` sang `action='store_true'`, dùng `args.ckpt` nếu truyền, dùng `output_path` cho thư mục lưu ảnh.

- [x] **8. Sửa `preprocess` mutate list đầu vào & thứ tự resize**
  - `DSALVANet/utils/data_preprocess.py`: copy `ori_boxes`, sửa `cv2.resize(ori_img, (w_rsz, h_rsz))`, dùng `torch.float32`.

- [x] **9. Thay `np.float` đã deprecated & chia cho 0 khi map hằng số**
  - `DSALVANet/utils/fsol_modules.py`, `PerSense_countr.py`, `PerSense_modules.py`.

- [x] **10. Giảm đọc file, sửa truyền nhầm hệ màu BGR/RGB & cache SAM encoder**
  - `countzes_core.py`: đọc ảnh 1 lần/iteration; cache SAM features (`sam_cache`) 1 lần/ảnh thay vì gọi `set_image` 5–6 lần; sửa `get_masks_and_boxes` & `preprocess` ở Stage 3 truyền đúng hệ màu (RGB cho SAM, BGR cho `preprocess`).
  - Rút gọn toàn bộ code trùng lặp của 5 entrypoint vào `countzes_core.py`.

---

## P2 — Điểm thuật toán cần kiểm chứng A/B trên GPU

- [ ] **11. Kiểm chứng A/B hai điểm logic có thể ảnh hưởng số đo paper**
  - Đã tách sẵn 2 cờ CLI trong `countzes_core.py` (mặc định giữ nguyên hành vi gốc):
    - `--fix_sses_inversion`: bỏ `1 - sim_clip` trong `SSES`.
    - `--single_instance_mode`: chọn `'paper'` (`1 < c < 2`), `'unit'` (`0.5 <= c <= 1.5`), hoặc `'le1'` (`0 < c <= 1.5`).
  - Cần chạy đối chiếu trên subset nhỏ khi có GPU + weights/data.

---

## Tiêu chí hoàn thành (Acceptance Criteria)
- [x] Tất cả file `.py` đã sửa pass `python -m py_compile`.
- [ ] Chạy thử được cả `--fsoc countr` và `--fsoc DSALVANet` kèm `--data`, `--outdir`, `--visualize` trên máy có GPU + weights/data.
- [ ] Không còn lỗi `RuntimeError` do `eval()`/`no_grad()`/`float64`/`np.float` trên môi trường GPU thực tế.
- [x] Cập nhật xong các checkbox và trạng thái trong [tasks/PROGRESS.md](PROGRESS.md).
