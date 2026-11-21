# Task 01: Sửa các bug trong pipeline CountZES & DSALVANet

- **Trạng thái:** `TODO`
- **Mục tiêu:** Khắc phục các lỗi crash, tràn VRAM, sai tọa độ feature, sai checkpoint và lỗi tổng hợp metric trên toàn bộ 5 script đánh giá (`countzes_fsc147.py`, `countzes_carpk.py`, `countzes_persense_D.py`, `countzes_mbm.py`, `countzes_vgg.py`) và các module phụ trợ trong `DSALVANet/`.

---

## P0 — Bug nghiêm trọng (Crash / Sai kết quả / Tràn VRAM)

- [ ] **1. Sửa xung đột `argparse` khi chạy `--fsoc DSALVANet`**
  - **File:** cả 5 file `countzes_*.py` (trong `get_arguments()` và nhánh `elif args.fsoc == 'DSALVANet':` ở `main()`).
  - **Vấn đề:** Tạo `parser_input = argparse.ArgumentParser()` thứ hai và gọi `parse_args()` sẽ crash `unrecognized arguments` nếu truyền kèm `--data`, `--outdir`, `--sam_type`, v.v.
  - **Cách sửa:** Gộp `--weight` (default `./DSALVANet/checkpoints/checkpoint_200.pth`) vào `get_arguments()`, xóa `parser_input`.

- [ ] **2. Bật `model.eval()`, `map_location`, và `torch.no_grad()` cho `DSALVANet`**
  - **File:** `DSALVANet/utils/model_helper.py` (`build_model`), `DSALVANet/models/DSALVANet.py` (`LVAM.reparameterize`, `MDM.reset_parameters`), và cả 5 file `countzes_*.py`.
  - **Vấn đề:**
    - `build_model` chỉ gọi `model.backbone.eval()`, không gọi `model.eval()`, khiến `LVAM` giữ `self.training=True` (gọi `.cuda()` cứng và thêm nhiễu ngẫu nhiên khi inference).
    - `torch.load(weight_path)` thiếu `map_location=device`.
    - Các lời gọi `counter_model(query, supports)` thiếu `with torch.no_grad():`, gây giữ computation graph trên VRAM.
  - **Cách sửa:** Thêm `map_location=device` và `model.eval()` trong `build_model`; bọc `with torch.no_grad():` quanh tất cả bước inference `counter_model`; sửa `torch.randn_like(std).to(std.device)` trong `LVAM.reparameterize`.

- [ ] **3. Sửa lỗi tọa độ & kích thước lưới trong `box_to_featvec` (Stage 3 — FCE)**
  - **File:** cả 5 file `countzes_*.py` (hàm `box_to_featvec` ở Stage 3).
  - **Vấn đề:** `anyup` trả về `hr_features` kích thước `(C, 256, 256)`, nhưng code lại clamp tọa độ box gốc `(W, H)` vào `256` rồi scale theo `(w_feat, h_feat)` của `full_feat` (`64 x 64`).
  - **Cách sửa:** Clamp box theo kích thước ảnh gốc `(W, H)` rồi scale theo `hr_features.shape` (`h_hr, w_hr = hr_features.shape[1], hr_features.shape[2]`).

- [ ] **4. Sửa checkpoint CounTR trên FSC-147**
  - **File:** `countzes_fsc147.py` (dòng load checkpoint CounTR trong `main()`).
  - **Vấn đề:** Đang load `./CounTR/output_allnew_dir/CARPK.pth` thay vì `./CounTR/output_allnew_dir/FSC147.pth`.

- [ ] **5. Sửa lỗi không cộng dồn metric `count_item`, `mae`, `mse` giữa các class**
  - **File:** `countzes_mbm.py`, `countzes_vgg.py`, `countzes_persense_D.py`.
  - **Vấn đề:** `count_item, mae, mse` là kiểu `int` truyền vào hàm `countzes(...)` trong vòng lặp `for obj_name in os.listdir(...)`, nên các lần gọi sau không cộng dồn metric của các thư mục class trước.
  - **Cách sửa:** Cho `countzes(...)` trả về `(count_item, mae, mse)` và cập nhật lại biến trong `main()`.

- [ ] **6. Bỏ `ThreadPoolExecutor` cho GPU inference ở Stage 2**
  - **File:** cả 5 file `countzes_*.py` (vòng lặp gọi `process_box`).
  - **Vấn đề:** Chạy 12 thread gọi cùng 1 model GPU đồng thời không tăng tốc và dễ xung đột/OOM.
  - **Cách sửa:** Chuyển sang vòng lặp tuần tự với `with torch.no_grad():`.

---

## P1 — Bug phụ, Tương thích & Tối ưu cơ bản

- [ ] **7. Sửa `--visualize`, `--ckpt` và đường dẫn lưu output**
  - Đổi `--visualize` từ `type=bool` sang `action='store_true'`.
  - Dùng `args.ckpt` khi `args.sam_type == 'vit_h'` (fallback mặc định `data/sam_vit_h_4b8939.pth` nếu không truyền).
  - Dùng `output_path` thay vì hardcode `'./outputs/countzes_fsc147'` trong `countzes_fsc147.py`.

- [ ] **8. Sửa `preprocess` mutate list đầu vào & thứ tự resize**
  - **File:** `DSALVANet/utils/data_preprocess.py`.
  - **Cách sửa:** Copy `ori_boxes = list(ori_boxes)` trước khi pad đủ 3 box; sửa `cv2.resize(ori_img, (w_rsz, h_rsz))`; đổi `rsz_boxes` sang `torch.float32`.

- [ ] **9. Thay `np.float` đã deprecated (NumPy >= 1.24)**
  - **File:** `DSALVANet/utils/fsol_modules.py`, `DSALVANet/utils/PerSense_countr.py`, `DSALVANet/utils/PerSense_modules.py`.
  - **Cách sửa:** Đổi `np.float` thành `float` (hoặc `np.float32`).

- [ ] **10. Giảm đọc file & chạy SAM encoder dư thừa trong mỗi vòng lặp ảnh**
  - Đọc ảnh 1 lần/iteration; đọc `FSC-147-D.json` (nếu dùng) 1 lần ngoài vòng lặp.
  - Cache SAM image embedding cho `test_image` 1 lần/ảnh thay vì gọi `predictor.set_image(test_image)` lặp lại 5–6 lần.

---

## P2 — Điểm thuật toán cần kiểm chứng A/B trước khi quyết định

- [ ] **11. Kiểm chứng A/B hai điểm logic có thể ảnh hưởng số đo paper**
  - `sim_clip = 1 - sim_clip` trong hàm `SSES` (đảo ngược similarity thành dissimilarity).
  - Điều kiện lọc single-instance `if 1 < c < 2:` (warning ghi `count<=1` nhưng code chỉ giữ `(1, 2)`).
  - **Yêu cầu:** Tách thành flag CLI (mặc định giữ theo code gốc để đối chiếu, bật cờ để test bản sửa) khi kiểm thử trên subset nhỏ.

---

## Tiêu chí hoàn thành (Acceptance Criteria)
- Tất cả file `.py` đã sửa pass `python -m py_compile`.
- Chạy thử được cả `--fsoc countr` và `--fsoc DSALVANet` kèm các flag `--data`, `--outdir`, `--visualize` mà không bị lỗi `argparse`.
- Không còn lỗi `RuntimeError` do `eval()`/`no_grad()`/`float64`/`np.float` trên môi trường test.
- Cập nhật xong các checkbox và trạng thái trong [tasks/PROGRESS.md](PROGRESS.md).
