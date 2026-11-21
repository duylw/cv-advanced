# AGENT.md

Hướng dẫn làm việc cho AI Agent và developer trong repository **CountZES**.

## 1. Bản chất dự án
- **CountZES** là framework **inference-only / training-free** cho bài toán **Zero-Shot Object Counting (ZOC)**.
- Pipeline dùng các model pretrained (**CLIP**, **GroundingDINO**, **SAM**, **AnyUp**) để tự động tìm 3 exemplar `[dae, dge, fce]`, rồi đưa vào counter few-shot có sẵn (**CounTR** hoặc **DSALVANet**).
- Không có quy trình training cho CountZES. (Các script train trong `CounTR/` là code gốc đi kèm từ repo CounTR.)

## 2. Quy trình làm việc với Task
1. Đọc [ARCHITECTURE.md](ARCHITECTURE.md) và [tasks/PROGRESS.md](tasks/PROGRESS.md) trước khi sửa code.
2. Mở file task đang thực hiện trong thư mục `tasks/` (hiện tại: [tasks/01-fix-bugs.md](tasks/01-fix-bugs.md)).
3. Sau khi hoàn thành từng hạng mục:
   - Đánh dấu `[x]` vào checklist trong file task tương ứng.
   - Cập nhật trạng thái, ngày/ghi chú và bước tiếp theo trong [tasks/PROGRESS.md](tasks/PROGRESS.md).

## 3. Quy ước kỹ thuật & An toàn code
- **Inference & VRAM:**
  - Mọi nhánh forward của counter (`CounTR`, `DSALVANet`) và các model phụ trợ phải chạy trong `model.eval()` và `with torch.no_grad():` (hoặc `torch.inference_mode()`).
  - Không dùng `ThreadPoolExecutor` để gọi song song cùng một GPU model.
- **Hệ tọa độ Bounding Box (rất dễ nhầm):**
  - `GroundingDINO` và `SAM` (`SSES`, `get_masks_and_boxes`, `box_to_featvec`) dùng **`[x0, y0, x1, y1]`** (`xyxy`).
  - `DSALVANet/utils/data_preprocess.py` (`preprocess`) và `CounTR/demo.py` (`load_image`) mong đợi **`[y0, x0, y1, x1]`** (`yxyx`).
  - Khi sửa code, kiểm tra kỹ bước chuyển đổi `[x0, y0, x1, y1]` ↔ `[y0, x0, y1, x1]`.
- **Bảo toàn hành vi thuật toán gốc:**
  - Các điểm liên quan trực tiếp đến logic bài báo (như `sim_clip = 1 - sim_clip` trong `SSES` hay ngưỡng lọc `1 < c < 2`) **không tự ý thay đổi mặc định** nếu chưa được phép; nên tách thành flag/cấu hình để A/B test với bản gốc.
- **Git & Dung lượng:**
  - Tuyệt đối không commit model weights (`*.pth`, `*.pt`), dữ liệu ảnh trong `data/`, hoặc ảnh output trong `outputs/`.

## 4. Kiểm tra tối thiểu sau mỗi lần sửa
- Chạy kiểm tra cú pháp toàn bộ file Python vừa sửa:
  ```bash
  python -m py_compile <cac_file_da_sua>
  ```
- Kiểm tra `git status` và `git diff` để đảm bảo không sót file rác/cache.
- Nếu chạy trên máy có GPU + weights/data, chạy smoke test trên một số lượng ảnh nhỏ trước khi chạy toàn bộ dataset.
