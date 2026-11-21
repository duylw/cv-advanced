---
name: safe-turbo-execution
description: >-
  Use this skill when running potentially risky, destructive, system-modifying, Git-remote,
  or heavy GPU/AWS operations while Turbo permission mode is enabled.
---

# Safe Turbo Execution Runbook

Quy trình kiểm tra an toàn trước và sau khi thực thi các tác vụ có rủi ro khi đang bật **Turbo permission mode**.

## 1. Phân loại mức độ rủi ro trước khi hành động

| Mức độ | Ví dụ hành động | Cách xử lý bắt buộc |
| :--- | :--- | :--- |
| **Xanh (An toàn)** | Đọc file, tìm kiếm, `git status`, `git diff`, `python -m py_compile`, chạy test nhỏ read-only | Thực thi bình thường. |
| **Vàng (Cẩn trọng)** | Sửa code nhiều file, `conda/pip install` trong env dự án, `git commit` | Kiểm tra đúng env/nhánh trước khi chạy; kiểm tra `git diff` trước khi commit. |
| **Đỏ (Nguy hiểm)** | Xóa file/thư mục (`rm -rf`, `Remove-Item -Recurse`), `git reset --hard` / `clean` / `push --force`, đụng `data/` hoặc weights, sửa ngoài workspace, chạy script từ URL | **DỪNG LẠI**, không gọi tool; liệt kê rõ lệnh/tác động và hỏi người dùng xác nhận bằng văn bản. |

## 2. Quy trình thực hiện cho thao tác mức Vàng & Đỏ
1. **Dry-run / Kiểm tra trước (Inspect first):**
   - Trước khi xóa hoặc dọn dẹp: chạy lệnh liệt kê danh sách file sẽ bị tác động (`Get-ChildItem`, `ls`, `git clean -n`) để kiểm chứng phạm vi.
   - Trước khi commit/push: chạy `git status` và `git diff --stat` để chắc chắn không dính `.pth`, `.pt`, `.env`, `*.pem`, `outputs/`, hay `data/`.
2. **Bảo vệ `data/` và checkpoints:**
   - Kiểm tra đường dẫn đích không trùng với `data/`, `DSALVANet/checkpoints/`, `CounTR/output_allnew_dir/`, `GroundingDINO/weights/`.
3. **Kiểm tra môi trường trước khi cài/chạy:**
   - Xác nhận đang ở đúng conda/venv (`conda info --envs` hoặc `which python`) trước khi `pip install`.
   - Khi chạy benchmark trên GPU/AWS, test thử số lượng ảnh nhỏ trước khi chạy toàn bộ dataset.
4. **Xác minh sau thực thi:**
   - Chạy `git status` để chắc chắn không có file ngoài ý muốn bị thay đổi hoặc xóa nhầm.
