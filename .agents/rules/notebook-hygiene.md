# Jupyter Notebook (.ipynb) Editing Hygiene

Quy ước bắt buộc khi tạo, chỉnh sửa hoặc hướng dẫn thao tác trên các file `.ipynb` (`colab_spike_test.ipynb`, `colab_run_pipeline.ipynb`, v.v.):

## 1. Tách Logic khỏi Notebook
- Giữ toàn bộ logic thuật toán, xử lý model và tương thích thư viện trong module `.py` (`countzes_core.py`).
- Notebook chỉ giữ phần setup môi trường, cấu hình tham số, tải dữ liệu mẫu và gọi hàm từ module `.py`.

## 2. Kiểm tra tính hợp lệ sau khi sửa `.ipynb`
- Sau mỗi lần sửa bất kỳ file `.ipynb` nào, bắt buộc chạy kiểm tra cấu trúc JSON và cú pháp Python AST (bỏ qua dòng magic `!` / `%`):
  ```bash
  python -c "import json, ast; f='<path>.ipynb'; d=json.load(open(f, encoding='utf-8')); [ast.parse('\n'.join((l[:len(l)-len(l.lstrip())]+'pass') if l.lstrip().startswith(('!','%')) else l for l in ''.join(c['source']).splitlines())) for c in d['cells'] if c['cell_type']=='code']; print('VALID')"
  ```

## 3. Tránh xung đột Buffer IDE & Output Bloat
- Khi file `.ipynb` đang mở trong IDE và Agent vừa sửa trực tiếp trên ổ đĩa: luôn nhắc người dùng dùng **File: Revert File** (hoặc đóng tab không lưu rồi mở lại) trước khi bấm Run hoặc `Ctrl+S`, tránh ghi đè buffer cũ lên thay đổi mới.
- Trước khi `git commit` các file `.ipynb`, hướng dẫn hoặc đảm bảo đã **Clear All Outputs** để tránh lưu chuỗi ảnh base64 nặng vào lịch sử Git; lưu ảnh kết quả ra `outputs/` hoặc Google Drive thay vì nhúng trong notebook.
