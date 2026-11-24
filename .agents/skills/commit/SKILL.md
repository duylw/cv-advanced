---
name: commit
description: >-
  Use this skill whenever the user invokes /commit or asks to inspect, filter,
  validate, stage, and create clean Git commits for the CountZES repository.
---

# Clean Git Commit Skill (`/commit`)

Quy trình tự động kiểm tra, lọc file hợp lệ, xác minh code/notebook và tạo commit sạch khi người dùng gõ **`/commit`** (hoặc yêu cầu commit thay đổi).

---

## 1. Kiểm tra hiện trạng (Inspect First)
Chạy lệnh read-only để nắm toàn bộ thay đổi hiện có:
```powershell
git status --short; git diff --stat
```
- Nếu working tree sạch (không có file `M`, `A`, `D`, hoặc `??` cần commit): thông báo ngắn gọn và dừng.

---

## 2. Lọc file hợp lệ & Bảo vệ dữ liệu (Tuyệt đối không `git add -A` mù quáng)

Phân loại mọi file thay đổi/chưa theo dõi trước khi `git add`:

- **KHÔNG bao giờ commit:**
  - Model weights: `*.pth`, `*.pt`, `DSALVANet/checkpoints/`, `CounTR/output_allnew_dir/`, `GroundingDINO/weights/`, `weights/`.
  - Dữ liệu ảnh & output chạy: `data/**/Images/`, `data/**/images_384_VarV2/`, `data/**/gt_density_map/`, `outputs/*` (trừ file `outputs/output.txt` nếu có).
  - Bí mật & tài liệu/nháp cá nhân local: `.env`, `*.pem`, `*.key`, token, `countzes.pdf`, `IMPROVEMENTS.md`, `__pycache__/`, `*.pyc`.
- Nếu xuất hiện file rác/tạm mới chưa có trong `.gitignore` (vd `.pyc`, log tạm, ảnh test cục bộ), bổ sung vào `.gitignore` (nếu hợp lý) hoặc bỏ qua không `git add` và báo rõ trong phản hồi.

---

## 3. Kiểm tra hợp lệ trước khi Commit (Pre-Commit Checks)

1. **Nếu có sửa file `.py`:**
   - Chạy kiểm tra cú pháp:
     ```powershell
     python -m py_compile resource_monitor.py countzes_core.py countzes_fsc147.py countzes_carpk.py countzes_persense_D.py countzes_mbm.py countzes_vgg.py
     ```
   - Nếu vừa thêm/đổi CLI flags trong `countzes_core.py` hoặc `countzes_*.py`, đối chiếu nhanh xem `USAGE.md` đã đồng bộ chưa (theo skill `countzes-usage-docs`).
2. **Nếu có sửa file `.ipynb` (theo rule `notebook-hygiene.md`):**
   - Kiểm tra JSON + Python AST hợp lệ (bỏ qua dòng magic `!` / `%`).
   - Kiểm tra xem cell code có còn `outputs` hoặc ảnh base64 nhúng sẵn không. Nếu notebook vừa chạy trên Colab sinh output nặng:
     - Hoặc hỏi/dọn sạch `outputs` + đặt `execution_count = null` (chỉ khi file đó không đang có thay đổi logic quan trọng chưa lưu trên IDE) rồi kiểm tra lại AST trước khi `git add`.
     - Nếu người dùng vừa chỉnh sửa notebook đang mở trên IDE, nhắc **File → Revert File** nếu đĩa và IDE có thể lệch nhau.

---

## 4. Stage & Commit theo Conventional Commits

- Stage tường minh từng đường dẫn hợp lệ (`git add <file1> <file2> ...`).
- Nếu các thay đổi gồm nhiều mảng hoàn toàn độc lập (ví dụ: sửa pipeline Python + thêm skill `.agents/`), có thể tách thành các commit nhỏ gọn theo đúng chủ đề, hoặc gộp 1 commit rõ nghĩa nếu cùng một tác vụ.
- Viết commit message súc tích theo chuẩn **Conventional Commits**:
  - `feat(<scope>): ...` (tính năng, option, notebook hoặc skill mới)
  - `fix(<scope>): ...` (sửa lỗi)
  - `docs(<scope>): ...` (tài liệu `USAGE.md`, `README.md`, `ARCHITECTURE.md`)
  - `chore(<scope>): ...` (`.gitignore`, cấu hình, `.agents/` rules/skills)

---

## 5. Xác minh & Quy tắc Push

1. Chạy `git status --short; git log -n 2 --stat` để xác nhận commit đã tạo sạch sẽ, đúng file mong muốn.
2. **Không tự ý `git push`** trừ khi người dùng yêu cầu rõ trong cùng lượt (ví dụ `/commit và push`, hoặc đã yêu cầu push trước đó) — tuân thủ `.agents/rules/turbo-safety.md`.
3. Tóm tắt ngắn gọn: hash commit, danh sách file đã commit, các file bỏ qua/ignored (nếu có), và nhắc `git push origin main` nếu người dùng sắp kéo code trên Colab.
