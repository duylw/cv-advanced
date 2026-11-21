---
trigger: always_on
description: Strict safety guardrails for running in Turbo permission mode to prevent destructive commands, data/weight loss, credential leaks, and unintended remote/system changes.
---

# Turbo Mode Safety Rule

Môi trường đang bật **Turbo permission mode** (lệnh terminal và thao tác file có thể được tự động phê duyệt). Agent **bắt buộc** tuân thủ các giới hạn an toàn sau trong mọi lượt thực thi:

## 1. Giới hạn phạm vi (Workspace Boundary)
- Chỉ đọc, tạo, sửa hoặc xóa file nằm bên trong workspace dự án hiện tại.
- **Không** chỉnh sửa, xóa hay đọc dữ liệu nhạy cảm ngoài workspace (`~/.ssh`, `~/.aws`, `~/.gemini`, System32, `/etc`, v.v.).

## 2. Bảo vệ Dữ liệu & Model Weights
- **Không** xóa, ghi đè hoặc di chuyển các thư mục dữ liệu và checkpoint (`data/`, `DSALVANet/checkpoints/`, `CounTR/output_allnew_dir/`, `GroundingDINO/weights/`, `weights/`), trừ khi người dùng chỉ định rõ ràng từng đường dẫn cụ thể.
- **Không** commit weights (`*.pth`, `*.pt`), dữ liệu ảnh trong `data/`, hoặc ảnh output trong `outputs/` lên Git.

## 3. Các lệnh phá hủy — Phải hỏi xác nhận trước khi chạy
Vì Turbo bỏ qua bước hỏi quyền của UI, Agent **phải dừng lại và hỏi người dùng bằng văn bản** trước khi thực thi bất kỳ hành động nào thuộc nhóm sau:
- **Xóa file/thư mục hàng loạt hoặc ép buộc:** `rm -rf`, `Remove-Item -Recurse -Force`, `del /s /q`, `rmdir /s`.
- **Git phá hủy dữ liệu hoặc lịch sử:** `git reset --hard`, `git clean -fd` / `-fdx`, `git checkout -- .`, `git restore .`, `git push --force` / `-f`, `git branch -D`.
- **Thao tác tiến trình hệ thống:** `taskkill /F`, `kill -9`, `Stop-Process` với process không do chính phiên làm việc hiện tại tạo ra.
- **Tải và chạy trực tiếp script từ Internet:** `curl ... | bash`, `wget ... | sh`, `iex (irm ...)`.

## 4. Bảo mật Secrets & Git Remote
- **Không** đọc, in ra hội thoại, hoặc ghi vào code/commit các file bí mật (`.env`, `*.pem`, `*.key`, AWS access key, GitHub token).
- **Không** tự ý `git push`, đổi `git remote`, hoặc xóa branch remote nếu người dùng chưa yêu cầu rõ trong lượt hiện tại.

## 5. An toàn Môi trường & Tài nguyên GPU (Local / AWS)
- Không cài/gỡ package vào Python toàn cục (global/system) ngoài môi trường ảo (`conda`/`venv`) của dự án.
- Trước khi chạy pipeline trên toàn bộ dataset lớn (ví dụ FSC-147 1,190 ảnh), ưu tiên kiểm tra cú pháp (`py_compile`) và smoke test trên quy mô nhỏ trước, hoặc hỏi xác nhận nếu tác vụ chạy dài/tốn chi phí cloud.
