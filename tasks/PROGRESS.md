# Task Progress Tracker

Bảng theo dõi tiến độ các task trong dự án **CountZES**.

## Tổng quan
| ID | Task | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :--- |
| `01` | [Sửa các bug trong pipeline & DSALVANet](01-fix-bugs.md) | `READY_FOR_GPU_TEST` | Đã sửa xong toàn bộ P0 & P1, tách `countzes_core.py`, pass `py_compile`. Chờ test runtime & A/B P2 trên máy GPU. |

---

## Trạng thái hiện tại
- **Task đang active:** `01-fix-bugs.md` (còn phần chạy kiểm thử thực tế trên máy GPU AWS + A/B test 2 cờ P2).
- **Môi trường local hiện tại:**
  - Chưa cài PyTorch/Conda và chưa tải datasets ảnh / checkpoints `.pth` trên máy Windows local.
  - Repo đã cấu hình `origin` → `https://github.com/duylw/cv-advanced.git` và `upstream` → `https://github.com/Muhammad-Ibraheem-Siddiqui/CountZES.git`.

## Lịch sử cập nhật
- **2026-11-21:** Review codebase; tạo `AGENT.md`, `ARCHITECTURE.md`, `tasks/`, và `.agents/` (Turbo safety rule/skill/hook).
- **2026-11-21:** Hoàn thành phần sửa code Task 01 (`countzes_core.py`, 5 entrypoint `countzes_*.py`, `DSALVANet/`, `CounTR/`) và kiểm tra cú pháp `py_compile` thành công.
