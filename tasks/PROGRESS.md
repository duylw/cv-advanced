# Task Progress Tracker

Bảng theo dõi tiến độ các task trong dự án **CountZES**.

## Tổng quan
| ID | Task | Trạng thái | Ghi chú |
| :--- | :--- | :--- | :--- |
| `01` | [Sửa các bug trong pipeline & DSALVANet](01-fix-bugs.md) | `TODO` | Đã audit xong toàn bộ lỗi kỹ thuật & logic; sẵn sàng thực hiện. |

---

## Trạng thái hiện tại
- **Task đang active:** `01-fix-bugs.md`
- **Môi trường local hiện tại:**
  - Chưa tải datasets ảnh và checkpoints `.pth` về `data/`, `DSALVANet/checkpoints/`, `CounTR/output_allnew_dir/`, `GroundingDINO/weights/`.
  - Đã tách remote gốc thành `upstream`; chờ cấu hình remote `origin` tới GitHub cá nhân để push code lên máy GPU.

## Lịch sử cập nhật
- **2026-11-21:** Review toàn bộ codebase, lập tài liệu `AGENT.md`, `ARCHITECTURE.md`, `tasks/01-fix-bugs.md` và `tasks/PROGRESS.md`.
