# Pha 3 — Ablation Study (100% Dataset)

- **Mục tiêu:** Tạo bảng Ablation khoa học **100% toàn bộ tập dữ liệu (không lấy mẫu con)**, mỗi hàng chỉ thêm **đúng 1 thay đổi duy nhất** để cô lập đóng góp thực tế của từng thành phần đã qua cổng Pha 2.
- **Chiến lược tiết kiệm GPU:** Chạy full bảng Ablation trước trên **`MBM` (44 ảnh, ~9–12p/run)** và **`VGG` (200 ảnh, ~35–55p/run)**, sau đó chạy ít nhất cặp đối chiếu cốt lõi (hoặc full bảng tùy tốc độ sau `SDES`) trên **`FSC-147 test` (1,190 ảnh)**.
- **Trạng thái:** `PENDING` (chờ Pha 2)

---

## Ma trận Ablation chuẩn (Mỗi bước +1 thay đổi duy nhất, cố định seed & tham số từ Pha 2)

| ID | Cấu hình | `le1` | `CLIP Fix` | `SDES` | `MD-FCE` | `SMDF` | `MBM` (44) | `VGG` (200) | `FSC-147 test` (1190) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A0` | Paper Baseline (`paper`, `raw`) | ✗ | ✗ | ✗ | ✗ | ✗ | ☐ | ☐ *(Colab `51.765`; chạy EC2 nếu cần đồng bộ tốc độ)* | ☐ |
| `A1` | `+ le1` *(hiệu chỉnh ngưỡng P2)* | ✓ | ✗ | ✗ | ✗ | ✗ | ☐ | ☐ | ☐ *(tùy budget)* |
| `A2` | `+ CLIP Fix` (`MaskCLIP + No-Crop`) | ✓ | ✓ | ✗ | ✗ | ✗ | ☐ | ☐ | ☐ *(tùy budget)* |
| `A3` | `+ SDES` *(Stage 2–3)* | ✓ | ✓ | ✓ | ✗ | ✗ | ☐ | ☐ | ☐ *(tùy budget)* |
| `A4` | `+ MD-FCE` *(Stage 3)* | ✓ | ✓ | ✓ | ✓ | ✗ | ☐ | ☐ | ☐ *(tùy budget)* |
| `A5` | **CountZES+ Full (`+ SMDF`)** | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ |

*(Ghi chú: Nếu ở Pha 2 Bước 2A có thành phần bị loại hoặc giữ lại `paper` thay vì `le1`, cập nhật lại bảng này trước khi bấm chạy full. Có thể bổ sung `A0 + module đơn lẻ` cho thành phần chủ lực nếu cần làm nổi bật thêm.)*

---

## Công việc chi tiết

- [ ] **3.1. Chạy Full Ablation trên `MBM` (44 ảnh × 6 runs ≈ 1 giờ)**
  - Dùng `countzes_mbm.py` với `--stage_samples -1 --visualize` (lưu toàn bộ 44 ảnh để phân tích sâu).
- [ ] **3.2. Chạy Full Ablation trên `VGG` (200 ảnh × 5–6 runs ≈ 4.5 giờ)**
  - Dùng `countzes_vgg.py` với `--stage_samples 15 --visualize` trong `tmux` (nhớ theo dõi `logs/resource_usage.log` và `s/img`).
- [ ] **3.3. Quyết định phạm vi `FSC-147 test` (1,190 ảnh)**
  - Tối thiểu: chạy `A0` (Baseline gốc) vs `A5` (**CountZES+ Full**) trên toàn bộ 1,190 ảnh `FSC-147 test`.
  - Lý tưởng (nếu tốc độ sau `SDES` ~10–11s/img tức ~3.5h/run): chạy đủ các nấc `A0 → A5` trên `FSC-147 test`.
- [ ] **3.4. Lưu lại `summary.json`, `predictions.csv`, `logs/resource_usage.log` cho từng run**
  - Đặt tên `--outdir` nhất quán: `ablation_<dataset>_<A0|A1|A2|A3|A4|A5>`.
