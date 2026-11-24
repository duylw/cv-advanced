# Pha 3 — Ablation Study (100% Dataset)

- **Mục tiêu:** Tạo bảng Ablation khoa học, **100% toàn bộ tập dữ liệu (không lấy mẫu con)**, cô lập đóng góp thực tế của từng thành phần đã chốt sau Pha 2.
- **Chiến lược tiết kiệm GPU:** Chạy full bảng Ablation trước trên **`VGG` (200 ảnh, ~55p/run)** và **`MBM` (44 ảnh, ~12p/run)**. Sau đó cân nhắc chạy cặp đối chiếu cốt lõi (hoặc full bảng) trên **`FSC-147 test` (1,190 ảnh)**.
- **Trạng thái:** `PENDING` (chờ Pha 2)

---

## Ma trận cấu hình Ablation (Bắt buộc cố định seed & tham số từ Pha 2)

| ID | Cấu hình | `MaskCLIP+NoCrop` | `SDES` | `MD-FCE` | `SMDF` | `VGG` (200) | `MBM` (44) | `FSC-147 test` (1190) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A0` | Reproduced Baseline (`raw`) | ✗ | ✗ | ✗ | ✗ | ☐ (đã có `51.765` trên Colab; chạy lại EC2 nếu đổi code) | ☐ | ☐ |
| `A1` | + CLIP Spatial Fix | ✓ | ✗ | ✗ | ✗ | ☐ | ☐ | ☐ |
| `A2` | + SDES | ✓ | ✓ | ✗ | ✗ | ☐ | ☐ | ☐ *(tùy budget)* |
| `A3` | + MD-FCE | ✓ | ✓ | ✓ | ✗ | ☐ | ☐ | ☐ *(tùy budget)* |
| `A4` | **Ours (Full = + SMDF)** | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ |

*(Mẹo: nếu cần chứng minh tác dụng đơn lẻ khi tắt từng nhánh, có thể thêm 1 dòng `A4 w/o SDES` trên `MBM`/`VGG`.)*

---

## Công việc chi tiết

- [ ] **3.1. Chạy Full Ablation trên `MBM` (44 ảnh × 5 runs ≈ 1 giờ)**
  - Dùng `countzes_mbm.py` với `--stage_samples -1 --visualize` (vì chỉ 44 ảnh, lưu hết để phân tích sâu).
- [ ] **3.2. Chạy Full Ablation trên `VGG` (200 ảnh × 4–5 runs ≈ 4.5 giờ)**
  - Dùng `countzes_vgg.py` với `--stage_samples 15 --visualize` trong `tmux` (nhớ theo dõi `logs/resource_usage.log` và `s/img`).
- [ ] **3.3. Quyết định phạm vi `FSC-147 test` (1,190 ảnh)**
  - Tối thiểu: chạy `A0` (hoặc `A1`) vs `A4 (Full)` trên `FSC-147 test` (1,190 ảnh) để có đối chiếu trực tiếp quy mô lớn.
  - Lý tưởng (nếu tốc độ sau `SDES` giảm còn ~10–12s/img và còn quota EC2): chạy đủ `A0 → A4` trên `FSC-147 test`.
- [ ] **3.4. Lưu lại `summary.json`, `predictions.csv`, `logs/resource_usage.log` cho từng run**
  - Đặt tên `--outdir` nhất quán: `ablation_<dataset>_<A0|A1|A2|A3|A4>`.
