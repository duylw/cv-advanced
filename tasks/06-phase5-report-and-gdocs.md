# Pha 5 — Phân tích, Trực quan hóa & Hoàn thiện Báo cáo

- **Mục tiêu:** Chuyển toàn bộ output thực nghiệm (`summary.json`, `predictions.csv`, `logs/resource_usage.log`, `stage_samples/*.png`) thành số liệu, biểu đồ, Google Doc hoàn chỉnh và báo cáo cuối cùng.
- **Trạng thái:** `PENDING` (chờ Pha 4)

---

## Checklist đầu ra

- [ ] **5.1. Thu thập & đồng bộ kết quả về Local / Drive**
  - `scp` các thư mục `outputs/<run>` (hoặc tải từ Drive) về lưu trữ an toàn; giữ nguyên `summary.json`, `predictions.csv`, `logs/`.
  - Nhớ **Stop EC2 instance** ngay sau khi kéo hết kết quả!
- [ ] **5.2. Phân tích định lượng & Paired Error Analysis**
  - So sánh ảnh-theo-ảnh từ `predictions.csv`: đếm số ảnh thắng/thua/hòa giữa `Baseline` vs `Ours`.
  - Trích `per_class_metrics` (trên `FSC-147 test` 29 classes) xem nhóm nào cải thiện mạnh nhất (thường là class mật độ cao / vật thể nhỏ / nền rối).
  - So sánh tài nguyên (`s/img`, Peak VRAM, Peak RAM) từ `logs/resource_usage.log`.
- [ ] **5.3. Phân tích định tính (`stage_samples` Before / After)**
  - Chọn 4–6 cặp ảnh tiêu biểu: (a) Case thành công rõ rệt (CLIP không còn bắt nhầm nền, `SDES` rải đều box), (b) Case failure còn tồn tại (để viết phần Limitations trung thực).
- [ ] **5.4. Cập nhật Google Doc đa tab (`1gE2K1dUJH4EObOuoqTOkJ_2LEFZoxi7moINtldBdYrw`)**
  - Dùng skill `countzes-drive-eval-to-doc` + `countzes-proposal-gdocs`.
  - Gỡ tiền tố `[Nháp]` trên các tab đã chứng minh hiệu quả, cập nhật số Ablation & Benchmark mới nhất vào tab `Overview` và từng tab đề xuất, chèn ảnh `stage_samples` mới.
- [ ] **5.5. Hoàn thiện Báo cáo / Slide thuyết trình**
  - Cấu trúc gợi ý: (1) Bối cảnh & Bug trong repo gốc → (2) Phân tích 4 điểm nghẽn (CLIP crop/pooling, exemplar trùng vị trí, nền trong FCE, fusion max) → (3) Phương pháp đề xuất (training-free) → (4) Ablation Study (`MBM` + `FSC-147` In-domain) → (5) Benchmark chính (`FSC-147` In-domain + `MBM`, kèm mốc Baseline `FSC-147` Cross-domain) & Tài nguyên → (6) Qualitative & Hạn chế.
