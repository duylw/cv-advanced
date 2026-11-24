---
name: countzes-drive-eval-to-doc
description: >-
  Use this skill whenever the user finishes an evaluation run on Google Colab and
  asks to automatically read logs, metrics, predictions, and stage images from
  Google Drive (MyDrive/CountZES/outputs/), analyze the results against baseline,
  and document the findings into the multi-tab Google Doc.
---

# CountZES Drive Read, Analyze & Document Skill

Skill này định nghĩa quy trình khép kín để **đọc kết quả thử nghiệm từ Google Drive** $\rightarrow$ **phân tích số liệu & ảnh từng giai đoạn** $\rightarrow$ **cập nhật báo cáo vào Google Doc nhiều tab** (kết hợp cùng skill `gdrive`, `gdocs`, và `countzes-proposal-gdocs`).

---

## 1. Cấu trúc Log & Evidence đang được lưu sẵn trên Google Drive

Mọi script đánh giá (`countzes_vgg.py`, `countzes_mbm.py`, `countzes_fsc147.py`, `countzes_carpk.py`, `countzes_persense_D.py`) đều gọi `run_recorded_evaluation()` trong `countzes_core.py` và lưu vào `/content/drive/MyDrive/CountZES/outputs/<OUTDIR>/`:

| File / Thư mục | Nội dung lưu lại | Ghi chú |
| :--- | :--- | :--- |
| `run_config.json` | Dataset, số mẫu, `sample_mode`, `sample_seed`, toàn bộ tham số, `git_commit`, CPU/RAM/GPU | Dùng để xác nhận điều kiện thí nghiệm |
| `summary.json` | `MAE`, `RMSE`, `succeeded`/`failed`, `avg_sec_per_image`, `per_class_metrics`, `resource_eval` | Dùng để lập bảng tổng hợp |
| `predictions.csv` / `.jsonl` | Theo từng ảnh: `gt_count`, `pred_count`, `abs_error`, `gdino_boxes_n`, `dae`, `count_stage2`, `pseudo_gt_kde`, `dge`, `count_stage3`, `single2_n`, `fce`, `duration_sec`, `status` | Dùng để tìm ảnh tốt nhất / lỗi nặng nhất & so sánh ghép cặp |
| `logs/resource_usage.log` / `.jsonl` | Thời gian, CPU %, System RAM, RSS, VRAM peak của `load_models` & `eval_*` | Ghi nối tiếp (`append`) nếu chạy lại trùng `OUTDIR` |
| `stage_samples/*.png` | Lưới `2×4` từng bước (`S1a → S3b → Final`) | Dùng kiểm tra định tính & chèn vào Google Doc |
| `<DATASET>/` | Ảnh density overlay kết quả cuối (khi bật `--visualize`) | Dùng minh họa bổ sung |

### Lưu ý tránh ghi đè nhầm kết quả giữa các lần thử
- Luôn đặt `OUTDIR` riêng cho từng thí nghiệm (tránh chạy đè lên cùng 1 tên thư mục nếu không muốn mất `summary.json`, `predictions.csv`, `stage_samples/` cũ).
- Nếu chạy trên AWS EC2 (không có Drive mount), cần đồng bộ `./outputs/<OUTDIR>/` lên Google Drive (hoặc copy về máy local) trước khi dùng luồng tự đọc Drive.

---

## 2. Yêu cầu công cụ trước khi chạy tự đọc Drive
- Cần cài sẵn và đăng nhập cả 2 plugin:
  1. **`gdrive`** (`antigravity-plugins-official/plugin/gdrive`) để liệt kê thư mục `CountZES/outputs/` và tải file log/ảnh từ Drive về thư mục tạm.
  2. **`gdocs`** (`antigravity-plugins-official/plugin/gdocs`) để cập nhật kết quả phân tích vào Google Doc.
- Nếu `gdrive` chưa được cài hoặc chưa đăng nhập, dừng lại hướng dẫn người dùng cài/đăng nhập trước rồi mới tiếp tục.

---

## 3. Quy trình thực hiện (4 Bước)

### Bước 1: Tìm & Tải log + ảnh từ Drive về thư mục Scratch
- **Tuyệt đối không** tải file output từ Drive vào workspace code (`d:\UIT\CV\CountZES`); luôn lưu vào `<appDataDir>/brain/<conversation-id>/scratch/eval_<OUTDIR>/`.
- Dùng `gdrive` tìm thư mục `CountZES` → `outputs` → `<OUTDIR>` (và đối chiếu với thư mục Baseline cần so sánh nếu chưa tải trước đó).
- Tải về scratch:
  1. `run_config.json`, `summary.json`, `predictions.csv`, `logs/resource_usage.log`.
  2. Một số ảnh trong `stage_samples/` (ưu tiên các ảnh nằm ở **Top sai số lớn nhất**, **Top chính xác nhất**, và các ảnh **cùng filename** với Baseline để đối chiếu Trước/Sau).

### Bước 2: Phân tích định lượng & định tính
1. **Kiểm tra tính hợp lệ của lượt chạy:**
   - Kiểm tra `failed == 0` (hoặc liệt kê rõ ảnh nào có `status != "ok"`).
   - Kiểm tra `timestamp_utc`, `num_samples`, `git_commit` trong `run_config.json`.
2. **Phân tích số liệu từ `summary.json` & `predictions.csv`:**
   - Tổng hợp `MAE`, `RMSE`, `avg_sec_per_image`, peak VRAM & System RAM.
   - Nếu là tập đa lớp (`per_class_metrics`), xác định Top class tốt nhất và Top class sai số cao nhất.
   - Soi các chỉ số trung gian trong `predictions.csv`: các ảnh có `gdino_boxes_n <= 1`, `single1_n == 0` hoặc `single2_n == 0`, và độ trôi giữa `count_stage2` → `pseudo_gt_kde` → `count_stage3` → `pred_count`.
3. **Kiểm tra trực quan ảnh lưới `2×4`:**
   - Mở trực tiếp các file `.png` trong `stage_samples/` bằng công cụ đọc file ảnh để xác nhận từng exemplar (`DAE`, `DGE`, `FCE`) có bám trúng vật thể đơn lẻ hay trượt ra nền/viền/cụm nhiều vật thể.

### Bước 3: Đối chiếu công bằng với Baseline
- **Kiểm tra khớp tập mẫu trước khi kết luận:**
  - Hai lần chạy chỉ so sánh trực tiếp được khi khớp `dataset`, `split`, và tập `filename` trong `predictions.csv`.
  - Nếu 2 lần chạy có số mẫu khác nhau, phải lọc giao nhau theo đúng tập `filename` chung trước khi tính `MAE` / `RMSE` đối chiếu, và nêu rõ số lượng mẫu chung đã so sánh.
- Trình bày trung thực cả **những mẫu/class cải thiện** lẫn **những mẫu bị tụt giảm (regressions)** nếu có.

### Bước 4: Cập nhật vào Google Doc theo chuẩn `countzes-proposal-gdocs`
- Kích hoạt skill **`countzes-proposal-gdocs`**:
  - Không viết code snippet, tên biến/hàm, hay danh sách cờ/tham số kỹ thuật vào nội dung tài liệu.
  - Cập nhật **Tab 1 (`Overview`)** với bảng tổng kết kết quả mới nhất.
  - Tạo mới (hoặc cập nhật) **Tab riêng của đề xuất tương ứng** với phần phân tích dễ hiểu, bảng so sánh Baseline vs Đề xuất, và **chèn trực tiếp ảnh `stage_samples/*.png` Trước / Sau** đã tải về scratch.
- Kiểm tra lại tài liệu bằng `gdocs readonly read` + `list-images` và gửi **link clickable** tới đúng tab vừa cập nhật kèm bản tóm tắt ngắn cho người dùng.
