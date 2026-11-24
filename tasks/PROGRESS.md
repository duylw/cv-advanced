# Task Progress Tracker — CountZES

> Master tracker gọn cho toàn bộ chiến thuật từ hiện tại → hoàn thành báo cáo.
> Khi làm việc ở pha nào, chỉ cần đọc `PROGRESS.md` + đúng file pha đó để tiết kiệm token.

## Lộ trình tổng thể

| ID | Pha / Tài liệu | Trạng thái | Mục tiêu cốt lõi |
| :--- | :--- | :--- | :--- |
| `01` | [Sửa bug pipeline & DSALVANet](01-fix-bugs.md) | `VERIFIED_DSALVANET` | P0/P1 & P2 A/B xong (`fix_sses_inversion=True`, `le1` thắng `20.21` MAE); còn `--fsoc countr` nếu dùng CounTR |
| `02` | [Pha 1: Code đề xuất (SDES, MD-FCE, SMDF)](02-phase1-implement-proposals.md) | `DONE` | Đã code (`a75c9c5`), docs (`61aabf4`) & pass smoke EC2 `5/5` (`57.26s`, VRAM `7.93 GB`) |
| `03` | [Pha 2: Dev Validation (Batch nhỏ)](03-phase2-dev-validation.md) | `NEXT` | Bước 2A thử độc lập từng module trên `B0` (`val n=58` + `MBM`); Bước 2B cộng dồn 1-thay-đổi/nấc |
| `04` | [Pha 3: Ablation Study (100% Dataset)](04-phase3-ablation-study.md) | `PENDING` | Chạy bảng Ablation 1-thay-đổi/hàng (`A0 → A5`) trên `MBM` (44) + `VGG` (200) (+ `FSC-147 test`) |
| `05` | [Pha 4: Benchmark chính (3 Domain)](05-phase4-main-benchmark.md) | `PENDING` | So Baseline vs Ours trên `FSC-147 test`, `CARPK test`, `VGG` + `MBM` |
| `06` | [Pha 5: Phân tích & Báo cáo / GDocs](06-phase5-report-and-gdocs.md) | `PENDING` | Tổng hợp số liệu, ảnh `stage_samples`, cập nhật Google Doc & viết báo cáo |
| `ref` | [EC2 Datasets & Run Cheatsheet](ec2-setup.md) | `ACTIVE_REF` | Lệnh `tmux`, tải `MBM`/`VGG`/`CARPK` lên EC2, `scp` output, nhắc tắt máy |

---

## Baseline & Kết quả đã đo (Reference Anchor)

| Nguồn / Run | Dataset (N) | Cài đặt | MAE | RMSE | Tốc độ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Paper (báo cáo gốc) | `FSC-147 test` (1190) | DSALVANet in-domain / cross | 15.77 / 21.09 | 91.40 / 110.14 | — |
| Paper (báo cáo gốc) | `CARPK` (459) / `PerSense-D` | DSALVANet | 7.24 / 12.29 | 9.22 / 28.32 | — |
| Paper (báo cáo gốc) | `VGG` (200) / `MBM` (44) | DSALVANet | 45.55 / 22.16 | 74.41 / 40.06 | — |
| Colab (`baseline_vgg_paper`) | `VGG` (200) | `raw`, `fix_sses_inversion=True` | 51.765 | 151.3083 | — |
| Colab (`baseline_vgg_paper_exact`) | `VGG` (200) | `raw`, `github_sses_inversion` | 69.465 | 89.3219 | — |
| Colab (`stratified_n29`, seed=42) | `FSC-147 test` (29) | `raw` | 13.4138 | 38.3428 | 15.56s/img |
| Colab (`stratified_n29`, seed=42) | `FSC-147 test` (29) | `maskclip` + `clip_no_crop` | **11.2069** | **29.9027** | 15.78s/img |
| EC2 T4 (`ec2_smoke_n5`, seed=42) | `FSC-147 test` (5) | `maskclip` + `clip_no_crop` | 4.2000 | 5.3852 | 16.94s/img |
| EC2 T4 (`p2_val29`, seed=42) | `FSC-147 val` (29) | `github_inv` vs `paper` vs `unit` vs `le1` | 26.069 / 23.448 / 21.966 / **20.207** | 88.44 / 86.15 / 85.67 / **85.20** | 9.85–13.27s/img |
| EC2 T4 (`ec2_smoke_all_proposals_n5`) | `FSC-147 val` (5) | `le1` + `maskclip+nocrop` + `sdes+md_fce+smdf` | 8.0000 | 15.3753 | 11.45s/img |

---

## Ngân sách Dataset & Thời gian ước lượng (EC2 `g4dn.xlarge` — T4 ~16s/img)

| Dataset | Số ảnh (Split) | Trên EC2? | ~Thời gian / 1 lượt Full | Vai trò |
| :--- | :---: | :---: | :---: | :--- |
| `MBM` | 44 | Chưa | ~12 phút | Dev validation + Ablation (Cell) |
| `FSC-147 val` (`n=58`) | 58 (2/class, seed 42) | Sẵn (`FSC-147`) | ~16 phút | Dev validation (không rò rỉ `test`) |
| `VGG` | 200 | Chưa | ~55 phút | Full Ablation + Main Benchmark (Cell) |
| `CARPK test` | 459 | Chưa | ~2.1 giờ | Main Benchmark (Aerial/Vehicle) |
| `FSC-147 test` | 1,190 | Sẵn (`FSC-147`) | ~5.5 giờ (chưa SDES) | Main Benchmark (Generic) + Ablation mở rộng |

---

## Trạng thái hiện tại & Việc làm ngay
- **Đã xong:**
  - Sửa `GroundingDINO/bertwarper.py` cho `transformers >= 5.x`; `MaskCLIP + No-Crop` đã cài & thắng `raw` trên `n=29` (`11.21` vs `13.41` MAE).
  - Setup xong EC2 `g4dn.xlarge` (`/opt/pytorch`, `FSC-147`, `checkpoint_200.pth`) + smoke test `5/5` pass + verify P2 (`le1` MAE `20.2069` vs `paper` `23.4483` vs `github_inv` `26.069`).
  - Có sẵn telemetry (`summary.json`, `predictions.csv`, `logs/resource_usage.log`, `stage_samples/`) và Google Doc tracker (`1gE2K1dUJH4EObOuoqTOkJ_2LEFZoxi7moINtldBdYrw`).
  - Pha 1 hoàn tất hoàn toàn: `--sdes`, `--md_fce`, `--smdf` đã commit (`a75c9c5`) & pass smoke EC2 `5/5` (`11.45s/img`, Peak VRAM `7,930.9 MB`).
- **Chưa xong / Next:**
  - Thực hiện **Pha 2 Bước 2A** ([03-phase2-dev-validation.md](03-phase2-dev-validation.md)): chạy 6 run độc lập trên `FSC-147 val` (`n=58`, `seed=42`) đối đầu thẳng `B0` (`paper` + `raw`) + tải `MBM` (44 ảnh).
  - Chuẩn bị tải `MBM`, `VGG`, `CARPK` lên EC2 theo [ec2-setup.md](ec2-setup.md).
