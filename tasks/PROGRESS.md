# Task Progress Tracker — CountZES

> Master tracker gọn cho toàn bộ chiến thuật từ hiện tại → hoàn thành báo cáo.
> Khi làm việc ở pha nào, chỉ cần đọc `PROGRESS.md` + đúng file pha đó để tiết kiệm token.

## Lộ trình tổng thể

| ID | Pha / Tài liệu | Trạng thái | Mục tiêu cốt lõi |
| :--- | :--- | :--- | :--- |
| `01` | [Sửa bug pipeline & DSALVANet](01-fix-bugs.md) | `VERIFIED` | P0/P1, P2 A/B (`fix_sses_inversion`) & `--fsoc countr` (`MBM` 44/44) đã verify |
| `02` | [Pha 1: Code đề xuất (SDES, MD-FCE, SMDF)](02-phase1-implement-proposals.md) | `DONE` | Đã code (`a75c9c5`, `2a93b48`), docs & pass smoke EC2 `5/5` (`11.45s/img`, VRAM `7.93 GB`) |
| `03` | [Pha 2: Dev Validation (Batch nhỏ)](03-phase2-dev-validation.md) | `DONE` | Đã xong Bước 2A & 2B (`val n=58`). **Chốt `CLIP Fix + SDES` trên `paper`** (`19.52 → 15.07` MAE); loại `le1` & `SMDF`; giữ `MD-FCE g=0` thử thêm trên `VGG`/`MBM` |
| `04` | [Pha 3: Ablation Study (100% Dataset)](04-phase3-ablation-study.md) | `NEXT` | Chạy bảng Ablation `A0 → A3` (100% dataset) trên `MBM` (44) + `FSC-147 test` (1,190, In-domain & Cross-domain) |
| `05` | [Pha 4: Benchmark chính (3 Track)](05-phase4-main-benchmark.md) | `PENDING` | So Baseline vs Ours trên `FSC-147 test` (In/Cross-domain) & `MBM` *(tạm gác `CARPK`/`VGG`)* |
| `06` | [Pha 5: Phân tích & Báo cáo / GDocs](06-phase5-report-and-gdocs.md) | `PENDING` | Tổng hợp số liệu, ảnh `stage_samples`, cập nhật Google Doc & viết báo cáo |
| `ref` | [EC2 Datasets & Run Cheatsheet](ec2-setup.md) | `ACTIVE_REF` | Lệnh `tmux`, tải `MBM`/`VGG`/`CARPK` lên EC2, `ec2_idle_stop.sh`, `scp` output |

---

## Baseline & Kết quả đã đo (Reference Anchor)

| Nguồn / Run | Dataset (N) | Cài đặt | MAE ↓ | RMSE ↓ | Tốc độ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Paper (báo cáo gốc) | `FSC-147 test` (1190) | DSALVANet in-domain / CounTR cross | 15.77 / 21.09 | 91.40 / 110.14 | — |
| Paper (báo cáo gốc) | `CARPK` (459) / `PerSense-D` | DSALVANet / CounTR | 7.24 / 12.29 | 9.22 / 28.32 | — |
| Paper (báo cáo gốc) | `VGG` (200) / `MBM` (44) | DSALVANet / CounTR | 45.55 / 22.16 | 74.41 / 40.06 | — |
| Colab (`baseline_vgg_paper`) | `VGG` (200) | `DSALVANet`, `box=0.20`, `pct=80`, `fix_sses_inversion=False` (`1-sim_clip`, GitHub) | 51.765 | 151.3083 | 22.909s/img |
| Colab (`baseline_vgg_paper_exact`) | `VGG` (200) | `DSALVANet`, `box=0.15`, `pct=90`, `fix_sses_inversion=True` (Paper exact) | 69.465 | 89.3219 | 16.617s/img |
| Colab (`baseline_mbm_paper`) | `MBM` (44) | `CounTR`, `box=0.15`, `pct=90`, `fix_sses_inversion=True` (Paper exact) | 50.9090 | 71.5297 | 14.825s/img |
| Colab (`baseline_mbm_github_box015`) | `MBM` (44) | `CounTR`, `box=0.15`, `pct=80`, `fix_sses_inversion=False` (`1-sim_clip`) | 25.0896 | 42.4355 | 16.086s/img |
| Colab (`baseline_mbm_github`) | `MBM` (44) | `CounTR`, `box=0.20`, `pct=80`, `fix_sses_inversion=False` (`1-sim_clip`, GitHub) | **19.5864** | **37.1069** | 15.607s/img |
| Colab (`stratified_n29`, seed=42) | `FSC-147 test` (29) | `raw` vs `maskclip+nocrop` | 13.414 / **11.207** | 38.34 / **29.90** | 15.6–15.8s/img |
| EC2 (`dev_2a_val58_B0`, seed=42) | `FSC-147 val` (58) | `paper`, `raw` (Paper Baseline) | 19.5172 | 63.7565 | 10.334s/img |
| EC2 (`dev_2a_val58_clip`, seed=42) | `FSC-147 val` (58) | `paper`, `MaskCLIP + No-Crop` | 16.6552 | 61.9828 | 10.482s/img |
| **EC2 (`dev_2b_val58_paper_clip_sdes`)** | **`FSC-147 val` (58)** | **`paper`, `MaskCLIP + No-Crop + SDES`** | **15.0690** | **61.2122** | **8.915s/img** |
| EC2 (`dev_2b_val58_paper_clip_sdes_mdfce0`) | `FSC-147 val` (58) | `+ MD-FCE (gamma=0.0)` | 15.1207 | 61.2315 | 8.938s/img |
| EC2 (`dev_2b_val58_paper_full_g0_085`) | `FSC-147 val` (58) | `+ SMDF (0.05/0.10/0.85)` *(loại)* | 17.6379 | 62.3487 | 8.945s/img |

---

## Ngân sách Dataset & Thời gian ước lượng (EC2 `g4dn.xlarge` — đo thực tế ~9–10.5s/img)

| Dataset / Track | Số ảnh (Split) | Trên EC2? | ~Thời gian / 1 lượt Full | Vai trò |
| :--- | :---: | :---: | :---: | :--- |
| `MBM` (`countr` + `FSC147.pth`) | 44 | Cần tải (`gdown`) | ~7 phút | **Active** — Cross-domain Cell |
| `FSC-147 test` (`DSALVANet`) | 1,190 | Sẵn (`FSC-147`) | ~3.0–3.5 giờ | **Active** — In-domain Generic |
| `FSC-147 test` (`countr` + `CARPK.pth`) | 1,190 | Cần tải `CARPK.pth` | ~3.0–3.5 giờ | **Active** — Cross-domain Generic |
| `VGG` (200) / `CARPK test` (459) | 200 / 459 | Chưa | ~33p / ~1.25h | *Tạm gác khỏi đợt test hiện tại* |

---

## Quyết định Baseline & Phạm vi cố định (Chốt `2026-11-26`)
> Không kiểm chứng thêm giữa Paper vs GitHub; từ Pha 3 trở đi, **mọi thực nghiệm xây và đo trực tiếp trên bộ mặc định GitHub gốc làm nền (`A0`)**:
> ```text
> --box_threshold 0.20 --text_threshold 0.15 --sses_start_percentile 80 --github_sses_inversion --single_instance_mode paper --clip_sim_mode raw
> ```
> - **3 track đang chạy hiện tại:**
>   1. `FSC-147 test` In-domain: `--fsoc DSALVANet` (`checkpoint_200.pth`)
>   2. `FSC-147 test` Cross-domain: `--fsoc countr --countr_ckpt ./CounTR/output_allnew_dir/CARPK.pth`
>   3. `MBM` Cross-domain: `--fsoc countr --countr_ckpt ./CounTR/output_allnew_dir/FSC147.pth`
> - *(Tạm gác `VGG`, `CARPK`, `PerSense-D` khỏi đợt chạy hiện tại. Loạt thử nghiệm Pha 2 `dev_2a/2b_val58_*` trước đó được đo trên nền `paper` `0.15 / 90 / fix_sses_inversion=True`.)*

---

## Trạng thái hiện tại & Việc làm ngay
- **Đã xong:**
  - Sửa bug pipeline + `bertwarper.py` (`transformers >= 5.x`) + telemetry `proposal_diagnostics` (`2a93b48`).
  - Hoàn tất Pha 1 & Pha 2 (`dev_2a_val58_*`, `dev_2b_val58_*` đều pass `58/58`).
  - Đã có sẵn Baseline `MBM` trên Colab (`baseline_mbm_github`: `19.5864 / 37.1069`).
- **Chưa xong / Next:**
  - Tải `data/mbm/Images` + `CounTR/output_allnew_dir/{FSC147,CARPK}.pth` lên `gpu-server`, smoke test 1 sample cho 3 track (`FSC-147 In`, `FSC-147 Cross`, `MBM`), rồi chạy Baseline & **Pha 3** ([04-phase3-ablation-study.md](04-phase3-ablation-study.md)).
