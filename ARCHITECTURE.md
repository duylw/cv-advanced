# ARCHITECTURE.md

Tài liệu mô tả kiến trúc tổng thể của dự án **CountZES** (*Counting via Zero-Shot Exemplar Discovery and Selection*).

## 1. Luồng xử lý 3 giai đoạn (Inference Pipeline)

```text
Input: (Ảnh RGB, Tên class text)
   │
   ▼
[Stage 1: DAE - Detection-Anchored Exemplar]
   ├── CLIP (openai/clip-vit-base-patch32) ──► CLIP patch similarity map (sim_clip)
   ├── GroundingDINO (SwinT-OGC)           ──► Candidate boxes (xyxy), confidences
   ├── Chọn box tốt nhất (confidence + entropy)
   └── SAM + SSES                          ──► Exemplar 1: `dae` + SAM similarity map (`sim_up`)
   │
   ▼
[Stage 2: DGE - Density-Guided Exemplar]
   ├── Counter (CounTR / DSALVANet) dùng `dae` ──► Density map ban đầu
   ├── Lấy peak cục bộ + lọc trong box GDINO  ──► Point prompts cho SAM
   ├── SAM sinh candidate boxes + lọc single-instance
   ├── Chạy lại Counter trên từng candidate + KDE (`cluster_counts`) ──► `pseudo_gt_count`
   └── Chọn box gần `pseudo_gt_count` + entropy thấp trên `sim_up`   ──► Exemplar 2: `dge`
   │
   ▼
[Stage 3: FCE - Feature-Consensus Exemplar]
   ├── Trích SAM feature của `dge` + chạy Counter lần nữa ──► Density map mới
   ├── SAM sinh candidate boxes mới + lọc container (`contains`) + lọc single-instance
   ├── AnyUp upsample SAM feature lên (256, 256)
   └── 2-means cosine clustering trên feature vector các box ──► Exemplar 3: `fce`
   │
   ▼
[Final Counting]
   └── Counter (CounTR / DSALVANet) nhận 3 exemplars `[dae, dge, fce]` ──► `Pred_Count` + MAE / RMSE
```

---

## 2. Cấu trúc thư mục chính

| Thành phần | Đường dẫn | Vai trò |
| :--- | :--- | :--- |
| **Entry scripts (5 datasets)** | `countzes_fsc147.py`, `countzes_carpk.py`, `countzes_persense_D.py`, `countzes_mbm.py`, `countzes_vgg.py` | Chạy đánh giá toàn bộ pipeline trên từng dataset. Hiện đang sao chép logic pipeline ở cả 5 file. |
| **Visualization helper** | `show.py` | Hàm vẽ mask, point, bbox cho matplotlib. |
| **DSALVANet** | `DSALVANet/models/DSALVANet.py` | Counter gồm `ResNet` (ResNet-50 `out_layers=[1,2,3]`), `DSAMMultiBlock` (với `MDM` dùng `deform_conv2d`), `LVAM`, `Regressor`. |
| **DSALVANet Utils** | `DSALVANet/utils/` | `model_helper.py` (load model), `data_preprocess.py` (resize 512x512 & pad đủ 3 box), `fsol_modules.py` (`vis_DSALVANet`), `PerSense_countr.py` (`vis_countr`). |
| **CounTR** | `CounTR/models_mae_cross.py`, `CounTR/demo.py` | Counter MAE ViT + Cross-Attention (`load_image`, `run_one_image` theo sliding window 384px). |
| **GroundingDINO** | `GroundingDINO/groundingdino/` | Open-vocabulary object detector (`Model.predict_with_classes`). |
| **SAM / MobileSAM** | `per_segment_anything/` | Segment Anything (`vit_h`, `vit_l`, `vit_b`, `vit_t`), dùng `SamPredictor` để trích feature & segment từ point prompts. |
| **Dữ liệu & Kết quả** | `data/`, `outputs/` | `data/` chứa metadata/ảnh các dataset; `outputs/` lưu ảnh trực quan hóa khi bật `--visualize`. |

---

## 3. Quy ước dữ liệu & Tọa độ quan trọng
- **GroundingDINO & SAM (`xyxy`):** `[x0, y0, x1, y1]` tính theo pixel ảnh gốc `(W, H)`.
- **DSALVANet (`preprocess`) & CounTR (`load_image`) (`yxyx`):** `[y0, x0, y1, x1]`. Khi truyền `[dae, dge, fce]` vào counter, pipeline chuyển đổi sang `[[y0, x0, y1, x1], ...]`.
- **AnyUp (`hr_features`):** Nhận `lr_features` `(1, C, 64, 64)` từ SAM và ảnh gốc chuẩn hóa ImageNet, xuất feature độ phân giải `(C, 256, 256)`.

---

## 4. Model Weights mặc định
- SAM ViT-H: `data/sam_vit_h_4b8939.pth` (hoặc MobileSAM: `weights/mobile_sam.pt` với `--sam_type vit_t`)
- GroundingDINO: `GroundingDINO/weights/groundingdino_swint_ogc.pth`
- DSALVANet: `DSALVANet/checkpoints/checkpoint_200.pth`
- CounTR: `CounTR/output_allnew_dir/FSC147.pth` và `CounTR/output_allnew_dir/CARPK.pth`
- CLIP (`openai/clip-vit-base-patch32`) & AnyUp (`wimmerth/anyup`): tự động tải qua HuggingFace / `torch.hub`.
