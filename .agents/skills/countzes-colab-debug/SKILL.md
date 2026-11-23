---
name: countzes-colab-debug
description: >-
  Use this skill when setting up, running, or troubleshooting CountZES notebooks
  and pipeline compatibility on Google Colab (Drive symlinks, GroundingDINO sys.path,
  transformers v4/v5 CLIP/BERT compatibility, and smoke-test image selection).
---

# CountZES Google Colab & Compatibility Runbook

Cheatsheet xử lý các lỗi thường gặp khi chạy CountZES trên Google Colab hoặc sau khi cập nhật thư viện.

## 1. Thiết lập Colab Idempotent (an toàn cho Run All)
- Lưu bền vững trên Google Drive tại `/content/drive/MyDrive/CountZES/{data,weights,outputs}` và tạo symlink về `./data`, `./outputs`, `./DSALVANet/checkpoints`, `./GroundingDINO/weights`, `./CounTR/output_allnew_dir`.
- Trước khi `wget` / `gdown` weights, luôn kiểm tra `os.path.exists(path) and os.path.getsize(path) > 0` để tránh tải trùng khi bấm *Run All*.
- Khi `git clone` bị hỏi username/lỗi trong Colab đối với private repo, thông báo rõ cần PAT hoặc chép thư mục/public mirror; nếu thư mục `/content/cv-advanced` đã có `.git`, chỉ `git pull` thay vì clone lại.

## 2. Lỗi tương thích thư viện hay gặp
1. **`ModuleNotFoundError: No module named 'groundingdino'`**:
   - Kiểm tra `countzes_core.py` đã thêm `<repo>/GroundingDINO` vào `sys.path` trước khi import `GroundingDINO.groundingdino...`.
2. **Lỗi `transformers >= 5.0`**:
   - `'BertModel' object has no attribute 'get_extended_attention_mask'`: cần tương thích trong `GroundingDINO/groundingdino/models/GroundingDINO/bertwarper.py` hoặc pin `"transformers>=4.31.0,<5.0.0"` trong cell cài đặt Colab.
   - `'BaseModelOutputWithPooling' object has no attribute 'norm'`: trong `compute_clip_similarity_map`, lấy `.pooler_output` nếu `get_text_features()` trả về `BaseModelOutputWithPooling`.
3. **AnyUp**:
   - `wimmerth/anyup` tải từ `torch.hub` (`/root/.cache/torch/hub/checkpoints/anyup_paper.pth`); là hành vi chuẩn để upsample SAM feature lên `(256, 256)` ở Stage 3 (`FCE`).

## 3. Chọn ảnh mẫu kiểm thử Stage 1–3
- Tránh URL từ `upload.wikimedia.org` trong script Python/Colab vì dễ bị chặn `HTTP Error 429: Too many requests`.
- Tránh ảnh quá ít đối tượng hoặc chồng lấn nhiều (dễ gây `Too few unique values for KDE` ở Stage 2 và `No single-instance boxes found`).
- Ưu tiên ảnh có nhiều đối tượng cùng loại, kích thước đồng đều từ nguồn raw ổn định (ví dụ `https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/orange.jpg` với `CLASS_PROMPT = "oranges"`).

## 4. Xác minh sau khi chỉnh sửa cho Colab
1. Chạy `python -m py_compile countzes_core.py` (và các script bị chạm tới).
2. Nếu có sửa notebook, xác nhận cú pháp JSON + Python AST hợp lệ.
3. Nhắc người dùng `git push` từ local rồi `!git pull` + restart/re-import kernel trên Colab nếu thay đổi nằm trong `.py`.
