# EC2 Datasets & Run Cheatsheet (`g4dn.xlarge`)

> Tài liệu tham khảo nhanh cho môi trường EC2 (Tesla T4 15GB, 4 vCPU, 15GB RAM, `/opt/pytorch`).
> **Lưu ý an toàn:** Không lưu IP công khai, key `.pem`, token hay secrets vào file này.

## 1. Kích hoạt môi trường mỗi phiên `tmux`

```bash
cd ~/cv-advanced
tmux new -s countzes        # hoặc: tmux attach -t countzes
source /opt/pytorch/bin/activate
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

---

## 2. Trạng thái Dataset trên EC2

| Dataset | Thư mục đích trên EC2 | Trạng thái | Ghi chú |
| :--- | :--- | :---: | :--- |
| `FSC-147` | `data/FSC-147/images_384_VarV2` | ✅ Sẵn sàng | `val=1286`, `test=1190` |
| `MBM` | `data/MBM/...` | ☐ Cần tải | 44 ảnh (~nhẹ, tải qua `gdown` / `scp` từ local) |
| `VGG` | `data/VGG/...` | ☐ Cần tải | 200 ảnh (`countzes_vgg.py`) |
| `CARPK` | `data/CARPK/...` | ☐ Cần tải | `test=459` (`countzes_carpk.py`) |

*(Kiểm tra chính xác tên thư mục con mà `countzes_mbm.py`, `countzes_vgg.py`, `countzes_carpk.py` kỳ vọng trước khi giải nén.)*

---

## 3. Đồng bộ code & kéo kết quả

```bash
# Trên EC2: kéo code mới sau khi sửa & push từ local
cd ~/cv-advanced && git pull

# Từ máy Windows local: kéo summary/logs/ảnh (tránh tải file nặng không cần thiết)
scp -r -i <PATH_TO_KEY> <USER>@<EC2_HOST>:~/cv-advanced/outputs/<RUN_NAME> ./outputs/
```

---

## 4. Tự động tắt máy khi rảnh (Idle Stop)

Đã có script [ec2_idle_stop.sh](../scripts/ec2_idle_stop.sh) (kiểm tra đồng thời GPU `>= 5%` và tiến trình `python.*countzes_`):

```bash
chmod +x scripts/ec2_idle_stop.sh

# 1. Test thử tự tắt sau 30 giây (check mỗi 5 giây):
bash scripts/ec2_idle_stop.sh --timeout 30

# 2. Chạy qua đêm (tự tắt sau 30 phút idle, mở ở 1 window tmux riêng):
bash scripts/ec2_idle_stop.sh --timeout 1800
```

*(Có thể thêm `--dry-run` để log thử mà chưa tắt máy thật; đảm bảo EC2 **Shutdown behavior** đang là `Stop`.)*

---

## 5. Quy tắc chạy dài trên EC2 (tránh rớt mạng & tốn tiền oan)

1. Luôn chạy các run > 15 phút bên trong `tmux` (`Ctrl+B` rồi `D` để detach trước khi đóng terminal).
2. Luôn chạy `n=5` smoke test sau mỗi lần `git pull` trước khi chạy full dataset.
3. Chạy xong đợt hoặc nghỉ làm việc → vào AWS Console **Stop** instance `g4dn.xlarge` (không Terminate để giữ nguyên `data/` và weights trên ổ 80GB).
