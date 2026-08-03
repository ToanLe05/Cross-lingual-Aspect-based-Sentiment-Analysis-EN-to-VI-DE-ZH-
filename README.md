# Cross-lingual Aspect-based Sentiment Analysis (EN to VI/DE/ZH)

Dự án này là bộ source code hoàn chỉnh để nghiên cứu việc chuyển giao tri thức Đánh giá Hình thái Cảm xúc (ABSA) từ ngôn ngữ giàu tài nguyên (Tiếng Anh - EN) sang các ngôn ngữ ít tài nguyên hơn (Tiếng Việt - VI, Tiếng Đức - DE, Tiếng Trung - ZH).

Dự án đánh giá 3 trường phái mô hình:
1. **AG-CAN:** Kiến trúc chuyên biệt (mBERT + Aspect-Guided Attention + Gated Residuals).
2. **XLM-R:** Mã hóa khổng lồ (Large-scale Encoder với Exact Masked Mean Pooling).
3. **mT5:** Mô hình tạo sinh (Generative Seq2Seq với Constrained Decoding/Label Scoring).

---

## 🌟 Tóm Tắt Quy Trình Chuẩn (A-Z)
Toàn bộ quy trình chạy thực nghiệm xoay quanh **3 Setting** chính:
*   **S1 (Zero-shot):** Train 100% trên Tiếng Anh (Chỉ cần chạy ĐÚNG 1 LẦN duy nhất).
*   **S2 (Few-shot):** Tải lại (Load) "não" của S1, rồi học thêm một số ít mẫu (50, 100, 200) của ngôn ngữ đích (Chạy 3 seeds để lấy trung bình).
*   **S3 (Full-target):** Tải lại "não" của S1, rồi học toàn bộ tập Train của ngôn ngữ đích (Chạy 3 seeds).

---

## Bước 1: Chuẩn Bị Môi Trường (Setup)
Bạn có thể thiết lập môi trường theo 2 cách: dùng Python thuần hoặc dùng Docker.

### Cách 1: Dùng Python thuần (Dành cho máy cá nhân / Colab)
Bạn cần cài đặt các thư viện Python. Đảm bảo bạn đang dùng môi trường có GPU (Nvidia/CUDA) để tiết kiệm thời gian chạy.
```bash
# Cài đặt các thư viện lõi
pip install -r requirements.txt
```
*(Nếu bạn chạy trên máy tính cá nhân, hãy đảm bảo đã cài sẵn driver CUDA cho PyTorch).*

### Cách 2: Dùng Docker (Cách chuyên nghiệp nhất để Share code)
Nếu bạn muốn gửi source code này cho thầy cô hoặc bạn bè mà không muốn họ phải vật lộn với việc cài đặt môi trường, hãy bảo họ cài **Docker Desktop** và gõ 2 lệnh sau:

```bash
# 1. Build môi trường ảo (Chỉ chạy 1 lần)
docker-compose build

# 2. Bắt đầu chạy code (Chèn "docker-compose run --rm absa" trước mọi lệnh python)
# Ví dụ thay vì gõ "python scripts/eval.py", hãy gõ:
docker-compose run --rm absa python scripts/eval.py
```
Với cách này, môi trường sẽ sạch 100% và chạy chuẩn xác trên mọi loại máy (Windows, Mac, Linux).

---

## Bước 2: Huấn luyện S1 (Khởi tạo Checkpoint Gốc)
**BẮT BUỘC PHẢI CHẠY BƯỚC NÀY ĐẦU TIÊN TRƯỚC KHI CHẠY S2/S3!**
S1 sẽ tiến hành đào tạo các model dựa trên tập dữ liệu tiếng Anh (`seed = 42`). Sau đó, hệ thống sẽ tự động lưu Checkpoint và test luôn trên cả 3 tập tiếng Việt (vi), Đức (de), Trung (zh).

**Lệnh chạy toàn bộ project (3 Models x 2 Domains):**
```bash
python scripts/train.py --setting s1 --targets vi de zh
```

**Hoặc chạy tách lẻ để chia việc cho team:**
```bash
# Người A chạy AG-CAN
python scripts/train.py --setting s1 --models ag_can --domains restaurant phone --targets vi de zh

# Người B chạy XLM-R
python scripts/train.py --setting s1 --models xlmr --domains restaurant phone --targets vi de zh

# Người C chạy mT5
python scripts/train.py --setting s1 --models mt5 --domains restaurant phone --targets vi de zh
```
Sau khi chạy xong, Model tốt nhất sẽ được lưu tại: `outputs/checkpoints/<model>/<domain>/s1/s1_seed42/best.pt`. **(Bạn có thể gửi file `best.pt` này cho bạn bè để họ chạy thẳng S2/S3 mà không cần cày lại S1).**

---

## Bước 3: Huấn Luyện S2 & S3 (Chuyển Giao Tri Thức)
Sau khi đã có file Checkpoint của S1 ở thư mục `outputs/checkpoints/`, bạn tiến hành chạy S2 (Few-shot) hoặc S3 (Full-target).
*Hệ thống sẽ tự động tìm Checkpoint S1, tải trọng số lên, và lặp lại việc train trên 3 mức Random Seeds (42, 123, 456).*

**Chạy S2 (Few-shot learning):**
```bash
python scripts/train.py --setting s2 --targets vi de zh
```
*(Mặc định nó sẽ tự động chạy các mốc N = 50, 100, 200 mẫu).*

**Chạy S3 (Full-target learning):**
```bash
python scripts/train.py --setting s3 --targets vi de zh
```

---

## Bước 4: Đánh Giá & Vẽ Biểu Đồ Báo Cáo
Sau khi 3 Setting (S1, S2, S3) đã chạy xong, toàn bộ kết quả điểm số (`Macro F1`, `Accuracy`) sẽ tụ hội dưới dạng các file `.json` bên trong thư mục `outputs/results/{setting}/`.

Để đọc số liệu và chuyển nó thành các **biểu đồ cực đẹp** dành cho Báo Cáo/Khóa luận, bạn chỉ cần gõ đúng 1 lệnh duy nhất:
```bash
python scripts/eval.py
```

Lúc này, hãy mở thư mục `outputs/figures/`, bạn sẽ thu hoạch được:
1. `macro_f1_comparison.png`: Biểu đồ so sánh điểm F1 của 3 Models.
2. `recovery_curves.png`: Biểu đồ đường chứng minh Few-shot (S2) giúp model mạnh lên nhanh như thế nào.
3. `gap_matrix.png`: Biểu đồ nhiệt (Heatmap) thể hiện sức ì đa ngôn ngữ.
4. `error_taxonomy.png`: Biểu đồ phân loại lỗi (Mô hình hay nhầm loại câu nào nhất?).

---

## 🛠️ Một số Lệnh Tùy Biến Nâng Cao (Dành cho Dev)

**1. Chỉ chạy 1 ngôn ngữ nhất định ở S2:**
```bash
python scripts/train.py --setting s2 --targets vi --models ag_can --domains restaurant
```

**2. Ghi đè lại (Xóa bỏ kết quả cũ và chạy lại từ đầu):**
Thêm cờ `--force` vào đuôi lệnh:
```bash
python scripts/train.py --setting s1 --targets vi de zh --force
```

**3. Chỉnh sửa siêu tham số (Hyperparameters):**
Mở file `config.yml` ra và đổi trực tiếp:
*   `batch_size`: Giảm xuống nếu GPU bị hết VRAM (OOM - Out of memory).
*   `epochs`: Tăng lên nếu mô hình chưa hội tụ.
*   `mt5_lr`: Learning Rate siêu nhỏ chuyên dụng cho mT5.
