# Bất động sản TP.HCM – Thu thập & Tiền xử lý dữ liệu Nhà Tốt

Pipeline thu thập và tiền xử lý dữ liệu tin rao bán bất động sản tại TP.HCM từ [Nhà Tốt](https://www.nhatot.com/) / [Chợ Tốt](https://www.chotot.com/), phục vụ đồ án Data Science **dự đoán giá** và **đánh giá cơ hội đầu tư**.

- **Phạm vi:** 22 quận/huyện TP.HCM (địa giới trước 07/2025) · căn hộ, nhà ở, đất · tin đang rao + tin đã gỡ/hết hạn
- **Lịch sử giá:** biểu đồ giá 13 tháng theo phường của Nhà Tốt (cột `bieu_do_gia`)
- **Đầu ra:** dataset sạch, chia theo quận / loại BĐS, và các bộ train/test đã sẵn sàng cho mô hình

> Repo **không chứa dữ liệu**, vì dữ liệu nặng khoảng 1 GB và có thông tin người đăng tin.
> Xem **[HUONG_DAN_LAY_DU_LIEU.md](HUONG_DAN_LAY_DU_LIEU.md)** để tự tải dữ liệu về (khoảng 30 phút).

## Kết quả lần chạy 25/09/2026

| | |
| :--- | :--- |
| Tin thô | 47.928 (22/22 quận, 0 trang lỗi) |
| Tin sạch | **43.490**: nhà ở 31.390 · căn hộ 6.217 · đất 5.883 |
| Biểu đồ giá 13 tháng | gắn được cho 43.420 tin |
| Kiểm tra nhanh (mô hình mặc định, tập test) | R² log giá: căn hộ 0,81 · nhà ở 0,87 · đất 0,80 |

<p align="center">
  <img src="reports/figures/03_price_distribution.png" width="85%"><br>
  <img src="reports/figures/05_growth_yield_nha_o.png" width="85%">
</p>

## Bắt đầu nhanh

```bash
git clone https://github.com/dqphong0302/chi-project.git
cd chi-project
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py          # cào + biểu đồ giá + tiền xử lý + train/test + biểu đồ
```

Cần Python ≥ 3.9 và kết nối Internet. Các chế độ chạy khác xem trong [HUONG_DAN_LAY_DU_LIEU.md](HUONG_DAN_LAY_DU_LIEU.md).

## Cấu trúc

```
chi-project/
├── run_pipeline.py          # Điểm chạy chính
├── src/
│   ├── config.py            # 22 quận, bảng giải mã, ngưỡng làm sạch
│   ├── crawler.py           # Cào tin rao (API ad-listing)
│   ├── market_price.py      # Biểu đồ giá 13 tháng theo phường
│   ├── storage.py           # Lưu/đọc từng lần cào, nhật ký snapshot
│   ├── preprocessor.py      # Trích xuất, làm sạch, sửa lỗi, đặc trưng
│   ├── model_prep.py        # Chia train/test, điền thiếu, mã hóa
│   ├── analyzer.py          # Thống kê theo quận × loại BĐS
│   └── figures.py           # Biểu đồ cho báo cáo / slide
├── reports/                 # Thống kê tổng hợp + biểu đồ (có sẵn trong repo)
├── slides/tien_xu_ly.pdf    # Slide báo cáo tiền xử lý (LaTeX Beamer)
├── GHI_CHU_TIEN_XU_LY.md    # Nhật ký quyết định tiền xử lý (làm gì, vì sao, bao nhiêu dòng)
├── BAO_CAO_TIEN_XU_LY.md    # Danh mục biến chi tiết
└── data/                    # (tạo ra khi chạy, không có trong repo)
```

## Pipeline

1. **Thu thập:** cào theo từng quận (tin đang rao + tin đã gỡ), có retry và thử lại trang lỗi. Mỗi lần cào lưu vào `data/raw/<run_id>/`, kèm `manifest.json`.
2. **Biểu đồ giá:** lấy đơn giá trung vị theo tháng, 13 tháng, cho từng phường × loại BĐS.
3. **Trích xuất:** giải mã theo nhãn chữ của API. Cùng một mã số có thể mang nghĩa khác nhau tùy loại BĐS.
4. **Làm sạch:** ngưỡng riêng cho từng loại BĐS; loại tin trùng (cả trùng mã tin lẫn cùng một BĐS do nhiều người đăng). Lý do loại từng dòng được lưu lại.
5. **Sửa từng ô:** giá trị phi lý được đặt thành NaN; thêm cờ độ chính xác tọa độ.
6. **Đặc trưng:** đặc điểm có cấu trúc, giá khu vực, tỷ suất cho thuê trích từ mô tả, cờ thiếu dữ liệu.
7. **Chuẩn bị mô hình:** mỗi loại BĐS một bộ; chia 80/20 theo người đăng; điền thiếu và mã hóa chỉ học từ tập train; biến mục tiêu `ln(giá)`.

Chi tiết từng bước: [GHI_CHU_TIEN_XU_LY.md](GHI_CHU_TIEN_XU_LY.md) · Danh mục biến: [BAO_CAO_TIEN_XU_LY.md](BAO_CAO_TIEN_XU_LY.md) · Slide: [slides/tien_xu_ly.pdf](slides/tien_xu_ly.pdf)

## Dùng dữ liệu cho mô hình

```python
import json, pandas as pd

d = "data/model_ready/nha_o/"          # hoặc can_ho/, dat/
feats = json.load(open(d + "features.json"))["feature_names_out"]
train = pd.read_parquet(d + "train.parquet")
test = pd.read_parquet(d + "test.parquet")

X_train, y_train = train[feats], train["target_log_price"]   # giá (tỷ đồng) = exp(y)
X_test, y_test = test[feats], test["target_log_price"]
```

## Lưu ý

- Giá trong dữ liệu là **giá rao bán**, không phải giá giao dịch thật.
- Tọa độ chỉ chính xác đến mức tâm con đường hoặc tâm phường.
- Dữ liệu lấy từ API công khai của Chợ Tốt, chỉ phục vụ học tập / nghiên cứu. Crawler có nghỉ giữa các request; vui lòng không giảm thời gian nghỉ.
- Biên dịch slide: `cd slides && tectonic tien_xu_ly.tex` (hoặc `xelatex`). Slide dùng font Arial.
