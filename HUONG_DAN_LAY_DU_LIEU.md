# Hướng dẫn lấy dữ liệu về

Repo không chứa dữ liệu. Làm theo các bước dưới đây để tự cào và tạo toàn bộ dữ liệu trên máy của bạn.

## 1. Cài đặt (làm 1 lần)

```bash
git clone https://github.com/dqphong0302/chi-project.git
cd chi-project

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Yêu cầu:
- Python ≥ 3.9
- Kết nối Internet ổn định
- Khoảng **1,5 GB** ổ đĩa trống (mỗi lần cào khoảng 300 MB dữ liệu thô)

## 2. Tải toàn bộ dữ liệu (khuyên dùng)

```bash
python run_pipeline.py
```

Lệnh này chạy lần lượt các bước sau, tổng cộng khoảng **30 phút**:

| Bước | Thời gian | Kết quả |
| :--- | :--- | :--- |
| 1. Cào tin rao 22 quận (đang rao + đã gỡ) | ~17 phút | `data/raw/<run_id>/raw_<quận>.json`, `manifest.json` |
| 2. Tải biểu đồ giá 13 tháng theo phường | ~8 phút | `data/raw/<run_id>/market_price_charts.json` |
| 3. Tiền xử lý | ~1 phút | `data/processed/...` |
| 4. Bài toán 1: chia train/test + mã hóa | < 1 phút | `data/model_ready/...` |
| 4b. Bài toán 2: dữ liệu phát hiện giá bất thường | < 1 phút | `data/anomaly_ready/...` |
| 5. Thống kê + biểu đồ | < 1 phút | `reports/...` |

`<run_id>` là thời điểm bắt đầu cào, ví dụ `20260925_120442`.

Khi chạy xong, kiểm tra xem có quận nào bị thiếu dữ liệu không:

```bash
python -c "import json,glob; m=json.load(open(sorted(glob.glob('data/raw/*/manifest.json'))[-1])); print({k:v['status'] for k,v in m['districts'].items() if v['status']!='ok'} or 'Đủ 22/22 quận')"
```

## 3. Các cách chạy khác

**Chạy thử nhanh** (2 trang mỗi quận, vài phút):

```bash
python run_pipeline.py --max-pages 2
```

**Chỉ cào một vài quận:**

```bash
python run_pipeline.py --districts 110 109 111
```

**Chỉ lấy tin đang rao** (bỏ tin đã gỡ / hết hạn):

```bash
python run_pipeline.py --active-only
```

**Chỉ lấy một loại BĐS** (`can_ho`, `nha_o`, `dat`):

```bash
python run_pipeline.py --category nha_o
```

**Chỉ cào, chưa xử lý:**

```bash
python run_pipeline.py --mode crawl-only
```

**Cào bổ sung các quận bị lỗi vào cùng lần cào cũ** (xem `run_id` trong tên thư mục `data/raw/`):

```bash
python run_pipeline.py --mode crawl-only --districts 119 120 --run-id 20260925_120442
python run_pipeline.py --mode market-price --run-id 20260925_120442
python run_pipeline.py --mode preprocess-only --run-id 20260925_120442
```

**Chỉ xử lý lại dữ liệu đã có** (sau khi sửa code tiền xử lý, không cần cào lại):

```bash
python run_pipeline.py --mode preprocess-only
```

**Chỉ chia lại train/test, tạo lại bộ bài toán 2 và vẽ lại biểu đồ:**

```bash
python run_pipeline.py --mode model-prep
```

## 4. Mã quận / huyện

| ID | Quận / Huyện | ID | Quận / Huyện |
| ---: | :--- | ---: | :--- |
| 96 | Quận 1 | 109 | Quận Bình Thạnh |
| 98 | Quận 3 | 110 | Quận Gò Vấp |
| 99 | Quận 4 | 111 | Quận Phú Nhuận |
| 100 | Quận 5 | 112 | Quận Tân Bình |
| 101 | Quận 6 | 113 | Quận Tân Phú |
| 102 | Quận 7 | 115 | Huyện Bình Chánh |
| 103 | Quận 8 | 116 | Huyện Củ Chi |
| 105 | Quận 10 | 117 | Huyện Hóc Môn |
| 106 | Quận 11 | 118 | Huyện Nhà Bè |
| 107 | Quận 12 | 119 | Thành phố Thủ Đức |
| 108 | Quận Bình Tân | 120 | Huyện Cần Giờ |

## 5. Dữ liệu nhận được

```
data/
├── raw/<run_id>/                     # Dữ liệu thô từng lần cào (JSON)
├── history/snapshots.csv.gz          # Nhật ký các lần cào (dùng để theo dõi giá theo thời gian)
├── processed/
│   ├── nhatot_tphcm_clean.csv        # Dataset sạch (mở được bằng Excel, UTF-8)
│   ├── nhatot_tphcm_clean.parquet    # Cùng nội dung, đọc nhanh bằng pandas
│   ├── by_district/<quận>.csv        # Chia theo 22 quận
│   ├── by_property_type/<loại>.csv   # can_ho / nha_o / dat
│   ├── dinh_dang_mau/<quận>.csv      # Đúng tên cột của bộ dữ liệu mẫu (tieu_de, gia_ban, ...)
│   ├── market_price_monthly.csv      # Giá khu vực theo tháng (13 tháng)
│   └── rejected_rows.csv             # Dòng bị loại + lý do
├── model_ready/<loại>/               # Bài toán 1
│   ├── train.parquet, test.parquet   # Đã điền thiếu + mã hóa (dùng ngay cho mô hình)
│   ├── train_raw.parquet, test_raw.parquet   # Chưa xử lý (để thử cách khác / dùng cho PySpark)
│   ├── split.csv                     # Tin nào thuộc train / test (dùng chung sklearn và PySpark)
│   ├── preprocessor.joblib           # Bộ tiền xử lý đã học trên train
│   ├── features.json                 # Danh sách biến
│   └── spark/                        # (sau khi chạy pyspark_prep/chuan_bi_spark.py)
└── anomaly_ready/<loại>/             # Bài toán 2
    ├── anomaly.parquet / .csv        # S2, S3, price_side, cột if_*, split, nhãn tham chiếu
    ├── model_features.parquet        # Đặc trưng mã hóa sẵn cho MỌI dòng (tính S1: model.predict)
    └── columns.json                  # Mô tả + cách tính S1, S4, điểm tổng hợp
```

Ý nghĩa từng cột: xem [BAO_CAO_TIEN_XU_LY.md](BAO_CAO_TIEN_XU_LY.md).

## 6. Tích lũy dữ liệu theo thời gian

Tin trên Chợ Tốt chỉ tồn tại khoảng 60 ngày. Muốn có chuỗi dữ liệu dài hơn, hãy **chạy lại hàng tuần**:

```bash
python run_pipeline.py
```

Mỗi lần chạy tạo một thư mục `data/raw/<run_id>/` mới và ghi nối tiếp vào `data/history/snapshots.csv.gz`. Nhờ vậy các cột `first_seen_at`, `n_snapshots`, `price_change_pct` sẽ phản ánh việc giá thay đổi qua các tuần.

Chạy tự động mỗi thứ Hai lúc 8 giờ sáng (macOS / Linux, `crontab -e`):

```
0 8 * * 1 cd /duong/dan/chi-project && .venv/bin/python run_pipeline.py >> crawl.log 2>&1
```

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
| :--- | :--- |
| Log báo `HTTP 429` hoặc `Thử lại lần ...` | Bình thường: crawler tự nghỉ rồi thử lại. Nếu lặp lại nhiều lần, hãy đợi 10–15 phút rồi chạy lại |
| `manifest.json` có quận `partial` / `failed` | Cào bổ sung quận đó bằng `--districts <id> --run-id <run_id>` (xem mục 3) |
| `Không tìm thấy lần cào nào` | Chưa cào lần nào: chạy `python run_pipeline.py` hoặc `--mode crawl-only` |
| `Chưa có biểu đồ giá cho lần cào này` | Chạy `python run_pipeline.py --mode market-price`, rồi `--mode preprocess-only` |
| Số tin khác với README | Bình thường: thị trường thay đổi mỗi ngày |
| Biểu đồ báo thiếu font Arial | Không ảnh hưởng: tự chuyển sang font DejaVu Sans |
