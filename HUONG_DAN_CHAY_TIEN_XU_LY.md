# Hướng dẫn chạy tiền xử lý – 2 hướng

Dự án chuẩn bị dữ liệu cho đề *"Dự đoán giá, xác định bất thường giá cho nhà ở"* theo **2 hướng**. Chọn hướng theo yêu cầu của giảng viên:

| | **Hướng A – Dữ liệu tự cào 22 quận** | **Hướng B – 3 file mẫu giảng viên cung cấp** |
| :--- | :--- | :--- |
| Khi nào dùng | Muốn điểm cộng "thu thập thêm dữ liệu các quận khác", dữ liệu nhiều và mới | Giảng viên chỉ yêu cầu phân tích đúng 3 file đã cấp |
| Đầu vào | API Chợ Tốt (cần Internet) | `quan-go-vap.csv`, `quan-binh-thanh.csv`, `quan-phu-nhuan.csv` |
| Lệnh chính | `python run_pipeline.py` | `python xu_ly_du_lieu_mau/xu_ly.py` |
| Thời gian | ~30 phút | vài giây |
| Kết quả (25/09/2026) | 47.928 → **43.490 tin sạch** (nhà ở 31.390) | 8.273 → **7.256 tin sạch** |
| Thư mục đầu ra | `data/processed/`, `data/model_ready/`, `data/anomaly_ready/` | `data/du_lieu_mau/` |

Hai hướng cho ra **cùng tên cột, cùng cách tạo đặc trưng, cùng tín hiệu S2/S3**. Vì vậy code mô hình viết cho hướng này chạy được cho hướng kia, chỉ cần đổi đường dẫn.

---

## Bước 0 – Chuẩn bị (làm 1 lần, chung cho cả 2 hướng)

```bash
git clone https://github.com/dqphong0302/chi-project.git
cd chi-project

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Yêu cầu: Python ≥ 3.9. Mọi lệnh bên dưới đều chạy **từ thư mục gốc `chi-project/`**.

---

## Hướng A – Dữ liệu tự cào 22 quận

### A1. Chạy toàn bộ (1 lệnh)

```bash
python run_pipeline.py
```

Lệnh này làm lần lượt các bước:

| # | Bước | Thời gian | Đầu ra |
| :--- | :--- | :--- | :--- |
| 1 | Cào tin rao 22 quận (đang rao + đã gỡ) | ~17 phút | `data/raw/<run_id>/raw_<quận>.json`, `manifest.json` |
| 2 | Tải biểu đồ giá 13 tháng theo phường | ~8 phút | `data/raw/<run_id>/market_price_charts.json` |
| 3 | Tiền xử lý: làm sạch, sửa lỗi, đặc trưng, tín hiệu S2/S3 | ~1 phút | `data/processed/` |
| 4 | Bài toán 1: chia train/test, điền thiếu, mã hóa | < 1 phút | `data/model_ready/<can_ho, nha_o, dat>/` |
| 5 | Bài toán 2: bộ dữ liệu phát hiện giá bất thường | < 1 phút | `data/anomaly_ready/<can_ho, nha_o, dat>/` |
| 6 | Thống kê theo quận + biểu đồ | < 1 phút | `reports/` |

**Bộ chính của đề là nhà ở:** `data/model_ready/nha_o/` và `data/anomaly_ready/nha_o/`.

### A2. Kiểm tra kết quả

```bash
# Có quận nào cào thiếu không?
python -c "import json,glob; m=json.load(open(sorted(glob.glob('data/raw/*/manifest.json'))[-1])); print({k:v['status'] for k,v in m['districts'].items() if v['status']!='ok'} or 'Đủ 22/22 quận')"

# Số tin sau từng bước
python -c "
import json
s=json.load(open('reports/preprocessing_stats.json')); m=json.load(open('reports/model_ready_summary.json'))
print('Tin thô:', s['n_raw'], '| Tin sạch:', s['n_final'], s['by_property_type'])
for t,v in m['splits'].items(): print(f'  {t}: train={v[\"n_train\"]}, test={v[\"n_test\"]}, R2(log)={m[\"baseline_check\"][t][\"r2_log_price\"]}')
"
```

Kết quả mẫu (25/09/2026; số sẽ khác theo ngày vì thị trường thay đổi):

```
Đủ 22/22 quận
Tin thô: 47928 | Tin sạch: 43490 {'nha_o': 31390, 'can_ho': 6217, 'dat': 5883}
  can_ho: train=4956, test=1253, R2(log)=0.816
  nha_o: train=25214, test=6087, R2(log)=0.872
  dat: train=4718, test=1120, R2(log)=0.788
```

### A3. Chạy lại một phần (không cần cào lại)

| Muốn làm | Lệnh |
| :--- | :--- |
| Chỉ cào, chưa xử lý | `python run_pipeline.py --mode crawl-only` |
| Cào bổ sung quận bị lỗi vào lần cào cũ | `python run_pipeline.py --mode crawl-only --districts 119 120 --run-id <run_id>` |
| Chỉ tải lại biểu đồ giá | `python run_pipeline.py --mode market-price` |
| Xử lý lại lần cào mới nhất (bước 3 → 6) | `python run_pipeline.py --mode preprocess-only` |
| Xử lý lại một lần cào cụ thể | `python run_pipeline.py --mode preprocess-only --run-id 20260925_120442` |
| Chỉ chia lại train/test + bài toán 2 + biểu đồ (bước 4 → 6) | `python run_pipeline.py --mode model-prep` |
| Chỉ in bảng thống kê | `python run_pipeline.py --mode report-only` |
| Chạy thử nhanh (2 trang mỗi quận) | `python run_pipeline.py --max-pages 2` |

Mã quận cho `--districts` và cách cào định kỳ hàng tuần: xem [HUONG_DAN_LAY_DU_LIEU.md](HUONG_DAN_LAY_DU_LIEU.md).

---

## Hướng B – Chỉ 3 file mẫu giảng viên cung cấp

### B1. Đặt 3 file vào đúng chỗ

3 file mẫu **không có trong repo** (dữ liệu giảng viên cấp chỉ dùng cho học tập). Chép chúng vào thư mục `Cung cap HV/` ở gốc dự án, **giữ nguyên tên file**:

```
chi-project/
└── Cung cap HV/
    ├── quan-go-vap.csv
    ├── quan-binh-thanh.csv
    └── quan-phu-nhuan.csv
```

### B2. Chạy

```bash
python xu_ly_du_lieu_mau/xu_ly.py
```

Nếu 3 file nằm ở chỗ khác, hoặc muốn ghi ra chỗ khác:

```bash
python xu_ly_du_lieu_mau/xu_ly.py --input "đường/dẫn/tới/thư mục 3 file" --output data/du_lieu_mau
```

Các bước script làm:
1. Đọc và gộp 3 file; bỏ cột số điện thoại.
2. Chuyển chuỗi sang số: `"8,6 tỷ"`, `"900 triệu"`, `"86,87 triệu/m²"`, `"99 m²"`, `"nhiều hơn 10 phòng"`…
3. Tách địa chỉ thành số nhà, đường, phường cũ, quận, phường mới.
4. Làm sạch: dòng rỗng, dòng trùng, tin trùng cùng một căn, giá trị phi lý.
5. Tạo đặc trưng: đặc điểm nhà, pháp lý, biểu đồ giá khu vực, từ khóa mô tả, cờ thiếu dữ liệu.
6. Chia train/test cho bài toán 1; tạo bộ dữ liệu và tín hiệu S2/S3 cho bài toán 2.

### B3. Kiểm tra kết quả

```bash
python -c "
import json
s=json.load(open('data/du_lieu_mau/stats.json'))
print('Dòng đọc:', s['n_raw'], '| Tin sạch:', s['n_clean'], s['by_district'])
print('  train/test:', s['model_ready']['n_train'], '/', s['model_ready']['n_test'], '| kiểm tra nhanh:', s['model_ready']['baseline_check'])
"
```

Kết quả mong đợi:

```
Dòng đọc: 8273 | Tin sạch: 7256 {'Quận Gò Vấp': 3837, 'Quận Bình Thạnh': 2307, 'Quận Phú Nhuận': 1112}
  train/test: 5802 / 1450 | kiểm tra nhanh: {'r2_log_price': 0.845, 'mae_billion': 1.435, 'rmse_billion': 2.79, 'median_ape_pct': 12.1}
```

### B4. Đầu ra

```
data/du_lieu_mau/
├── du_lieu_mau_clean.csv / .parquet     7.256 tin sạch
├── rejected_rows.csv                    dòng bị loại + lý do
├── thong_ke_theo_quan.csv               thống kê theo quận
├── stats.json                           số liệu từng bước
├── model_ready/                         bài toán 1 (train/test, split.csv, preprocessor.joblib, features.json)
└── anomaly_ready/                       bài toán 2 (anomaly.parquet, model_features.parquet)
```

Chi tiết từng bước và bảng đổi tên cột: [xu_ly_du_lieu_mau/README.md](xu_ly_du_lieu_mau/README.md).

---

## Bước sau tiền xử lý – PySpark (chung cho cả 2 hướng)

Đề yêu cầu làm cả sklearn và PySpark. Script dưới đây dựng pipeline Spark ML trên **cùng tập train/test** với sklearn.

**1. Cài Java 17 hoặc 21** (Spark không chạy với Java ≥ 23):

```bash
java -version                                   # kiểm tra phiên bản hiện tại
brew install openjdk@21                         # macOS
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
```

Trên Windows: cài [Eclipse Temurin 21](https://adoptium.net/), rồi đặt biến môi trường `JAVA_HOME` trỏ tới thư mục cài.

**2. Cài PySpark và chạy:**

```bash
pip install pyspark

# Hướng A (nhà ở; đổi nha_o thành can_ho / dat nếu cần)
python pyspark_prep/chuan_bi_spark.py --dir data/model_ready/nha_o

# Hướng B
python pyspark_prep/chuan_bi_spark.py --dir data/du_lieu_mau/model_ready
```

Kết quả mong đợi:

```
Spark: train=25214, test=6087, số chiều vector đặc trưng=7799 -> data/model_ready/nha_o/spark        # hướng A
Spark: train=5802, test=1450, số chiều vector đặc trưng=879 -> data/du_lieu_mau/model_ready/spark    # hướng B
```

Đầu ra: `<thư mục>/spark/train.parquet`, `test.parquet` (cột `features`, `label` = ln(giá tỷ)), và `spark/pipeline_model/`.

---

## Dùng dữ liệu đã chuẩn bị ở bước mô hình

| | Hướng A (nhà ở) | Hướng B |
| :--- | :--- | :--- |
| Bài toán 1 – sklearn | `data/model_ready/nha_o/` | `data/du_lieu_mau/model_ready/` |
| Bài toán 1 – PySpark | `data/model_ready/nha_o/spark/` | `data/du_lieu_mau/model_ready/spark/` |
| Bài toán 2 | `data/anomaly_ready/nha_o/` | `data/du_lieu_mau/anomaly_ready/` |

Code mẫu đọc dữ liệu và tính S1, S4, điểm tổng hợp: xem mục "Dùng dữ liệu cho bước mô hình" trong [README.md](README.md).

---

## Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách xử lý |
| :--- | :--- |
| `FileNotFoundError: ... Cung cap HV/quan-go-vap.csv` | Hướng B: chưa chép 3 file, hoặc sai tên file. Kiểm tra lại mục B1, hoặc dùng `--input` |
| `Không tìm thấy lần cào nào` | Hướng A: chưa cào. Chạy `python run_pipeline.py` |
| Log báo `HTTP 429` / `Thử lại lần ...` | Bình thường: crawler tự nghỉ rồi thử lại |
| `manifest.json` có quận `partial` / `failed` | Cào bổ sung quận đó (mục A3) |
| `Chưa có biểu đồ giá cho lần cào này` | `python run_pipeline.py --mode market-price`, rồi `--mode preprocess-only` |
| Spark báo `getSubject is not supported` | Đang dùng Java ≥ 23 → cài Java 21, đặt `JAVA_HOME` |
| Spark báo `JAVA_HOME is not set` / không tìm thấy Java | Chưa cài Java hoặc chưa đặt `JAVA_HOME` |
| Spark báo `Illegal Parquet type ... TIMESTAMP(NANOS)` | File tạo bằng code cũ → chạy lại tiền xử lý (A: `--mode preprocess-only`; B: `xu_ly.py`) |
| Số tin khác với hướng dẫn | Hướng A: bình thường, thị trường thay đổi mỗi ngày. Hướng B: kiểm tra lại đúng 3 file gốc |
