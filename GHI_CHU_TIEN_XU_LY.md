# GHI CHÚ TIỀN XỬ LÝ DỮ LIỆU BĐS TP.HCM (NHÀ TỐT)

Nhật ký các quyết định trong khâu thu thập và tiền xử lý: **làm gì, vì sao, ảnh hưởng bao nhiêu dòng**.
Số liệu lấy từ lần cào `20260925_120442` (25/09/2026). Chạy lại pipeline thì xem số mới trong
`reports/preprocessing_stats.json`, `reports/model_ready_summary.json` và `reports/anomaly_ready_summary.json`.

Đề bài (`topic_Price_Prediction_AnomalyDetection.pdf`): **nhà ở**; bài toán 1 dự đoán giá (≥ 4 mô hình ở cả sklearn
lẫn PySpark; đánh giá MAE, RMSE, R²; feature importance); bài toán 2 phát hiện giá bất thường bằng điểm tổng hợp
0–100 từ 4 tín hiệu (Residual-Z, Min/Max, P10–P90, Isolation Forest). Cộng điểm nếu thu thập thêm quận khác.

Mô tả chi tiết từng biến nằm trong [BAO_CAO_TIEN_XU_LY.md](BAO_CAO_TIEN_XU_LY.md). Slide nằm ở [slides/tien_xu_ly.pdf](slides/tien_xu_ly.pdf).

---

## 0. Tóm tắt nhanh

| Bước | Kết quả |
| :--- | :--- |
| Thu thập | 47.928 tin thô, 22/22 quận, 0 trang lỗi (21.961 đang rao + tin đã gỡ / hết hạn) |
| Làm sạch | Loại 4.438 tin → **43.490 tin sạch** (nhà ở 31.390 · căn hộ 6.217 · đất 5.883) |
| Sửa từng ô | ~325 giá trị phi lý chuyển thành NaN (tọa độ, kích thước, số phòng, số tầng, số WC) |
| Bổ sung | Biểu đồ giá 13 tháng cho 43.420 tin; giá thuê trích từ mô tả cho 2.994 tin |
| Bài toán 1 | 3 bộ train/test theo loại BĐS (nhà ở là bộ chính), 75–97 biến sau mã hóa; `split.csv` dùng chung sklearn / PySpark |
| Kiểm tra nhanh | Mô hình mặc định: R² (log giá) = 0,82 căn hộ · 0,87 nhà ở · 0,79 đất → dữ liệu đủ thông tin |
| Bài toán 2 | Nhà ở 31.567 tin (gồm 177 tin giá phi lý giữ lại); S2 Min/Max và S3 P10–P90 đã tính; ma trận Isolation Forest sẵn |
| 3 file mẫu | Xử lý riêng ở `xu_ly_du_lieu_mau/`: 8.273 → 7.256 tin, R² (log giá) 0,85 |

---

## 1. Thu thập (`src/crawler.py`, `src/market_price.py`)

1. **Cào theo từng quận** (API `ad-listing`, `area=<id quận>`), 50 tin/trang, lấy hết mọi trang.
   - *Vì sao:* API giới hạn số tin mỗi truy vấn theo `total` của quận; cào theo quận vừa lấy đủ vừa chia sẵn.
2. **Lấy cả tin đã gỡ / hết hạn** (`include_expired_ads=true`) → dữ liệu gần gấp đôi (≈24.700 → 47.928).
   - Tin đang rao sống khoảng 60 ngày; tin đã gỡ lùi thêm khoảng 1 tháng. Không lấy được tin 2 năm trước.
   - `is_removed = 1`: tin đã rời thị trường (đã bán hoặc hết hạn, API không phân biệt).
3. **Retry + backoff; trang lỗi được thử lại; `manifest.json`** ghi trạng thái từng quận.
   - *Vì sao:* bản cũ gặp lỗi mạng sẽ trả trang rỗng, dừng quận đó mà không báo, làm mất dữ liệu âm thầm.
4. **Biểu đồ giá** (API `market-price/charts`, bắt buộc header `Accept: application/json; version=2`):
   đơn giá trung vị theo tháng trong 13 tháng, cho từng phường × loại BĐS × phân nhóm.
   - Cấp đường phố trả về y hệt cấp phường nên chỉ gọi cấp phường (792 lần gọi).
   - Đây chính là cột `bieu_do_gia` của bộ dữ liệu mẫu giảng viên cung cấp.
5. **Không thu thập số điện thoại** (`dien_thoai` trong mẫu): dữ liệu cá nhân, không cần cho bài toán.

## 2. Trích xuất (`raw_ads_to_dataframe`)

- **Giải mã theo nhãn chữ của API (`feature_params`)**, không tự đoán mã số.
  - *Lỗi phát hiện:* cùng một mã nhưng nghĩa khác nhau theo loại BĐS. Pháp lý mã 6 = "Sổ hồng riêng" (căn hộ) nhưng = "Giấy tờ viết tay" (nhà/đất). Bản cũ đã ghi "Chưa xác định" cho toàn bộ căn hộ có sổ hồng riêng.
  - `pty_characteristics` cũng vậy: mã 4 = "Nhà tóp hậu" (nhà ở) nhưng = "Chưa có thổ cư" (đất).
- **Đơn vị hecta → m²** (`size_unit = 2`).
- **Ngày đăng thật = `orig_list_time`**, vì `list_time` đổi mỗi lần tin được "đẩy".
- Giữ cả **phường cũ** (`ward_name`) và **phường mới sau sáp nhập 07/2025** (`ward_name_new`).

## 3. Làm sạch (`clean_data`) – loại cả dòng

| Lý do | Số dòng | Ghi chú |
| :--- | ---: | :--- |
| Trùng mềm (nhiều tin cùng 1 BĐS) | 3.263 | Cùng loại, phường, giá, diện tích, số phòng, số tầng, tầng → giữ tin đăng sớm nhất. Số bản trùng lưu ở `n_duplicate_posts` |
| Văn phòng / mặt bằng (ngoài phạm vi) | 521 | Cấu trúc giá khác hẳn nhà để ở |
| Thiếu giá / diện tích / quận | 437 | Không thể làm biến mục tiêu |
| Chợ Tốt đánh dấu giá không hợp lệ | 146 | `is_price_not_valid` |
| Ngoài ngưỡng giá / diện tích / đơn giá | 71 | Ngưỡng **riêng theo loại BĐS** (xem `CLEANING_THRESHOLDS` trong `config.py`) |
| Diện tích sử dụng > 20 lần diện tích đất | 9 | Nhập sai |

- Ngưỡng đơn giá tối thiểu của đất là 0,2 triệu/m². Bản cũ dùng chung 5 triệu/m² nên loại oan khoảng 11% tin đất (chủ yếu Củ Chi).
- Mọi dòng bị loại được lưu kèm lý do trong `data/processed/rejected_rows.csv`, có thể kiểm tra lại.

## 4. Sửa từng ô (`fix_data_quality`) – không loại dòng, đặt NaN

| Vấn đề | Số dòng | Xử lý |
| :--- | ---: | :--- |
| Tọa độ nằm ngoài TP.HCM | 63 | Đặt NaN |
| Chiều ngang ≤ 0 hoặc > 50 m | 100 | Đặt NaN |
| Chiều dài ≤ 0 hoặc > 200 m | 17 | Đặt NaN |
| Căn hộ > 6 phòng ngủ | 54 | Đặt NaN |
| Nhà ở > 10 tầng | 49 | Đặt NaN |
| Số WC > số phòng ngủ + 5 | 42 | Đặt NaN |
| Diện tích lệch ngang × dài quá 2 lần | 472 | Gắn cờ `size_dim_mismatch` (có thể do nở hậu / diện tích công nhận) |

**Độ chính xác tọa độ (`geo_precision`):** Chợ Tốt không trả tọa độ thật của căn nhà.
- `street` (32.061 tin): tọa độ là tâm con đường trong phường.
- `ward` (11.366 tin): một điểm dùng chung cho nhiều đường, tức là tâm phường.
- → Khi làm mô hình, tọa độ chỉ nên xem ở mức đường / phường. Không nên tính khoảng cách tới tiện ích ở mức mét.

## 5. Đặc trưng (`engineer_features`, `add_market_price_features`)

- `price_per_m2`, `price_per_m2_living`, `total_floor_area_est`, `price_segment`.
- Cờ đặc điểm `char_*` (có cấu trúc) + `is_frontage`; `txt_*` là tín hiệu phụ từ văn bản (đã loại "gần mặt tiền").
- **Cờ thiếu `*_missing`** thay cho việc điền giá trị thiếu ở bước này.
- **Giá thuê trích từ mô tả** ("đang cho thuê 40tr/tháng", "dòng tiền 12tr/tháng"):
  `rent_million_per_month`, `gross_rental_yield_pct` (chỉ giữ tỷ suất trong khoảng 0,3–20%/năm).
  Có 2.994 tin; tỷ suất gộp trung vị **3,0%/năm**, hợp lý với thị trường TP.HCM.
- **Giá khu vực:** `bieu_do_gia` (13 giá trị), `area_median_price_per_m2`, `area_yoy_change_pct`,
  `area_12m_growth_pct`, `area_growth_smoothed_pct` (trung bình 3 tháng cuối so với 3 tháng đầu, giảm nhiễu).
- **Min/max biểu đồ giá** (`area_chart_min`, `area_chart_max`): biên độ đơn giá trung vị 13 tháng của khu vực.

## 6. Chuẩn bị cho mô hình (`src/model_prep.py` → `data/model_ready/<loại>/`)

1. **Mỗi loại BĐS một bộ dữ liệu riêng**: đơn giá căn hộ (m² sàn), nhà (m² đất) và đất không so sánh được với nhau.
2. **Biến mục tiêu `target_log_price = ln(giá tỷ đồng)`**: phân phối đơn giá lệch phải (xem biểu đồ 03).
3. **Chia 80/20 theo nhóm người đăng (`account_id`)**: tin của cùng một môi giới rất giống nhau.
   Nếu chúng nằm ở cả train và test thì kết quả đánh giá sẽ đẹp giả.
4. **Loại ngoại lai cực đoan chỉ trên train** (3 × IQR của log đơn giá theo quận, ngưỡng tính từ train):
   loại 8 tin căn hộ, 89 tin nhà ở, 45 tin đất. Tập test giữ nguyên để đánh giá trung thực.
5. **Tiền xử lý fit trên train, áp dụng cho test** (`preprocessor.joblib`):
   - số: điền trung vị + chuẩn hóa · nhị phân: điền giá trị phổ biến nhất
   - phân loại ít giá trị: điền `missing` + One-Hot (gộp nhóm hiếm < 20 mẫu)
   - phường / đường / dự án: **Target Encoding có cross-fitting** (tránh rò rỉ nhãn)
6. **Không dùng làm biến đầu vào:**
   - Biến rò rỉ, tính từ chính giá: `price_per_m2`, tín hiệu bất thường `grp_*` / `s2_*` / `s3_*` / `price_side`, và giá thuê nêu trong tin.
   - Biến chỉ biết sau khi đăng: `days_on_market`, `is_removed`, `is_bumped`.

| Loại | Train | Test | Người đăng train / test | Biến sau mã hóa | R² (log giá) | Sai số trung vị |
| :--- | ---: | ---: | :---: | ---: | ---: | ---: |
| Căn hộ | 4.956 | 1.253 | 2.003 / 502 | 91 | 0,82 | 12% |
| Nhà ở | 25.214 | 6.087 | 6.758 / 1.693 | 97 | 0,87 | 14% |
| Đất | 4.718 | 1.120 | 1.955 / 495 | 75 | 0,79 | 20% |

Kiểm tra nhanh dùng `HistGradientBoostingRegressor` mặc định, **không tinh chỉnh**. Mục đích chỉ là xác nhận dữ liệu có đủ thông tin, không phải kết quả mô hình cuối.

**Đánh giá theo đơn vị gốc (tỷ đồng):** mô hình học trên log(giá), nên phải đổi về tỷ bằng `exp()` trước khi tính MAE / RMSE như đề yêu cầu.
Với nhà ở: MAE 1,87 tỷ, RMSE 7,23 tỷ, R² (thang tỷ) 0,63. RMSE và R² thang tỷ bị vài căn rất đắt kéo xuống,
nên báo cáo **cả hai thang** và thêm sai số trung vị.

## 6b. PySpark (`pyspark_prep/chuan_bi_spark.py`)

- Đề yêu cầu làm cả sklearn và PySpark. Để so sánh công bằng, **hai môi trường phải dùng cùng một tập test**:
  script đọc `train_raw` / `test_raw.parquet` (đúng cách chia của `split.csv`) và dựng pipeline Spark ML
  (Imputer → StringIndexer + OneHotEncoder → VectorAssembler → StandardScaler), fit trên train.
- `preprocessor.joblib` là của sklearn nên Spark không dùng được. Spark cũng không có Target Encoding ở mọi phiên bản,
  nên phường / đường / dự án được One-Hot (vector thưa).
- Kết quả lưu ở `data/model_ready/nha_o/spark/{train,test}.parquet` (cột `features`, `label`) và `spark/pipeline_model/`.

## 6c. Chuẩn bị cho bài toán 2 (`src/anomaly_signals.py`, `src/anomaly_prep.py` → `data/anomaly_ready/<loại>/`)

1. **Giữ lại các tin bị loại vì giá**: 146 tin Chợ Tốt đánh dấu giá không hợp lệ và các tin vượt ngưỡng giá / đơn giá
   (tổng 189 tin; nhà ở 177). Bài toán 1 bỏ chúng, nhưng chúng chính là các ca bất thường cần phát hiện.
   Cột `in_clean_dataset = 0`, `split = chi_bai_toan_2`.
2. **Nhóm tương đồng** = Phường × Loại BĐS × Phân nhóm (nhà hẻm / mặt phố / biệt thự / liền kề…), cần ≥ 30 tin;
   nhóm nhỏ hơn thì lùi lên Quận × Phân nhóm → Quận → Toàn TP. 34.014 tin so sánh được ở cấp phường.
3. **S3 – P10–P90** (đúng slide 26–27): khoảng cách ra ngoài [P10, P90] của nhóm, chuẩn hóa min-max về [0, 1].
   Khác đề một điểm: dùng khoảng cách **tương đối** (chia cho P10 / P90) thay vì tuyệt đối, vì đơn giá Quận 1 và Củ Chi
   chênh nhau hơn 20 lần. Chặn trần ở phân vị 99 để vài tin nhập sai số 0 không làm các tin khác dồn về 0.
4. **S2 – Min/Max** (slide 24–25): đề gợi ý khung sàn / trần "dựa trên dữ liệu lịch sử hoặc chuyên gia".
   Dữ liệu lịch sử lấy từ **biểu đồ giá 13 tháng** của đúng phường và phân nhóm: sàn = 0,4 × đáy, trần = 2,5 × đỉnh
   (biểu đồ là giá trung vị nên phải nới rộng). Có biểu đồ cho 43.609/43.679 tin; còn lại dùng P1 / P99 của nhóm.
   Kết quả nhà ở: 610 tin vi phạm (1,9%).
5. **S1 – Residual-Z và S4 – Isolation Forest** cần mô hình → chuẩn bị sẵn:
   - `split` giống bài toán 1: huấn luyện mô hình giá trên train rồi dự đoán cho mọi dòng để tính phần dư.
   - Cột `if_*` (đã điền thiếu): log đơn giá, log diện tích, tỷ lệ đơn giá so với nhóm và khu vực, số phòng, số tầng,
     số phòng / 100 m² (bắt ca "nhà 20 m² mà 10 phòng ngủ" như slide 28)…
   - Hàm `residual_z_score`, `isolation_forest_score`, `composite_score` cài đúng công thức slide 22–30.
6. **Nhãn tham chiếu để đánh giá**: `label_chotot_invalid_price` (146 tin nhà ở). Kiểm tra cho thấy đây là tin **giá quá thấp**
   (giá trung vị bằng 68% nhóm; 44% nằm dưới P10, so với 10% ở tin thường). S2 bắt 8,2% số tin này, so với 1,9% ở tin thường
   (gấp 4 lần). Nhãn không đầy đủ (chỉ gồm tin Chợ Tốt bắt được), nên dùng để tham khảo precision / recall của top-k%.

## 6d. Xử lý riêng 3 file mẫu (`xu_ly_du_lieu_mau/`)

Dùng khi giảng viên chỉ yêu cầu phân tích 3 file cấp sẵn. Chi tiết: [xu_ly_du_lieu_mau/README.md](xu_ly_du_lieu_mau/README.md).
- Chuyển chuỗi sang số ("8,6 tỷ", "900 triệu", "1,2 tỷ/m²", "nhiều hơn 10 phòng"…), tách địa chỉ (100% dòng), bỏ số điện thoại.
- 8.273 dòng → loại 309 dòng rỗng, 26 trùng hoàn toàn, 677 trùng cùng một căn… → **7.256 tin**.
- Cùng tên cột, cùng cách tạo đặc trưng, cùng tín hiệu S2/S3 với pipeline chính. Chia train/test ngẫu nhiên phân tầng theo quận
  (bộ mẫu không có mã người đăng). Kiểm tra nhanh: R² (log giá) 0,85, sai số trung vị 12%.

Cách dùng:

```python
import json, pandas as pd
d = "data/model_ready/nha_o/"
feats = json.load(open(d + "features.json"))["feature_names_out"]
train, test = pd.read_parquet(d + "train.parquet"), pd.read_parquet(d + "test.parquet")
X_train, y_train = train[feats], train["target_log_price"]   # giá dự đoán = exp(y)
```

Cần thử cách xử lý khác (vd. không chuẩn hóa, điền thiếu kiểu khác) thì dùng `train_raw.parquet` / `test_raw.parquet` (chưa điền thiếu, chưa mã hóa).

## 7. Hạn chế đã biết (ghi vào báo cáo)

- **Giá rao ≠ giá giao dịch**: giá chào bán thường cao hơn giá chốt.
- **Chưa có chuỗi thời gian dài của từng tin**: cần cào định kỳ (`python run_pipeline.py` hàng tuần) để tích lũy snapshot.
- **Biểu đồ giá cấp phường nhiễu** khi phường ít tin (vd. Cần Giờ ra +156%). Nên dùng bản làm mượt, và bỏ qua quận có < 100 tin.
- **Tin đã gỡ ≠ đã bán**: có thể chỉ là hết hạn đăng.
- **Nhãn tham chiếu cho bài toán 2 chỉ phủ trường hợp "giá quá thấp"** (tin Chợ Tốt đánh dấu); chưa có nhãn cho "giá quá cao".
- **Hệ số khung Min/Max (0,4 / 2,5) là lựa chọn của nhóm**, chỉnh trong `src/anomaly_signals.py` nếu cần.
- **Phạm vi là TP.HCM cũ** (22 quận/huyện), không gồm Bình Dương và Bà Rịa - Vũng Tàu cũ.
- **23 tin đất (0,4%) có đơn giá < 1 triệu/m²**, phần lớn là đất nông nghiệp vùng ven: vẫn giữ, vì đó là giá thật của phân khúc này; 5 tin đất thổ cư trong số này nên xem lại.

## 8. Lệnh chạy

```bash
python run_pipeline.py                      # cào + biểu đồ giá + tiền xử lý + chuẩn bị mô hình + biểu đồ
python run_pipeline.py --mode preprocess-only   # chỉ xử lý lại lần cào mới nhất
python run_pipeline.py --mode model-prep        # chỉ chia train/test + bộ bài toán 2 + vẽ biểu đồ
python pyspark_prep/chuan_bi_spark.py --dir data/model_ready/nha_o   # tiền xử lý phía PySpark
python xu_ly_du_lieu_mau/xu_ly.py               # chỉ xử lý 3 file mẫu của giảng viên
cd slides && tectonic tien_xu_ly.tex            # biên dịch slide
```
