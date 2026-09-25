# Xử lý riêng bộ dữ liệu mẫu (3 file giảng viên cung cấp)

Dùng khi chỉ cần phân tích **đúng 3 file** trong bộ dữ liệu mẫu: `quan-go-vap.csv`, `quan-binh-thanh.csv`, `quan-phu-nhuan.csv` (mô tả trong "Mô tả bộ dữ liệu Nhà Tốt.pdf"). Thư mục này **không cần cào dữ liệu**.

> 3 file mẫu **không có trong repo**: đề bài ghi rõ dữ liệu cấp chỉ dùng cho học tập, không chia sẻ.
> Hãy tự chép 3 file vào thư mục `Cung cap HV/` ở gốc dự án (hoặc chỉ đường dẫn bằng `--input`).

## Chạy

```bash
python xu_ly_du_lieu_mau/xu_ly.py                                   # đọc "Cung cap HV/", ghi ra data/du_lieu_mau/
python xu_ly_du_lieu_mau/xu_ly.py --input "đường/dẫn/3 file" --output data/du_lieu_mau
```

Chạy mất vài giây. Sau đó, nếu muốn làm bằng PySpark:

```bash
python pyspark_prep/chuan_bi_spark.py --dir data/du_lieu_mau/model_ready
```

## Các bước xử lý

| Bước | Nội dung |
| :--- | :--- |
| 1. Đọc | Gộp 3 file (8.273 dòng); **bỏ cột `dien_thoai`** (dữ liệu cá nhân) |
| 2. Chuyển chuỗi → số | `"8,6 tỷ"` → 8,6; `"900 triệu"` → 0,9 tỷ; `"86,87 triệu/m²"` → 86,87; `"1,2 tỷ/m²"` → 1.200 tr/m²; `"99 m²"`, `"5.2 m"` → số; `"nhiều hơn 10 phòng"` → 11 + cờ `rooms_over_10` |
| 3. Tách địa chỉ | số nhà, đường, phường cũ, quận, phường mới sau sáp nhập (tách đúng 100% dòng) |
| 4. Làm sạch | Loại 309 dòng rỗng, 3 dòng thiếu giá/diện tích, 26 dòng trùng hoàn toàn, 677 tin trùng cùng một căn (cùng phường, giá, diện tích, số phòng, số tầng), 2 dòng diện tích sử dụng phi lý → **còn 7.256 tin** |
| 5. Sửa từng ô | Chiều ngang/dài phi lý, trên 10 tầng, số WC vượt số phòng ngủ quá 5 → NaN; kiểm tra đơn giá niêm yết khớp giá/diện tích |
| 6. Đặc trưng | `dac_diem` → cờ `char_*`; pháp lý → `legal_group`; `bieu_do_gia` → giá khu vực tháng gần nhất, min/max, tăng giá 12 tháng (thô và làm mượt); từ khóa mặt tiền / hẻm xe hơi; giá thuê → tỷ suất cho thuê; cờ thiếu `*_missing` |
| 7. Bài toán 1 | Chia ngẫu nhiên 80/20, phân tầng theo quận (bộ mẫu không có mã người đăng); loại ngoại lai cực đoan chỉ trên train; điền thiếu + mã hóa fit trên train; biến mục tiêu `ln(giá tỷ)` |
| 8. Bài toán 2 | Tín hiệu S2 (Min/Max từ `bieu_do_gia`) và S3 (P10–P90 theo phường × loại nhà) đã tính sẵn; ma trận `if_*` cho Isolation Forest; hàm tính S1, S4 và điểm tổng hợp nằm ở `src/anomaly_prep.py` |

Tên cột đầu ra **giống hệt pipeline chính** (`price_billion`, `size`, `rooms`, `ward_name`, `char_hem_xe_hoi`…), nên cùng một đoạn code mô hình chạy được cho cả 3 file mẫu và dữ liệu cào 22 quận. Ý nghĩa từng cột: [../BAO_CAO_TIEN_XU_LY.md](../BAO_CAO_TIEN_XU_LY.md).

| Cột gốc | Cột sau xử lý |
| :--- | :--- |
| tieu_de / mo_ta | subject / body |
| gia_ban | price_billion (tỷ), price |
| don_gia, gia_m2 | listed_price_per_m2 (niêm yết); price_per_m2 (tự tính = giá / diện tích) |
| dien_tich, dien_tich_dat | size, land_size |
| dien_tich_su_dung | living_size |
| dia_chi | address, street_number, street_name, ward_name, district_name, ward_name_new |
| loai_hinh | house_type |
| giay_to_phap_ly | legal_document, legal_group, has_secure_legal |
| so_phong_ngu / so_phong_ve_sinh / tong_so_tang | rooms (+ rooms_over_10) / toilets / floors |
| tinh_trang_noi_that / huong_cua_chinh | furnishing / direction |
| dac_diem | char_hem_xe_hoi, char_no_hau, char_nha_nat, … |
| chieu_ngang / chieu_dai | width / length |
| bieu_do_gia | bieu_do_gia (giữ nguyên) + area_median_price_per_m2, area_chart_min/max, area_12m_growth_pct, area_growth_smoothed_pct |
| ma_can, ten_phan_khu_lo | bỏ (trống 99,7%) |
| dien_thoai | bỏ (dữ liệu cá nhân) |

## Đầu ra (`data/du_lieu_mau/`)

```
du_lieu_mau_clean.csv / .parquet     7.256 tin sạch
rejected_rows.csv                    dòng bị loại + lý do
thong_ke_theo_quan.csv               giá, diện tích, đơn giá trung vị theo quận
stats.json                           số liệu từng bước
model_ready/                         bài toán 1: train/test (5.802 / 1.450), split.csv, preprocessor.joblib, features.json
model_ready/spark/                   (sau khi chạy chuan_bi_spark.py) vector đặc trưng cho PySpark
anomaly_ready/anomaly.parquet/.csv   bài toán 2: S2, S3, price_side, cột if_*, split
```

## Kết quả lần chạy 25/09/2026

| Quận | Số tin | Giá trung vị | Đơn giá trung vị |
| :--- | ---: | ---: | ---: |
| Gò Vấp | 3.837 | 6,3 tỷ | 126 tr/m² |
| Bình Thạnh | 2.307 | 6,9 tỷ | 152 tr/m² |
| Phú Nhuận | 1.112 | 8,3 tỷ | 197 tr/m² |

Kiểm tra nhanh bằng mô hình mặc định (`HistGradientBoosting`, không tinh chỉnh) trên tập test: R² (log giá) = **0,85**, MAE = 1,44 tỷ, RMSE = 2,79 tỷ, sai số trung vị 12%. Dữ liệu đủ thông tin cho bài toán dự đoán giá.

Bài toán 2 (7.256 tin): 50 tin vi phạm Min/Max; theo khoảng P10–P90 có 761 tin giá quá thấp, 745 tin giá quá cao.

## Khác biệt so với dữ liệu cào 22 quận

| | 3 file mẫu | Dữ liệu cào (run_pipeline.py) |
| :--- | :--- | :--- |
| Phạm vi | 3 quận, chỉ nhà ở | 22 quận, nhà ở + căn hộ + đất |
| Chia train/test | Ngẫu nhiên, phân tầng theo quận | Theo người đăng (`account_id`), tránh rò rỉ giữa tin của cùng một môi giới |
| Thời gian đăng, tin đã gỡ | Không có | Có |
| Tọa độ, dự án, mã người đăng | Không có | Có |
| Nhãn tham chiếu cho bài toán 2 | Không có | `label_chotot_invalid_price` (146 tin nhà ở) |
