"""
Tín hiệu phát hiện giá bất thường theo đề bài (topic_Price_Prediction_AnomalyDetection, slide 21–30).

Đề bài dùng 4 tín hiệu, mỗi tín hiệu chuẩn hóa về [0, 1], rồi cộng có trọng số thành điểm 0–100:
    S1 Residual-Z       cần mô hình dự đoán giá    -> tính ở bước modeling (chuẩn bị sẵn khóa & cách tính)
    S2 Min/Max          khung sàn/trần theo khu vực & loại hình   -> TÍNH Ở ĐÂY
    S3 P10–P90          khoảng tin cậy của nhóm tương đồng        -> TÍNH Ở ĐÂY
    S4 Isolation Forest  học máy không giám sát    -> tính ở bước modeling (chuẩn bị sẵn ma trận đặc trưng)

Nhóm tương đồng: Phường x Loại BĐS x Phân nhóm (nhà hẻm / mặt phố / biệt thự...; chung cư / duplex...;
đất thổ cư / nông nghiệp...). Nhóm ít hơn MIN_GROUP_N tin thì lùi dần lên Quận x Phân nhóm -> Quận -> Toàn TP.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Phân nhóm dùng để so sánh "cùng loại hình"
SUBTYPE_COL = {"nha_o": "house_type", "can_ho": "apartment_type", "dat": "land_type"}
MIN_GROUP_N = 30

# S2 Min/Max: khung = [hệ số thấp x đáy biểu đồ giá 13 tháng, hệ số cao x đỉnh biểu đồ giá 13 tháng]
# của đúng phường & phân nhóm (dữ liệu lịch sử của Nhà Tốt). Biểu đồ là đơn giá TRUNG VỊ theo tháng nên
# phải nới rộng để không bắt nhầm nhà đẹp/xấu hợp lệ. Không có biểu đồ -> dùng P1/P99 của nhóm.
MINMAX_LOW_FACTOR = 0.4
MINMAX_HIGH_FACTOR = 2.5

# S3: khoảng cách tương đối ra ngoài [P10, P90], chuẩn hóa min-max; chặn trần ở phân vị 99 của khoảng cách
# để một vài tin nhập sai số 0 không làm các tin khác về ~0.
S3_CAP_QUANTILE = 0.99

GROUP_LEVELS = [
    ("phuong", ["ward_id", "property_type", "subtype"]),
    ("quan_phan_nhom", ["district_name", "property_type", "subtype"]),
    ("quan", ["district_name", "property_type"]),
    ("toan_tp", ["property_type"]),
]


def compute_price_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["subtype"] = None
    for ptype, col in SUBTYPE_COL.items():
        m = df["property_type"] == ptype
        if col in df and m.any():
            df.loc[m, "subtype"] = df.loc[m, col]
    df["subtype"] = df["subtype"].fillna("khong_ro")

    x = df["price_per_m2"]
    df["grp_level"] = None
    for name in ["grp_key", "grp_n", "grp_p01", "grp_p10", "grp_p50", "grp_p90", "grp_p99"]:
        df[name] = np.nan if name != "grp_key" else None

    # Chọn nhóm nhỏ nhất có đủ MIN_GROUP_N tin
    for level, keys in GROUP_LEVELS:
        todo = df["grp_level"].isna()
        if not todo.any():
            break
        g = df.groupby(keys, dropna=False)["price_per_m2"]
        n = g.transform("size")
        ok = todo & ((n >= MIN_GROUP_N) | (level == "toan_tp"))
        df.loc[ok, "grp_level"] = level
        df.loc[ok, "grp_key"] = df.loc[ok, keys].astype(str).agg(" | ".join, axis=1)
        df.loc[ok, "grp_n"] = n[ok]
        for name, q in [("grp_p01", 0.01), ("grp_p10", 0.10), ("grp_p50", 0.50), ("grp_p90", 0.90), ("grp_p99", 0.99)]:
            df.loc[ok, name] = g.transform("quantile", q)[ok]

    # ---- S3: ngoài khoảng tin cậy P10–P90 ----
    below = (df["grp_p10"] - x) / df["grp_p10"]
    above = (x - df["grp_p90"]) / df["grp_p90"]
    d = np.where(x < df["grp_p10"], below, np.where(x > df["grp_p90"], above, 0.0))
    df["s3_distance"] = np.round(d, 4)
    cap = pd.Series(d[d > 0]).quantile(S3_CAP_QUANTILE) if (d > 0).any() else 1.0
    df["s3_percentile"] = np.round(np.clip(d / cap, 0, 1), 4)

    # ---- S2: vi phạm khung Min/Max ----
    has_chart = df["area_chart_min"].notna() & df["area_chart_max"].notna() if "area_chart_min" in df else pd.Series(False, index=df.index)
    chart_low = df.get("area_chart_min", np.nan) * MINMAX_LOW_FACTOR
    chart_high = df.get("area_chart_max", np.nan) * MINMAX_HIGH_FACTOR
    df["minmax_low"] = np.where(has_chart, chart_low, df["grp_p01"]).round(2)
    df["minmax_high"] = np.where(has_chart, chart_high, df["grp_p99"]).round(2)
    df["minmax_source"] = np.where(has_chart, "bieu_do_gia", "p01_p99_nhom")
    df["s2_minmax"] = ((x < df["minmax_low"]) | (x > df["minmax_high"])).astype(int)

    # Hướng lệch: giá quá thấp / quá cao (dùng để chia danh sách kết quả như đề bài)
    low = (x < df["grp_p10"]) | (x < df["minmax_low"])
    high = (x > df["grp_p90"]) | (x > df["minmax_high"])
    df["price_side"] = np.select([low, high], ["qua_thap", "qua_cao"], default="binh_thuong")

    df = df.drop(columns=["subtype"])
    logger.info(f"Tín hiệu bất thường: S2 Min/Max vi phạm {int(df['s2_minmax'].sum())} tin; "
                f"S3 ngoài P10–P90 {int((df['s3_distance'] > 0).sum())} tin; "
                f"nhóm tương đồng: {df['grp_level'].value_counts().to_dict()}")
    return df
