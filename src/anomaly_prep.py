"""
Chuẩn bị dữ liệu cho BÀI TOÁN 2 – phát hiện giá bất thường (đề bài slide 20–31).

Đầu ra: data/anomaly_ready/<loại>/
    anomaly.parquet / anomaly.csv   mỗi dòng 1 tin, gồm:
        - thông tin hiển thị (quận, phường, tiêu đề, giá, diện tích, đơn giá...)
        - split: train / test (giống hệt bài toán 1) / train_outlier_removed / chi_bai_toan_2
        - S2 (s2_minmax) và S3 (s3_percentile) ĐÃ TÍNH, price_side (quá thấp / quá cao)
        - if_*: ma trận đặc trưng đã điền thiếu cho Isolation Forest (S4)
        - nhãn tham chiếu để đánh giá: label_chotot_invalid_price, in_clean_dataset
    columns.json    mô tả cột + cách tính S1, S4 và điểm tổng hợp

Các hàm residual_z_score / isolation_forest_score / composite_score cài đúng công thức của đề,
để nhóm làm mô hình gọi sau khi có mô hình dự đoán giá.
"""

import json
import logging
from typing import Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_DIR, INCLUDED_PROPERTY_TYPES, PARQUET_KWARGS, PROCESSED_DATA_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

ANOMALY_READY_DIR = DATA_DIR / "anomaly_ready"
MODEL_READY_DIR = DATA_DIR / "model_ready"

DISPLAY_COLS = [
    "ad_id", "account_id", "listing_status", "district_name", "ward_name", "ward_name_new", "street_name",
    "subject", "property_type", "house_type", "apartment_type", "land_type", "legal_document",
    "price_billion", "size", "price_per_m2", "rooms", "toilets", "floors", "width", "length",
    "area_chart_title", "area_median_price_per_m2", "area_chart_min", "area_chart_max",
]
SIGNAL_COLS = [
    "grp_level", "grp_key", "grp_n", "grp_p10", "grp_p50", "grp_p90",
    "minmax_low", "minmax_high", "minmax_source", "s2_minmax",
    "s3_distance", "s3_percentile", "price_side",
]
LABEL_COLS = ["in_clean_dataset", "reject_reason", "label_chotot_invalid_price"]


def _if_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng cho Isolation Forest: tìm 'sự kết hợp kỳ lạ' (vd. nhà 20 m² mà 10 phòng ngủ) và giá lệch nhóm."""
    size = df["size"].where(df["size"] > 0)
    f = pd.DataFrame({
        "if_log_price_per_m2": np.log(df["price_per_m2"]),
        "if_log_size": np.log(size),
        "if_ratio_to_group_median": df["price_per_m2"] / df["grp_p50"],
        "if_ratio_to_area_median": df["price_per_m2"] / df["area_median_price_per_m2"],
        "if_rooms": df["rooms"],
        "if_toilets": df["toilets"],
        "if_floors": df["floors"],
        "if_width": df["width"],
        "if_length": df["length"],
        "if_rooms_per_100m2": df["rooms"] / size * 100,
        "if_living_to_land_ratio": df["living_size"] / size,
    }, index=df.index)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[:, f.notna().any()]                  # bỏ cột trống hoàn toàn (vd. số tầng với căn hộ)
    return f.fillna(f.median())                     # IsolationForest không nhận NaN


def prepare_property_type(base: pd.DataFrame, ptype: str) -> Dict:
    df = base[base["property_type"] == ptype].copy()
    split_file = MODEL_READY_DIR / ptype / "split.csv"
    if split_file.exists():
        split = pd.read_csv(split_file).set_index("ad_id")["split"]
        df["split"] = df["ad_id"].map(split)
    else:
        df["split"] = np.nan
    df["split"] = np.where(df["split"].notna(), df["split"],
                           np.where(df["in_clean_dataset"] == 1, "train_outlier_removed", "chi_bai_toan_2"))

    out_df = pd.concat([df[DISPLAY_COLS + ["split"] + LABEL_COLS + SIGNAL_COLS].reset_index(drop=True),
                        _if_features(df).reset_index(drop=True)], axis=1)
    out_df["target_log_price"] = np.log(out_df["price_billion"])

    out = ANOMALY_READY_DIR / ptype
    out.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out / "anomaly.parquet", **PARQUET_KWARGS)
    model_features_for_all_rows(df, MODEL_READY_DIR / ptype, out)
    out_df.to_csv(out / "anomaly.csv", index=False, encoding="utf-8-sig")

    lab = out_df["label_chotot_invalid_price"] == 1
    summary = {
        "property_type": ptype,
        "n_rows": len(out_df),
        "split": out_df["split"].value_counts().to_dict(),
        "n_label_chotot_invalid_price": int(lab.sum()),
        "n_s2_minmax": int(out_df["s2_minmax"].sum()),
        "n_s3_outside_p10_p90": int((out_df["s3_distance"] > 0).sum()),
        "price_side": out_df["price_side"].value_counts().to_dict(),
        # Kiểm tra nhanh: tỷ lệ tin bị Chợ Tốt đánh dấu giá không hợp lệ được S2 / S3 bắt
        "s2_rate_on_invalid": round(float(out_df.loc[lab, "s2_minmax"].mean()), 3) if lab.any() else None,
        "s2_rate_on_valid": round(float(out_df.loc[~lab, "s2_minmax"].mean()), 3),
        "s3_mean_on_invalid": round(float(out_df.loc[lab, "s3_percentile"].mean()), 3) if lab.any() else None,
        "s3_mean_on_valid": round(float(out_df.loc[~lab, "s3_percentile"].mean()), 3),
        "if_feature_columns": [c for c in out_df.columns if c.startswith("if_")],
    }
    with open(out / "columns.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "huong_dan": USAGE}, f, ensure_ascii=False, indent=2)
    logger.info(f"[{ptype}] bài toán 2: {len(out_df)} tin (S2={summary['n_s2_minmax']}, "
                f"ngoài P10–P90={summary['n_s3_outside_p10_p90']}, nhãn Chợ Tốt={summary['n_label_chotot_invalid_price']}) -> {out}")
    return summary


def model_features_for_all_rows(df: pd.DataFrame, model_dir, out) -> None:
    """
    Mã hóa MỌI dòng của bộ bài toán 2 bằng bộ tiền xử lý đã fit ở bài toán 1 -> model_features.parquet
    (cùng cột feature_names_out). Nhờ vậy tính S1 chỉ cần: y_hat = model.predict(X[feats]).
    """
    from src.model_prep import _prepare_frame
    if not (model_dir / "preprocessor.joblib").exists():
        return
    pre = joblib.load(model_dir / "preprocessor.joblib")
    frame = _prepare_frame(df)
    cols = list(pre.feature_names_in_)
    for c in cols:
        if frame[c].dtype == object:
            frame[c] = frame[c].where(frame[c].notna(), None)
    X = pre.transform(frame[cols]).reset_index(drop=True)
    X.insert(0, "ad_id", frame["ad_id"].values)
    X.to_parquet(out / "model_features.parquet", **PARQUET_KWARGS)


USAGE = {
    "S1_residual_z": "Huấn luyện mô hình giá trên split=='train' (target_log_price). Dự đoán cho MỌI dòng "
                     "bằng model_features.parquet (đã mã hóa, cùng cột feature_names_out của bài toán 1), "
                     "e = y - y_hat; Z = (e - mean_e_train) / std_e_train; S1 = min(|Z|, 3) / 3. "
                     "Dùng hàm residual_z_score().",
    "S2_min_max": "Đã tính: s2_minmax = 1 nếu đơn giá < minmax_low hoặc > minmax_high.",
    "S3_p10_p90": "Đã tính: s3_percentile = khoảng cách tương đối ra ngoài [grp_p10, grp_p90], chuẩn hóa về [0,1].",
    "S4_isolation_forest": "IsolationForest().fit(các cột if_*); S4 = (Score_max - Score) / (Score_max - Score_min) "
                           "với Score = decision_function. Dùng hàm isolation_forest_score().",
    "composite": "Total = 100 * (w1*S1 + w2*S2 + w3*S3 + w4*S4) / (w1+w2+w3+w4); top-k% điểm cao nhất = bất thường; "
                 "price_side cho biết quá thấp hay quá cao. Dùng hàm composite_score().",
    "danh_gia": "label_chotot_invalid_price = 1 là tin Chợ Tốt tự đánh dấu giá không hợp lệ -> nhãn tham chiếu "
                "(không đầy đủ) để đo precision/recall của top-k%.",
}


# ----------------------------------------------------------------------------
# Hàm tính S1, S4, điểm tổng hợp theo công thức của đề (dùng ở bước modeling)
# ----------------------------------------------------------------------------
def residual_z_score(y_true: np.ndarray, y_pred: np.ndarray, train_mask: Optional[np.ndarray] = None,
                     z_cap: float = 3.0) -> np.ndarray:
    """S1 = min(|Z|, z_cap) / z_cap, Z tính từ phần dư; trung bình/độ lệch chuẩn lấy trên tập train."""
    e = np.asarray(y_true) - np.asarray(y_pred)
    ref = e[train_mask] if train_mask is not None else e
    z = (e - ref.mean()) / ref.std()
    return np.clip(np.abs(z), 0, z_cap) / z_cap


def isolation_forest_score(decision_function_values: np.ndarray) -> np.ndarray:
    """S4 = (Score_max - Score) / (Score_max - Score_min): điểm càng thấp (càng bất thường) -> S4 càng gần 1."""
    s = np.asarray(decision_function_values)
    return (s.max() - s) / (s.max() - s.min())


def composite_score(s1: Sequence[float], s2: Sequence[float], s3: Sequence[float], s4: Sequence[float],
                    weights: Sequence[float] = (0.4, 0.2, 0.2, 0.2)) -> np.ndarray:
    """Điểm tổng hợp 0–100."""
    w = np.asarray(weights, dtype=float)
    stack = np.vstack([s1, s2, s3, s4]).astype(float)
    return 100 * (w[:, None] * stack).sum(axis=0) / w.sum()


def run() -> Dict:
    base = pd.read_parquet(PROCESSED_DATA_DIR / "nhatot_tphcm_anomaly_base.parquet")
    result = {p: prepare_property_type(base, p) for p in INCLUDED_PROPERTY_TYPES
              if (base["property_type"] == p).sum() >= 30}
    with open(REPORTS_DIR / "anomaly_ready_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
