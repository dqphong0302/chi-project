"""
Chuẩn bị dữ liệu cho mô hình dự đoán giá (bước cuối của tiền xử lý).

Cho từng loại BĐS (can_ho, nha_o, dat):
1. Chọn biến: chỉ đặc điểm BĐS + vị trí + giá khu vực. Loại các biến rò rỉ (tính từ chính giá bán)
   và biến chỉ biết sau khi đăng tin (số ngày đăng, đã gỡ...).
2. Chia train/test 80/20 theo NHÓM người đăng (account_id): tin của cùng một môi giới thường gần giống
   nhau, nếu nằm ở cả train và test thì đánh giá sẽ lạc quan giả.
3. Loại ngoại lai cực đoan CHỈ trên train (ngưỡng tính từ train, 3 x IQR của log đơn giá theo quận).
4. Tiền xử lý fit trên train rồi áp dụng cho test:
   - số: điền trung vị + chuẩn hóa (StandardScaler)
   - nhị phân: điền giá trị phổ biến nhất
   - phân loại ít giá trị: điền "missing" + One-Hot (gộp nhóm hiếm < 20 mẫu)
   - phân loại nhiều giá trị (phường, đường, dự án): Target Encoding có cross-fitting (tránh rò rỉ)
5. Biến mục tiêu: log(giá tỷ đồng) -> phân phối gần chuẩn hơn; đổi lại bằng exp().
"""

import json
import logging
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer

from src.config import DATA_DIR, INCLUDED_PROPERTY_TYPES, PROCESSED_DATA_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

MODEL_READY_DIR = DATA_DIR / "model_ready"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_OUTLIER_IQR_K = 3.0
MIN_CATEGORY_FREQ = 20
MIN_ROWS_PER_TYPE = 100      # ít hơn thì bỏ qua loại BĐS đó (vd. khi chạy thử --max-pages)
MIN_ACCOUNTS_PER_TYPE = 10

NUMERIC = [
    "size", "living_size", "total_floor_area_est", "width", "length", "rooms", "toilets", "floors",
    "floor_number", "latitude", "longitude",
    "area_median_price_per_m2", "area_yoy_change_pct", "area_growth_smoothed_pct",
]
BINARY = [
    "is_frontage", "is_alley_address", "has_secure_legal", "has_project", "txt_mat_tien", "txt_hem_xe_hoi",
    "size_dim_mismatch",
    "char_mat_tien", "char_hem_xe_hoi", "char_no_hau", "char_top_hau", "char_dinh_quy_hoach",
    "char_chua_hoan_cong", "char_nha_nat", "char_tho_cu_toan_bo", "char_tho_cu_1_phan",
    "char_chua_co_tho_cu", "char_khong_tho_cu", "char_dat_chua_chuyen_tho",
    "rooms_missing", "toilets_missing", "floors_missing", "width_missing", "length_missing",
    "living_size_missing", "legal_group_missing", "direction_missing", "furnishing_missing",
]
CATEGORICAL = [
    "district_name", "district_type", "house_type", "apartment_type", "land_type", "legal_group",
    "furnishing", "direction", "balcony_direction", "property_status", "geo_precision", "is_main_street",
]
HIGH_CARDINALITY = ["ward_key", "street_key", "project_name"]

# Không dùng làm biến đầu vào (ghi lại để giải thích trong báo cáo)
EXCLUDED_LEAKAGE = [
    "price", "price_billion", "price_per_m2", "price_per_m2_living", "chotot_price_per_m2", "price_segment",
    "ref_median_price_per_m2", "price_deviation_pct", "is_price_outlier", "outlier_type",
    "is_investment_opportunity", "first_seen_price", "price_change_pct",
    "rent_million_per_month", "gross_rental_yield_pct",  # nêu kèm giá trong tin, có tương quan trực tiếp với giá
]
EXCLUDED_POST_LISTING = ["days_on_market", "is_removed", "listing_status", "is_bumped", "n_snapshots", "is_sticky"]


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ward_key"] = df["ward_id"].astype("Int64").astype(str)
    df["street_key"] = df["ward_key"] + "|" + df["street_name"].fillna("?")
    df["is_main_street"] = df["is_main_street"].map({True: "yes", False: "no"})
    df["target_log_price"] = np.log(df["price_billion"])
    return df


def _usable(cols: List[str], train: pd.DataFrame) -> List[str]:
    """Bỏ cột không tồn tại, trống hoàn toàn hoặc chỉ có 1 giá trị trong train (không mang thông tin)."""
    return [c for c in cols if c in train and train[c].notna().any() and train[c].nunique(dropna=True) > 1]


def _drop_train_outliers(train: pd.DataFrame) -> pd.DataFrame:
    log_ppm2 = np.log(train["price_per_m2"])
    q1 = log_ppm2.groupby(train["district_name"]).transform(lambda s: s.quantile(0.25))
    q3 = log_ppm2.groupby(train["district_name"]).transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    keep = log_ppm2.between(q1 - TRAIN_OUTLIER_IQR_K * iqr, q3 + TRAIN_OUTLIER_IQR_K * iqr)
    return train[keep].copy()


def build_preprocessor(num: List[str], binary: List[str], cat: List[str], high: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_FREQ,
                                         sparse_output=False)),
            ]), cat),
            ("te", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("target", TargetEncoder(target_type="continuous", random_state=RANDOM_STATE)),
            ]), high),
        ],
        verbose_feature_names_out=True,
    ).set_output(transform="pandas")


def prepare_property_type(df: pd.DataFrame, ptype: str) -> Dict:
    data = _prepare_frame(df[df["property_type"] == ptype])
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(data, groups=data["account_id"].fillna(-1)))
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    n_train_before = len(train)
    train = _drop_train_outliers(train)

    num, binary = _usable(NUMERIC, train), _usable(BINARY, train)
    cat, high = _usable(CATEGORICAL, train), _usable(HIGH_CARDINALITY, train)
    for col in cat + high:  # kiểu chuỗi đồng nhất cho encoder
        train[col] = train[col].astype("object").where(train[col].notna(), None)
        test[col] = test[col].astype("object").where(test[col].notna(), None)

    pre = build_preprocessor(num, binary, cat, high)
    X_train = pre.fit_transform(train[num + binary + cat + high], train["target_log_price"])
    X_test = pre.transform(test[num + binary + cat + high])

    out = MODEL_READY_DIR / ptype
    out.mkdir(parents=True, exist_ok=True)
    id_cols = ["ad_id", "account_id", "district_name", "ward_name", "listing_status", "price_billion",
               "price_per_m2", "target_log_price"]
    train.to_parquet(out / "train_raw.parquet", index=False)  # chưa điền thiếu / mã hóa (để EDA, thử cách khác)
    test.to_parquet(out / "test_raw.parquet", index=False)
    for X, raw, name in [(X_train, train, "train"), (X_test, test, "test")]:
        pd.concat([X.reset_index(drop=True), raw[id_cols].reset_index(drop=True)], axis=1) \
            .to_parquet(out / f"{name}.parquet", index=False)
    joblib.dump(pre, out / "preprocessor.joblib")

    summary = {
        "property_type": ptype,
        "n_total": len(data),
        "n_train": len(train),
        "n_train_outliers_removed": n_train_before - len(train),
        "n_test": len(test),
        "n_accounts_train": int(train["account_id"].nunique()),
        "n_accounts_test": int(test["account_id"].nunique()),
        "n_features_in": {"numeric": num, "binary": binary, "categorical": cat, "high_cardinality": high},
        "n_features_out": X_train.shape[1],
        "feature_names_out": list(X_train.columns),
        "target": "target_log_price = ln(price_billion)",
        "id_columns_not_features": id_cols,
    }
    with open(out / "features.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"[{ptype}] train={len(train)} (bỏ {summary['n_train_outliers_removed']} ngoại lai), "
                f"test={len(test)}, {X_train.shape[1]} biến sau mã hóa -> {out}")
    return summary


def baseline_check(ptype: str) -> Dict:
    """Kiểm tra nhanh dữ liệu có đủ thông tin: mô hình mặc định HistGradientBoosting (không tinh chỉnh)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_percentage_error, r2_score

    out = MODEL_READY_DIR / ptype
    feats = json.load(open(out / "features.json", encoding="utf-8"))["feature_names_out"]
    train, test = pd.read_parquet(out / "train.parquet"), pd.read_parquet(out / "test.parquet")
    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE).fit(train[feats], train["target_log_price"])
    pred = model.predict(test[feats])
    return {
        "property_type": ptype,
        "r2_log_price": round(r2_score(test["target_log_price"], pred), 3),
        "mape_price_pct": round(mean_absolute_percentage_error(np.exp(test["target_log_price"]), np.exp(pred)) * 100, 1),
        "median_ape_pct": round(float(np.median(np.abs(np.exp(pred) / np.exp(test["target_log_price"]) - 1))) * 100, 1),
    }


def run() -> Dict:
    df = pd.read_parquet(PROCESSED_DATA_DIR / "nhatot_tphcm_clean.parquet")
    types = []
    for p in INCLUDED_PROPERTY_TYPES:
        sub = df[df["property_type"] == p]
        if len(sub) < MIN_ROWS_PER_TYPE or sub["account_id"].nunique() < MIN_ACCOUNTS_PER_TYPE:
            logger.warning(f"[{p}] chỉ có {len(sub)} tin / {sub['account_id'].nunique()} người đăng -> "
                           f"bỏ qua bước chia train/test (cần >= {MIN_ROWS_PER_TYPE} tin, >= {MIN_ACCOUNTS_PER_TYPE} người đăng).")
        else:
            types.append(p)
    summaries = {p: prepare_property_type(df, p) for p in types}
    checks = {p: baseline_check(p) for p in types}
    for c in checks.values():
        logger.info(f"Kiểm tra nhanh [{c['property_type']}]: R²(log giá)={c['r2_log_price']}, "
                    f"MAPE={c['mape_price_pct']}%, sai số trung vị={c['median_ape_pct']}%")
    result = {"splits": {p: {k: v for k, v in s.items() if k != "feature_names_out"} for p, s in summaries.items()},
              "baseline_check": checks}
    with open(REPORTS_DIR / "model_ready_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
