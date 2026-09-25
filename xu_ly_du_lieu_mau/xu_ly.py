"""
Tiền xử lý RIÊNG bộ dữ liệu mẫu giảng viên cung cấp (3 file: quan-go-vap.csv, quan-binh-thanh.csv,
quan-phu-nhuan.csv – xem "Mô tả bộ dữ liệu Nhà Tốt.pdf"). Dùng khi chỉ cần phân tích đúng 3 file này.

Chạy (từ thư mục gốc dự án):
    python xu_ly_du_lieu_mau/xu_ly.py
    python xu_ly_du_lieu_mau/xu_ly.py --input "Cung cap HV" --output data/du_lieu_mau

Đầu ra dùng CÙNG tên cột với pipeline chính (run_pipeline.py) nên code mô hình chạy được cho cả hai bộ.
Số điện thoại (dien_thoai) bị bỏ ngay khi đọc: dữ liệu cá nhân, không cần cho bài toán.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import anomaly_signals                                       # noqa: E402
from src.anomaly_prep import DISPLAY_COLS, SIGNAL_COLS, _if_features  # noqa: E402
from src.config import CLEANING_THRESHOLDS                            # noqa: E402
from src.market_price import _smoothed_growth                         # noqa: E402
from src.model_prep import build_preprocessor                         # noqa: E402
from src.preprocessor import MAX_GROSS_YIELD_PCT, TXT_HEM_XE_HOI, TXT_MAT_TIEN, TXT_RENT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("du_lieu_mau")

FILES = {
    "quan-go-vap.csv": "Quận Gò Vấp",
    "quan-binh-thanh.csv": "Quận Bình Thạnh",
    "quan-phu-nhuan.csv": "Quận Phú Nhuận",
}
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Cột "dac_diem" -> cờ char_* (cùng tên với pipeline chính)
DAC_DIEM_FLAGS = {
    "Mặt tiền": "mat_tien", "Hẻm xe hơi": "hem_xe_hoi", "Nhà nở hậu": "no_hau", "Nhà tóp hậu": "top_hau",
    "Nhà dính quy hoạch / lộ giới": "dinh_quy_hoach", "Nhà chưa hoàn công": "chua_hoan_cong",
    "Nhà nát": "nha_nat", "Đất chưa chuyển thổ": "dat_chua_chuyen_tho", "Hiện trạng khác": "hien_trang_khac",
}
LEGAL_GROUPS = {
    "Đã có sổ": "co_so_rieng", "Đang chờ sổ": "cho_so", "Sổ chung / công chứng vi bằng": "so_chung_vi_bang",
    "Giấy tờ viết tay": "giay_tay", "Không có sổ": "khong_so",
}
WARD_PREFIX = re.compile(r"^(Phường|Xã|Thị trấn)\b")


def parse_address(addr: str) -> dict:
    """'[số nhà,] Đường X, Phường Y, Quận Z, Tp Hồ Chí Minh (Phường Mới, TP Hồ Chí Minh mới)'."""
    if not isinstance(addr, str) or not addr.strip():
        return {}
    main, _, new = addr.partition("(")
    parts = [p.strip() for p in main.split(",") if p.strip()]
    i = next((k for k, p in enumerate(parts) if WARD_PREFIX.match(p)), None)
    if i is None:
        return {}
    return {
        "street_number": ", ".join(parts[: max(i - 1, 0)]) or None,
        "street": parts[i - 1] if i >= 1 else None,
        "ward": parts[i],
        "district": parts[i + 1] if i + 1 < len(parts) and parts[i + 1].startswith("Quận") else None,
        "ward_new": new.split(",")[0].strip() or None,
    }

# Biến đưa vào mô hình giá (tập con của pipeline chính, chỉ những cột có trong bộ mẫu)
NUMERIC = ["size", "living_size", "total_floor_area_est", "width", "length", "rooms", "toilets", "floors",
           "area_median_price_per_m2", "area_growth_smoothed_pct", "area_chart_min", "area_chart_max"]
BINARY = ["is_frontage", "has_secure_legal", "txt_mat_tien", "txt_hem_xe_hoi", "size_dim_mismatch", "rooms_over_10",
          "char_hem_xe_hoi", "char_no_hau", "char_top_hau", "char_dinh_quy_hoach", "char_chua_hoan_cong",
          "char_nha_nat", "char_dat_chua_chuyen_tho",
          "rooms_missing", "toilets_missing", "floors_missing", "width_missing", "length_missing",
          "living_size_missing", "legal_group_missing", "direction_missing", "furnishing_missing"]
CATEGORICAL = ["district_name", "house_type", "legal_group", "furnishing", "direction"]
HIGH_CARDINALITY = ["ward_key", "street_key"]


# ----------------------------------------------------------------------------
# Đọc & chuyển chuỗi -> số
# ----------------------------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    """'8,6' -> 8.6 ; '5.2' -> 5.2 (bộ mẫu dùng dấu phẩy cho giá, dấu chấm cho m / m²)."""
    return pd.to_numeric(s.str.extract(r"([\d.,]+)")[0].str.replace(",", ".", regex=False), errors="coerce")


def parse_money_billion(s: pd.Series) -> pd.Series:
    v = _num(s)
    return np.where(s.str.contains("triệu", na=False), v / 1000, v)          # "900 triệu" -> 0,9 tỷ


def parse_price_per_m2_million(s: pd.Series) -> pd.Series:
    v = _num(s)
    return np.where(s.str.contains("tỷ/m", na=False), v * 1000, v)           # "1,2 tỷ/m²" -> 1200 tr/m²


def load(input_dir: Path) -> pd.DataFrame:
    frames = []
    for fname, district in FILES.items():
        f = pd.read_csv(input_dir / fname, dtype=str)
        frames.append(f.assign(source_file=fname, source_district=district))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop(columns=["dien_thoai"], errors="ignore")                  # dữ liệu cá nhân
    raw.insert(0, "row_id", np.arange(len(raw)))
    logger.info(f"Đọc {len(raw)} dòng từ {len(FILES)} file: {raw['source_file'].value_counts().to_dict()}")
    return raw


def to_structured(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"ad_id": raw["row_id"], "source_file": raw["source_file"]})
    df["subject"] = raw["tieu_de"].fillna("").str.strip()
    df["body"] = raw["mo_ta"].fillna("").str.strip()
    df["property_type"] = "nha_o"
    df["house_type"] = raw["loai_hinh"]
    df["price_billion"] = parse_money_billion(raw["gia_ban"])
    df["price"] = df["price_billion"] * 1e9
    df["listed_price_per_m2"] = parse_price_per_m2_million(raw["don_gia"])
    df["size"] = _num(raw["dien_tich"])
    df["land_size"] = _num(raw["dien_tich_dat"])
    df["living_size"] = _num(raw["dien_tich_su_dung"])
    df["rooms"] = _num(raw["so_phong_ngu"])
    df["rooms_over_10"] = raw["so_phong_ngu"].str.contains("nhiều hơn", case=False, na=False).astype(int)
    df.loc[df["rooms_over_10"] == 1, "rooms"] = 11                          # quy ước giống API: 11 = "hơn 10"
    df["toilets"] = _num(raw["so_phong_ve_sinh"])
    df.loc[raw["so_phong_ve_sinh"].str.contains("nhiều hơn", case=False, na=False), "toilets"] = 11
    df["floors"] = _num(raw["tong_so_tang"])
    df["width"] = _num(raw["chieu_ngang"])
    df["length"] = _num(raw["chieu_dai"])
    df["legal_document"] = raw["giay_to_phap_ly"]
    df["legal_group"] = raw["giay_to_phap_ly"].map(LEGAL_GROUPS)
    df["furnishing"] = raw["tinh_trang_noi_that"]
    df["direction"] = raw["huong_cua_chinh"]

    # Địa chỉ: "Đường X, Phường Y, Quận Z, Tp Hồ Chí Minh (Phường Mới, TP Hồ Chí Minh mới)"
    addr = pd.DataFrame(raw["dia_chi"].map(parse_address).tolist(), index=raw.index,
                        columns=["street_number", "street", "ward", "district", "ward_new"])
    df["address"] = raw["dia_chi"].str.strip()
    df["street_number"] = addr["street_number"]
    df["street_name"] = addr["street"].str.replace(r"^Đường\s+", "", regex=True).str.strip()
    df["ward_name"] = addr["ward"].str.strip()
    df["district_name"] = addr["district"].str.strip().fillna(raw["source_district"])
    df["ward_name_new"] = addr["ward_new"].str.strip()
    df["ward_id"] = df["district_name"] + " | " + df["ward_name"].fillna("?")

    # Đặc điểm -> cờ
    tags = raw["dac_diem"].fillna("").str.split(r",\s*")
    for label, flag in DAC_DIEM_FLAGS.items():
        df[f"char_{flag}"] = tags.map(lambda t, lb=label: int(lb in t))

    # Biểu đồ giá khu vực (bieu_do_gia): chuỗi đơn giá trung vị theo tháng, phần tử cuối = tháng gần nhất
    series = raw["bieu_do_gia"].fillna("[]").map(lambda s: [v for v in json.loads(s) if v] if s.startswith("[") else [])
    df["bieu_do_gia"] = raw["bieu_do_gia"]
    df["area_chart_n_months"] = series.map(len)
    df["area_median_price_per_m2"] = series.map(lambda v: v[-1] if v else np.nan)
    df["area_chart_min"] = series.map(lambda v: min(v) if v else np.nan)
    df["area_chart_max"] = series.map(lambda v: max(v) if v else np.nan)
    df["area_12m_growth_pct"] = series.map(lambda v: round((v[-1] - v[0]) / v[0] * 100, 2) if len(v) >= 2 else np.nan)
    df["area_growth_smoothed_pct"] = series.map(lambda v: _smoothed_growth(v) if v else np.nan)
    return df


# ----------------------------------------------------------------------------
# Làm sạch
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame, stats: dict):
    reason = pd.Series(pd.NA, index=df.index, dtype="object")

    def reject(mask, why):
        mask = mask.fillna(False) & reason.isna()
        reason[mask] = why
        stats["rejected"][why] = int(mask.sum())
        logger.info(f"  - Loại {int(mask.sum()):>5} dòng: {why}")

    reject(df["price_billion"].isna() & df["size"].isna() & (df["subject"] == ""), "dong_rong")
    reject(df["price_billion"].isna() | df["size"].isna(), "thieu_gia_hoac_dien_tich")
    content = ["subject", "body", "price_billion", "size", "address", "rooms", "floors", "width", "length"]
    reject(df[content].duplicated(keep="first"), "trung_hoan_toan")

    df["price_per_m2"] = df["price_billion"] * 1000 / df["size"]
    th = CLEANING_THRESHOLDS["nha_o"]
    reject(~df["price"].between(*th["price"]), "gia_ngoai_nguong")
    reject(~df["size"].between(*th["size"]), "dien_tich_ngoai_nguong")
    reject(~df["price_per_m2"].between(*th["price_per_m2"]), "don_gia_ngoai_nguong")
    reject(df["living_size"] > df["size"] * 20, "dien_tich_su_dung_bat_thuong")

    dup_key = ["ward_id", "price_billion", "size", "rooms", "floors"]
    valid = reason.isna()
    df["n_duplicate_posts"] = 1
    df.loc[valid, "n_duplicate_posts"] = df.loc[valid, dup_key].fillna(-1).groupby(dup_key)["size"].transform("size").values
    reject(valid & df[dup_key].fillna(-1).duplicated(keep="first"), "trung_mem_nhieu_tin_cung_nha")

    df["reject_reason"] = reason
    return df


def fix_and_engineer(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    fixes = stats["quality_fixes"]
    for col, bad, why in [
        ("width", (df["width"] <= 0) | (df["width"] > 50), "chieu_ngang_phi_ly"),
        ("length", (df["length"] <= 0) | (df["length"] > 200), "chieu_dai_phi_ly"),
        ("floors", df["floors"] > 10, "qua_10_tang"),
        ("toilets", df["toilets"] > df["rooms"].fillna(0) + 5, "so_wc_vuot_so_phong_ngu_qua_5"),
    ]:
        fixes[why] = int(bad.sum())
        df.loc[bad, col] = np.nan

    # Đơn giá niêm yết khác đơn giá tự tính > 5% -> nhập liệu không nhất quán
    df["price_per_m2_inconsistent"] = ((df["listed_price_per_m2"] - df["price_per_m2"]).abs()
                                       > 0.05 * df["price_per_m2"]).astype(int)
    fixes["don_gia_niem_yet_lech_don_gia_tinh"] = int(df["price_per_m2_inconsistent"].sum())
    ratio = df["size"] / (df["width"] * df["length"])
    df["size_dim_mismatch"] = ((ratio < 0.5) | (ratio > 2)).astype(int)

    df["total_floor_area_est"] = df["living_size"].fillna(df["size"] * df["floors"])
    df["price_per_m2_living"] = df["price_billion"] * 1000 / df["living_size"]
    df["has_secure_legal"] = (df["legal_group"] == "co_so_rieng").astype(int)
    df["is_frontage"] = ((df["house_type"] == "Nhà mặt phố, mặt tiền") | (df["char_mat_tien"] == 1)).astype(int)
    text = (df["subject"] + " " + df["body"]).str.lower()
    df["txt_mat_tien"] = text.str.contains(TXT_MAT_TIEN).astype(int)
    df["txt_hem_xe_hoi"] = text.str.contains(TXT_HEM_XE_HOI).astype(int)
    rent = pd.to_numeric((df["subject"] + " . " + df["body"]).str.extract(TXT_RENT)[0]
                         .str.replace(",", ".", regex=False), errors="coerce")
    yld = rent * 12 / 1000 / df["price_billion"] * 100
    ok = yld.between(0.3, MAX_GROSS_YIELD_PCT)
    df["rent_million_per_month"] = rent.where(ok)
    df["gross_rental_yield_pct"] = yld.where(ok).round(2)
    for col in ["rooms", "toilets", "floors", "width", "length", "living_size", "legal_group", "direction", "furnishing"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df["ward_key"] = df["ward_id"]
    df["street_key"] = df["ward_id"] + " | " + df["street_name"].fillna("?")
    df["target_log_price"] = np.log(df["price_billion"])
    return df


# ----------------------------------------------------------------------------
# Chuẩn bị cho mô hình (bài toán 1) & bài toán 2
# ----------------------------------------------------------------------------
def model_ready(clean_df: pd.DataFrame, out: Path) -> dict:
    # Bộ mẫu không có mã người đăng -> chia ngẫu nhiên 80/20, phân tầng theo quận
    rng = np.random.default_rng(RANDOM_STATE)
    is_test = pd.Series(False, index=clean_df.index)
    for _, idx in clean_df.groupby("district_name").groups.items():
        idx = np.array(idx)
        is_test[rng.choice(idx, size=int(round(len(idx) * TEST_SIZE)), replace=False)] = True
    train, test = clean_df[~is_test].copy(), clean_df[is_test].copy()

    # Loại ngoại lai cực đoan chỉ trên train (3 x IQR log đơn giá theo quận)
    lp = np.log(train["price_per_m2"])
    q1 = lp.groupby(train["district_name"]).transform(lambda s: s.quantile(0.25))
    q3 = lp.groupby(train["district_name"]).transform(lambda s: s.quantile(0.75))
    keep = lp.between(q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))
    n_out = int((~keep).sum())
    train = train[keep].copy()

    usable = lambda cols: [c for c in cols if train[c].notna().any() and train[c].nunique(dropna=True) > 1]  # noqa: E731
    num, binary, cat, high = usable(NUMERIC), usable(BINARY), usable(CATEGORICAL), usable(HIGH_CARDINALITY)
    for col in cat + high:
        train[col] = train[col].astype("object").where(train[col].notna(), None)
        test[col] = test[col].astype("object").where(test[col].notna(), None)
    pre = build_preprocessor(num, binary, cat, high)
    X_train = pre.fit_transform(train[num + binary + cat + high], train["target_log_price"])
    X_test = pre.transform(test[num + binary + cat + high])

    d = out / "model_ready"
    d.mkdir(parents=True, exist_ok=True)
    ids = ["ad_id", "district_name", "ward_name", "price_billion", "price_per_m2", "target_log_price"]
    train.to_parquet(d / "train_raw.parquet", index=False)
    test.to_parquet(d / "test_raw.parquet", index=False)
    for X, raw, name in [(X_train, train, "train"), (X_test, test, "test")]:
        pd.concat([X.reset_index(drop=True), raw[ids].reset_index(drop=True)], axis=1).to_parquet(d / f"{name}.parquet", index=False)
    pd.concat([train[["ad_id"]].assign(split="train"), test[["ad_id"]].assign(split="test")]).to_csv(d / "split.csv", index=False)
    import joblib
    joblib.dump(pre, d / "preprocessor.joblib")
    info = {"n_train": len(train), "n_train_outliers_removed": n_out, "n_test": len(test),
            "features_in": {"numeric": num, "binary": binary, "categorical": cat, "high_cardinality": high},
            "n_features_out": X_train.shape[1], "feature_names_out": list(X_train.columns),
            "target": "target_log_price = ln(price_billion)", "split": "ngẫu nhiên 80/20 phân tầng theo quận (không có mã người đăng)"}
    with open(d / "features.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Kiểm tra nhanh dữ liệu có đủ thông tin
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    feats = list(X_train.columns)
    pred = HistGradientBoostingRegressor(random_state=RANDOM_STATE).fit(X_train, train["target_log_price"]).predict(X_test)
    y, p = np.exp(test["target_log_price"].values), np.exp(pred)
    info["baseline_check"] = {
        "r2_log_price": round(r2_score(test["target_log_price"], pred), 3),
        "mae_billion": round(float(np.mean(np.abs(p - y))), 3),
        "rmse_billion": round(float(np.sqrt(np.mean((p - y) ** 2))), 3),
        "median_ape_pct": round(float(np.median(np.abs(p / y - 1)) * 100), 1),
    }
    logger.info(f"model_ready: train={len(train)} (bỏ {n_out} ngoại lai), test={len(test)}, {len(feats)} biến; "
                f"kiểm tra nhanh {info['baseline_check']}")
    return {k: v for k, v in info.items() if k != "feature_names_out"}


def anomaly_ready(work: pd.DataFrame, split: pd.Series, out: Path) -> dict:
    work = work.copy()
    work["split"] = work["ad_id"].map(split)
    work["split"] = np.where(work["split"].notna(), work["split"],
                             np.where(work["in_clean_dataset"] == 1, "train_outlier_removed", "chi_bai_toan_2"))
    work["label_chotot_invalid_price"] = np.nan   # bộ mẫu không có cờ này
    for c in DISPLAY_COLS:
        if c not in work:
            work[c] = np.nan
    res = pd.concat([work[DISPLAY_COLS + ["split", "in_clean_dataset", "reject_reason", "label_chotot_invalid_price"]
                          + SIGNAL_COLS].reset_index(drop=True),
                     _if_features(work).reset_index(drop=True)], axis=1)
    res["target_log_price"] = np.log(res["price_billion"])
    d = out / "anomaly_ready"
    d.mkdir(parents=True, exist_ok=True)
    res.to_parquet(d / "anomaly.parquet", index=False)
    res.to_csv(d / "anomaly.csv", index=False, encoding="utf-8-sig")
    # Mã hóa mọi dòng bằng bộ tiền xử lý của bài toán 1 -> tính S1 chỉ cần model.predict()
    import joblib
    pre = joblib.load(out / "model_ready" / "preprocessor.joblib")
    frame = work.copy()
    cols = list(pre.feature_names_in_)
    for c in cols:
        if frame[c].dtype == object:
            frame[c] = frame[c].where(frame[c].notna(), None)
    X = pre.transform(frame[cols]).reset_index(drop=True)
    X.insert(0, "ad_id", frame["ad_id"].values)
    X.to_parquet(d / "model_features.parquet", index=False)
    info = {"n_rows": len(res), "split": res["split"].value_counts().to_dict(),
            "n_s2_minmax": int(res["s2_minmax"].sum()), "n_s3_outside_p10_p90": int((res["s3_distance"] > 0).sum()),
            "price_side": res["price_side"].value_counts().to_dict(),
            "minmax_source": res["minmax_source"].value_counts().to_dict(),
            "if_feature_columns": [c for c in res.columns if c.startswith("if_")]}
    logger.info(f"anomaly_ready: {len(res)} tin, S2={info['n_s2_minmax']}, ngoài P10–P90={info['n_s3_outside_p10_p90']}")
    return info


def main():
    ap = argparse.ArgumentParser(description="Tiền xử lý riêng 3 file dữ liệu mẫu giảng viên cung cấp")
    ap.add_argument("--input", default=str(ROOT / "Cung cap HV"), help="Thư mục chứa 3 file CSV mẫu")
    ap.add_argument("--output", default=str(ROOT / "data" / "du_lieu_mau"), help="Thư mục đầu ra")
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"rejected": {}, "quality_fixes": {}}

    raw = load(inp)
    stats["n_raw"] = len(raw)
    df = clean(to_structured(raw), stats)

    # Giữ các tin bị loại vì GIÁ cho bài toán 2 (giống pipeline chính)
    anomaly_keep = df["reject_reason"].isin(["gia_ngoai_nguong", "don_gia_ngoai_nguong"])
    df["in_clean_dataset"] = df["reject_reason"].isna().astype(int)
    work = df[(df["in_clean_dataset"] == 1) | anomaly_keep].copy()
    work = fix_and_engineer(work, stats)
    work = anomaly_signals.compute_price_signals(work)

    clean_df = work[work["in_clean_dataset"] == 1].reset_index(drop=True)
    stats["n_clean"] = len(clean_df)
    stats["by_district"] = clean_df["district_name"].value_counts().to_dict()
    stats["missing_pct"] = (clean_df[["rooms", "toilets", "floors", "width", "length", "living_size", "legal_group",
                                      "furnishing", "direction", "bieu_do_gia"]].isna().mean() * 100).round(1).to_dict()

    clean_df.to_csv(out / "du_lieu_mau_clean.csv", index=False, encoding="utf-8-sig")
    clean_df.to_parquet(out / "du_lieu_mau_clean.parquet", index=False)
    df[df["reject_reason"].notna()].drop(columns=["body"]).to_csv(out / "rejected_rows.csv", index=False, encoding="utf-8-sig")
    summary = (clean_df.groupby("district_name")
               .agg(so_tin=("ad_id", "size"), gia_tv_ty=("price_billion", "median"), dien_tich_tv=("size", "median"),
                    don_gia_tv=("price_per_m2", "median"), ty_le_co_so=("has_secure_legal", "mean"),
                    tang_gia_khu_vuc_tv=("area_growth_smoothed_pct", "median"))
               .round(2))
    summary.to_csv(out / "thong_ke_theo_quan.csv", encoding="utf-8-sig")

    stats["model_ready"] = model_ready(clean_df, out)
    split = pd.read_csv(out / "model_ready" / "split.csv").set_index("ad_id")["split"]
    stats["anomaly_ready"] = anomaly_ready(work, split, out)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + summary.to_string() + "\n")
    print(f"Đầu ra: {out}/ (du_lieu_mau_clean.csv/.parquet, rejected_rows.csv, thong_ke_theo_quan.csv, "
          f"model_ready/, anomaly_ready/, stats.json)")


if __name__ == "__main__":
    main()
