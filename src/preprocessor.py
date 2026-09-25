"""
Module Tiền Xử Lý Dữ Liệu Bất Động Sản (Data Preprocessing Pipeline).
- Trích xuất trường có cấu trúc (giải mã theo nhãn feature_params của API, theo từng loại BĐS)
- Làm sạch theo ngưỡng riêng từng loại BĐS, ghi lại lý do loại bỏ (rejected_rows.csv)
- Loại tin trùng (trùng list_id và trùng "mềm" do nhiều môi giới đăng cùng 1 BĐS)
- KHÔNG điền giá trị thiếu (imputation) ở đây: giữ NaN + cột cờ *_missing,
  việc điền thiếu làm ở bước modeling SAU KHI chia train/test để tránh rò rỉ dữ liệu.
- Nhãn tham chiếu theo nhóm (quận x loại BĐS): ngoại lai, cơ hội đầu tư (chỉ dùng cho EDA,
  KHÔNG dùng làm biến đầu vào mô hình dự đoán giá vì được tính từ chính giá bán).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import market_price, storage
from src.config import (
    ALL_CHARACTERISTIC_FLAGS,
    APARTMENT_TYPE_LABELS,
    BY_DISTRICT_DIR,
    BY_TYPE_DIR,
    CLEANING_THRESHOLDS,
    COMMERCIAL_TYPE_LABELS,
    DIRECTION_LABELS,
    DISTRICTS_TPHCM,
    DROP_SOFT_DUPLICATES,
    FURNISHING_LABELS,
    HOUSE_TYPE_LABELS,
    INCLUDED_PROPERTY_TYPES,
    INVESTMENT_DISCOUNT_PCT,
    LAND_TYPE_LABELS,
    LEGAL_GROUPS,
    LEGAL_LABELS,
    MIN_GROUP_SIZE,
    PROCESSED_DATA_DIR,
    PROPERTY_STATUS_LABELS,
    PROPERTY_TYPES,
    PTY_CHARACTERISTICS,
    SECURE_LEGAL_GROUPS,
    SIZE_UNIT_TO_M2,
    TIMEZONE,
)

logger = logging.getLogger(__name__)

# Từ khóa trong tiêu đề/mô tả (chỉ là tín hiệu phụ, ưu tiên cờ char_* có cấu trúc).
# Loại trừ các cụm "gần/sát/cách/ra mặt tiền" vốn KHÔNG phải nhà mặt tiền.
TXT_MAT_TIEN = re.compile(r"(?<!gần )(?<!sát )(?<!cách )(?<!ra )(?<!thông )mặt tiền")
TXT_HEM_XE_HOI = re.compile(r"hẻm xe hơi|hẻm ô ?tô|hẻm oto|\bhxh\b|(?:xe hơi|ô ?tô|oto) (?:vào|đỗ|đậu|ngủ) (?:tận )?(?:nhà|cửa)")

# Thu nhập cho thuê nhắc trong tin (vd. "đang cho thuê 40tr/tháng", "dòng tiền 12 triệu/tháng")
TXT_RENT = re.compile(
    r"(?:cho thuê|thu nhập|dòng tiền|doanh thu)[^.\n]{0,25}?(\d+(?:[.,]\d+)?)\s*(?:tr|triệu|trieu|m)\b"
    r"[^.\n]{0,8}?(?:/|\s)?\s*(?:tháng|th\b)", re.IGNORECASE)
MAX_GROSS_YIELD_PCT = 20.0

# Khung tọa độ TP.HCM (cũ); 1 tọa độ dùng chung cho >= N tên đường khác nhau = tâm phường
HCM_BBOX = {"lat": (10.3, 11.2), "lon": (106.3, 107.1)}
APPROX_COORD_MIN_STREETS = 3

MISSING_FLAG_COLS = ["rooms", "toilets", "floors", "width", "length", "living_size",
                     "legal_group", "direction", "furnishing", "latitude"]


def _feature_labels(ad: Dict[str, Any]) -> Dict[str, str]:
    """Nhãn chữ do Chợ Tốt hiển thị cho từng trường (nguồn giải mã đáng tin cậy nhất)."""
    labels: Dict[str, str] = {}
    for section in (ad.get("feature_params") or {}).values():
        for item in section.get("items", []) if isinstance(section, dict) else []:
            if item.get("id") and item.get("value") is not None:
                labels.setdefault(item["id"], str(item["value"]).strip())
    return labels


def _decode(ad: Dict[str, Any], labels: Dict[str, str], field: str, mapping: Dict[int, str]) -> Optional[str]:
    code = ad.get(field)
    if code is None:
        return None
    label = labels.get(field)
    # Một số tin có nhãn là chính mã số ("1") -> dùng bảng dự phòng
    if label and not label.isdigit():
        return label.replace("Hướng ", "")
    return mapping.get(code, f"code_{code}")


def _to_datetime(ms: Any) -> pd.Timestamp:
    if not ms:
        return pd.NaT
    return pd.to_datetime(ms, unit="ms", utc=True).tz_convert(TIMEZONE)


class RealEstatePreprocessor:
    """Pipeline tiền xử lý dữ liệu Bất động sản Nhà Tốt TP.HCM."""

    def __init__(self):
        # Thống kê từng bước, lưu ra reports/preprocessing_stats.json (dùng cho ghi chú & slide)
        self.stats: Dict[str, Any] = {"rejected": {}, "quality_fixes": {}}

    # ------------------------------------------------------------------
    # 1. Trích xuất
    # ------------------------------------------------------------------
    def raw_ads_to_dataframe(self, raw_ads: List[Dict[str, Any]]) -> pd.DataFrame:
        records = []
        for ad in raw_ads:
            labels = _feature_labels(ad)
            category = ad.get("category")
            ptype = PROPERTY_TYPES.get(category, f"cg_{category}")

            legal_map = LEGAL_LABELS["can_ho"] if ptype == "can_ho" else LEGAL_LABELS["default"]
            legal_label = _decode(ad, labels, "property_legal_document", legal_map)

            char_map = PTY_CHARACTERISTICS.get(ptype, PTY_CHARACTERISTICS["nha_o"])
            char_codes = ad.get("pty_characteristics") or []
            chars = {char_map.get(c, f"code_{c}") for c in char_codes}

            size_unit = ad.get("size_unit")
            size_factor = SIZE_UNIT_TO_M2.get(size_unit, 1.0)
            size = ad.get("size")
            seller = ad.get("seller_info") or {}
            district_id = ad.get("area")

            record = {
                "ad_id": ad.get("list_id"),
                "chotot_ad_id": ad.get("ad_id"),
                "run_id": ad.get("_run_id"),
                "crawled_at": ad.get("_crawled_at"),
                "subject": str(ad.get("subject") or "").strip(),
                "body": str(ad.get("body") or "").strip(),
                # Phân loại
                "category_id": category,
                "category_name": ad.get("category_name"),
                "property_type": ptype,
                "house_type": _decode(ad, labels, "house_type", HOUSE_TYPE_LABELS),
                "apartment_type": _decode(ad, labels, "apartment_type", APARTMENT_TYPE_LABELS),
                "land_type": _decode(ad, labels, "land_type", LAND_TYPE_LABELS),
                "commercial_type": _decode(ad, labels, "commercial_type", COMMERCIAL_TYPE_LABELS),
                "project_id": ad.get("projectid"),
                "project_name": (ad.get("pty_project_name") or None),
                "property_status": _decode(ad, labels, "property_status", PROPERTY_STATUS_LABELS),
                # Vị trí (địa giới cũ: quận/phường; địa giới mới sau 07/2025: ward_name_new)
                "district_id": district_id,
                "district_name": ad.get("area_name"),
                "district_type": DISTRICTS_TPHCM.get(district_id, {}).get("type"),
                "ward_id": ad.get("ward"),
                "ward_name": ad.get("ward_name"),
                "ward_name_new": ad.get("ward_name_v3"),
                "street_name": ad.get("street_name"),
                "street_id": ad.get("unique_street_id"),
                "street_number": (str(ad.get("street_number")).strip() or None) if ad.get("street_number") else None,
                "address": ", ".join(x for x in [ad.get("street_name") and f"Đường {ad['street_name']}", ad.get("ward_name"),
                                                 ad.get("area_name"), ad.get("region_name")] if x)
                           + (f" ({ad['ward_name_v3']}, TP Hồ Chí Minh mới)" if ad.get("ward_name_v3") else ""),
                "is_main_street": ad.get("is_main_street"),
                "latitude": ad.get("latitude"),
                "longitude": ad.get("longitude"),
                # Giá & diện tích
                "price": ad.get("price"),
                "is_price_not_valid": bool(ad.get("is_price_not_valid", False)),
                "chotot_price_per_m2": ad.get("price_million_per_m2"),
                "size": size * size_factor if size is not None else None,
                "size_unit_raw": ad.get("size_unit_string"),
                "living_size": ad.get("living_size"),
                "width": ad.get("width"),
                "length": ad.get("length"),
                # Cấu trúc
                "rooms": ad.get("rooms"),  # 11 = "nhiều hơn 10 PN"
                "toilets": ad.get("toilets"),
                "floors": ad.get("floors"),
                "floor_number": ad.get("floornumber"),
                "block": ad.get("block"),
                "unit_number": ad.get("unitnumber"),
                "direction": _decode(ad, labels, "direction", DIRECTION_LABELS),
                "balcony_direction": _decode(ad, labels, "balconydirection", DIRECTION_LABELS),
                "furnishing": _decode(ad, labels, "furnishing_sell", FURNISHING_LABELS),
                # Pháp lý
                "legal_document": legal_label,
                "legal_group": LEGAL_GROUPS.get(legal_label) if legal_label else None,
                # Người đăng & chất lượng tin
                "account_id": ad.get("account_id"),
                "is_company_ad": bool(ad.get("company_ad")),   # True = môi giới / công ty
                "is_shop": ad.get("shop") is not None,
                "is_verified": bool(ad.get("is_shop_verified", False)),
                "seller_live_ads": seller.get("live_ads"),
                "seller_sold_ads": seller.get("sold_ads"),
                "number_of_images": ad.get("number_of_images"),
                "has_video": bool(ad.get("has_video") or ad.get("videos")),
                "is_sticky": bool(ad.get("is_sticky", False)),
                "listing_status": ad.get("status"),   # active = đang rao; deleted = đã gỡ / hết hạn
                # Thời gian
                "list_time_ms": ad.get("list_time"),
                "orig_list_time_ms": ad.get("orig_list_time") or ad.get("list_time"),
            }
            for flag in ALL_CHARACTERISTIC_FLAGS:
                record[f"char_{flag}"] = int(flag in chars)
            records.append(record)

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df["list_time"] = df["list_time_ms"].map(_to_datetime)
        df["orig_list_time"] = df["orig_list_time_ms"].map(_to_datetime)
        crawled = pd.to_datetime(df["crawled_at"], utc=True, errors="coerce").dt.tz_convert(TIMEZONE)
        df["crawled_at"] = crawled
        df["is_removed"] = (df["listing_status"] == "deleted").astype(int)
        # Chỉ tính cho tin đang rao; tin đã gỡ không biết thời điểm gỡ nên để NaN
        df["days_on_market"] = ((crawled - df["orig_list_time"]).dt.total_seconds() / 86400).round(1).where(df["is_removed"] == 0)
        df["is_bumped"] = (df["list_time_ms"] != df["orig_list_time_ms"]).astype(int)
        df["post_year"] = df["orig_list_time"].dt.year
        df["post_month"] = df["orig_list_time"].dt.month
        return df

    # ------------------------------------------------------------------
    # 2. Làm sạch (ghi lại lý do loại bỏ từng dòng)
    # ------------------------------------------------------------------
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n0 = len(df)
        self.stats["n_raw"] = n0
        df = df.drop_duplicates(subset=["ad_id"], keep="last").copy()
        self.stats["rejected"]["trung_ad_id"] = n0 - len(df)
        logger.info(f"Loại trùng ad_id: {n0} -> {len(df)} dòng.")

        numeric_cols = ["price", "size", "living_size", "width", "length", "rooms", "toilets", "floors",
                        "floor_number", "latitude", "longitude", "chotot_price_per_m2"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Các trường không áp dụng cho loại BĐS -> NaN (không phải "thiếu")
        df.loc[df["property_type"] == "dat", ["rooms", "toilets", "floors", "living_size"]] = np.nan
        df.loc[df["property_type"] == "can_ho", ["floors", "width", "length"]] = np.nan

        df["price_billion"] = (df["price"] / 1e9).round(3)
        df["price_per_m2"] = (df["price"] / df["size"] / 1e6).round(3)  # triệu/m²

        reason = pd.Series(pd.NA, index=df.index, dtype="object")

        def reject(mask: pd.Series, why: str) -> None:
            mask = mask.fillna(False) & reason.isna()
            reason[mask] = why
            self.stats["rejected"][why] = int(mask.sum())
            logger.info(f"  - Loại {int(mask.sum()):>6} dòng: {why}")

        reject(~df["property_type"].isin(INCLUDED_PROPERTY_TYPES), "loai_bds_khong_thuoc_pham_vi")
        reject(df["is_price_not_valid"], "gia_bi_chotot_danh_dau_khong_hop_le")
        reject(df["price"].isna() | df["size"].isna() | df["district_name"].isna(), "thieu_gia_dien_tich_quan")

        for ptype, th in CLEANING_THRESHOLDS.items():
            is_type = df["property_type"] == ptype
            for col, (lo, hi) in th.items():
                reject(is_type & ~df[col].between(lo, hi), f"{col}_ngoai_nguong_{ptype}")

        reject((df["property_type"] == "nha_o") & (df["living_size"] > df["size"] * 20), "dien_tich_su_dung_bat_thuong")

        # Trùng "mềm": cùng loại, phường, giá, diện tích, số phòng, số tầng -> nhiều khả năng cùng 1 BĐS
        dup_key = ["property_type", "ward_id", "price", "size", "rooms", "floors", "floor_number"]
        valid = reason.isna()
        key_df = df.loc[valid, dup_key].fillna(-1)
        df["n_duplicate_posts"] = 1
        df.loc[valid, "n_duplicate_posts"] = key_df.groupby(dup_key)["price"].transform("size").values
        if DROP_SOFT_DUPLICATES:
            order = df.loc[valid].sort_values("orig_list_time_ms").index
            is_dup = df.loc[order, dup_key].fillna(-1).duplicated(keep="first")
            reject(pd.Series(is_dup.values, index=order).reindex(df.index, fill_value=False), "trung_mem_nhieu_tin_cung_bds")

        rejected = df[reason.notna()].assign(reject_reason=reason[reason.notna()])
        clean = df[reason.isna()].copy()
        logger.info(f"Sau làm sạch: còn {len(clean)} / {len(df)} dòng ({len(rejected)} dòng bị loại).")
        self.stats["n_clean"] = len(clean)
        return clean, rejected

    def fix_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sửa giá trị sai ở từng ô (không loại cả dòng): đặt NaN cho giá trị phi lý và gắn cờ để mô hình biết.
        """
        df = df.copy()
        fixes = self.stats["quality_fixes"]

        def to_nan(mask: pd.Series, cols: List[str], why: str) -> None:
            mask = mask.fillna(False)
            df.loc[mask, cols] = np.nan
            fixes[why] = int(mask.sum())
            logger.info(f"  - Sửa {int(mask.sum()):>6} dòng: {why}")

        # Tọa độ ngoài TP.HCM
        lat, lon = df["latitude"], df["longitude"]
        to_nan(~lat.between(*HCM_BBOX["lat"]) | ~lon.between(*HCM_BBOX["lon"]),
               ["latitude", "longitude"], "toa_do_ngoai_tphcm")
        # Chợ Tốt không trả tọa độ chính xác: mỗi (phường, đường) có 1 tọa độ = tâm con đường;
        # tin không có đường được đặt ở tâm phường (1 điểm dùng chung cho nhiều đường).
        n_streets = df.groupby(["latitude", "longitude"])["street_name"].transform("nunique")
        df["geo_precision"] = np.select(
            [df["latitude"].isna(), n_streets >= APPROX_COORD_MIN_STREETS], ["none", "ward"], default="street")
        for level, n in df["geo_precision"].value_counts().items():
            fixes[f"toa_do_muc_{level}"] = int(n)

        # Kích thước phi lý (gõ nhầm đơn vị / thừa số 0)
        to_nan((df["width"] <= 0) | (df["width"] > 50), ["width"], "chieu_ngang_phi_ly")
        to_nan((df["length"] <= 0) | (df["length"] > 200), ["length"], "chieu_dai_phi_ly")
        ratio = df["size"] / (df["width"] * df["length"])
        df["size_dim_mismatch"] = ((ratio < 0.5) | (ratio > 2)).astype(int)
        fixes["dien_tich_lech_ngang_x_dai_gan_co"] = int(df["size_dim_mismatch"].sum())

        # Cấu trúc phi lý theo loại BĐS
        to_nan((df["property_type"] == "can_ho") & (df["rooms"] > 6), ["rooms"], "can_ho_qua_6_phong_ngu")
        to_nan((df["property_type"] == "nha_o") & (df["floors"] > 10), ["floors"], "nha_o_qua_10_tang")
        to_nan(df["toilets"] > df["rooms"].fillna(0) + 5, ["toilets"], "so_wc_vuot_so_phong_ngu_qua_5")
        return df

    # ------------------------------------------------------------------
    # 3. Đặc trưng
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["price_segment"] = pd.cut(
            df["price_billion"], bins=[0, 3, 7, 15, np.inf], right=False,
            labels=["Bình dân (< 3 tỷ)", "Trung cấp (3 - 7 tỷ)", "Cao cấp (7 - 15 tỷ)", "Hạng sang (>= 15 tỷ)"],
        ).astype(str)

        # Đơn giá theo diện tích sử dụng (nhà ở): so sánh công bằng hơn giữa nhà 1 tầng và nhiều tầng
        df["price_per_m2_living"] = (df["price"] / df["living_size"] / 1e6).round(3)
        df["total_floor_area_est"] = np.where(
            df["property_type"] == "nha_o", df["living_size"].fillna(df["size"] * df["floors"]), df["size"]
        )

        df["has_secure_legal"] = df["legal_group"].isin(SECURE_LEGAL_GROUPS).astype(int)
        df["is_frontage"] = ((df["house_type"] == HOUSE_TYPE_LABELS[1]) | (df["char_mat_tien"] == 1)).astype(int)
        df["is_alley_address"] = df["street_number"].fillna("").str.contains("/").astype(int)
        df["has_project"] = df["project_name"].notna().astype(int)

        text = (df["subject"] + " " + df["body"]).str.lower()
        df["txt_mat_tien"] = text.str.contains(TXT_MAT_TIEN).astype(int)
        df["txt_hem_xe_hoi"] = text.str.contains(TXT_HEM_XE_HOI).astype(int)

        # Thu nhập cho thuê & tỷ suất cho thuê gộp (%/năm) – biến quan trọng cho đánh giá đầu tư
        rent = pd.to_numeric((df["subject"] + " . " + df["body"]).str.extract(TXT_RENT)[0]
                             .str.replace(",", ".", regex=False), errors="coerce")
        gross_yield = rent * 12 / 1000 / df["price_billion"] * 100
        valid_rent = gross_yield.between(0.3, MAX_GROSS_YIELD_PCT)
        df["rent_million_per_month"] = rent.where(valid_rent)
        df["gross_rental_yield_pct"] = gross_yield.where(valid_rent).round(2)

        for col in MISSING_FLAG_COLS:
            df[f"{col}_missing"] = df[col].isna().astype(int)
        return df

    def add_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gắn thông tin theo thời gian từ nhật ký snapshot (các lần cào trước)."""
        snaps = storage.load_snapshots()
        if snaps.empty:
            df["first_seen_at"], df["n_snapshots"], df["first_seen_price"] = pd.NaT, 1, df["price"]
        else:
            snaps = snaps.sort_values("crawled_at")
            hist = snaps.groupby("list_id").agg(
                first_seen_at=("crawled_at", "first"),
                n_snapshots=("run_id", "nunique"),
                first_seen_price=("price", "first"),
            )
            df = df.merge(hist, left_on="ad_id", right_index=True, how="left")
            df["n_snapshots"] = df["n_snapshots"].fillna(1).astype(int)
            df["first_seen_price"] = df["first_seen_price"].fillna(df["price"])
        df["price_change_pct"] = ((df["price"] - df["first_seen_price"]) / df["first_seen_price"] * 100).round(2)
        return df

    # ------------------------------------------------------------------
    # 4. Nhãn tham chiếu: ngoại lai & cơ hội đầu tư (theo quận x loại BĐS)
    # ------------------------------------------------------------------
    def detect_anomalies_and_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        grp = df.groupby(["district_name", "property_type"])["price_per_m2"]
        size = grp.transform("size")
        city = df.groupby("property_type")["price_per_m2"]

        use_local = size >= MIN_GROUP_SIZE
        stats = {}
        for name, q in [("q1", 0.25), ("median", 0.5), ("q3", 0.75)]:
            stats[name] = grp.transform("quantile", q).where(use_local, city.transform("quantile", q))
        iqr = stats["q3"] - stats["q1"]
        lower, upper = stats["q1"] - 1.5 * iqr, stats["q3"] + 1.5 * iqr

        df["ref_group"] = np.where(use_local, df["district_name"] + " | " + df["property_type"], "TP.HCM | " + df["property_type"])
        df["ref_group_size"] = np.where(use_local, size, city.transform("size"))
        df["ref_median_price_per_m2"] = stats["median"].round(2)
        df["price_deviation_pct"] = ((df["price_per_m2"] - stats["median"]) / stats["median"] * 100).round(1)

        low, high = df["price_per_m2"] < lower, df["price_per_m2"] > upper
        df["is_price_outlier"] = (low | high).astype(int)
        df["outlier_type"] = np.select([low, high], ["Ngoại lai thấp", "Ngoại lai cao"], default="Bình thường")

        risky = (df["char_dinh_quy_hoach"] == 1) | (df["char_khong_tho_cu"] == 1) | (df["char_chua_co_tho_cu"] == 1)
        df["is_investment_opportunity"] = (
            (df["price_deviation_pct"] <= INVESTMENT_DISCOUNT_PCT)
            & (df["has_secure_legal"] == 1)
            & ~low           # quá rẻ bất thường -> nghi tin ảo, không tính là cơ hội
            & ~risky
        ).astype(int)

        logger.info(f"Ngoại lai: {df['is_price_outlier'].sum()}, cơ hội đầu tư (theo luật): {df['is_investment_opportunity'].sum()}")
        return df

    # ------------------------------------------------------------------
    # 5. Lưu
    # ------------------------------------------------------------------
    def save_outputs(self, df: pd.DataFrame, rejected: pd.DataFrame) -> Dict[str, str]:
        out = {}
        path = PROCESSED_DATA_DIR / "nhatot_tphcm_clean.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        df.to_parquet(PROCESSED_DATA_DIR / "nhatot_tphcm_clean.parquet", index=False)
        out["all"] = str(path)

        rejected.drop(columns=["body"]).to_csv(PROCESSED_DATA_DIR / "rejected_rows.csv", index=False, encoding="utf-8-sig")

        for old in list(BY_DISTRICT_DIR.glob("*.csv")) + list(BY_TYPE_DIR.glob("*.csv")):
            old.unlink()  # file dẫn xuất của lần chạy trước, tạo lại hoàn toàn bên dưới
        slug_lookup = {info["name"]: info["slug"] for info in DISTRICTS_TPHCM.values()}
        for district_name, group in df.groupby("district_name"):
            slug = slug_lookup.get(district_name, re.sub(r"[^\w]+", "_", district_name.lower()))
            group.to_csv(BY_DISTRICT_DIR / f"{slug}.csv", index=False, encoding="utf-8-sig")
        for ptype, group in df.groupby("property_type"):
            group.to_csv(BY_TYPE_DIR / f"{ptype}.csv", index=False, encoding="utf-8-sig")

        logger.info(f"Đã lưu {len(df)} dòng sạch tại {path} (+ .parquet), "
                    f"chia theo quận -> {BY_DISTRICT_DIR}, theo loại BĐS -> {BY_TYPE_DIR}")
        return out

    def export_sample_format(self, df: pd.DataFrame) -> None:
        """
        Xuất thêm theo đúng tên cột của bộ dữ liệu mẫu (Mô tả bộ dữ liệu Nhà Tốt.pdf), mỗi quận 1 file.
        Khác mẫu: giá trị đã ở dạng số (tỷ, m², triệu/m²) thay vì chuỗi "8,6 tỷ"; không thu thập số điện thoại.
        """
        out_dir = PROCESSED_DATA_DIR / "dinh_dang_mau"
        out_dir.mkdir(exist_ok=True)
        for old in out_dir.glob("*.csv"):
            old.unlink()
        sample = pd.DataFrame({
            "ad_id": df["ad_id"],
            "loai_bds": df["property_type"],
            "tieu_de": df["subject"],
            "gia_ban": df["price_billion"],                 # tỷ
            "don_gia": df["price_per_m2"],                  # triệu/m²
            "dien_tich": df["size"],                        # m²
            "dia_chi": df["address"],
            "mo_ta": df["body"],
            "loai_hinh": df["house_type"].fillna(df["apartment_type"]).fillna(df["land_type"]),
            "dien_tich_dat": df["size"].where(df["property_type"] != "can_ho"),
            "dien_tich_su_dung": df["living_size"].where(df["property_type"] == "nha_o", df["size"].where(df["property_type"] == "can_ho")),
            "gia_m2": df["price_per_m2"],
            "giay_to_phap_ly": df["legal_document"],
            "so_phong_ngu": df["rooms"],
            "so_phong_ve_sinh": df["toilets"],
            "tong_so_tang": df["floors"],
            "tinh_trang_noi_that": df["furnishing"],
            "huong_cua_chinh": df["direction"],
            "dac_diem": df[[c for c in df.columns if c.startswith("char_")]].apply(
                lambda r: ", ".join(c[5:] for c, v in r.items() if v == 1) or None, axis=1),
            "chieu_ngang": df["width"],
            "chieu_dai": df["length"],
            "ma_can": df["unit_number"],
            "ten_phan_khu_lo": df["block"],
            "bieu_do_gia": df.get("bieu_do_gia"),
        })
        slug_lookup = {info["name"]: info["slug"].replace("_", "-") for info in DISTRICTS_TPHCM.values()}
        for district_name, group in sample.groupby(df["district_name"]):
            group.to_csv(out_dir / f"{slug_lookup.get(district_name, district_name)}.csv", index=False, encoding="utf-8-sig")
        logger.info(f"Đã xuất định dạng giống bộ mẫu -> {out_dir}")

    def add_market_price_features(self, df: pd.DataFrame, raw_ads: List[Dict[str, Any]],
                                  charts: List[Dict[str, Any]]) -> pd.DataFrame:
        """Gắn biểu đồ giá 13 tháng của phường (cột bieu_do_gia như bộ dữ liệu mẫu) và tăng trưởng giá khu vực."""
        if not charts:
            logger.warning("Chưa có biểu đồ giá cho lần cào này (chạy --mode market-price). Bỏ qua.")
            return df
        feats = market_price.chart_features(charts, raw_ads)
        df = df.merge(feats, on="ad_id", how="left")
        market_price.charts_to_monthly(charts).to_csv(market_price.MONTHLY_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"Gắn biểu đồ giá cho {df['bieu_do_gia'].notna().sum()}/{len(df)} tin; "
                    f"bảng giá theo tháng -> {market_price.MONTHLY_FILE}")
        return df

    def process(self, raw_ads: List[Dict[str, Any]], charts: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        df_raw = self.raw_ads_to_dataframe(raw_ads)
        if df_raw.empty:
            logger.warning("Không có dữ liệu thô để xử lý.")
            return df_raw

        df_clean, rejected = self.clean_data(df_raw)
        df_clean = self.fix_data_quality(df_clean)
        df = self.engineer_features(df_clean)
        df = self.add_history_features(df)
        df = self.add_market_price_features(df, raw_ads, charts or [])
        df = self.detect_anomalies_and_opportunities(df)
        df = df.drop(columns=["list_time_ms", "orig_list_time_ms", "commercial_type"])
        self.save_outputs(df, rejected)
        self.export_sample_format(df)
        self.save_stats(df)
        return df

    def save_stats(self, df: pd.DataFrame) -> None:
        from src.config import REPORTS_DIR
        st = self.stats
        st["n_final"] = len(df)
        st["by_property_type"] = df["property_type"].value_counts().to_dict()
        st["by_status"] = df["listing_status"].value_counts().to_dict()
        st["n_with_chart"] = int(df["bieu_do_gia"].notna().sum()) if "bieu_do_gia" in df else 0
        st["n_with_rent"] = int(df["rent_million_per_month"].notna().sum())
        st["median_gross_yield_pct"] = float(df["gross_rental_yield_pct"].median())
        st["n_outliers"] = int(df["is_price_outlier"].sum())
        st["n_opportunities"] = int(df["is_investment_opportunity"].sum())
        st["orig_list_time_range"] = [str(df["orig_list_time"].min().date()), str(df["orig_list_time"].max().date())]
        st["run_id"] = str(df["run_id"].iloc[0])
        with open(REPORTS_DIR / "preprocessing_stats.json", "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2, default=str)
