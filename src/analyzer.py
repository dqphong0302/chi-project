"""
Module Báo cáo & Thống kê Khám phá Dữ liệu (Exploratory Data Analysis Report).
Thống kê theo Quận/Huyện x Loại BĐS (không gộp căn hộ, nhà ở, đất vì đơn giá/m² khác bản chất).
"""

import logging

import pandas as pd

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


class DistrictAnalyzer:
    """Tạo báo cáo phân tích tổng quan theo Quận/Huyện tại TP.HCM."""

    @staticmethod
    def generate_district_summary(df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            df.groupby(["district_name", "property_type"])
            .agg(**{
                "Số tin": ("ad_id", "size"),
                "Giá TV (Tỷ)": ("price_billion", "median"),
                "Giá P25 (Tỷ)": ("price_billion", lambda s: s.quantile(0.25)),
                "Giá P75 (Tỷ)": ("price_billion", lambda s: s.quantile(0.75)),
                "Diện tích TV (m2)": ("size", "median"),
                "Đơn giá TV (Tr/m2)": ("price_per_m2", "median"),
                "Tỷ lệ có sổ (%)": ("has_secure_legal", lambda s: s.mean() * 100),
                "Tỷ lệ môi giới (%)": ("is_company_ad", lambda s: s.mean() * 100),
                "Số ngày đăng TV": ("days_on_market", "median"),
                "Tỷ lệ tin đã gỡ (%)": ("is_removed", lambda s: s.mean() * 100),
                "Số cơ hội đầu tư": ("is_investment_opportunity", "sum"),
                "Số tin ngoại lai": ("is_price_outlier", "sum"),
            })
            .round(2)
            .reset_index()
            .rename(columns={"district_name": "Quận / Huyện", "property_type": "Loại BĐS"})
            .sort_values(["Loại BĐS", "Đơn giá TV (Tr/m2)"], ascending=[True, False])
        )
        summary.to_csv(REPORTS_DIR / "eda_summary_by_district.csv", index=False, encoding="utf-8-sig")

        pivot = df.pivot_table(index="district_name", columns="property_type", values="price_per_m2",
                               aggfunc="median").round(1)
        pivot.to_csv(REPORTS_DIR / "median_price_per_m2_district_x_type.csv", encoding="utf-8-sig")

        missing = (df.isna().mean() * 100).round(1).rename("pct_missing").to_frame()
        missing.to_csv(REPORTS_DIR / "missing_values.csv", encoding="utf-8-sig")

        logger.info(f"Đã xuất báo cáo EDA vào: {REPORTS_DIR}")
        return summary
