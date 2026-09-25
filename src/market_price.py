"""
Biểu đồ giá thị trường của Nhà Tốt (cột `bieu_do_gia` trong bộ dữ liệu mẫu).

API: gateway.chotot.com/api-pty/public/market-price/charts (bắt buộc header Accept: application/json; version=2)
Trả về đơn giá trung vị / trung bình (triệu/m²) theo THÁNG trong 13 tháng gần nhất cho từng
Phường x Loại BĐS, tách thêm theo loại nhà (nhà ở), tình trạng bàn giao (căn hộ)...
Đây là nguồn lịch sử giá 12 tháng, bù cho việc tin rao chỉ tồn tại ~60 ngày.

Dữ liệu cấp phường được tính trên toàn thị trường khu vực, nên có thể dùng làm biến vị trí
cho mô hình dự đoán giá, và dùng tăng trưởng giá theo năm để so sánh khu vực đầu tư.
"""

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from src import storage
from src.config import DEFAULT_HEADERS, PROCESSED_DATA_DIR, PROPERTY_TYPES, REGION_TPHCM_V2, TIMEZONE

logger = logging.getLogger(__name__)

CHARTS_URL = "https://gateway.chotot.com/api-pty/public/market-price/charts"
CHART_CATEGORIES = (1010, 1020, 1040)
CHARTS_FILE = "market_price_charts.json"
MONTHLY_FILE = PROCESSED_DATA_DIR / "market_price_monthly.csv"


class MarketPriceCrawler:
    def __init__(self, timeout: int = 30, max_retries: int = 4, delay: Tuple[float, float] = (0.3, 0.6)):
        self.session = requests.Session()
        self.session.headers.update({**DEFAULT_HEADERS, "Accept": "application/json; version=2"})
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay

    def fetch_chart(self, area_v2: int, ward: int, category: int) -> Optional[Dict[str, Any]]:
        """Trả về data của biểu đồ; {} nếu khu vực không có dữ liệu (404); None nếu lỗi mạng."""
        params = {
            "region": REGION_TPHCM_V2, "category": category, "area": area_v2, "ward": ward,
            "include_current": "true", "bubble_ward_empty_street_price": "true", "disable_get_near_by": "true",
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(CHARTS_URL, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json().get("data") or {}
                if r.status_code == 404:
                    return {}
                reason = f"HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as e:
                reason = f"{type(e).__name__}: {e}"
            wait = min(60.0, 2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(f"[chart area={area_v2} ward={ward} cg={category}] {reason}. Thử lại sau {wait:.1f}s")
            time.sleep(wait)
        return None

    def crawl_for_run(self, run_id: Optional[str] = None) -> str:
        """Lấy biểu đồ giá cho mọi (quận, phường, loại BĐS) xuất hiện trong một lần cào."""
        run_id = run_id or storage.latest_run_id()
        ads = storage.load_run(run_id)
        keys = sorted({
            (ad["area_v2"], ad["ward"], ad["category"])
            for ad in ads
            if ad.get("area_v2") and ad.get("ward") and ad.get("category") in CHART_CATEGORIES
        })
        logger.info(f"Lấy biểu đồ giá cho {len(keys)} tổ hợp Phường x Loại BĐS...")

        results, failed = [], []
        for area_v2, ward, category in tqdm(keys, desc="Biểu đồ giá", unit="phường"):
            data = self.fetch_chart(area_v2, ward, category)
            if data is None:
                failed.append((area_v2, ward, category))
            else:
                results.append({"area_v2": area_v2, "ward": ward, "category": category, "data": data})
            time.sleep(random.uniform(*self.delay))

        path = storage.run_dir(run_id) / CHARTS_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"results": results, "failed": failed}, f, ensure_ascii=False)
        logger.info(f"Đã lưu {len(results)} biểu đồ ({len(failed)} lỗi) vào {path}")
        return run_id


def load_charts(run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    run_id = run_id or storage.latest_run_id()
    path = storage.RAW_DATA_DIR / run_id / CHARTS_FILE
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["results"]


def charts_to_monthly(charts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Bảng dài: mỗi dòng = 1 tháng của 1 biểu đồ (phường x loại BĐS x phân nhóm)."""
    rows = []
    for item in charts:
        for chart in (item["data"] or {}).get("charts", []):
            for point in chart.get("value") or []:
                rows.append({
                    "area_v2": item["area_v2"],
                    "ward_id": item["ward"],
                    "category_id": item["category"],
                    "property_type": PROPERTY_TYPES.get(item["category"]),
                    "chart_type_key": chart.get("type_key"),
                    "chart_type_id": chart.get("type_id"),
                    "chart_title": chart.get("title"),
                    "month": pd.to_datetime(point["time"], unit="s", utc=True).tz_convert(TIMEZONE).strftime("%Y-%m"),
                    "median_price_per_m2": point.get("median"),
                    "mean_price_per_m2": point.get("mean"),
                    "last_month_volatility_pct": chart.get("last_month_volatility"),
                    "last_year_volatility_pct": chart.get("last_year_volatility"),
                })
    return pd.DataFrame(rows)


def _smoothed_growth(series: List[Optional[float]], window: int = 3) -> Optional[float]:
    vals = [v for v in series if v]
    if len(vals) < 2 * window:
        return None
    head, tail = sum(vals[:window]) / window, sum(vals[-window:]) / window
    return round((tail - head) / head * 100, 2)


def chart_features(charts: List[Dict[str, Any]], raw_ads: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Gắn biểu đồ giá khu vực vào từng tin. Chọn biểu đồ khớp phân nhóm của tin
    (vd. nhà hẻm -> biểu đồ 'Nhà ngõ, hẻm' của phường), nếu không có thì dùng biểu đồ tổng (type_id 0).
    """
    lookup: Dict[Tuple, Dict[str, Any]] = {}
    for item in charts:
        for chart in (item["data"] or {}).get("charts", []):
            if chart.get("value"):
                lookup[(item["ward"], item["category"], chart.get("type_key"), chart.get("type_id"))] = chart

    type_keys = {k[2] for k in lookup}
    rows = []
    for ad in raw_ads:
        ward, category = ad.get("ward"), ad.get("category")
        chart = None
        for key in type_keys:
            if ad.get(key) is not None and (ward, category, key, ad.get(key)) in lookup:
                chart = lookup[(ward, category, key, ad.get(key))]
                break
        if chart is None:
            chart = next((lookup[k] for k in [(ward, category, t, 0) for t in type_keys] if k in lookup), None)
        if chart is None:
            continue
        series = [p.get("median") for p in chart["value"]]
        first, last = series[0], series[-1]
        rows.append({
            "ad_id": ad.get("list_id"),
            "bieu_do_gia": json.dumps(series),
            "area_chart_title": chart.get("title"),
            "area_median_price_per_m2": chart.get("median"),
            "area_mom_change_pct": round(chart["last_month_volatility"], 2) if chart.get("last_month_volatility") is not None else None,
            "area_yoy_change_pct": round(chart["last_year_volatility"], 2) if chart.get("last_year_volatility") is not None else None,
            "area_12m_growth_pct": round((last - first) / first * 100, 2) if first and last else None,
            # Làm mượt: TB 3 tháng cuối so với TB 3 tháng đầu (giảm nhiễu khi phường ít tin)
            "area_growth_smoothed_pct": _smoothed_growth(series),
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["ad_id"])
