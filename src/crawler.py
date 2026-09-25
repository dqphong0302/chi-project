"""
Module Crawler dữ liệu Bất động sản Nhà Tốt / Chợ Tốt.
Cào theo từng Quận/Huyện tại TP.HCM, phân trang, retry có backoff, thử lại trang lỗi,
lưu file thô ngay sau mỗi quận và ghi manifest trạng thái.

Lưu ý: tin trên Chợ Tốt chỉ tồn tại ~60 ngày, API không trả tin cũ hơn.
Muốn có dữ liệu theo thời gian -> chạy cào định kỳ (vd. hàng tuần); mỗi lần cào là 1 snapshot.
"""

import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from tqdm import tqdm

from src import storage
from src.config import (
    CATEGORIES,
    CHOTOT_API_URL,
    DEFAULT_HEADERS,
    DISTRICTS_TPHCM,
    PAGE_SIZE,
    REGION_TPHCM_V2,
    TIMEZONE,
)

logger = logging.getLogger(__name__)


class NhaTotCrawler:
    """Crawler chuyên biệt cho dữ liệu BĐS Nhà Tốt tại TP.HCM."""

    def __init__(self, timeout: int = 20, max_retries: int = 5, delay: Tuple[float, float] = (0.3, 0.7),
                 include_expired: bool = True):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay
        # Lấy cả tin đã hết hạn / đã gỡ (status="deleted", lùi thêm ~1 tháng) -> gần gấp đôi dữ liệu,
        # và cho biết tin nào đã rời thị trường (đã bán hoặc hết hạn).
        self.include_expired = include_expired

    def fetch_page(self, district_id: int, category: int, offset: int) -> Optional[Dict[str, Any]]:
        """Lấy 1 trang tin rao bán. Trả về None nếu thất bại sau khi đã retry (KHÔNG trả trang rỗng giả)."""
        params = {
            "region_v2": REGION_TPHCM_V2,
            "area": district_id,
            "cg": category,
            "st": "s",  # 's' = Mua bán, bỏ qua 'u' = Cho thuê
            "limit": PAGE_SIZE,
            "o": offset,
        }
        if self.include_expired:
            params["include_expired_ads"] = "true"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(CHOTOT_API_URL, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                reason = f"HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as e:
                reason = f"{type(e).__name__}: {e}"

            wait = min(60.0, 2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(f"[area={district_id} o={offset}] {reason}. Thử lại lần {attempt}/{self.max_retries} sau {wait:.1f}s")
            time.sleep(wait)

        logger.error(f"[area={district_id} o={offset}] Thất bại sau {self.max_retries} lần thử.")
        return None

    def crawl_district(
        self,
        district_id: int,
        run_id: str,
        category: int = CATEGORIES["all_bds"],
        max_pages: Optional[int] = None,
        max_age_days: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Cào 1 quận. Trả về (danh sách tin, thông tin trạng thái cho manifest)."""
        info = DISTRICTS_TPHCM.get(district_id, {"name": f"Area {district_id}", "slug": f"area_{district_id}"})
        crawled_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat()
        cutoff_ms = int((time.time() - max_age_days * 86400) * 1000) if max_age_days else None

        first = self.fetch_page(district_id, category, 0)
        if first is None:
            return [], {"name": info["name"], "status": "failed", "total": None, "fetched": 0, "failed_offsets": [0]}

        total = int(first.get("total") or 0)
        n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        if max_pages:
            n_pages = min(n_pages, max_pages)
        logger.info(f"--- {info['name']} (ID {district_id}): total={total}, sẽ cào {n_pages} trang ---")

        ads_by_id: Dict[Any, Dict[str, Any]] = {}
        failed_offsets: List[int] = []
        stopped_by_age = False

        def collect(page: Dict[str, Any]) -> bool:
            """Thêm tin vào kết quả. Trả về True nếu cả trang đã cũ hơn cutoff (dừng phân trang)."""
            page_ads = page.get("ads", [])
            for ad in page_ads:
                if cutoff_ms and ad.get("list_time", 0) < cutoff_ms:
                    continue
                ad["_run_id"] = run_id
                ad["_crawled_at"] = crawled_at
                ads_by_id[ad.get("list_id")] = ad  # trùng do trang bị lệch khi có tin mới -> ghi đè
            # Tin đẩy/ưu tiên có thể cũ nhưng nằm đầu trang, nên chỉ dừng khi CẢ trang đều cũ
            return bool(cutoff_ms and page_ads and all(a.get("list_time", 0) < cutoff_ms for a in page_ads))

        stopped_by_age = collect(first)

        pbar = tqdm(range(1, n_pages), desc=info["name"], unit="trang", leave=False)
        for page_idx in pbar:
            if stopped_by_age:
                break
            offset = page_idx * PAGE_SIZE
            page = self.fetch_page(district_id, category, offset)
            if page is None:
                failed_offsets.append(offset)
                continue
            if not page.get("ads"):
                break  # hết dữ liệu thật sự
            stopped_by_age = collect(page)
            time.sleep(random.uniform(*self.delay))

        # Thử lại các trang lỗi một lần nữa sau khi nghỉ
        if failed_offsets:
            logger.info(f"{info['name']}: thử lại {len(failed_offsets)} trang lỗi sau 30s...")
            time.sleep(30)
            still_failed = []
            for offset in failed_offsets:
                page = self.fetch_page(district_id, category, offset)
                if page is None:
                    still_failed.append(offset)
                else:
                    collect(page)
            failed_offsets = still_failed

        ads = list(ads_by_id.values())
        status = "ok" if not failed_offsets else "partial"
        if failed_offsets:
            logger.error(f"{info['name']}: còn {len(failed_offsets)} trang lỗi {failed_offsets} -> dữ liệu CHƯA ĐỦ.")
        logger.info(f"Hoàn thành {info['name']}: {len(ads)}/{total} tin.")

        return ads, {
            "name": info["name"],
            "status": status,
            "total": total,
            "fetched": len(ads),
            "failed_offsets": failed_offsets,
            "stopped_by_age": stopped_by_age,
            "crawled_at": crawled_at,
        }

    def crawl(
        self,
        district_ids: Optional[Iterable[int]] = None,
        category: int = CATEGORIES["all_bds"],
        max_pages: Optional[int] = None,
        max_age_days: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Cào danh sách quận (mặc định cả 22 quận). Mỗi quận được lưu file ngay khi cào xong."""
        run_id = run_id or storage.new_run_id()
        district_ids = list(district_ids or DISTRICTS_TPHCM.keys())
        manifest = storage.load_manifest(run_id)
        manifest.update({"run_id": run_id, "category": category, "max_pages": max_pages,
                         "max_age_days": max_age_days, "include_expired": self.include_expired})

        logger.info(f"Lần cào {run_id}: {len(district_ids)} quận/huyện -> {storage.run_dir(run_id)}")
        all_ads: List[Dict[str, Any]] = []
        for district_id in district_ids:
            ads, status = self.crawl_district(district_id, run_id, category, max_pages, max_age_days)
            slug = DISTRICTS_TPHCM.get(district_id, {}).get("slug", f"area_{district_id}")
            if ads:
                storage.save_district(run_id, slug, ads)
            manifest["districts"][str(district_id)] = status
            storage.save_manifest(run_id, manifest)
            all_ads.extend(ads)
            time.sleep(random.uniform(*self.delay))

        bad = {k: v["status"] for k, v in manifest["districts"].items() if v["status"] != "ok"}
        if bad:
            logger.warning(f"Các quận chưa cào đầy đủ: {bad}. Có thể chạy lại với --districts ... --run-id {run_id}")
        logger.info(f"Lần cào {run_id} hoàn tất: {len(all_ads)} tin.")
        return run_id, all_ads
