"""
Lưu trữ dữ liệu thô theo từng lần cào (run) và nhật ký snapshot qua thời gian.

Cấu trúc:
    data/raw/<run_id>/raw_<slug>.json   # tin thô của từng quận (ghi ngay sau khi cào xong quận)
    data/raw/<run_id>/manifest.json     # trạng thái từng quận: total, số tin, trang lỗi
    data/history/snapshots.csv.gz       # mỗi dòng = 1 tin tại 1 lần cào (để theo dõi giá / thời gian tồn tại)
"""

import gzip
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import HISTORY_DIR, RAW_DATA_DIR, TIMEZONE

logger = logging.getLogger(__name__)

RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")
SNAPSHOT_FILE = HISTORY_DIR / "snapshots.csv.gz"
SNAPSHOT_COLUMNS = [
    "run_id", "crawled_at", "list_id", "area", "category", "ward",
    "price", "size", "list_time", "orig_list_time", "account_id",
]


def new_run_id() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y%m%d_%H%M%S")


def run_dir(run_id: str) -> Path:
    d = RAW_DATA_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_district(run_id: str, slug: str, ads: List[Dict[str, Any]]) -> Path:
    path = run_dir(run_id) / f"raw_{slug}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ads, f, ensure_ascii=False)
    return path


def load_manifest(run_id: str) -> Dict[str, Any]:
    path = RAW_DATA_DIR / run_id / "manifest.json"
    if not path.exists():
        return {"run_id": run_id, "districts": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(run_id: str, manifest: Dict[str, Any]) -> None:
    with open(run_dir(run_id) / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def list_runs() -> List[str]:
    """Danh sách run_id (sắp xếp tăng dần theo thời gian) có ít nhất 1 file quận."""
    return sorted(
        p.name for p in RAW_DATA_DIR.iterdir()
        if p.is_dir() and RUN_ID_PATTERN.match(p.name) and any(p.glob("raw_*.json"))
    )


def latest_run_id() -> str:
    runs = list_runs()
    if not runs:
        raise FileNotFoundError("Không tìm thấy lần cào nào trong data/raw/<run_id>/. Hãy chạy bước crawl trước.")
    return runs[-1]


def run_crawled_at(run_id: str) -> str:
    """Thời điểm cào (ISO) suy ra từ run_id, dùng cho dữ liệu cũ chưa có trường _crawled_at."""
    return datetime.strptime(run_id, "%Y%m%d_%H%M%S").replace(tzinfo=ZoneInfo(TIMEZONE)).isoformat()


def load_run(run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Gộp toàn bộ file quận của một lần cào (mặc định: lần mới nhất)."""
    run_id = run_id or latest_run_id()
    folder = RAW_DATA_DIR / run_id
    files = sorted(folder.glob("raw_*.json"))
    if not files:
        raise FileNotFoundError(f"Lần cào {run_id} không có file raw_*.json nào.")

    fallback_crawled_at = run_crawled_at(run_id)
    ads: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for ad in json.load(f):
                ad.setdefault("_run_id", run_id)
                ad.setdefault("_crawled_at", fallback_crawled_at)
                ads.append(ad)

    manifest = load_manifest(run_id)
    partial = [k for k, v in manifest.get("districts", {}).items() if v.get("status") != "ok"]
    if partial:
        logger.warning(f"Lần cào {run_id} có quận chưa đầy đủ: {partial}")
    logger.info(f"Đã đọc {len(ads)} tin thô từ {len(files)} file của lần cào {run_id}.")
    return ads


# ---------------------------------------------------------------------------
# Nhật ký snapshot (time series)
# ---------------------------------------------------------------------------
def ads_to_snapshot(ads: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [{
        "run_id": ad.get("_run_id"),
        "crawled_at": ad.get("_crawled_at"),
        "list_id": ad.get("list_id"),
        "area": ad.get("area"),
        "category": ad.get("category"),
        "ward": ad.get("ward"),
        "price": ad.get("price"),
        "size": ad.get("size"),
        "list_time": ad.get("list_time"),
        "orig_list_time": ad.get("orig_list_time"),
        "account_id": ad.get("account_id"),
    } for ad in ads]
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS).drop_duplicates(subset=["run_id", "list_id"])


def append_snapshots(ads: List[Dict[str, Any]]) -> None:
    """Ghi thêm snapshot của một lần cào; bỏ qua nếu run_id đó đã có trong nhật ký."""
    snap = ads_to_snapshot(ads)
    if snap.empty:
        return
    if SNAPSHOT_FILE.exists():
        existing_runs = set(pd.read_csv(SNAPSHOT_FILE, usecols=["run_id"], dtype=str)["run_id"])
        snap = snap[~snap["run_id"].astype(str).isin(existing_runs)]
        if snap.empty:
            logger.info("Snapshot của lần cào này đã có trong nhật ký, bỏ qua.")
            return
        with gzip.open(SNAPSHOT_FILE, "at", encoding="utf-8", newline="") as f:
            snap.to_csv(f, header=False, index=False)
    else:
        snap.to_csv(SNAPSHOT_FILE, index=False, compression="gzip")
    logger.info(f"Đã ghi {len(snap)} snapshot vào {SNAPSHOT_FILE}")


def load_snapshots() -> pd.DataFrame:
    if not SNAPSHOT_FILE.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    return pd.read_csv(SNAPSHOT_FILE, dtype={"run_id": str})
