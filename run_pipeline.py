"""
Script điều phối toàn bộ Pipeline:
Cào dữ liệu Nhà Tốt -> Tiền xử lý -> Chia theo Quận / Loại BĐS -> Báo cáo EDA.

Cách sử dụng:
1. Cào toàn bộ 22 quận + tiền xử lý (mặc định, ~25.000 tin, ~10 phút):
   python run_pipeline.py

2. Chạy thử nhanh (2 trang/quận):
   python run_pipeline.py --max-pages 2

3. Cào lại một số quận bị lỗi vào CÙNG lần cào trước đó:
   python run_pipeline.py --mode crawl-only --districts 119 120 --run-id 20260925_111024

4. Chỉ lấy biểu đồ giá 13 tháng theo phường cho lần cào mới nhất:
   python run_pipeline.py --mode market-price

5. Chỉ tiền xử lý lại lần cào mới nhất (hoặc --run-id cụ thể):
   python run_pipeline.py --mode preprocess-only

Lưu ý: Chợ Tốt chỉ giữ tin ~60 ngày. Để có dữ liệu theo thời gian, hãy chạy lệnh 1 định kỳ
(vd. hàng tuần); mỗi lần cào được ghi vào data/history/snapshots.csv.gz.
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import market_price, storage
from src.analyzer import DistrictAnalyzer
from src.config import BY_DISTRICT_DIR, BY_TYPE_DIR, CATEGORIES, PROCESSED_DATA_DIR, REPORTS_DIR
from src.crawler import NhaTotCrawler
from src.preprocessor import RealEstatePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def main():
    parser = argparse.ArgumentParser(description="Chợ Tốt / Nhà Tốt Real Estate Data Pipeline")
    parser.add_argument("--mode", choices=["all", "crawl-only", "market-price", "preprocess-only", "model-prep", "report-only"], default="all")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Số trang tối đa mỗi quận (50 tin/trang). 0 = cào toàn bộ (mặc định).")
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="Chỉ lấy tin đăng trong N ngày gần nhất (mặc định: không giới hạn).")
    parser.add_argument("--active-only", action="store_true",
                        help="Chỉ lấy tin đang rao (mặc định lấy cả tin đã hết hạn / đã gỡ).")
    parser.add_argument("--category", choices=list(CATEGORIES.keys()), default="all_bds")
    parser.add_argument("--districts", nargs="+", type=int, default=None,
                        help="ID quận cần cào (vd: --districts 96 98 102). Để trống = 22 quận.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Lần cào dùng cho preprocess (mặc định: mới nhất), hoặc lần cào cần ghi bổ sung khi crawl.")
    args = parser.parse_args()

    run_id = args.run_id

    if args.mode in ["all", "crawl-only"]:
        crawler = NhaTotCrawler(include_expired=not args.active_only)
        run_id, _ = crawler.crawl(
            district_ids=args.districts,
            category=CATEGORIES[args.category],
            max_pages=args.max_pages or None,
            max_age_days=args.max_age_days,
            run_id=run_id,
        )
        # Ghi snapshot từ toàn bộ dữ liệu của run (kể cả khi chỉ cào bổ sung vài quận)
        storage.append_snapshots(storage.load_run(run_id))
        if args.mode == "crawl-only":
            return

    # Biểu đồ giá 13 tháng theo phường (cột bieu_do_gia của bộ dữ liệu mẫu)
    if args.mode in ["all", "market-price"]:
        run_id = market_price.MarketPriceCrawler().crawl_for_run(run_id)

    if args.mode in ["all", "preprocess-only"]:
        raw_ads = storage.load_run(run_id)
        df_clean = RealEstatePreprocessor().process(raw_ads, market_price.load_charts(run_id or storage.latest_run_id()))
        if df_clean.empty:
            logger.error("Dữ liệu sau khi tiền xử lý rỗng.")
            return

    # Chia train/test + điền thiếu + mã hóa cho mô hình
    if args.mode in ["all", "preprocess-only", "model-prep"]:
        from src import model_prep
        model_prep.run()
        from src import anomaly_prep
        anomaly_prep.run()
        from src import figures
        figures.run()

    clean_file = PROCESSED_DATA_DIR / "nhatot_tphcm_clean.parquet"
    if not clean_file.exists():
        logger.error(f"Không tìm thấy file sạch: {clean_file}")
        return
    df_clean = pd.read_parquet(clean_file)
    summary_df = DistrictAnalyzer.generate_district_summary(df_clean)

    with pd.option_context("display.max_rows", 200, "display.width", 250):
        print("\n" + summary_df.to_string(index=False) + "\n")
    print("CÁC FILE ĐẦU RA:")
    print(f" 1. Dataset sạch:            {PROCESSED_DATA_DIR / 'nhatot_tphcm_clean.csv'} (+ .parquet)")
    print(f" 2. Dòng bị loại + lý do:    {PROCESSED_DATA_DIR / 'rejected_rows.csv'}")
    print(f" 3. Chia theo quận:          {BY_DISTRICT_DIR}/")
    print(f" 4. Chia theo loại BĐS:      {BY_TYPE_DIR}/")
    print(f" 5. Báo cáo EDA:             {REPORTS_DIR}/")
    print(f" 6. Định dạng giống bộ mẫu:  {PROCESSED_DATA_DIR / 'dinh_dang_mau'}/")
    print(f" 7. Giá khu vực theo tháng:  {market_price.MONTHLY_FILE}")
    print(f" 8. Nhật ký snapshot:        {storage.SNAPSHOT_FILE}")


if __name__ == "__main__":
    main()
