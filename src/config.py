"""
Cấu hình hệ thống Cào & Tiền xử lý dữ liệu Bất động sản TP.HCM từ Nhà Tốt / Chợ Tốt.
Đồ án tốt nghiệp Data Science: Dự đoán giá & Phát hiện bất thường (Anomaly Detection).
"""

from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Cấu hình thư mục dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"              # Mỗi lần cào = 1 thư mục con data/raw/<run_id>/
HISTORY_DIR = DATA_DIR / "history"           # Nhật ký snapshot qua các lần cào (dữ liệu theo thời gian)
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BY_DISTRICT_DIR = PROCESSED_DATA_DIR / "by_district"
BY_TYPE_DIR = PROCESSED_DATA_DIR / "by_property_type"
REPORTS_DIR = BASE_DIR / "reports"

for d in [RAW_DATA_DIR, HISTORY_DIR, PROCESSED_DATA_DIR, BY_DISTRICT_DIR, BY_TYPE_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMEZONE = "Asia/Ho_Chi_Minh"

# API Chợ Tốt / Nhà Tốt
CHOTOT_API_URL = "https://gateway.chotot.com/v1/public/ad-listing"
PAGE_SIZE = 50

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.nhatot.com/",
    "Origin": "https://www.nhatot.com",
}

# Mã vùng TP. Hồ Chí Minh (phạm vi TP.HCM cũ - 22 quận/huyện trước sáp nhập 01/07/2025)
REGION_TPHCM_V2 = 13000

# Danh mục BĐS Nhà Tốt (tham số cg của API)
CATEGORIES = {
    "all_bds": 1000,
    "can_ho": 1010,
    "nha_o": 1020,
    "van_phong": 1030,
    "dat": 1040,
}

# Mã danh mục -> loại BĐS chuẩn hóa
PROPERTY_TYPES = {
    1010: "can_ho",
    1020: "nha_o",
    1030: "van_phong_mat_bang",
    1040: "dat",
}

# Chỉ giữ các loại BĐS để ở / đầu tư đất trong dataset sạch.
# Văn phòng/mặt bằng (1030) có cấu trúc giá khác hẳn -> loại ra (vẫn lưu trong rejected_rows.csv).
INCLUDED_PROPERTY_TYPES = ["can_ho", "nha_o", "dat"]

# Danh sách 22 Quận / Huyện / TP thuộc TP.HCM trên hệ thống Chợ Tốt
DISTRICTS_TPHCM = {
    96: {"name": "Quận 1", "slug": "quan_1", "type": "inner"},
    98: {"name": "Quận 3", "slug": "quan_3", "type": "inner"},
    99: {"name": "Quận 4", "slug": "quan_4", "type": "inner"},
    100: {"name": "Quận 5", "slug": "quan_5", "type": "inner"},
    101: {"name": "Quận 6", "slug": "quan_6", "type": "inner"},
    102: {"name": "Quận 7", "slug": "quan_7", "type": "inner"},
    103: {"name": "Quận 8", "slug": "quan_8", "type": "inner"},
    105: {"name": "Quận 10", "slug": "quan_10", "type": "inner"},
    106: {"name": "Quận 11", "slug": "quan_11", "type": "inner"},
    107: {"name": "Quận 12", "slug": "quan_12", "type": "suburban"},
    108: {"name": "Quận Bình Tân", "slug": "quan_binh_tan", "type": "suburban"},
    109: {"name": "Quận Bình Thạnh", "slug": "quan_binh_thanh", "type": "inner"},
    110: {"name": "Quận Gò Vấp", "slug": "quan_go_vap", "type": "inner"},
    111: {"name": "Quận Phú Nhuận", "slug": "quan_phu_nhuan", "type": "inner"},
    112: {"name": "Quận Tân Bình", "slug": "quan_tan_binh", "type": "inner"},
    113: {"name": "Quận Tân Phú", "slug": "quan_tan_phu", "type": "inner"},
    115: {"name": "Huyện Bình Chánh", "slug": "huyen_binh_chanh", "type": "rural"},
    116: {"name": "Huyện Củ Chi", "slug": "huyen_cu_chi", "type": "rural"},
    117: {"name": "Huyện Hóc Môn", "slug": "huyen_hoc_mon", "type": "rural"},
    118: {"name": "Huyện Nhà Bè", "slug": "huyen_nha_be", "type": "suburban"},
    119: {"name": "Thành phố Thủ Đức", "slug": "tp_thu_duc", "type": "satellite_city"},
    120: {"name": "Huyện Cần Giờ", "slug": "huyen_can_gio", "type": "rural"},
}

# ---------------------------------------------------------------------------
# BẢNG GIẢI MÃ (đối chiếu với nhãn trong feature_params của API, 09/2026).
# Ưu tiên dùng nhãn chữ trong feature_params; các bảng dưới chỉ là dự phòng.
# LƯU Ý: pháp lý và pty_characteristics dùng cùng mã số nhưng NGHĨA KHÁC theo loại BĐS.
# ---------------------------------------------------------------------------
LEGAL_LABELS = {
    "can_ho": {1: "Đã có sổ", 2: "Đang chờ sổ", 4: "Hợp đồng đặt cọc", 5: "Hợp đồng mua bán", 6: "Sổ hồng riêng"},
    "default": {1: "Đã có sổ", 2: "Đang chờ sổ", 3: "Giấy tờ khác", 4: "Không có sổ",
                5: "Sổ chung / công chứng vi bằng", 6: "Giấy tờ viết tay"},
}

# Nhãn pháp lý -> nhóm chuẩn hóa
LEGAL_GROUPS = {
    "Đã có sổ": "co_so_rieng",
    "Sổ hồng riêng": "co_so_rieng",
    "Đang chờ sổ": "cho_so",
    "Sổ chung / công chứng vi bằng": "so_chung_vi_bang",
    "Hợp đồng mua bán": "hop_dong",
    "Hợp đồng đặt cọc": "hop_dong",
    "Giấy tờ viết tay": "giay_tay",
    "Không có sổ": "khong_so",
    "Giấy tờ khác": "khac",
}
SECURE_LEGAL_GROUPS = {"co_so_rieng"}

PTY_CHARACTERISTICS = {
    "nha_o": {1: "mat_tien", 2: "hem_xe_hoi", 3: "no_hau", 4: "top_hau", 5: "dinh_quy_hoach",
              6: "chua_hoan_cong", 7: "nha_nat", 8: "dat_chua_chuyen_tho", 9: "hien_trang_khac"},
    "dat": {1: "mat_tien", 2: "hem_xe_hoi", 3: "no_hau", 4: "chua_co_tho_cu", 5: "tho_cu_1_phan",
            6: "tho_cu_toan_bo", 7: "khong_tho_cu", 8: "dat_chua_chuyen_tho", 9: "hien_trang_khac"},
}
# Tất cả cờ đặc điểm sẽ có cột char_<tên> (0/1) trong dataset
ALL_CHARACTERISTIC_FLAGS = sorted({v for m in PTY_CHARACTERISTICS.values() for v in m.values()})

HOUSE_TYPE_LABELS = {1: "Nhà mặt phố, mặt tiền", 2: "Nhà biệt thự", 3: "Nhà ngõ, hẻm", 4: "Nhà phố liền kề"}
DIRECTION_LABELS = {1: "Đông", 2: "Tây", 3: "Nam", 4: "Bắc", 5: "Đông Bắc", 6: "Đông Nam", 7: "Tây Bắc", 8: "Tây Nam"}
FURNISHING_LABELS = {1: "Nội thất cao cấp", 2: "Nội thất đầy đủ", 3: "Hoàn thiện cơ bản", 4: "Bàn giao thô"}
APARTMENT_TYPE_LABELS = {1: "Chung cư", 2: "Căn hộ dịch vụ, mini", 3: "Duplex", 4: "Penthouse",
                         5: "Tập thể, cư xá", 6: "Officetel"}
LAND_TYPE_LABELS = {1: "Đất thổ cư", 2: "Đất nền dự án", 3: "Đất công nghiệp", 4: "Đất nông nghiệp"}
PROPERTY_STATUS_LABELS = {1: "Chưa bàn giao", 2: "Đã bàn giao"}
COMMERCIAL_TYPE_LABELS = {1: "Shophouse", 2: "Officetel", 3: "Văn phòng", 4: "Mặt bằng kinh doanh"}

# size_unit: None/1 = m², 2 = hecta
SIZE_UNIT_TO_M2 = {None: 1.0, 1: 1.0, 2: 10_000.0}

# ---------------------------------------------------------------------------
# NGƯỠNG LÀM SẠCH THEO TỪNG LOẠI BĐS
# price: VNĐ, size: m², price_per_m2: triệu VNĐ/m²
# ---------------------------------------------------------------------------
CLEANING_THRESHOLDS = {
    "can_ho": {"price": (300e6, 200e9), "size": (15, 500), "price_per_m2": (10, 800)},
    "nha_o": {"price": (300e6, 1000e9), "size": (8, 10_000), "price_per_m2": (1, 2_000)},
    "dat": {"price": (100e6, 1000e9), "size": (20, 500_000), "price_per_m2": (0.2, 2_000)},
}

# Loại tin trùng "mềm" (cùng 1 BĐS do nhiều môi giới đăng lại)
DROP_SOFT_DUPLICATES = True

# Tham số tín hiệu bất thường (nhóm tương đồng, khung Min/Max, P10–P90): xem src/anomaly_signals.py
