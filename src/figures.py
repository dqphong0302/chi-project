"""
Sinh biểu đồ (PNG) cho báo cáo / slide tiền xử lý -> reports/figures/.
Bảng màu: 3 màu phân loại đã kiểm tra khả năng phân biệt cho người mù màu (xanh dương, cam, xanh ngọc);
dải tuần tự 1 màu xanh dương; phân kỳ xanh <-> đỏ với trung điểm xám.
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

FIG_DIR = REPORTS_DIR / "figures"
TYPE_COLORS = {"nha_o": "#2a78d6", "can_ho": "#eb6834", "dat": "#1baf7a"}
TYPE_LABELS = {"nha_o": "Nhà ở", "can_ho": "Căn hộ", "dat": "Đất"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})

REASON_LABELS = {
    "trung_mem_nhieu_tin_cung_bds": "Trùng mềm (nhiều tin cùng 1 BĐS)",
    "loai_bds_khong_thuoc_pham_vi": "Văn phòng / mặt bằng (ngoài phạm vi)",
    "thieu_gia_dien_tich_quan": "Thiếu giá / diện tích / quận",
    "gia_bi_chotot_danh_dau_khong_hop_le": "Chợ Tốt đánh dấu giá không hợp lệ",
}


def _save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def fig_rejections(stats):
    rej = {k: v for k, v in stats["rejected"].items() if v > 0}
    other = sum(v for k, v in rej.items() if k not in REASON_LABELS)
    items = [(REASON_LABELS[k], v) for k, v in rej.items() if k in REASON_LABELS] + [("Ngoài ngưỡng giá / diện tích / đơn giá", other)]
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh([i[0] for i in items], [i[1] for i in items], color=BLUE, height=0.55)
    for y, (_, v) in enumerate(items):
        ax.text(v + 40, y, f"{v:,}".replace(",", "."), va="center", color=INK2, fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Số dòng bị loại")
    ax.set_title(f"Loại {sum(rej.values()):,} / {stats['n_raw']:,} tin thô".replace(",", "."))
    ax.tick_params(axis="y", colors=INK2, length=0)
    _save(fig, "01_rejections.png")


def fig_missing(df):
    cols = {"rooms": "Phòng ngủ", "toilets": "WC", "floors": "Số tầng", "width": "Ngang", "length": "Dài",
            "living_size": "DT sử dụng", "legal_group": "Pháp lý", "furnishing": "Nội thất",
            "direction": "Hướng", "project_name": "Dự án", "street_name": "Tên đường", "latitude": "Tọa độ"}
    applicable = {"can_ho": ["rooms", "toilets", "legal_group", "furnishing", "direction", "project_name", "street_name", "latitude"],
                  "nha_o": ["rooms", "toilets", "floors", "width", "length", "living_size", "legal_group", "furnishing", "direction", "street_name", "latitude"],
                  "dat": ["width", "length", "legal_group", "direction", "street_name", "latitude"]}
    mat = pd.DataFrame(index=list(cols.values()), columns=[TYPE_LABELS[t] for t in TYPE_LABELS], dtype=float)
    for t, c in applicable.items():
        sub = df[df.property_type == t]
        for col in c:
            mat.loc[cols[col], TYPE_LABELS[t]] = sub[col].isna().mean() * 100
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.imshow(mat.fillna(-1).values.astype(float), cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    ax.imshow(np.where(mat.isna(), 1, np.nan), cmap=LinearSegmentedColormap.from_list("na", [NEUTRAL, NEUTRAL]), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            ax.text(j, i, "không áp dụng" if pd.isna(v) else f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    color=MUTED if pd.isna(v) else ("white" if v > 55 else INK))
    ax.set_xticks(range(mat.shape[1]), mat.columns, color=INK2)
    ax.set_yticks(range(mat.shape[0]), mat.index, color=INK2)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Tỷ lệ thiếu dữ liệu (%) theo loại BĐS")
    _save(fig, "02_missing.png")


def fig_price_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    for ax, t in zip(axes, TYPE_LABELS):
        v = df.loc[df.property_type == t, "price_per_m2"]
        bins = np.logspace(np.log10(v.quantile(0.002)), np.log10(v.quantile(0.998)), 40)
        ax.hist(v, bins=bins, color=TYPE_COLORS[t], edgecolor=SURFACE, linewidth=0.6)
        ax.set_xscale("log")
        ax.axvline(v.median(), color=INK, linewidth=1, linestyle="--")
        ax.text(v.median(), ax.get_ylim()[1] * 0.92, f"  trung vị {v.median():.0f}", color=INK, fontsize=9)
        ax.set_title(f"{TYPE_LABELS[t]} (n={len(v):,})".replace(",", "."), fontsize=11)
        ax.set_xlabel("Đơn giá (triệu/m², thang log)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Số tin")
    fig.suptitle("Phân phối đơn giá lệch phải -> dùng log(giá) làm biến mục tiêu", x=0.01, ha="left",
                 fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    _save(fig, "03_price_distribution.png")


def fig_district_price(df):
    sub = df[df.property_type == "nha_o"]
    med = sub.groupby("district_name")["price_per_m2"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(med.index, med.values, color=BLUE, height=0.6)
    for y, v in enumerate(med.values):
        ax.text(v + 3, y, f"{v:.0f}", va="center", color=INK2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.set_xlabel("Đơn giá trung vị (triệu/m² đất)")
    ax.set_title("Nhà ở: đơn giá trung vị theo quận/huyện")
    _save(fig, "04_district_price_nha_o.png")


def fig_growth_yield(df):
    sub = df[df.property_type == "nha_o"]
    g = sub.groupby("district_name").agg(growth=("area_growth_smoothed_pct", "median"),
                                         yld=("gross_rental_yield_pct", "median"),
                                         n_yld=("gross_rental_yield_pct", "count"),
                                         n=("ad_id", "size"))
    g["growth"] = g["growth"].where(g["n"] >= 100)  # quá ít tin -> chuỗi giá phường rất nhiễu
    g = g.sort_values("growth", na_position="first")
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    ax = axes[0]
    colors = [BLUE if v >= 0 else RED for v in g.growth.fillna(0)]
    ax.barh(g.index, g.growth, color=colors, height=0.6)
    ax.axvline(0, color=AXIS, linewidth=1)
    for y, v in enumerate(g.growth):
        if pd.isna(v):
            ax.text(0.3, y, "ít dữ liệu", va="center", color=MUTED, fontsize=9)
        else:
            ax.text(v + (0.3 if v >= 0 else -0.3), y, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", color=INK2, fontsize=9)
    ax.set_title("Tăng giá khu vực 12 tháng (làm mượt)")
    ax.set_xlabel("% (TB 3 tháng cuối so với 3 tháng đầu)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax = axes[1]
    y = g.yld.where(g.n_yld >= 15)
    ax.barh(g.index, y, color="#1c5cab", height=0.6)
    for i, (v, n) in enumerate(zip(y, g.n_yld)):
        ax.text((0 if pd.isna(v) else v) + 0.08, i, "ít dữ liệu" if pd.isna(v) else f"{v:.1f}%  (n={n})",
                va="center", color=MUTED if pd.isna(v) else INK2, fontsize=9)
    ax.set_title("Tỷ suất cho thuê gộp (trích từ tin)")
    ax.set_xlabel("%/năm, trung vị")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Nhà ở: tín hiệu đầu tư theo quận", x=0.01, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    _save(fig, "05_growth_yield_nha_o.png")


def fig_market_trend():
    m = pd.read_csv(PROCESSED_DATA_DIR / "market_price_monthly.csv")
    m = m[(m.property_type == "nha_o") & (m.chart_type_id == 0)]
    dmap = pd.read_parquet(PROCESSED_DATA_DIR / "nhatot_tphcm_clean.parquet",
                           columns=["ward_id", "district_name"]).drop_duplicates("ward_id")
    m = m.merge(dmap, on="ward_id")
    picks = {"Quận Gò Vấp": "#2a78d6", "Quận Bình Thạnh": "#eb6834", "Quận Phú Nhuận": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for d, c in picks.items():
        s = m[m.district_name == d].groupby("month")["median_price_per_m2"].median()
        ax.plot(s.index, s.values, color=c, linewidth=2, marker="o", markersize=4)
        ax.text(len(s) - 0.7, s.values[-1], d.replace("Quận ", ""), color=INK2, va="center", fontsize=10)
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylabel("triệu/m² (trung vị các phường)")
    ax.set_title("Biểu đồ giá 13 tháng (nhà ở) - 3 quận của bộ dữ liệu mẫu")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", visible=False)
    _save(fig, "06_market_trend.png")


def fig_baseline(summary):
    chk = summary["baseline_check"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    types = list(TYPE_LABELS)
    vals = [chk[t]["r2_log_price"] for t in types]
    ax.bar([TYPE_LABELS[t] for t in types], vals, color=[TYPE_COLORS[t] for t in types], width=0.5)
    for i, t in enumerate(types):
        ax.text(i, vals[i] + 0.03, f"R² {vals[i]:.2f}\nsai số TV {chk[t]['median_ape_pct']:.0f}%",
                ha="center", va="bottom", color=INK2, fontsize=11)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("R² (log giá)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Kiểm tra nhanh trên tập test (mô hình mặc định)")
    _save(fig, "07_baseline.png")


def run():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DATA_DIR / "nhatot_tphcm_clean.parquet")
    stats = json.load(open(REPORTS_DIR / "preprocessing_stats.json", encoding="utf-8"))
    summary = json.load(open(REPORTS_DIR / "model_ready_summary.json", encoding="utf-8"))
    fig_rejections(stats)
    fig_missing(df)
    fig_price_distribution(df)
    fig_district_price(df)
    fig_growth_yield(df)
    fig_market_trend()
    fig_baseline(summary)
    logger.info(f"Đã lưu biểu đồ vào {FIG_DIR}")


if __name__ == "__main__":
    run()
