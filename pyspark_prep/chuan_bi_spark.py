"""
Tiền xử lý phía PySpark – dùng CÙNG tập train/test với sklearn để so sánh 2 môi trường công bằng
(đề bài: "Thực hiện trên cả 2 môi trường ML truyền thống và PySpark, ít nhất 4 models cho mỗi môi trường").

Đọc train_raw.parquet / test_raw.parquet + features.json do pipeline tạo ra, dựng Spark ML Pipeline:
    số        -> Imputer(median)
    nhị phân  -> Imputer(mode)
    phân loại (kể cả phường / đường / dự án) -> điền "missing" -> StringIndexer(handleInvalid="keep") -> OneHotEncoder
    -> VectorAssembler -> StandardScaler
Pipeline chỉ fit trên train rồi áp dụng cho test. Lưu ra <thư mục>/spark/{train,test}.parquet (cột features, label)
và <thư mục>/spark/pipeline_model/.

Cách chạy (cần: pip install pyspark, và Java 17 hoặc 21 – Java >= 23 báo lỗi "getSubject is not supported";
khi đó cài JDK 21 rồi `export JAVA_HOME=<đường dẫn JDK 21>`):
    python pyspark_prep/chuan_bi_spark.py --dir data/model_ready/nha_o          # dữ liệu cào 22 quận
    python pyspark_prep/chuan_bi_spark.py --dir data/du_lieu_mau/model_ready    # 3 file dữ liệu mẫu

Sau đó huấn luyện mô hình, ví dụ:
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
    train = spark.read.parquet(".../spark/train.parquet")   # label = ln(giá tỷ đồng) -> đổi về tỷ bằng exp()
"""

import argparse
import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_pipeline(num, binary, cat):
    stages = []
    if num:
        stages.append(Imputer(inputCols=num, outputCols=[f"{c}__imp" for c in num], strategy="median"))
    if binary:
        stages.append(Imputer(inputCols=binary, outputCols=[f"{c}__imp" for c in binary], strategy="mode"))
    if cat:
        stages.append(StringIndexer(inputCols=cat, outputCols=[f"{c}__idx" for c in cat], handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[f"{c}__idx" for c in cat], outputCols=[f"{c}__ohe" for c in cat],
                                    handleInvalid="keep"))
    assembled = [f"{c}__imp" for c in num + binary] + [f"{c}__ohe" for c in cat]
    stages.append(VectorAssembler(inputCols=assembled, outputCol="features_raw", handleInvalid="keep"))
    stages.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True))
    return Pipeline(stages=stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/model_ready/nha_o", help="Thư mục model_ready (có features.json)")
    args = ap.parse_args()
    d = Path(args.dir)
    info = json.load(open(d / "features.json", encoding="utf-8"))["features_in"]
    num, binary = info["numeric"], info["binary"]
    cat = info["categorical"] + info["high_cardinality"]   # Spark: phường/đường/dự án cũng One-Hot (vector thưa)

    spark = SparkSession.builder.appName("nhatot_prep").master("local[*]") \
        .config("spark.sql.session.timeZone", "Asia/Ho_Chi_Minh").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    def load(name):
        df = spark.read.parquet(str(d / f"{name}_raw.parquet"))
        cols = ["ad_id", "target_log_price"] + num + binary + cat
        df = df.select(*cols)
        for c in num + binary:
            df = df.withColumn(c, F.col(c).cast("double"))
        for c in cat:
            df = df.withColumn(c, F.coalesce(F.col(c).cast("string"), F.lit("missing")))
        return df.withColumnRenamed("target_log_price", "label")

    train, test = load("train"), load("test")
    model = build_pipeline(num, binary, cat).fit(train)
    out = d / "spark"
    for name, df in [("train", train), ("test", test)]:
        model.transform(df).select("ad_id", "features", "label").write.mode("overwrite").parquet(str(out / f"{name}.parquet"))
    model.write().overwrite().save(str(out / "pipeline_model"))

    tr = spark.read.parquet(str(out / "train.parquet"))
    n_feat = tr.first()["features"].size
    print(f"Spark: train={tr.count()}, test={spark.read.parquet(str(out / 'test.parquet')).count()}, "
          f"số chiều vector đặc trưng={n_feat} -> {out}")
    spark.stop()


if __name__ == "__main__":
    main()
