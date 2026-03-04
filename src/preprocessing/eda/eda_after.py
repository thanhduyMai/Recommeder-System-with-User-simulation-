# -*- coding: utf-8 -*-
"""
EDA_full.py

Exploratory Data Analysis (EDA) cho dữ liệu Amazon Ratings + Metadata
"""

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_unixtime, to_date, year, month, count, desc,
    when, mean, isnan
)

# =========================================================
# 1️⃣ Tạo Spark session
# =========================================================
def create_spark_session(app_name="AmazonEDA"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# =========================================================
# 2️⃣ Load dữ liệu từ Parquet
# =========================================================
def load_parquet(spark, parquet_path):
    """
    Load data từ parquet (staging layer)
    """
    df = spark.read.parquet(parquet_path)
    return df


# =========================================================
# 3️⃣ Kiểm tra nulls
# =========================================================
def check_nulls(df, name="DataFrame"):
    """
    Hiển thị số lượng nulls trên từng cột
    """
    print(f"\n===== NULL CHECK: {name} =====")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()


def basic_eda_ratings(df):
    """
    Phân tích mô tả cơ bản cho dữ liệu ratings
    """
    print("\n===== SCHEMA (RATINGS) =====")
    df.printSchema()
    check_nulls(df, "df_ratings")

    print("\n===== THỐNG KÊ RATING =====")
    if 'rating' in df.columns:
        df.select('rating').describe().show()
    else:
        print("⚠️ Không có cột rating — implicit feedback dataset?")

    print("\n===== TOP 10 ITEM CÓ NHIỀU REVIEW NHẤT =====")
    if 'asin' in df.columns:
        df.groupBy('asin') \
          .agg(count('*').alias('review_count')) \
          .orderBy(desc('review_count')) \
          .show(10, truncate=False)

    print("\n===== TOP 10 USER CÓ NHIỀU REVIEW NHẤT =====")
    if 'user_id' in df.columns:
        df.groupBy('user_id') \
          .agg(count('*').alias('review_count')) \
          .orderBy(desc('review_count')) \
          .show(10, truncate=False)

    # Distinct count
    if 'user_id' in df.columns:
        distinct_users = df.select('user_id').distinct().count()
        print(f"Distinct users: {distinct_users}")

    if 'asin' in df.columns:
        distinct_items = df.select('asin').distinct().count()
        print(f"Distinct items: {distinct_items}")

    total_interactions = df.count()

    if 'user_id' in df.columns:
        avg_actions_user = total_interactions / distinct_users
        print(f"Trung bình actions/user: {avg_actions_user:.2f}")

    if 'asin' in df.columns:
        avg_actions_item = total_interactions / distinct_items
        print(f"Trung bình actions/item: {avg_actions_item:.2f}")

# =========================================================
# =========================================================
# 8️⃣ Long-tail & Zipfian analysis
# =========================================================
def long_tail_analysis(df):
    """
    Phân tích long-tail cho items
    """
    print("\n===== LONG-TAIL / ZIPFIAN ANALYSIS =====")

    item_counts = df.groupBy('asin').agg(count('*').alias('cnt')) \
                    .orderBy(desc('cnt'))

    total_interactions = df.count()

    for p in [0.01, 0.05, 0.10]:
        top_k = int(item_counts.count() * p)
        top_interactions = item_counts.limit(top_k) \
            .agg({'cnt': 'sum'}).collect()[0][0]

        ratio = (top_interactions / total_interactions) * 100
        print(f"Top {int(p*100)}% items account for {ratio:.2f}% interactions")


# =========================================================
# 9️⃣ Sparsity analysis
# =========================================================
def sparsity_analysis(df):
    """
    Tính sparsity của user-item matrix
    """
    print("\n===== SPARSITY ANALYSIS =====")

    num_users = df.select('user_id').distinct().count()
    num_items = df.select('asin').distinct().count()
    num_interactions = df.count()

    sparsity = 1 - (num_interactions / (num_users * num_items))

    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Interactions: {num_interactions}")
    print(f"Sparsity: {sparsity:.6f}")

# =========================================================
# 🔟 Cold-start ratio
# =========================================================
def cold_start_analysis(df, threshold=5):
    """
    Phân tích cold-start user & item
    threshold: K trong K-core
    """
    print("\n===== COLD-START ANALYSIS =====")

    user_cnt = df.groupBy('user_id').agg(count('*').alias('cnt'))
    item_cnt = df.groupBy('asin').agg(count('*').alias('cnt'))

    total_users = user_cnt.count()
    total_items = item_cnt.count()

    # CHÚ Ý: < threshold, KHÔNG PHẢI <=
    cold_users = user_cnt.filter(col('cnt') < threshold).count()
    cold_items = item_cnt.filter(col('cnt') < threshold).count()

    print(f"Cold users (<{threshold}): {cold_users} ({cold_users/total_users*100:.2f}%)")
    print(f"Cold items (<{threshold}): {cold_items} ({cold_items/total_items*100:.2f}%)")


# =========================================================
# 6️⃣ EDA cho metadata
# =========================================================
def basic_eda_meta(df):
    """
    EDA cơ bản cho metadata
    """
    print("\n===== SCHEMA (META) =====")
    df.printSchema()
    check_nulls(df, "df_meta")

    print("\n===== THỐNG KÊ CỘT PRICE =====")
    if 'price' in df.columns:
        df.select('price').describe().show()
    else:
        print("⚠️ Không có cột price.")
    
    if "price" in df.columns:
        total_rows = df.count()
        missing_price = df.filter(col("price").isNull()).count()
        missing_ratio = (missing_price / total_rows) * 100 if total_rows > 0 else 0

        print(f"\n💰 Tổng số dòng: {total_rows}")
        print(f"💸 Số dòng thiếu price: {missing_price} ({missing_ratio:.2f}%)")

    print("\n===== TOP CATEGORY =====")
    if 'main_category' in df.columns:
        df.groupBy('main_category').agg(count('*').alias('count')) \
          .orderBy(desc('count')).show(10, truncate=False)

    print("\n===== DISTINCT COUNTS =====")
    for col_name in ['main_category', 'store']:
        if col_name in df.columns:
            cnt = df.select(col_name).distinct().count()
            print(f"{col_name}: {cnt} distinct")

# =========================================================
# 7️⃣ Main pipeline
# =========================================================
def main():
    spark = create_spark_session()

    ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
    meta_path = "hdfs:///DATALAKE/MyData/staging/filtered/meta_Beauty_and_Personal_Care_filtered"

    df_ratings = load_parquet(spark, ratings_path)
    df_meta = load_parquet(spark, meta_path)

    print("\n===== BẮT ĐẦU EDA =====")

    # EDA ratings
    basic_eda_ratings(df_ratings)
    long_tail_analysis(df_ratings)
    sparsity_analysis(df_ratings)
    cold_start_analysis(df_ratings, threshold=5)

    # EDA metadata
    basic_eda_meta(df_meta)
    final_users = df_ratings.select("user_id").distinct().count()
    final_items = df_meta.select("asin").distinct().count()
    spark.stop()


if __name__ == "__main__":
    main()
