# -*- coding: utf-8 -*-
"""
ALS_optimized.py
Huấn luyện mô hình ALS (implicit feedback) với tham số tối ưu hơn
và tính NDCG@10 + HitRate@10 theo phương pháp Leave-One-Out.

- ĐÃ SỬA: Áp dụng logic "Split then Filter".
  1. Load data, chỉ lọc verified_purchase.
  2. Split train/test (cả hai đều chứa mọi mức rating).
  3. Áp dụng lọc rating >= 4 CHỈ cho df_train để huấn luyện.
  4. Đánh giá trên df_test (gốc, không lọc rating).
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.recommendation import ALS


# ==============================================================
# 1️⃣ Spark Session
# ==============================================================
def create_spark_session(app_name="ALS_Optimized_Implicit_FAIR"):
    # (Hàm này giữ nguyên, không thay đổi)
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .config("spark.executor.instances", "6")
        .config("spark.executor.cores", "6")
        .config("spark.executor.memoryOverhead", "1024")
        .config("spark.storage.memoryFraction", "0.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ==============================================================
# 2️⃣ Data Loading & Splitting
# ==============================================================
def load_ratings_data(spark, file_path):
    # (Hàm này giữ nguyên, không thay đổi)
    return spark.read.parquet(file_path)


def split_last_k_out(df, k=1, user_col="user_id", time_col="event_date", tie_breaker_col="asin"):
    # (Hàm này giữ nguyên, không thay đổi)
    w = Window.partitionBy(user_col).orderBy(F.asc(time_col), F.asc(tie_breaker_col))
    df_ranked = df.withColumn("rank", F.row_number().over(w))
    df_max = df_ranked.groupBy(user_col).agg(F.max("rank").alias("max_rank"))
    df_join = df_ranked.join(df_max, on=user_col, how="inner")

    boundary_col = F.when(F.col("max_rank") > F.lit(k),
                          F.col("max_rank") - F.lit(k)) \
                     .otherwise(F.col("max_rank") - F.lit(1))
    df_with_boundary = df_join.withColumn("train_upper_rank", boundary_col)

    df_train = df_with_boundary.filter(F.col("rank") <= F.col("train_upper_rank")) \
                               .drop("rank", "max_rank", "train_upper_rank")
    df_test = df_with_boundary.filter(F.col("rank") > F.col("train_upper_rank")) \
                              .drop("rank", "max_rank", "train_upper_rank")
    return df_train, df_test


# ==============================================================
# 3️⃣ Evaluation Metrics (Spark 2.4 compatible)
# ==============================================================
def compute_metrics(spark, recs_df, test_df, k_recs=10):
    # (Hàm này giữ nguyên, không thay đổi)
    window_spec = Window.partitionBy("user_id").orderBy(F.desc("predicted_rating"))
    recs_ranked = recs_df.withColumn("rank", F.row_number().over(window_spec)) \
                         .filter(F.col("rank") <= k_recs)

    gt_items = test_df.select("user_id", "asin").withColumnRenamed("asin", "gt_item")

    hits_df = recs_ranked.join(
        gt_items,
        (recs_ranked.user_id == gt_items.user_id) & (recs_ranked.asin == gt_items.gt_item),
        "left"
    ).select(recs_ranked.user_id, recs_ranked.asin, recs_ranked.predicted_rating,
             recs_ranked.rank, gt_items.gt_item)

    dcg_df = hits_df.filter(F.col("gt_item").isNotNull()) \
                    .withColumn("dcg_item", 1.0 / F.log2(F.col("rank") + F.lit(1.0))) \
                    .groupBy("user_id").agg(F.sum("dcg_item").alias("dcg"))

    gt_per_user = test_df.groupBy("user_id").agg(F.count("asin").alias("num_gt_items"))
    idcg_list = [(n, float(sum([1.0 / np.log2(i + 1) for i in range(1, n + 1)]))) for n in range(1, k_recs + 1)]
    idcg_lookup_df = spark.createDataFrame(idcg_list, ["lookup_size", "idcg"])

    idcg_df = gt_per_user.withColumn("lookup_size", F.least(F.col("num_gt_items"), F.lit(k_recs))) \
                         .join(idcg_lookup_df, "lookup_size", "left") \
                         .select("user_id", "idcg")

    metrics_per_user = idcg_df.join(dcg_df, "user_id", "left").fillna(0.0, subset=["dcg"]) \
                              .withColumn("ndcg", F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(0.0)) \
                              .withColumn("hit", F.when(F.col("dcg") > 0, 1.0).otherwise(0.0))

    result = metrics_per_user.agg(F.mean("hit").alias("hit_rate"), F.mean("ndcg").alias("ndcg")).collect()[0]
    return float(result["hit_rate"]), float(result["ndcg"])


# ==============================================================
# 4️⃣ MAIN PIPELINE
# ==============================================================
if __name__ == "__main__":
    spark = create_spark_session()
    spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")

    ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
    df_ratings = load_ratings_data(spark, ratings_path)

    # ‼️ THAY ĐỔI: Chỉ lọc verified_purchase. Bỏ lọc rating >= 4 ở đây.
    df_ratings_filtered = df_ratings.filter(
       (F.col("verified_purchase") == True)
    )
    # (Cũng bỏ .withColumn("confidence", ...) ở đây, sẽ thêm sau)

    # ‼️ THAY ĐỔI: Chia dữ liệu từ df_ratings_filtered (chưa lọc rating)
    df_train, df_test = split_last_k_out(
        df_ratings_filtered, 
        k=1, 
        user_col="user_id", 
        time_col="event_date", 
        tie_breaker_col="asin"
    )
    
    # df_test chứa ground truth (mọi rating), checkpoint nó
    df_test = df_test.checkpoint(eager=True)
    df_train.cache()
    
    # Giải phóng các dataframe không cần nữa
    df_ratings.unpersist()
    df_ratings_filtered.unpersist()

    # ‼️ THÊM MỚI: Bây giờ mới áp dụng lọc rating >= 4 CHỈ cho df_train
    print("Applying rating >= 4 filter ONLY to df_train...")
    df_train_positive = df_train.filter(
        F.col("rating") >= 4
    ).withColumn("confidence", F.lit(1.0))
    
    df_train_positive.cache()

    # Mapping
    # ‼️ THAY ĐỔI: user_mapping chỉ học từ df_train_positive
    user_mapping_train = df_train_positive.select("user_id").distinct().rdd.zipWithIndex() \
        .toDF(["tmp", "userIndex"]) \
        .select(F.col("tmp.user_id").alias("user_id"), F.col("userIndex").cast("long"))

    # ‼️ GIỮ NGUYÊN (VÀ ĐÚNG): item_mapping phải biết TẤT CẢ item
    # từ cả df_train (gốc) và df_test (gốc)
    data_items_all = df_train.select("asin").union(df_test.select("asin")).distinct()
    
    item_mapping_all = data_items_all.rdd.zipWithIndex() \
        .map(lambda x: (x[0]["asin"], x[1])) \
        .toDF(["asin", "itemIndex"]) \
        .select("asin", F.col("itemIndex").cast("long"))

    # ‼️ THAY ĐỔI: df_train_idx được tạo từ df_train_positive
    df_train_idx = df_train_positive.join(user_mapping_train, "user_id", "inner") \
                           .join(item_mapping_all, "asin", "inner") \
                           .select("user_id", "asin", "confidence", "userIndex", "itemIndex")

    df_train_idx = df_train_idx.repartition(200).cache()
    
    # ‼️ THAY ĐỔI: Giải phóng cả hai df_train
    df_train.unpersist()
    df_train_positive.unpersist()


    print("✅ Data ready. Training ALS (optimized implicit model)...")

    als = ALS(
        userCol="userIndex",
        itemCol="itemIndex",
        ratingCol="confidence",
        implicitPrefs=True,
        alpha=40.0,           
        rank=200,
        regParam=0.1,
        maxIter=10,
        coldStartStrategy="drop"
    )
    als.setCheckpointInterval(5)

    # ‼️ (GIỮ NGUYÊN): Huấn luyện trên df_train_idx (đã lọc)
    model = als.fit(df_train_idx)
    df_train_idx.unpersist()
    print("✅ Model training completed.")

    # Recommendations
    recs = model.recommendForAllUsers(10)
    recs_exp = recs.select("userIndex", F.explode("recommendations").alias("rec")) \
                   .select("userIndex", F.col("rec.itemIndex").alias("itemIndex"),
                           F.col("rec.rating").alias("predicted_rating"))

    recs_final = recs_exp.join(user_mapping_train, "userIndex") \
                         .join(item_mapping_all, "itemIndex") \
                         .select("user_id", "asin", "predicted_rating")

    # Evaluate
    # ‼️ (GIỮ NGUYÊN): Đánh giá trên df_test (gốc, không lọc rating)
    test_for_eval = df_test.join(recs_final.select("user_id").distinct(), "user_id", "left")
    hit_rate, ndcg = compute_metrics(spark, recs_final, test_for_eval, k_recs=10)
    print(f"🎯 Evaluation results — HitRate@10 = {hit_rate:.6f}, NDCG@10 = {ndcg:.6f}")

    spark.stop()