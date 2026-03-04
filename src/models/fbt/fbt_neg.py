# fbt.py (FAIR version)
# -*- coding: utf-8 -*-
"""
FBT (Frequently Bought Together) baseline recommender — FAIR version
- Tự tính từ rating data (user_id, parent_asin)
- ĐÃ SỬA: Dùng logic split (LOO) theo event_date để "fair" với ALS
- ĐÃ SỬA (Lần 2): Di chuyển bộ lọc rating >= 4 CHỈ áp dụng cho train set (Split then Filter)
- ‼️ THAY ĐỔI: KHÔNG TỰ TẠO negative sampling. 
  Tải candidate set (1 pos + 100 neg) đã được tính toán trước
  từ file Parquet (do precompute_candidates.py tạo ra).
"""

import numpy as np
from pyspark.sql import SparkSession, functions as F, Window
import random

# ==============================================================
# 1️⃣ Spark Session (Giữ nguyên)
# ==============================================================
def create_spark_session(app_name="FBT_From_Ratings_Optimized"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.instances", "6")
        .config("spark.executor.cores", "6")
        .config("spark.sql.shuffle.partitions", "400")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ==============================================================
# 2️⃣ Compute Metrics (Giữ nguyên)
# ==============================================================
def compute_metrics(spark, recs_df, test_df, k_recs=10):
    # (Hàm này giữ nguyên, không thay đổi)
    window_spec = Window.partitionBy("user_id").orderBy(F.desc("predicted_rating"))
    recs_ranked = (
        recs_df.withColumn("rank", F.row_number().over(window_spec))
        .filter(F.col("rank") <= k_recs)
    )

    gt_items = test_df.select("user_id", "parent_asin").withColumnRenamed(
        "parent_asin", "gt_item"
    )

    hits_df = (
        recs_ranked.join(
            gt_items,
            (recs_ranked.user_id == gt_items.user_id)
            & (recs_ranked.parent_asin == gt_items.gt_item),
            "left",
        )
        .select(
            recs_ranked.user_id,
            recs_ranked.parent_asin,
            recs_ranked.predicted_rating,
            recs_ranked.rank,
            gt_items.gt_item,
        )
    )

    dcg_df = (
        hits_df.filter(F.col("gt_item").isNotNull())
        .withColumn("dcg_item", 1.0 / F.log2(F.col("rank") + F.lit(1.0)))
        .groupBy("user_id")
        .agg(F.sum("dcg_item").alias("dcg"))
    )

    gt_per_user = test_df.groupBy("user_id").agg(
        F.count("parent_asin").alias("num_gt_items")
    )

    idcg_list = [
        (n, float(sum([1.0 / np.log2(i + 1) for i in range(1, n + 1)])))
        for n in range(1, k_recs + 1)
    ]
    idcg_lookup_df = spark.createDataFrame(idcg_list, ["lookup_size", "idcg"])

    idcg_df = (
        gt_per_user.withColumn(
            "lookup_size", F.least(F.col("num_gt_items"), F.lit(k_recs))
        )
        .join(idcg_lookup_df, "lookup_size", "left")
        .select("user_id", "idcg")
    )

    metrics_per_user = (
        idcg_df.join(dcg_df, "user_id", "left")
        .fillna(0.0, subset=["dcg"])
        .withColumn(
            "ndcg",
            F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(0.0),
        )
        .withColumn("hit", F.when(F.col("dcg") > 0, 1.0).otherwise(0.0))
    )

    result = (
        metrics_per_user.agg(
            F.mean("hit").alias("hit_rate"), F.mean("ndcg").alias("ndcg")
        ).collect()[0]
    )
    return float(result["hit_rate"]), float(result["ndcg"])

# ==============================================================
# 2.5 ‼️ HÀM SPLIT (Giữ nguyên)
# ==============================================================
def split_last_k_out(df, k=1, user_col="user_id", time_col="event_date", tie_breaker_col="parent_asin"):
    """
    Hàm Leave-One-Out (LOO) dựa trên thời gian.
    """
    # (Hàm này giữ nguyên, không thay đổi)
    df_with_tie_breaker = df.withColumn("tie_breaker", F.col(tie_breaker_col))
    
    w = Window.partitionBy(user_col).orderBy(F.asc(time_col), F.asc("tie_breaker"))
    df_ranked = df_with_tie_breaker.withColumn("rank", F.row_number().over(w))
    df_max = df_ranked.groupBy(user_col).agg(F.max("rank").alias("max_rank"))
    df_join = df_ranked.join(df_max, on=user_col, how="inner")

    boundary_col = F.when(F.col("max_rank") > F.lit(k),
                          F.col("max_rank") - F.lit(k)) \
                     .otherwise(F.col("max_rank") - F.lit(1))
    df_with_boundary = df_join.withColumn("train_upper_rank", boundary_col)

    df_train = df_with_boundary.filter(F.col("rank") <= F.col("train_upper_rank")) \
                               .drop("rank", "max_rank", "train_upper_rank", "tie_breaker")
    df_test = df_with_boundary.filter(F.col("rank") > F.col("train_upper_rank")) \
                              .drop("rank", "max_rank", "train_upper_rank", "tie_breaker")
    return df_train, df_test

# ==============================================================
# 3️⃣ Main pipeline
# ==============================================================
if __name__ == "__main__":
    spark = create_spark_session()

    # ----------------------------------------------------------
    # 📦 Load rating data (Giữ nguyên)
    # ----------------------------------------------------------
    ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
    
    # ‼️ THAY ĐỔI: Path để TẢI candidates (từ precompute_candidates.py)
    candidates_load_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates.parquet"

    print("Loading data...")
    df = (
        spark.read.parquet(ratings_path)
        # ‼️ (GIỮ NGUYÊN) Bỏ lọc rating >= 4 ở đây
        .filter(F.col("verified_purchase") == True)
        .select("user_id", "parent_asin", "event_date", "rating") 
        .distinct()
        .dropna()
    )
    df.cache()

    # ----------------------------------------------------------
    # 🧩 Step 1: Giới hạn user dense (Giữ nguyên)
    # ----------------------------------------------------------
    print("Filtering dense users (<= 174 items)...")
    user_counts = df.groupBy("user_id").agg(F.countDistinct("parent_asin").alias("num_items"))
    
    df_filtered = (
        df.join(user_counts, "user_id")
        .filter(F.col("num_items") <= 174) 
    ).drop("num_items")
    
    df_filtered.cache()
    df.unpersist()

    # ----------------------------------------------------------
    # 🧩 Step 2: Split train/test (Giữ nguyên)
    # ----------------------------------------------------------
    print("Splitting data using 'split_last_k_out' (time-based)...")
    df_train, df_test = split_last_k_out(
        df_filtered,
        k=1,
        user_col="user_id",
        time_col="event_date",
        tie_breaker_col="parent_asin"
    )

    df_train.cache()
    df_test.cache() # Đây là Ground Truth
    df_filtered.unpersist() 
    
    # ----------------------------------------------------------
    # 🧩 Step 3: Tạo cặp sản phẩm (A, B) TỪ DỮ LIỆU TRAIN (Giữ nguyên)
    # ----------------------------------------------------------
    print("Data split. Building FBT model ONLY on df_train (rating >= 4)...")
    
    # ‼️ (GIỮ NGUYÊN) Lọc (Filter) CHỈ cho df_train
    df_train_positive = df_train.filter(F.col("rating") >= 4).cache()
    
    user_pairs = (
        df_train_positive.alias("a") 
        .join(
            df_train_positive.alias("b"), 
            (F.col("a.user_id") == F.col("b.user_id"))
            & (F.col("a.parent_asin") < F.col("b.parent_asin")),
            "inner",
        )
        .select(
            F.col("a.parent_asin").alias("asin_A"),
            F.col("b.parent_asin").alias("asin_B")
        )
        .repartition(400)
        .checkpoint(eager=True)
    )

    # ----------------------------------------------------------
    # 🧩 Step 4: Đếm tần suất (Giữ nguyên)
    # ----------------------------------------------------------
    fbt_pairs = (
        user_pairs.groupBy("asin_A", "asin_B")
        .agg(F.count("*").alias("co_count"))
        .filter(F.col("co_count") >= 3)
        .checkpoint(eager=True)
    )

    # ----------------------------------------------------------
    # 🧩 Step 5: ‼️ THAY ĐỔI: TẢI CANDIDATES
    # ----------------------------------------------------------
    print(f"Loading precomputed candidates from {candidates_load_path}...")
    
    # Tải file parquet (File này có cột 'asin' - chính là parent)
    candidates_df_full = spark.read.parquet(candidates_load_path)
    
    # [FIX] Đổi tên 'asin' -> 'parent_asin' để khớp với FBT
    candidates_df = candidates_df_full.select(
        "user_id", 
        F.col("asin").alias("parent_asin") # <--- QUAN TRỌNG: THÊM DÒNG NÀY
    ).distinct()


    # ----------------------------------------------------------
    # 🧩 Step 6: Gợi ý FBT cho candidates (Giữ nguyên logic)
    # ----------------------------------------------------------
    
    # Rename để join với fbt_pairs
    candidates_renamed = candidates_df.withColumnRenamed("parent_asin", "asin_B")

    # Item mồi từ df_train_positive (per user)
    user_item_train = df_train_positive.select("user_id", F.col("parent_asin").alias("asin_A")).distinct()

    # Join để tìm pairs
    fbt_recs = user_item_train.join(candidates_renamed, "user_id") \
                              .join(fbt_pairs, ["asin_A", "asin_B"], "left") \
                              .groupBy("user_id", "asin_B") \
                              .agg(F.max(F.coalesce(F.col("co_count"), F.lit(0))).alias("predicted_rating")) \
                              .withColumnRenamed("asin_B", "parent_asin")

    # Giải phóng
    df_train.unpersist()
    df_train_positive.unpersist()
    candidates_df.unpersist()

    # ------------------------------------------------------
    # 🧮 Step 7: Evaluate performance (Giữ nguyên)
    # ------------------------------------------------------
    # (Đánh giá trên df_test - Ground truth)
    
    print("Calculating metrics...")
    hit_fbt, ndcg_fbt = compute_metrics(spark, fbt_recs, df_test, k_recs=10)
    print("="*30)
    print(f"🛒 FBT — HitRate@10 = {hit_fbt:.6f}, NDCG@10 = {ndcg_fbt:.6f}")
    print("="*30)

    spark.stop()