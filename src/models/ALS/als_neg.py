# als_eval_fair.py (Updated for Dual Candidates)
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.recommendation import ALS
import random

# ==============================================================
# 1️⃣ Spark Session
# ==============================================================
def create_spark_session(app_name="ALS_Fair_Benchmark"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "400")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# ==============================================================
# 2️⃣ Helper Functions
# ==============================================================
def load_ratings_data(spark, file_path):
    return spark.read.parquet(file_path)

def split_last_k_out(df, k=1, user_col="user_id", time_col="event_date", tie_breaker_col="asin"):
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

def compute_metrics(spark, recs_df, test_df, k_recs=10):
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

    ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
    
    
    candidates_load_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates_dual.parquet"

    # 1. Load & Filter Data
    df_ratings = load_ratings_data(spark, ratings_path) \
        .filter(F.col("verified_purchase") == True) \
        .select("user_id", "parent_asin", "rating", "event_date") \
        .withColumnRenamed("parent_asin", "asin") # Đổi tên thành asin để khớp logic chung

    # 2. Split Train/Test
    df_train, df_test = split_last_k_out(
        df_ratings, 
        k=1, 
        user_col="user_id", 
        time_col="event_date", 
        tie_breaker_col="asin"
    )
    
    df_test = df_test.checkpoint(eager=True)
    df_train.cache()

    # 3. Chuẩn bị dữ liệu Train (Chỉ lấy rating >= 4 cho ALS implicit)
    print("Preparing Training Data (Rating >= 4.0)...")
    df_train_positive = df_train.filter(
        F.col("rating") >= 4
    ).withColumn("confidence", F.lit(1.0))
    
    df_train_positive.cache()

    # 4. Mapping User/Item Index (Cần thiết cho Spark ALS)
    # Gom toàn bộ User/Item từ cả Train và Test để đánh index không bị sót
    user_mapping = df_ratings.select("user_id").distinct().rdd.zipWithIndex() \
        .toDF(["tmp", "userIndex"]) \
        .select(F.col("tmp.user_id").alias("user_id"), F.col("userIndex").cast("long"))

    item_mapping = df_ratings.select("asin").distinct().rdd.zipWithIndex() \
        .map(lambda x: (x[0]["asin"], x[1])) \
        .toDF(["asin", "itemIndex"]) \
        .select("asin", F.col("itemIndex").cast("long"))

    # Join Index vào Train Data
    df_train_idx = df_train_positive.join(user_mapping, "user_id", "inner") \
                           .join(item_mapping, "asin", "inner") \
                           .select("userIndex", "itemIndex", "confidence")

    df_train_idx = df_train_idx.repartition(200).cache()
    
    print("✅ Data ready. Training ALS...")

    # [Cấu hình ALS]
    als = ALS(
        userCol="userIndex",
        itemCol="itemIndex",
        ratingCol="confidence",
        implicitPrefs=True,
        alpha=40.0,           
        rank=64,             
        regParam=0.1,
        maxIter=15,          
        coldStartStrategy="drop"
    )
    als.setCheckpointInterval(5)

    model = als.fit(df_train_idx)
    df_train_idx.unpersist()
    print("✅ Model training completed.")

    # ----------------------------------------------------------
    # 5. TẢI CANDIDATES & PREDICT
    # ----------------------------------------------------------
    print(f"Loading candidates from {candidates_load_path}...")
    
    # ‼️ [QUAN TRỌNG] Chỉ load cột ID, bỏ qua các cột Vector nặng nề
    candidates_df_full = spark.read.parquet(candidates_load_path)
    candidates_df = candidates_df_full.select("user_id", "asin").distinct()
    
    # Join Index cho Candidates
    candidates_idx = candidates_df.join(user_mapping, "user_id", "inner") \
                                  .join(item_mapping, "asin", "inner") \
                                  .select("user_id", "asin", "userIndex", "itemIndex")

    # Predict
    print("Predicting on candidates...")
    predictions = model.transform(candidates_idx)
    
    # Fill NaN = 0 (Cho những item cold-start mà ALS không biết)
    recs_final = predictions.withColumn("predicted_rating", F.coalesce(F.col("prediction"), F.lit(0.0))) \
                            .select("user_id", "asin", "predicted_rating")

    # ----------------------------------------------------------
    # 6. EVALUATE
    # ----------------------------------------------------------
    print("Evaluating...")
    
    # ‼️ [FIX FAIRNESS] Bỏ bộ lọc rating >= 4.0
    # Dùng toàn bộ df_test làm Ground Truth (vì Candidate đã fix cứng 1 Positive item rồi)
    
    test_users = candidates_df.select("user_id").distinct()
    test_for_eval = df_test.join(test_users, "user_id", "inner")
    
    user_count = test_for_eval.count()
    print(f"Evaluating on {user_count} users...")

    if user_count > 0:
        hit_rate, ndcg = compute_metrics(spark, recs_final, test_for_eval, k_recs=10)
        print("="*40)
        print(f"🎯 ALS RESULTS (Fair Benchmark)")
        print(f"   HitRate@10 : {hit_rate:.6f}")
        print(f"   NDCG@10    : {ndcg:.6f}")
        print("="*40)
    else:
        print("⚠️ No users left in evaluation set!")

    spark.stop()