# precompute_candidates_dual.py
# Cập nhật: Hỗ trợ Dual Vectors (Title + Review)
# Tối ưu: Load trực tiếp Vector có sẵn (không chạy UDF thừa)
# Logic: Leave-One-Out + 100 Random Unseen Negatives

import random
import numpy as np
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import ArrayType, StringType
# Vẫn cần import VectorUDT để định nghĩa UDF nếu cần (dù ở đây ta dùng vector có sẵn)
from pyspark.ml.linalg import Vectors, VectorUDT

# ==============================================================
# 1️⃣ SETUP SPARK
# ==============================================================
print("Starting Precompute Candidates (Dual Vectors - Optimized)...")
spark = SparkSession.builder \
    .appName("Precompute_Candidates_Dual_Optimized") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

# --- Paths ---
ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
meta_path = "hdfs:///DATALAKE/MyData/staging/filtered/meta_Beauty_and_Personal_Care_filtered"

# Path tới 2 Feature Store
title_w2v_path = "hdfs:///DATALAKE/MyData/feature_store/item_title_w2v"
review_w2v_path = "hdfs:///DATALAKE/MyData/feature_store/item_review_agg_w2v"

# Path lưu kết quả
candidates_save_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates_dual.parquet"

# ==============================================================
# 2️⃣ LOAD DỮ LIỆU (PARENT ASIN & DUAL VECTORS)
# ==============================================================
print("Loading Data...")

# 1. Ratings: Lấy PARENT_ASIN và đổi tên thành 'asin'
ratings = spark.read.parquet(ratings_path) \
    .filter(F.col("verified_purchase") == True) \
    .select(
        "user_id", 
        F.col("parent_asin").alias("asin"), 
        "rating", 
        "helpful_vote", 
        "event_date"
    ) \
    .dropna()

# 2. Meta
meta = spark.read.parquet(meta_path) \
    .select("parent_asin", "main_category", "price") \
    .withColumnRenamed("parent_asin", "item_id_meta") \
    .dropna()

# 3. Title Vectors (Đã là Vector, không cần UDF)
print("Loading Title Vectors (Direct)...")
title_df = spark.read.parquet(title_w2v_path) \
    .select(
        F.col("parent_asin").alias("asin_title"),
        F.col("title_vec").alias("item_title_vec") # Đổi tên chuẩn
    )

# 4. Review Vectors (Đã là Vector, không cần UDF)
print("Loading Review Vectors (Direct)...")
# [QUAN TRỌNG] Tên cột gốc của bạn là 'item_vec' (theo log check schema)
review_df = spark.read.parquet(review_w2v_path) \
    .select(
        F.col("parent_asin").alias("asin_review"),
        F.col("item_vec").alias("item_review_vec") # Đổi tên chuẩn
    )

# 5. JOIN ALL
print("Joining All Tables...")
data = ratings.join(meta, ratings["asin"] == meta["item_id_meta"]).drop("item_id_meta") \
              .join(title_df, ratings["asin"] == title_df["asin_title"]).drop("asin_title") \
              .join(review_df, ratings["asin"] == review_df["asin_review"]).drop("asin_review")

data.cache()

# ==============================================================
# 3️⃣ CREATE CANDIDATES (1 POS + 100 NEGS)
# ==============================================================
# Hàm Split LOO
def split_last_k_out(df, k=1, user_col="user_id", time_col="event_date", tie_breaker_col="asin"):
    w = Window.partitionBy(user_col).orderBy(F.asc(time_col), F.asc(tie_breaker_col))
    df_ranked = df.withColumn("rank", F.row_number().over(w))
    df_max = df_ranked.groupBy(user_col).agg(F.max("rank").alias("max_rank"))
    df_join = df_ranked.join(df_max, on=user_col, how="inner")
    
    boundary_col = F.when(F.col("max_rank") > F.lit(k), F.col("max_rank") - F.lit(k)).otherwise(F.col("max_rank") - F.lit(1))
    df_with_boundary = df_join.withColumn("train_upper_rank", boundary_col)
    
    df_train = df_with_boundary.filter(F.col("rank") <= F.col("train_upper_rank")).drop("rank", "max_rank", "train_upper_rank")
    df_test = df_with_boundary.filter(F.col("rank") > F.col("train_upper_rank")).drop("rank", "max_rank", "train_upper_rank")
    return df_train, df_test

print("Splitting Train/Test for LOO...")
df_train, df_test = split_last_k_out(data, k=1, user_col="user_id", time_col="event_date", tie_breaker_col="asin")

# Broadcast tất cả Items để chọn random
# Chỉ cần lấy cột ID để broadcast
all_asins_df = data.select("asin").distinct()
all_asins_list = [row.asin for row in all_asins_df.collect()]
all_asins_broadcast = spark.sparkContext.broadcast(all_asins_list)

# Cache bảng Features đầy đủ để join sau cùng (tránh join lặp lại)
all_items_features = data.select("asin", "main_category", "item_title_vec", "item_review_vec").distinct().cache()

# Tạo lịch sử mua (để loại trừ khi random)
history_train = df_train.groupBy("user_id").agg(F.collect_set("asin").alias("history"))

print("Generating 100 Random Negatives per User...")
num_neg = 100

@F.udf(ArrayType(StringType()))
def sample_negatives_udf(history, positive):
    all_items = all_asins_broadcast.value
    # Loại trừ Lịch sử cũ + Món hiện tại
    excluded = set(history) | {positive}
    available = list(set(all_items) - excluded)
    if not available: return []
    return random.sample(available, min(num_neg, len(available)))

# Join Test với History
df_test_setup = df_test.join(history_train, "user_id", "left").withColumn("history", F.coalesce(F.col("history"), F.array()))

# Thực hiện Sampling
df_test_neg = df_test_setup.withColumn("neg_asins", sample_negatives_udf("history", "asin"))

# --- Gộp thành Candidates ---
# 1. Positive (Ground Truth)
pos_cand = df_test.select("user_id", "asin", "rating", "helpful_vote").withColumn("label", F.lit(1.0))

# 2. Negative (Random)
neg_cand = df_test_neg.select("user_id", F.explode("neg_asins").alias("asin")) \
    .withColumn("rating", F.lit(0.0)) \
    .withColumn("helpful_vote", F.lit(0.0)) \
    .withColumn("label", F.lit(0.0))

# 3. Union & Add Features
# Join lại với bảng Feature đã cache để lấy Title Vec và Review Vec
print("Finalizing Candidates Table...")
final_candidates = pos_cand.union(neg_cand).join(all_items_features, "asin", "inner")

# ==============================================================
# 4️⃣ SAVE
# ==============================================================
print(f"Saving Candidates to {candidates_save_path}...")
final_candidates.write.mode("overwrite").parquet(candidates_save_path)
print("✅ PRECOMPUTE CANDIDATES DONE.")

spark.stop()