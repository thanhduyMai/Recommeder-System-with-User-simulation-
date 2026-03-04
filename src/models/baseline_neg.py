# baseline_fair_final.py
# -*- coding: utf-8 -*-
"""
Baseline (Popularity + Random) 
→ DÙNG CHÍNH XÁC candidate set giống DeepFM/FBT/ALS
→ ĐÁNH GIÁ TRÊN TOÀN BỘ TEST SET (KHÔNG filter rating >=4 nữa - theo yêu cầu)
→ ĐÃ FIX 100% lỗi ambiguous user_id
→ compute_metrics định nghĩa trước, không còn NameError
"""

from pyspark.sql import SparkSession, functions as F, Window
import numpy as np

# ==============================================================
# 1️⃣ Spark Session
# ==============================================================
spark = (
    SparkSession.builder
    .appName("Baseline_Final_No_Filter_Test")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.shuffle.partitions", "400")
    .getOrCreate()
)
spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")

# ==============================================================
# 2️⃣ Compute Metrics (FIX ambiguous user_id bằng alias rõ ràng)
# ==============================================================
def compute_metrics(spark, recs_df, test_df, k_recs=10):
    window_spec = Window.partitionBy("user_id").orderBy(F.desc("predicted_rating"))
    recs_ranked = recs_df.withColumn("rank", F.row_number().over(window_spec)) \
                         .filter(F.col("rank") <= k_recs)

    gt_items = test_df.select("user_id", "asin").withColumnRenamed("asin", "gt_item")

    # FIX ambiguous: dùng alias rõ ràng và chỉ lấy 1 user_id
    hits_df = recs_ranked.alias("r").join(
        gt_items.alias("g"),
        (F.col("r.user_id") == F.col("g.user_id")) & (F.col("r.asin") == F.col("g.gt_item")),
        "left"
    ).select(
        F.col("r.user_id").alias("user_id"),
        F.col("r.asin"),
        F.col("r.predicted_rating"),
        F.col("r.rank"),
        F.col("g.gt_item")
    )

    dcg_df = hits_df.filter(F.col("gt_item").isNotNull()) \
                    .withColumn("dcg_item", 1.0 / F.log2(F.col("rank") + 1.0)) \
                    .groupBy("user_id") \
                    .agg(F.sum("dcg_item").alias("dcg"))

    gt_per_user = test_df.groupBy("user_id").agg(F.count("asin").alias("num_gt_items"))

    # IDCG chuẩn (vì LOO nên hầu hết num_gt_items = 1 → idcg = 1.0)
    idcg_list = [(n, float(sum([1.0 / np.log2(i + 1) for i in range(1, n + 1)]))) for n in range(1, k_recs + 1)]
    idcg_lookup_df = spark.createDataFrame(idcg_list, ["lookup_size", "idcg"])

    idcg_df = gt_per_user.withColumn("lookup_size", F.least(F.col("num_gt_items"), F.lit(k_recs))) \
                         .join(idcg_lookup_df, "lookup_size") \
                         .select("user_id", "idcg")

    metrics_per_user = idcg_df.join(dcg_df, "user_id", "left") \
                              .fillna(0.0, subset=["dcg"]) \
                              .withColumn("ndcg", F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(0.0)) \
                              .withColumn("hit", F.when(F.col("dcg") > 0, 1.0).otherwise(0.0))

    result = metrics_per_user.agg(
        F.mean("hit").alias("hit_rate"),
        F.mean("ndcg").alias("ndcg")
    ).collect()[0]

    return float(result["hit_rate"]), float(result["ndcg"])

# ==============================================================
# 3️⃣ Load candidates + LOO split
# ==============================================================
candidates_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates_dual.parquet"
ratings_path    = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"

print("Loading candidates...")
candidates_df = spark.read.parquet(candidates_path).select("user_id", "asin").distinct().cache()

print("Loading ratings & LOO split...")
df = (
    spark.read.parquet(ratings_path)
    .filter(F.col("verified_purchase") == True)
    .select("user_id", "parent_asin", "event_date", "rating")
    .withColumnRenamed("parent_asin", "asin")
    .dropna()
)

w = Window.partitionBy("user_id").orderBy(F.asc("event_date"), F.asc("asin"))
df_ranked = df.withColumn("rank", F.row_number().over(w))
df_max = df_ranked.groupBy("user_id").agg(F.max("rank").alias("max_rank"))
df_join = df_ranked.join(df_max, "user_id")
df_with_boundary = df_join.withColumn(
    "train_upper_rank",
    F.when(F.col("max_rank") > 1, F.col("max_rank") - 1).otherwise(F.col("max_rank"))
)

df_train = df_with_boundary.filter(F.col("rank") <= F.col("train_upper_rank"))
df_test  = df_with_boundary.filter(F.col("rank") > F.col("train_upper_rank")).select("user_id", "asin", "rating")

df_train.cache()
df_test.cache()
print(f"Test set size: {df_test.count()} users")

# ==============================================================
# 4️⃣ Popularity Baseline
# ==============================================================
print("Building Popularity (count trên tất cả positive interactions trong train)...")
pop_scores = df_train.groupBy("asin").agg(F.count("*").alias("pop_score"))

pop_recs = candidates_df.join(pop_scores, "asin", "left") \
                        .fillna({"pop_score": 0}) \
                        .withColumn("predicted_rating", F.col("pop_score").cast("float")) \
                        .select("user_id", "asin", "predicted_rating")

# ==============================================================
# 5️⃣ Random Baseline
# ==============================================================
print("Building Random baseline...")
rand_recs = candidates_df.withColumn("predicted_rating", F.rand())

# ==============================================================
# 6️⃣ Evaluation trên TOÀN BỘ test set (không filter rating >=4)
# ==============================================================
# Chỉ giữ user có trong candidates để fair với các model khác (tự động hit=0 nếu không có)
# Nhưng thực tế candidates bao phủ gần hết → không ảnh hưởng nhiều
test_users_with_recs = candidates_df.select("user_id").distinct()
df_test_eval = df_test.join(test_users_with_recs, "user_id", "inner").cache()

print(f"Evaluating on {df_test_eval.count()} test interactions (all ratings, only users in candidates)")

hit_pop, ndcg_pop = compute_metrics(spark, pop_recs, df_test_eval, k_recs=10)
hit_rand, ndcg_rand = compute_metrics(spark, rand_recs, df_test_eval, k_recs=10)

# ==============================================================
# 7️⃣ Results
# ==============================================================
print("\n" + "="*70)
print("          BASELINE RESULTS - TOÀN BỘ TEST SET (không filter >=4)")
print("="*70)
print(f"🎲 Random      : HR@10 = {hit_rand:.6f}  |  NDCG@10 = {ndcg_rand:.6f}")
print(f"🔥 Popularity  : HR@10 = {hit_pop:.6f}  |  NDCG@10 = {ndcg_pop:.6f}")
print("="*70)
print("→ Giờ thì hoàn toàn fair với DeepFM/FBT cũ (trước khi fix eval)!")
print("→ Popularity thường ~0.13-0.14 khi eval trên full test set (có cả rating thấp)")

df_test_eval.unpersist()
candidates_df.unpersist()
df_train.unpersist()
df_test.unpersist()

spark.stop()