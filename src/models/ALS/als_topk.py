
import numpy as np
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.recommendation import ALS, ALSModel

# 1. SETUP
spark = SparkSession.builder \
    .appName("ALS_Generate_TopK") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")

# 2. PATHS
ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
# Dùng chung đề thi với DeepFM
candidates_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates_dual.parquet"
# Output riêng cho ALS
output_path = "hdfs:///DATALAKE/MyData/predictions/als_topk.parquet"

# 3. TRAIN ALS (Nhanh nên train lại luôn cho tiện)
print("🚀 Loading Data & Training ALS...")
df_ratings = spark.read.parquet(ratings_path) \
    .filter(F.col("verified_purchase") == True) \
    .select("user_id", "parent_asin", "rating") \
    .withColumnRenamed("parent_asin", "asin")

# Lọc rating >= 4 (Logic Positive của ALS)
df_train = df_ratings.filter(F.col("rating") >= 4).withColumn("confidence", F.lit(1.0))

# Indexing (Bắt buộc cho ALS)
user_indexer = df_ratings.select("user_id").distinct().rdd.zipWithIndex().toDF(["tmp", "userIndex"]).select(F.col("tmp.user_id"), F.col("userIndex").cast("long"))
item_indexer = df_ratings.select("asin").distinct().rdd.zipWithIndex().map(lambda x: (x[0]["asin"], x[1])).toDF(["asin", "itemIndex"]).select("asin", F.col("itemIndex").cast("long"))

df_train_idx = df_train.join(user_indexer, "user_id").join(item_indexer, "asin")

# Train Model
als = ALS(
    userCol="userIndex", itemCol="itemIndex", ratingCol="confidence",
    implicitPrefs=True, alpha=40.0, rank=64, regParam=0.1, maxIter=15, coldStartStrategy="drop"
)
model = als.fit(df_train_idx)
print("✅ ALS Trained.")

# 4. PREDICT TRÊN CANDIDATES
print("🔮 Predicting on Candidates...")
# Load Candidate (Chỉ lấy ID)
candidates = spark.read.parquet(candidates_path).select("user_id", "asin").distinct()

# Map Index
candidates_idx = candidates.join(user_indexer, "user_id").join(item_indexer, "asin")

# Predict
predictions = model.transform(candidates_idx)
# Fill 0 cho cold-start
recs = predictions.withColumn("score", F.coalesce(F.col("prediction"), F.lit(0.0)))

# 5. LẤY TOP 10 & SAVE
print("💾 Saving Top-10 ALS Recommendations...")
window_spec = Window.partitionBy("user_id").orderBy(F.desc("score"))

top_k_als = recs.withColumn("rank", F.row_number().over(window_spec)) \
    .filter(F.col("rank") <= 10) \
    .groupBy("user_id") \
    .agg(F.collect_list("asin").alias("recommendations"))

top_k_als.write.mode("overwrite").parquet(output_path)
print(f"✅ DONE! Saved to: {output_path}")

spark.stop()