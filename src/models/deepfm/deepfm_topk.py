# File: deepfm_train_v2_with_save.py
import os
import random
import numpy as np

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import FloatType, ArrayType, StringType, StructType
# [SPARK 2.4] Import VectorUDT từ linalg là BẮT BUỘC
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler, OneHotEncoderEstimator # Spark 2.4 dùng OneHotEncoderEstimator

# BigDL Imports
from bigdl.dllib.nnframes import NNEstimator, NNModel
from bigdl.dllib.nn.layer import (
    Sequential, Linear, ReLU, ConcatTable, CAddTable, Square, Sum, Sigmoid
)
from bigdl.dllib.optim.optimizer import Adam, EveryEpoch, Loss
from bigdl.dllib.nn.criterion import BCECriterion
from bigdl.dllib.nncontext import init_nncontext
from bigdl.dllib.utils.common import *

# Fix Import Summary
try:
    from bigdl.dllib.optim.optimizer import TrainSummary, ValidationSummary
except ImportError:
    from bigdl.visualization.tensorboard import TrainSummary, ValidationSummary

# ==============================================================
# 1️⃣ CÁC HÀM HỖ TRỢ & UDF (CHUẨN SPARK 2.4)
# ==============================================================

def normalize_vec_impl(vec):
    if vec is None: return None
    if hasattr(vec, "toArray"): vec = vec.toArray()
    norm = np.linalg.norm(vec)
    if norm == 0: return [float(v) for v in vec]
    return [float(v / float(norm)) for v in vec]

def to_dense_vector_impl(x):
    if x is None: return None
    return Vectors.dense(x)

vector_dim = 64
zero_list = [0.0] * vector_dim
def fill_null_vector_impl(v):
    if v is None: return Vectors.dense(zero_list)
    return v

# Đăng ký UDF
normalize_udf = F.udf(normalize_vec_impl, ArrayType(FloatType()))
to_vector_udf = F.udf(to_dense_vector_impl, VectorUDT())
fill_null_vec_udf = F.udf(fill_null_vector_impl, VectorUDT())

# Hàm LOO Split
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

# ==============================================================
# 2️⃣ Spark Session
# ==============================================================
print("Starting DeepFM (Spark 2.4 Safe Mode)...")
spark = SparkSession.builder \
    .appName("DeepFM_Spark2.4_Optimized") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

init_nncontext(app_name="DeepFM_Spark2.4_Optimized", spark_conf=spark.sparkContext.getConf())
spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")

# --- Paths ---
ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
meta_path = "hdfs:///DATALAKE/MyData/staging/filtered/meta_Beauty_and_Personal_Care_filtered"
title_w2v_path = "hdfs:///DATALAKE/MyData/feature_store/item_title_w2v"
review_w2v_path = "hdfs:///DATALAKE/MyData/feature_store/item_review_agg_w2v"

# ==============================================================
# 3️⃣ Tải và Xử lý Dữ liệu
# ==============================================================
print("Loading data...")
ratings = spark.read.parquet(ratings_path) \
    .filter(F.col("verified_purchase") == True) \
    .select("user_id", "parent_asin", "rating", "helpful_vote", "event_date") \
    .dropna()

item_features_meta = spark.read.parquet(meta_path) \
    .select("parent_asin", "main_category", "price") \
    .withColumnRenamed("parent_asin", "item_id_meta") \
    .dropna()

# Load Vectors
print("Processing Vectors...")
item_title_df = spark.read.parquet(title_w2v_path)
if "title_vec" in item_title_df.columns:
    item_title_df = item_title_df.withColumnRenamed("title_vec", "item_title_vec").withColumnRenamed("parent_asin", "asin_title")

item_title_df = item_title_df.withColumn("item_title_vec", to_vector_udf(F.col("item_title_vec")))

item_review_df = spark.read.parquet(review_w2v_path)
if "item_vec" in item_review_df.columns:
    src_col = "item_vec"
else:
    src_col = "review_vec"
    
item_review_df = item_review_df \
    .withColumnRenamed(src_col, "item_review_vec") \
    .withColumnRenamed("parent_asin", "asin_review") \
    .withColumnRenamed("item_id", "asin_review") 

item_review_df = item_review_df.withColumn("item_review_vec", to_vector_udf(F.col("item_review_vec")))

# Join
print("Joining Data Tables...")
data = ratings.join(item_features_meta, ratings["parent_asin"] == item_features_meta["item_id_meta"]).drop("item_id_meta") \
              .join(item_title_df, ratings["parent_asin"] == item_title_df["asin_title"]).drop("asin_title") \
              .join(item_review_df, ratings["parent_asin"] == item_review_df["asin_review"]).drop("asin_review") \
              .withColumnRenamed("parent_asin", "asin")

data.cache()

all_items_features = data.select("asin", "main_category", "item_title_vec", "item_review_vec").distinct().cache()
all_asins_list = all_items_features.select("asin").distinct().collect()
all_asins_broadcast = spark.sparkContext.broadcast([row.asin for row in all_asins_list])

# ==============================================================
# 4️⃣ Split & History Calculation
# ==============================================================
print("Splitting LOO...")
df_train, df_test = split_last_k_out(data, k=1, user_col="user_id", time_col="event_date", tie_breaker_col="asin")
df_train = df_train.repartition(200).cache()
data.unpersist()

# History
to_array_udf_manual = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
df_train_arr = df_train.withColumn("title_arr", to_array_udf_manual("item_title_vec")) \
                       .withColumn("review_arr", to_array_udf_manual("item_review_vec"))

def mean_pool_impl(list_vecs):
    if not list_vecs: return [0.0] * vector_dim
    valid_vecs = [v for v in list_vecs if v is not None]
    if not valid_vecs: return [0.0] * vector_dim
    return np.mean(np.array(valid_vecs), axis=0).tolist()

mean_pool_udf = F.udf(mean_pool_impl, ArrayType(FloatType()))

history_df = df_train_arr.groupBy("user_id").agg(
    mean_pool_udf(F.collect_list("title_arr")).alias("history_title_vec"),
    mean_pool_udf(F.collect_list("review_arr")).alias("history_review_vec")
)

history_df = history_df.withColumn("history_title_vec", to_vector_udf(F.col("history_title_vec"))) \
                       .withColumn("history_review_vec", to_vector_udf(F.col("history_review_vec")))
history_df.cache()

# ==============================================================
# 5️⃣ Pipeline
# ==============================================================
print("Configuring pipeline...")
category_indexer = StringIndexer(inputCol="main_category", outputCol="category_idx", handleInvalid="keep")
category_encoder = OneHotEncoderEstimator(inputCols=["category_idx"], outputCols=["category_vec"])
continuous_assembler = VectorAssembler(inputCols=["helpful_vote"], outputCol="continuous_vec")
scaler = MinMaxScaler(inputCol="continuous_vec", outputCol="continuous_scaled_vec")

final_assembler = VectorAssembler(
    inputCols=["category_vec", "continuous_scaled_vec", "item_title_vec", "item_review_vec", "history_title_vec", "history_review_vec"],
    outputCol="features"
)

pipeline = Pipeline(stages=[category_indexer, category_encoder, continuous_assembler, scaler, final_assembler])

df_train_for_fit = df_train.join(history_df, "user_id", "left")
df_train_for_fit = df_train_for_fit.withColumn("history_title_vec", fill_null_vec_udf("history_title_vec")) \
                                   .withColumn("history_review_vec", fill_null_vec_udf("history_review_vec"))

print("Fitting pipeline...")
pipeline_model = pipeline.fit(df_train_for_fit)

# ==============================================================
# 6️⃣ Creating Training Data
# ==============================================================
print("Creating training candidates...")
df_train_positive = df_train.filter(F.col("rating") >= 4.0)
full_history = df_train.groupBy("user_id").agg(F.collect_set("asin").alias("interacted_items"))
df_train_setup = df_train_positive.join(full_history, "user_id", "left")

print("⚡ Caching setup...")
df_train_setup = df_train_setup.repartition(400).checkpoint(eager=True) 

num_neg = 2
@F.udf(ArrayType(StringType()))
def sample_unseen_negatives(interacted_items):
    all_items = all_asins_broadcast.value
    excluded = set(interacted_items) if interacted_items else set()
    available = list(set(all_items) - excluded)
    if not available: return []
    return random.sample(available, min(num_neg, len(available)))

print("Running Negative Sampling...")
df_train_with_neg = df_train_setup.withColumn("neg_asins", sample_unseen_negatives(F.col("interacted_items")))

pos_rows = df_train_with_neg.select("user_id", "asin", "helpful_vote").withColumn("label", F.lit(1.0))
neg_rows = df_train_with_neg.select("user_id", F.explode("neg_asins").alias("asin")).withColumn("label", F.lit(0.0)).withColumn("helpful_vote", F.lit(0.0))

print("Merging...")
train_candidates = pos_rows.unionByName(neg_rows) \
    .join(all_items_features, "asin", "left") \
    .fillna({"main_category": "unknown"}) \
    .join(history_df, "user_id", "left")

train_candidates = train_candidates.withColumn("history_title_vec", fill_null_vec_udf(F.col("history_title_vec"))) \
                                   .withColumn("history_review_vec", fill_null_vec_udf(F.col("history_review_vec"))) \
                                   .fillna({"helpful_vote": 0.0})

print("⚡ Caching raw candidates...")
train_candidates = train_candidates.repartition(400).checkpoint(eager=True)

print("Applying Pipeline...")
train_features_df = pipeline_model.transform(train_candidates).select("features", "label")
train_features_df = train_features_df.checkpoint(eager=True)
print(f"Final Train Size: {train_features_df.count()}")

# ==============================================================
# 7️⃣ Model Training
# ==============================================================
sample_row = train_features_df.head()
n_features = sample_row.features.size
print(f"✅ Auto-detected features dimension: {n_features}")

k_latent = 10
fm_first = Sequential().add(Linear(n_features, 1))
fm_second = Sequential().add(Linear(n_features, k_latent)).add(Square()).add(Sum(dimension=2))
dnn = Sequential().add(Linear(n_features, 128)).add(ReLU()).add(Linear(128, 64)).add(ReLU()).add(Linear(64, 1))
model = Sequential().add(ConcatTable().add(fm_first).add(fm_second).add(dnn)).add(CAddTable()).add(Sigmoid())

log_dir = "/tmp/bigdl_dual_vec_logs"
if not os.path.exists(log_dir): os.makedirs(log_dir)
app_name = "DeepFM_Final_Run"

print("Splitting Train/Val...")
train_df, val_df = train_features_df.randomSplit([0.8, 0.2], seed=42)

criterion = BCECriterion()
estimator = NNEstimator(model, criterion) \
    .setBatchSize(4104) \
    .setMaxEpoch(20) \
    .setOptimMethod(Adam(1e-4)) \
    .setTrainSummary(TrainSummary(log_dir, app_name)) \
    .setValidationSummary(ValidationSummary(log_dir, app_name)) \
    .setValidation(trigger=EveryEpoch(), val_df=val_df, val_method=[Loss(criterion)], batch_size=4104)

trained_model = estimator.fit(train_df)
print(f"Training Done. Logs saved to {log_dir}")

# ==============================================================
# 8️⃣ EVALUATION (Dùng Precomputed Candidates)
# ==============================================================
print("\n" + "="*50)
print("🚀 STARTING EVALUATION PHASE")
print("="*50)

candidates_load_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates_dual.parquet"
print(f"Loading precomputed candidates from {candidates_load_path}...")
candidates_df = spark.read.parquet(candidates_load_path)

candidates_subset = candidates_df.select(
    "user_id", "asin", "label", 
    "main_category", "helpful_vote", 
    "item_title_vec", "item_review_vec" 
)

eval_df = candidates_subset.join(history_df, "user_id", "left")
eval_df = eval_df.withColumn("history_title_vec", fill_null_vec_udf(F.col("history_title_vec"))) \
                 .withColumn("history_review_vec", fill_null_vec_udf(F.col("history_review_vec"))) \
                 .fillna({"main_category": "unknown", "helpful_vote": 0.0})

print("Transforming candidates...")
eval_features = pipeline_model.transform(eval_df).select("user_id", "asin", "features", "label")

print("Predicting scores...")
predictions = trained_model.transform(eval_features)

get_score = F.udf(lambda v: float(v[0]), FloatType())
recs_df = predictions.withColumn("predicted_rating", get_score(F.col("prediction"))) \
                     .select("user_id", "asin", "predicted_rating", "label")

print("Calculating Ranking Metrics...")
K = 10
window_spec = Window.partitionBy("user_id").orderBy(F.desc("predicted_rating"))
ranked_df = recs_df.withColumn("rank", F.row_number().over(window_spec))

metrics_df = ranked_df.filter(F.col("rank") <= K).groupBy("user_id").agg(
    F.sum("label").alias("is_hit"), 
    F.sum(F.expr("IF(label=1, 1.0/log2(rank+1), 0.0)")).alias("ndcg_val")
)

final_metrics = metrics_df.agg(F.mean("is_hit").alias("hit_rate"), F.mean("ndcg_val").alias("ndcg")).collect()[0]

print("\n" + "="*40)
print("📊 FINAL DEEPFM EVALUATION RESULTS")
print(f"   HitRate@{K}  : {final_metrics['hit_rate']:.6f}")
print(f"   NDCG@{K}     : {final_metrics['ndcg']:.6f}")
print("="*40 + "\n")

# ==============================================================
# 9️⃣ [NEW] SAVE TOP-K RECOMMENDATIONS FOR RECSIM
# ==============================================================
print("\n" + "="*50)
print("💾 SAVING TOP-10 RECOMMENDATIONS TO HDFS")
print("="*50)

recs_save_path = "hdfs:///DATALAKE/MyData/predictions/deepfm_topk.parquet"

# Tái sử dụng ranked_df đã tính ở trên (Lọc Top 10, GroupBy User)
final_topk_df = ranked_df.filter(F.col("rank") <= 10) \
                         .groupBy("user_id") \
                         .agg(F.collect_list("asin").alias("recommendations"))

print(f"Writing parquet to: {recs_save_path}...")
final_topk_df.write.mode("overwrite").parquet(recs_save_path)
print("✅ Successfully saved Top-K recommendations!")

spark.stop()