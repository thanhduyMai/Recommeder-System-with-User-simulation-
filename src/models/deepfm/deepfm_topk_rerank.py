# File: deepfm_topk.py
# UPDATE: Train Full Data (Negative Sampling) -> Load ALS Candidates -> Rerank -> Save

import os
import random
import numpy as np

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import FloatType, ArrayType, StringType, StructType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler, OneHotEncoderEstimator

# BigDL Imports
from bigdl.dllib.nnframes import NNEstimator, NNModel
from bigdl.dllib.nn.layer import (
    Sequential, Linear, ReLU, ConcatTable, CAddTable, Square, Sum, Sigmoid
)
from bigdl.dllib.optim.optimizer import Adam, EveryEpoch, Loss
from bigdl.dllib.nn.criterion import BCECriterion
from bigdl.dllib.nncontext import init_nncontext
from bigdl.dllib.utils.common import *

try:
    from bigdl.dllib.optim.optimizer import TrainSummary, ValidationSummary
except ImportError:
    from bigdl.visualization.tensorboard import TrainSummary, ValidationSummary

# ==============================================================
# 1️⃣ CÁC HÀM HỖ TRỢ & UDF
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

normalize_udf = F.udf(normalize_vec_impl, ArrayType(FloatType()))
to_vector_udf = F.udf(to_dense_vector_impl, VectorUDT())
fill_null_vec_udf = F.udf(fill_null_vector_impl, VectorUDT())

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
# 2️⃣ SETUP SPARK
# ==============================================================
print("Starting DeepFM Reranking Service (Full Training Mode)...")
spark = SparkSession.builder \
    .appName("DeepFM_Rerank_Full") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()

init_nncontext(app_name="DeepFM_Rerank_Full", spark_conf=spark.sparkContext.getConf())
spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")

# PATHS
ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
meta_path = "hdfs:///DATALAKE/MyData/staging/filtered/meta_Beauty_and_Personal_Care_filtered"
title_w2v_path = "hdfs:///DATALAKE/MyData/feature_store/item_title_w2v"
review_w2v_path = "hdfs:///DATALAKE/MyData/feature_store/item_review_agg_w2v"

# INPUT/OUTPUT
als_recs_path = "hdfs:///DATALAKE/MyData/predictions/als_topk.parquet"
deepfm_output_path = "hdfs:///DATALAKE/MyData/predictions/deepfm_topk.parquet"

# ==============================================================
# 3️⃣ LOAD DATA & FEATURES
# ==============================================================
print("Loading & Preprocessing Data...")
ratings = spark.read.parquet(ratings_path).filter(F.col("verified_purchase") == True).select("user_id", "parent_asin", "rating", "helpful_vote", "event_date").dropna()
meta = spark.read.parquet(meta_path).select("parent_asin", "main_category", "price").withColumnRenamed("parent_asin", "item_id_meta").dropna()

# Vectors
title = spark.read.parquet(title_w2v_path)
if "title_vec" in title.columns: title = title.withColumnRenamed("title_vec", "item_title_vec").withColumnRenamed("parent_asin", "asin_title")
title = title.withColumn("item_title_vec", to_vector_udf(F.col("item_title_vec")))

review = spark.read.parquet(review_w2v_path)
src_col = "item_vec" if "item_vec" in review.columns else "review_vec"
review = review.withColumnRenamed(src_col, "item_review_vec").withColumnRenamed("parent_asin", "asin_review").withColumnRenamed("item_id", "asin_review")
review = review.withColumn("item_review_vec", to_vector_udf(F.col("item_review_vec")))

# Join
data = ratings.join(meta, ratings["parent_asin"] == meta["item_id_meta"]).drop("item_id_meta") \
              .join(title, ratings["parent_asin"] == title["asin_title"]).drop("asin_title") \
              .join(review, ratings["parent_asin"] == review["asin_review"]).drop("asin_review") \
              .withColumnRenamed("parent_asin", "asin")

data.cache()

# Cache All Items Features (để lát nữa join với ALS Candidates)
all_items_features = data.select("asin", "main_category", "item_title_vec", "item_review_vec").distinct().cache()
all_asins_list = all_items_features.select("asin").distinct().collect()
all_asins_broadcast = spark.sparkContext.broadcast([row.asin for row in all_asins_list])

# ==============================================================
# 4️⃣ HISTORY CALCULATION
# ==============================================================
print("Calculating User History...")
df_train, df_test = split_last_k_out(data, k=1)
df_train = df_train.repartition(200).cache()
data.unpersist()

# Helper for mean pooling
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
# 5️⃣ PIPELINE CONFIGURATION
# ==============================================================
print("Configuring Pipeline...")
category_indexer = StringIndexer(inputCol="main_category", outputCol="category_idx", handleInvalid="keep")
category_encoder = OneHotEncoderEstimator(inputCols=["category_idx"], outputCols=["category_vec"])
continuous_assembler = VectorAssembler(inputCols=["helpful_vote"], outputCol="continuous_vec")
scaler = MinMaxScaler(inputCol="continuous_vec", outputCol="continuous_scaled_vec")
final_assembler = VectorAssembler(
    inputCols=["category_vec", "continuous_scaled_vec", "item_title_vec", "item_review_vec", "history_title_vec", "history_review_vec"],
    outputCol="features"
)
pipeline = Pipeline(stages=[category_indexer, category_encoder, continuous_assembler, scaler, final_assembler])

# Prepare Data for Fit (Join History first)
df_train_for_fit = df_train.join(history_df, "user_id", "left") \
    .withColumn("history_title_vec", fill_null_vec_udf("history_title_vec")) \
    .withColumn("history_review_vec", fill_null_vec_udf("history_review_vec"))

pipeline_model = pipeline.fit(df_train_for_fit)

# ==============================================================
# 6️⃣ FULL TRAINING DATA PREPARATION (NEGATIVE SAMPLING)
# ==============================================================
print("Creating FULL Training Candidates (with Neg Sampling)...")

# 1. Positive Samples (Rating >= 4)
df_train_positive = df_train.filter(F.col("rating") >= 4.0)

# 2. History Setup (cho Neg Sampling)
full_history = df_train.groupBy("user_id").agg(F.collect_set("asin").alias("interacted_items"))
df_train_setup = df_train_positive.join(full_history, "user_id", "left")

print("⚡ Caching setup...")
df_train_setup = df_train_setup.repartition(400).checkpoint(eager=True) 

# 3. UDF Neg Sampling
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

# 4. Expand & Label
pos_rows = df_train_with_neg.select("user_id", "asin", "helpful_vote").withColumn("label", F.lit(1.0))
neg_rows = df_train_with_neg.select("user_id", F.explode("neg_asins").alias("asin")) \
    .withColumn("label", F.lit(0.0)) \
    .withColumn("helpful_vote", F.lit(0.0))

# 5. Merge & Join Features
print("Merging Positives & Negatives...")
train_candidates = pos_rows.unionByName(neg_rows) \
    .join(all_items_features, "asin", "left") \
    .fillna({"main_category": "unknown"}) \
    .join(history_df, "user_id", "left")

train_candidates = train_candidates \
    .withColumn("history_title_vec", fill_null_vec_udf(F.col("history_title_vec"))) \
    .withColumn("history_review_vec", fill_null_vec_udf(F.col("history_review_vec"))) \
    .fillna({"helpful_vote": 0.0})

print("⚡ Caching Raw Training Data...")
train_candidates = train_candidates.repartition(400).checkpoint(eager=True)

# 6. Transform Features
print("Applying Pipeline Transform...")
train_features_df = pipeline_model.transform(train_candidates).select("features", "label")
train_features_df = train_features_df.checkpoint(eager=True)
print(f"✅ Final Train Size: {train_features_df.count()}")

# ==============================================================
# 7️⃣ MODEL TRAINING
# ==============================================================
# Detect dimensions
sample_row = train_features_df.head()
n_features = sample_row.features.size
print(f"Features Dimension: {n_features}")

# Architecture (DeepFM)
k_latent = 10
fm_first = Sequential().add(Linear(n_features, 1))
fm_second = Sequential().add(Linear(n_features, k_latent)).add(Square()).add(Sum(dimension=2))
dnn = Sequential().add(Linear(n_features, 128)).add(ReLU()).add(Linear(128, 64)).add(ReLU()).add(Linear(64, 1))
model = Sequential().add(ConcatTable().add(fm_first).add(fm_second).add(dnn)).add(CAddTable()).add(Sigmoid())

# Train
criterion = BCECriterion()
estimator = NNEstimator(model, criterion) \
    .setBatchSize(4104) \
    .setMaxEpoch(20) \
    .setOptimMethod(Adam(1e-4))

print("🚀 Training DeepFM Model...")
trained_model = estimator.fit(train_features_df)
print("✅ Model Trained successfully.")

# ==============================================================
# 8️⃣ LOAD ALS CANDIDATES & RERANK
# ==============================================================
print("\n" + "="*50)
print("🚀 STARTING RERANKING PHASE (ALS -> DEEPFM)")
print("="*50)

print(f"Loading ALS Candidates from: {als_recs_path}")
try:
    df_als = spark.read.parquet(als_recs_path)
    # Explode: [A, B, C] -> Row(A), Row(B), Row(C)
    candidates_df = df_als.select(
        F.col("user_id"), 
        F.explode("recommendations").alias("asin")
    )
    print(f"   -> ALS Candidates loaded. Users: {df_als.count()}")
except Exception as e:
    print(f"❌ Error loading ALS: {e}")
    spark.stop()
    exit()

print("Enriching Candidates with Features...")
# 1. Join Item Features (Left Join để tránh mất candidates nếu thiếu meta)
candidates_enriched = candidates_df.join(all_items_features, "asin", "left") \
    .fillna({"main_category": "unknown"}) 

# 2. Join User History
candidates_final = candidates_enriched.join(history_df, "user_id", "left")

# 3. Fill Nulls (Rất quan trọng cho Inference)
candidates_final = candidates_final \
    .withColumn("history_title_vec", fill_null_vec_udf(F.col("history_title_vec"))) \
    .withColumn("history_review_vec", fill_null_vec_udf(F.col("history_review_vec"))) \
    .withColumn("item_title_vec", fill_null_vec_udf(F.col("item_title_vec"))) \
    .withColumn("item_review_vec", fill_null_vec_udf(F.col("item_review_vec"))) \
    .fillna({"helpful_vote": 0.0}) \
    .withColumn("label", F.lit(0.0)) # Dummy label cho Pipeline

print("Transforming Candidates...")
final_features = pipeline_model.transform(candidates_final).select("user_id", "asin", "features")

print("🔮 DeepFM Predicting Scores...")
predictions = trained_model.transform(final_features)

get_score = F.udf(lambda v: float(v[0]), FloatType())
recs_scored = predictions.withColumn("score", get_score(F.col("prediction")))

print("Selecting Top-10 DeepFM Reranked Items...")
window_spec = Window.partitionBy("user_id").orderBy(F.desc("score"))

final_topk_df = recs_scored.withColumn("rank", F.row_number().over(window_spec)) \
    .filter(F.col("rank") <= 10) \
    .groupBy("user_id") \
    .agg(F.collect_list("asin").alias("recommendations"))

print(f"💾 Saving to: {deepfm_output_path}")
final_topk_df.write.mode("overwrite").parquet(deepfm_output_path)

print("✅ DONE! Hybrid RecSys Pipeline Complete.")
spark.stop()