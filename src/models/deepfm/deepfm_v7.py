# File: deepfm_train_final_spark24.py
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

# [SPARK 2.4] Định nghĩa UDF theo kiểu cũ để tránh lỗi serialization
def normalize_vec_impl(vec):
    if vec is None:
        return None
    # Convert Vector to array/list if needed
    if hasattr(vec, "toArray"):
        vec = vec.toArray()
    
    norm = np.linalg.norm(vec)
    if norm == 0:
        return [float(v) for v in vec]
    norm_float = float(norm)
    return [float(v / norm_float) for v in vec]

# [SPARK 2.4] UDF chuyển Array -> DenseVector an toàn
def to_dense_vector_impl(x):
    if x is None:
        return None
    return Vectors.dense(x)

# [SPARK 2.4] UDF Fill Null Vector (Quan trọng nhất)
# Cần hardcode dimension nếu biết trước (ở đây giả sử 64)
# Nếu dimension khác, hãy sửa số 64
vector_dim = 64
zero_list = [0.0] * vector_dim

def fill_null_vector_impl(v):
    if v is None:
        return Vectors.dense(zero_list)
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
# 3️⃣ Tải và Xử lý Dữ liệu (ĐÃ SỬA TÊN CỘT)
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

# --- Load & Process Title Vectors ---
print("Processing Title Vectors...")
item_title_df = spark.read.parquet(title_w2v_path)

# [FIX] In schema ra để kiểm tra nếu lỗi
print("Title Schema:")
item_title_df.printSchema()

# [FIX] Đổi tên dựa trên log lỗi của bạn (title_vec -> item_title_vec)
item_title_df = item_title_df \
    .withColumnRenamed("title_vec", "item_title_vec") \
    .withColumnRenamed("parent_asin", "asin_title") 

# Normalize (Array) -> Convert to Vector
item_title_df = item_title_df.withColumn("item_title_vec", normalize_udf(F.col("item_title_vec"))) \
                             .withColumn("item_title_vec", to_vector_udf(F.col("item_title_vec")))

# --- Load & Process Review Vectors ---
print("Processing Review Vectors...")
item_review_df = spark.read.parquet(review_w2v_path)

# [FIX] In schema ra để kiểm tra
print("Review Schema:")
item_review_df.printSchema()


if "review_vec" in item_review_df.columns:
    src_col = "review_vec"
elif "word2vec" in item_review_df.columns:
    src_col = "word2vec"
else:
    # Fallback: Lấy cột array/vector đầu tiên tìm thấy trừ cột ID
    cols = item_review_df.columns
    src_col = [c for c in cols if "asin" not in c and "id" not in c][0]
    print(f"Warning: Auto-detected review vector column: {src_col}")

item_review_df = item_review_df \
    .withColumnRenamed(src_col, "item_review_vec") \
    .withColumnRenamed("parent_asin", "asin_review") \
    .withColumnRenamed("item_id", "asin_review") # Handle trường hợp tên là item_id

# Normalize (Array) -> Convert to Vector
item_review_df = item_review_df.withColumn("item_review_vec", normalize_udf(F.col("item_review_vec"))) \
                               .withColumn("item_review_vec", to_vector_udf(F.col("item_review_vec")))

# --- JOIN ---
print("Joining Data Tables...")

data = ratings.join(item_features_meta, ratings["parent_asin"] == item_features_meta["item_id_meta"]).drop("item_id_meta") \
              .join(item_title_df, ratings["parent_asin"] == item_title_df["asin_title"]).drop("asin_title") \
              .join(item_review_df, ratings["parent_asin"] == item_review_df["asin_review"]).drop("asin_review") \
              .withColumnRenamed("parent_asin", "asin")

data.cache()

# Broadcast Items for Neg Sampling
all_items_features = data.select("asin", "main_category", "item_title_vec", "item_review_vec").distinct().cache()
all_asins_list = all_items_features.select("asin").distinct().collect()
all_asins_broadcast = spark.sparkContext.broadcast([row.asin for row in all_asins_list])
# ==============================================================
# 4️⃣ Split & History Calculation
# ==============================================================
print("Splitting LOO...")
df_train, df_test = split_last_k_out(data, k=1, user_col="user_id", time_col="event_date", tie_breaker_col="asin")
df_train = df_train.repartition(200).cache()
df_test.cache()
data.unpersist()

# --- Tính toán History Mean ---
# [SPARK 2.4] Convert Vector -> Array để dùng collect_list
to_array_udf_manual = F.udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))

df_train_arr = df_train.withColumn("title_arr", to_array_udf_manual("item_title_vec")) \
                       .withColumn("review_arr", to_array_udf_manual("item_review_vec"))

# UDF Mean Pool cho List of Lists
def mean_pool_impl(list_vecs):
    if not list_vecs: 
        return [0.0] * vector_dim
    # Spark 2.4 đôi khi trả về None trong list, cần filter
    valid_vecs = [v for v in list_vecs if v is not None]
    if not valid_vecs:
        return [0.0] * vector_dim
    return np.mean(np.array(valid_vecs), axis=0).tolist()

mean_pool_udf = F.udf(mean_pool_impl, ArrayType(FloatType()))

history_df = df_train_arr.groupBy("user_id").agg(
    mean_pool_udf(F.collect_list("title_arr")).alias("history_title_vec"),
    mean_pool_udf(F.collect_list("review_arr")).alias("history_review_vec")
)

# Convert lại thành Vector
history_df = history_df.withColumn("history_title_vec", to_vector_udf(F.col("history_title_vec"))) \
                       .withColumn("history_review_vec", to_vector_udf(F.col("history_review_vec")))
history_df.cache()

# ==============================================================
# 5️⃣ Pipeline (Spark 2.4: OneHotEncoderEstimator)
# ==============================================================
print("Configuring pipeline...")
category_indexer = StringIndexer(inputCol="main_category", outputCol="category_idx", handleInvalid="keep")

# [SPARK 2.4 UPDATE] Dùng OneHotEncoderEstimator thay vì OneHotEncoder (deprecated/khác behavior)
category_encoder = OneHotEncoderEstimator(inputCols=["category_idx"], outputCols=["category_vec"])

continuous_assembler = VectorAssembler(inputCols=["helpful_vote"], outputCol="continuous_vec")
scaler = MinMaxScaler(inputCol="continuous_vec", outputCol="continuous_scaled_vec")

final_assembler = VectorAssembler(
    inputCols=[
        "category_vec", 
        "continuous_scaled_vec", 
        "item_title_vec",       
        "item_review_vec",      
        "history_title_vec",    
        "history_review_vec"    
    ],
    outputCol="features"
)

pipeline = Pipeline(stages=[category_indexer, category_encoder, continuous_assembler, scaler, final_assembler])

df_train_for_fit = df_train.join(history_df, "user_id", "left")
# Fill null tạm thời cho history trước khi fit pipeline 
df_train_for_fit = df_train_for_fit.withColumn("history_title_vec", fill_null_vec_udf("history_title_vec")) \
                                   .withColumn("history_review_vec", fill_null_vec_udf("history_review_vec"))

print("Fitting pipeline...")
pipeline_model = pipeline.fit(df_train_for_fit)
# ==============================================================
# 6️⃣ Tạo Training Data (Unseen Negatives) 
# ==============================================================
print("Creating training candidates...")

# 1. Positive
df_train_positive = df_train.filter(F.col("rating") >= 4.0)

# 2. History Exclusion
full_history = df_train.groupBy("user_id").agg(F.collect_set("asin").alias("interacted_items"))
df_train_setup = df_train_positive.join(full_history, "user_id", "left")

# ⚡️ [TỐI ƯU QUAN TRỌNG] Checkpoint trung gian tại đây!
# Lý do: Tính toán History (groupBy) và Join rất nặng. 
# Lưu lại kết quả này xuống đĩa trước khi chạy UDF Random để tránh Spark tính lại từ đầu nếu thiếu RAM.
print("⚡️ Caching intermediate setup (Positives + History) to speed up UDF...")
df_train_setup = df_train_setup.repartition(400).checkpoint(eager=True) 

# 3. Neg Sampling UDF
num_neg = 2
@F.udf(ArrayType(StringType()))
def sample_unseen_negatives(interacted_items):
    all_items = all_asins_broadcast.value
    excluded = set(interacted_items) if interacted_items else set()
    available = list(set(all_items) - excluded)
    if not available: return []
    return random.sample(available, min(num_neg, len(available)))

print("Running Negative Sampling UDF...")
df_train_with_neg = df_train_setup.withColumn("neg_asins", sample_unseen_negatives(F.col("interacted_items")))

# 4. Expand & Label
# [FIX] Positive: Phải select cả "helpful_vote"
pos_rows = df_train_with_neg.select("user_id", "asin", "helpful_vote") \
    .withColumn("label", F.lit(1.0))

# [FIX] Negative: Phải tạo cột "helpful_vote" giả (giá trị 0.0)
neg_rows = df_train_with_neg.select("user_id", F.explode("neg_asins").alias("asin")) \
    .withColumn("label", F.lit(0.0)) \
    .withColumn("helpful_vote", F.lit(0.0))

# 5. Merge & Join Features
print("Merging Positives and Negatives...")
train_candidates = pos_rows.unionByName(neg_rows) \
    .join(all_items_features, "asin", "left") \
    .fillna({"main_category": "unknown"}) \
    .join(history_df, "user_id", "left")

# 6. Fill Null Vectors bằng UDF chuẩn
print("Filling null vectors...")
train_candidates = train_candidates.withColumn("history_title_vec", fill_null_vec_udf(F.col("history_title_vec"))) \
                                   .withColumn("history_review_vec", fill_null_vec_udf(F.col("history_review_vec")))
    
# [QUAN TRỌNG] Fill null cho cột helpful_vote
train_candidates = train_candidates.fillna({"helpful_vote": 0.0})

# ⚡️ [TỐI ƯU LẦN 2] Checkpoint dữ liệu thô trước khi đưa vào Pipeline
# Giúp Pipeline (VectorAssembler) chạy nhanh hơn vì không phải đợi Join nữa
print("⚡️ Caching raw training candidates before Pipeline...")
train_candidates = train_candidates.repartition(400).checkpoint(eager=True)

# 7. Transform
print("Applying Pipeline Transform...")
train_features_df = pipeline_model.transform(train_candidates).select("features", "label")

# Checkpoint cuối cùng (để sẵn sàng cho training)
print("Finalizing Training Dataset...")
train_features_df = train_features_df.checkpoint(eager=True)
print(f"Final Train Size: {train_features_df.count()}")
# ==============================================================
# 7️⃣ Model Training
# ==============================================================
sample_row = train_features_df.head()
n_features = sample_row.features.size
print(f"✅ Auto-detected features dimension: {n_features}")

# DeepFM Architecture
k_latent = 10
fm_first = Sequential().add(Linear(n_features, 1))
fm_second = Sequential().add(Linear(n_features, k_latent)).add(Square()).add(Sum(dimension=2))
dnn = Sequential() \
    .add(Linear(n_features, 128)).add(ReLU()) \
    .add(Linear(128, 64)).add(ReLU()) \
    .add(Linear(64, 1))

model = Sequential() \
    .add(ConcatTable().add(fm_first).add(fm_second).add(dnn)) \
    .add(CAddTable()) \
    .add(Sigmoid())

# Train
print("Starting training...")
log_dir = "/tmp/bigdl_dual_vec_logs"
if not os.path.exists(log_dir): os.makedirs(log_dir)
app_name = "DeepFM_Final_Run" # <--- Định nghĩa tên App

print("Splitting Train/Validation set...")
train_df, val_df = train_features_df.randomSplit([0.8, 0.2], seed=42)

train_summary = TrainSummary(log_dir=log_dir, app_name=app_name)
val_summary = ValidationSummary(log_dir=log_dir, app_name=app_name)
criterion = BCECriterion()

estimator = NNEstimator(model, criterion) \
    .setBatchSize(4104) \
    .setMaxEpoch(20) \
    .setOptimMethod(Adam(1e-4)) \
    .setTrainSummary(train_summary) \
    .setValidationSummary(val_summary) \
    .setValidation(
        trigger=EveryEpoch(),
        val_df=val_df,
        val_method=[Loss(criterion)], 
        batch_size=4104
    )

trained_model = estimator.fit(train_df)
print(f"Training Done. Logs saved to {log_dir}")


# ==============================================================
# 8️⃣ ĐÁNH GIÁ (EVALUATION) - PHIÊN BẢN DUAL VECTOR CHUẨN
# ==============================================================
print("\n" + "="*50)
print("🚀 STARTING EVALUATION PHASE")
print("="*50)

# Path tới file candidates bạn vừa tạo xong
candidates_load_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates_dual.parquet"

# 1. Load Candidates (Đã có sẵn Title/Review Vec chuẩn từ file precompute)
print(f"Loading precomputed candidates from {candidates_load_path}...")
candidates_df = spark.read.parquet(candidates_load_path)

# 2. Join với User History (Lấy từ biến history_df đang có trong RAM)
# Candidates thiếu thông tin lịch sử, phải join vào mới chạy model được.
print("Joining Candidates with User History...")

# Chỉ select các cột cần thiết để tránh lỗi trùng tên
candidates_subset = candidates_df.select(
    "user_id", "asin", "label", 
    "main_category", "helpful_vote", 
    "item_title_vec", "item_review_vec" 
)

# Join Left để giữ lại toàn bộ đề thi (candidates)
eval_df = candidates_subset.join(history_df, "user_id", "left")

# 3. [QUAN TRỌNG] Fill Null Vectors (Fix lỗi Cold Start / Null Join)
# Dùng lại UDF 'fill_null_vec_udf' đã định nghĩa ở đầu file train
print("Handling Nulls for History Vectors...")

eval_df = eval_df.withColumn("history_title_vec", fill_null_vec_udf(F.col("history_title_vec"))) \
                 .withColumn("history_review_vec", fill_null_vec_udf(F.col("history_review_vec")))

# Fill null cho các cột phụ (nếu cần)
eval_df = eval_df.fillna({"main_category": "unknown", "helpful_vote": 0.0})

# 4. Transform qua Pipeline (Tái sử dụng pipeline đã fit lúc train)
print("Transforming candidates through Feature Pipeline...")
# Pipeline sẽ tự động biến đổi Category -> OneHot, Scale Helpful Vote, v.v.
eval_features = pipeline_model.transform(eval_df).select("user_id", "asin", "features", "label")

# 5. Dự đoán (Predict)
print("Predicting scores with Trained Model...")
predictions = trained_model.transform(eval_features)

# Trích xuất điểm số (probability) từ vector dự đoán (DeepFM output vector size 1 hoặc 2)
get_score = F.udf(lambda v: float(v[0]), FloatType())
recs_df = predictions.withColumn("predicted_rating", get_score(F.col("prediction"))) \
                     .select("user_id", "asin", "predicted_rating", "label")

# 6. Tính Metrics (HR@10, NDCG@10)
print("Calculating Ranking Metrics (NDCG@10, HR@10)...")

K = 10
window_spec = Window.partitionBy("user_id").orderBy(F.desc("predicted_rating"))

# Xếp hạng
ranked_df = recs_df.withColumn("rank", F.row_number().over(window_spec))

# Lọc Top K
top_k_df = ranked_df.filter(F.col("rank") <= K)

# Tính Hit & NDCG
metrics_df = top_k_df.groupBy("user_id").agg(
    F.sum("label").alias("is_hit"), 
    F.sum(F.expr("IF(label=1, 1.0/log2(rank+1), 0.0)")).alias("ndcg_val")
)

# Trung bình toàn bộ user
final_metrics = metrics_df.agg(
    F.mean("is_hit").alias("hit_rate"),
    F.mean("ndcg_val").alias("ndcg")
).collect()[0]

print("\n" + "="*40)
print("📊 FINAL DEEPFM EVALUATION RESULTS")
print("="*40)
print(f"   Candidate Set: 1 Pos + 100 Negs")
print(f"   HitRate@{K}  : {final_metrics['hit_rate']:.6f}")
print(f"   NDCG@{K}     : {final_metrics['ndcg']:.6f}")
print("="*40 + "\n")

# Dừng Spark
spark.stop()