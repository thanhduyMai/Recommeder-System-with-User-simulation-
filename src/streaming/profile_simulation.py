# -*- coding: utf-8 -*-


import sys
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import FloatType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT

# =========================================================
# 1️⃣ SPARK SESSION
# =========================================================
def create_spark_session(app_name="RecSim_Full_Coverage"):
    spark = (
      SparkSession.builder
      .appName(app_name)
      .config("spark.driver.memory", "8g")
      .config("spark.executor.memory", "8g")
      .config("spark.yarn.executor.memoryOverhead", "4g")
      .config("spark.sql.shuffle.partitions", "800")
      .config("spark.driver.maxResultSize", "4g")
      .config("spark.sql.execution.arrow.enabled", "true")
      .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# =========================================================
# 2️⃣ UDFS
# =========================================================

# Gộp 2 vector: (v1 + v2) / 2. Nếu null thì coi là 0.
def merge_vectors_impl(v1, v2):
    if not v1: v1 = [0.0] * 64
    if not v2: v2 = [0.0] * 64
    return [(float(a) + float(b)) / 2.0 for a, b in zip(v1, v2)]

# Trung bình lịch sử
def mean_pool_impl(list_vecs):
    if not list_vecs: return [0.0] * 64
    valid_vecs = [v for v in list_vecs if v is not None]
    if not valid_vecs: return [0.0] * 64
    return np.mean(np.array(valid_vecs), axis=0).tolist()

# Convert VectorUDT -> Array
def to_arr_impl(v):
    if v is not None: return v.toArray().tolist()
    return [0.0] * 64

merge_udf = F.udf(merge_vectors_impl, ArrayType(FloatType()))
mean_pool_udf = F.udf(mean_pool_impl, ArrayType(FloatType()))
to_arr = F.udf(to_arr_impl, ArrayType(FloatType()))

# =========================================================
# 3️⃣ MAIN FLOW
# =========================================================
def main():
    spark = create_spark_session()
    
    # --- PATHS ---
    ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
    meta_path = "hdfs:///DATALAKE/MyData/staging/filtered/meta_Beauty_and_Personal_Care_filtered"
    title_path = "hdfs:///DATALAKE/MyData/feature_store/item_title_w2v"
    review_path = "hdfs:///DATALAKE/MyData/feature_store/item_review_agg_w2v"
    output_path = "hdfs:///DATALAKE/MyData/recsim/catalogs/"

    print("⏳ Loading Data...")
    
    # 1. Ratings
    df_ratings = spark.read.parquet(ratings_path) \
        .filter(F.col("verified_purchase") == True) \
        .select("user_id", "parent_asin")

    # 2. Vectors (Convert to Array & Rename)
    print("   -> Processing Title Vectors...")
    df_title = spark.read.parquet(title_path)
    if "title_vec" in df_title.columns:
        df_title = df_title.withColumn("arr_title", to_arr("title_vec"))
    else:
        df_title = df_title.withColumn("arr_title", to_arr("item_title_vec"))
    df_title = df_title.withColumnRenamed("parent_asin", "asin_t").select("asin_t", "arr_title")

    print("   -> Processing Review Vectors...")
    df_review = spark.read.parquet(review_path)
    vec_col = "item_vec" if "item_vec" in df_review.columns else "review_vec"
    if vec_col not in df_review.columns: vec_col = "item_review_vec"
    
    df_review = df_review.withColumn("arr_review", to_arr(vec_col))
    df_review = df_review.withColumnRenamed("parent_asin", "asin_r") \
                         .withColumnRenamed("item_id", "asin_r") \
                         .select("asin_r", "arr_review")

    # =====================================================
    # 3. DOCUMENT CATALOG (FULL OUTER JOIN - KEEP ALL)
    # =====================================================
    print("🔨 Building Document Catalog (Full Coverage)...")
    
    # FULL OUTER JOIN: Lấy cả Title-only và Review-only
    df_dual = df_title.join(df_review, df_title.asin_t == df_review.asin_r, "full_outer")
    
    # Hợp nhất ID: Lấy cái nào không null
    df_dual = df_dual.withColumn("product_id", F.coalesce(F.col("asin_t"), F.col("asin_r")))
    
    # Tính vector gộp (UDF tự xử lý null thành 0)
    df_docs = df_dual.withColumn("sim_content_vec", merge_udf("arr_title", "arr_review")) \
        .select("product_id", "sim_content_vec")

    # Join Meta lấy giá (Left join để không mất item nếu thiếu meta)
    df_meta = spark.read.parquet(meta_path).select("parent_asin", "price")
    df_docs = df_docs.join(df_meta, df_docs.product_id == df_meta.parent_asin, "left").drop("parent_asin")

    print(f"💾 Saving Document Catalog ({df_docs.count()} items)...")
    df_docs.write.mode("overwrite").parquet(output_path + "document_catalog.parquet")
    
    # =====================================================
    # 4. USER CATALOG (LEFT JOIN - KEEP ALL USERS)
    # =====================================================
    print("🔨 Building User Catalog (Full History)...")
    
    # LEFT JOIN: Giữ lại mọi hành vi mua, kể cả món không có vector
    df_user_hist = df_ratings.join(df_docs, df_ratings.parent_asin == df_docs.product_id, "left")
    
    # GroupBy tính trung bình (Bỏ qua null)
    df_users = df_user_hist.groupBy("user_id").agg(
        mean_pool_udf(F.collect_list("sim_content_vec")).alias("user_sim_vec")
    )

    print(f"💾 Saving User Catalog ({df_users.count()} users)...")
    df_users.repartition(10).write.mode("overwrite").parquet(output_path + "active_users.parquet")
    
    print("✅ ALL DONE! Catalogs are maximized.")
    spark.stop()

if __name__ == "__main__":
    main()