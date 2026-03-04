# -*- coding: utf-8 -*-
"""
Phân tích phân phối số lượng sản phẩm mỗi user (num_items)
+ Phân tích phân phối co_count để chọn threshold cho FBT
→ Lưu dữ liệu vào Feature Store
"""

from pyspark.sql import SparkSession, functions as F, Window

# ==============================================================  
# 1️⃣ SPARK SESSION  
# ==============================================================  
spark = (
    SparkSession.builder
    .appName("Analyze_User_Item_and_CoCount_Distribution")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.executor.instances", "4")
    .config("spark.executor.cores", "4")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")

# ==============================================================  
# 2️⃣ LOAD DATA  
# ==============================================================  
ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"

df = (
    spark.read.parquet(ratings_path)
    .select("user_id", "parent_asin")
    .dropna(subset=["user_id", "parent_asin"])
    .distinct()
)

# ==============================================================  
# 3️⃣ PHÂN TÍCH SỐ LƯỢNG ITEM MỖI USER  
# ==============================================================  
user_counts = (
    df.groupBy("user_id")
      .agg(F.countDistinct("parent_asin").alias("num_items"))
      .checkpoint(eager=True)
)

# Summary (Q1, median, Q3, max, mean)
summary_stats = (
    user_counts
    .select(
        F.expr("percentile(num_items, array(0.25, 0.5, 0.75))").alias("quartiles"),
        F.max("num_items").alias("max_items"),
        F.mean("num_items").alias("mean_items")
    )
    .collect()[0]
)

quartiles = summary_stats["quartiles"]
print("\n📊 [USER ITEM SUMMARY]")
print(f"  Mean num_items/user = {summary_stats['mean_items']:.2f}")
print(f"  Q1 = {quartiles[0]:.1f}, Median = {quartiles[1]:.1f}, Q3 = {quartiles[2]:.1f}, Max = {summary_stats['max_items']}")
print("------------------------------------------------------")

# Percentiles
percentiles = [0.9, 0.95, 0.99, 0.995, 0.999]
exprs = [F.expr(f"percentile(num_items, {p})").alias(f"p{int(p*1000)}") for p in percentiles]
percentile_row = user_counts.select(exprs).collect()[0]

print("📈 [USER ITEM PERCENTILES]")
for p in percentiles:
    print(f"  {int(p*100)}% users có ≤ {percentile_row[f'p{int(p*1000)}']:.1f} sản phẩm")

# ==============================================================  
# 4️⃣ TẠO CẶP SẢN PHẨM (ITEM PAIRS)  
# ==============================================================  
df_pairs = (
    df.alias("a")
      .join(df.alias("b"), "user_id")
      .where(F.col("a.parent_asin") < F.col("b.parent_asin"))
      .select(
          F.col("a.parent_asin").alias("asin_A"),
          F.col("b.parent_asin").alias("asin_B")
      )
      .checkpoint(eager=True)
)

print("\n🔧 Generated item pairs from user interactions")

# ==============================================================  
# 5️⃣ TÍNH co_count  
# ==============================================================  
co_counts = (
    df_pairs.groupBy("asin_A", "asin_B")
            .agg(F.count("*").alias("co_count"))
            .checkpoint(eager=True)
)

print(f"🔧 co_counts computed: {co_counts.count():,} item pairs")

# ==============================================================  
# 6️⃣ PHÂN TÍCH DISTRIBUTION co_count  
# ==============================================================  
print("\n📊 [CO_COUNT SUMMARY]")

# Tổng quan
summary_cc = co_counts.select(
    F.mean("co_count").alias("mean_cc"),
    F.expr("percentile(co_count, array(0.25, 0.5, 0.75))").alias("quartiles")
).collect()[0]

cc_q = summary_cc["quartiles"]
print(f"  Mean co_count = {summary_cc['mean_cc']:.2f}")
print(f"  Q1 = {cc_q[0]}, Median = {cc_q[1]}, Q3 = {cc_q[2]}")

# Percentiles
percentiles_cc = [0.9, 0.95, 0.99, 0.995]
exprs_cc = [F.expr(f"percentile(co_count, {p})").alias(f"p{int(p*1000)}") for p in percentiles_cc]
row_cc = co_counts.select(exprs_cc).collect()[0]

print("\n📈 [CO_COUNT PERCENTILES]")
for p in percentiles_cc:
    print(f"  {int(p*100)}% item-pairs có co_count ≤ {row_cc[f'p{int(p*1000)}']}")

# Buckets
print("\n📦 [CO_COUNT BUCKETS]")
buckets = (
    co_counts
    .select(
        F.when(F.col("co_count") == 1, "1")
         .when(F.col("co_count") == 2, "2")
         .when(F.col("co_count").between(3, 4), "3-4")
         .when(F.col("co_count").between(5, 9), "5-9")
         .otherwise(">=10")
         .alias("bucket")
    )
)
bucket_dist = buckets.groupBy("bucket").count()
bucket_dist.show(truncate=False)

# ==============================================================  
# 7️⃣ LƯU FEATURE STORE  
# ==============================================================  
user_count_path = "hdfs:///DATALAKE/MyData/feature_store/user_item_counts"
co_count_path = "hdfs:///DATALAKE/MyData/feature_store/item_pair_co_counts"

user_counts.write.mode("overwrite").parquet(user_count_path)
co_counts.write.mode("overwrite").parquet(co_count_path)

print(f"\n✅ Saved:")
print(f"  user_item_counts   → {user_count_path}")
print(f"  item_pair_co_count → {co_count_path}")

spark.stop()
