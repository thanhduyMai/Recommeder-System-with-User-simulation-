# -*- coding: utf-8 -*-
"""
Phân tích phân phối số lượng sản phẩm mỗi user (num_items)
→ Dùng để chọn ngưỡng lọc hợp lý cho FBT
→ Tự lưu user_counts ra HDFS để tái sử dụng
"""

from pyspark.sql import SparkSession, functions as F

# ==============================================================
# 1️⃣ Spark Session
# ==============================================================
spark = (
    SparkSession.builder
    .appName("Analyze_User_Item_Distribution_Spark")
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
# 2️⃣ Load data (5-core filtered)
# ==============================================================
ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
df = (
    spark.read.parquet(ratings_path)
    .select("user_id", "parent_asin")
    .dropna(subset=["user_id", "parent_asin"])
    .distinct()
)

# ==============================================================
# 3️⃣ Đếm số item mỗi user
# ==============================================================
user_counts = (
    df.groupBy("user_id")
      .agg(F.countDistinct("parent_asin").alias("num_items"))
      .checkpoint(eager=True)
)

# ==============================================================
# 4️⃣ Tính thống kê tổng quan
# ==============================================================
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
print("\n📊 [SUMMARY]")
print(f"  Mean num_items/user = {summary_stats['mean_items']:.2f}")
print(f"  Q1 = {quartiles[0]:.1f}, Median = {quartiles[1]:.1f}, Q3 = {quartiles[2]:.1f}, Max = {summary_stats['max_items']}")
print("------------------------------------------------------")

# ==============================================================
# 5️⃣ Phân phối phần trăm user (percentiles)
# ==============================================================
percentiles = [0.9, 0.95, 0.99, 0.995, 0.999]
exprs = [F.expr(f"percentile(num_items, {p})").alias(f"p{int(p*1000)}") for p in percentiles]
percentile_row = user_counts.select(exprs).collect()[0]

print("📈 [PERCENTILES]")
for p in percentiles:
    print(f"  {int(p*100)}% user có ≤ {percentile_row[f'p{int(p*1000)}']:.1f} sản phẩm")

# ==============================================================
# 6️⃣ Lưu bảng user_counts ra HDFS (tái sử dụng trong FBT)
# ==============================================================
output_path = "hdfs:///DATALAKE/MyData/feature_store/user_item_counts"
user_counts.write.mode("overwrite").parquet(output_path)
print(f"\n✅ Đã lưu user_counts ra: {output_path}")

spark.stop()
