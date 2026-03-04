# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, when, regexp_replace
from pyspark.sql.types import DoubleType


def create_spark_session(app_name="RatingsEDA"):
    """Tạo SparkSession với cấu hình cơ bản"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_parquet(spark, path):
    """Load parquet file từ HDFS"""
    print(f"\nĐang đọc dữ liệu từ: {path}")
    return spark.read.parquet(path)


def check_nulls(df, name="DataFrame"):
    """Kiểm tra số lượng null trên từng cột"""
    print(f"\n===== Số lượng null trong {name} =====")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()


from pyspark.sql import functions as F

from pyspark.storagelevel import StorageLevel  # Thêm dòng này


def filter_min_reviews(df, spark, min_reviews=5, max_iter=20):
    """
    Iteratively filter users and items until both have >= min_reviews interactions.
    This ensures a k-core dataset.
    """
    print(f"\n===== Iterative filtering for users >= {min_reviews} and items >= {min_reviews} =====")
    current_df = df.repartition(100)
    spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark-checkpoints")  # Set checkpoint dir (thay bằng path HDFS của bạn nếu cần)
    
    iteration = 0
    while True:
        iteration += 1
        if iteration > max_iter:
            print(f"🛑 Đạt max iteration {max_iter}, dừng để tránh loop lâu.")
            break
        
        print(f"Starting iteration {iteration}...")
        prev_count = current_df.count()
        
        # Filter users >= min_reviews
        user_counts = current_df.groupBy("user_id").agg(F.count("*").alias("user_count")).filter(F.col("user_count") >= min_reviews)
        valid_users = user_counts.select("user_id").distinct()
        current_df = current_df.join(valid_users, on="user_id", how="inner").checkpoint(eager=True)  # Checkpoint để break lineage
        
        # Filter items >= min_reviews
        item_counts = current_df.groupBy("asin").agg(F.count("*").alias("item_count")).filter(F.col("item_count") >= min_reviews)
        valid_items = item_counts.select("asin").distinct()
        current_df = current_df.join(valid_items, on="asin", how="inner").checkpoint(eager=True)  # Checkpoint lại
        
        new_count = current_df.count()
        change = prev_count - new_count
        print(f"Iteration {iteration}: Interactions = {new_count} (change: {change})")
        
        # Stop if no more changes or change small
        if change < 10 or new_count == prev_count:
            break
    
    # Final stats
    final_users = current_df.select("user_id").distinct().count()
    final_items = current_df.select("asin").distinct().count()
    print(f"✅ Final: {final_users} users, {final_items} items, {new_count} interactions")
    
    return current_df

def clean_meta(df_meta,df_ratings):
    """Làm sạch dữ liệu metadata"""
    print("\n===== Đang xử lý dữ liệu  =====")
    check_nulls(df_meta, name="df_meta (trước khi fillna)")
    check_nulls(df_ratings, name= "df_ratings trước khi fillna")

    fill_map = {}
    if "main_category" in df_meta.columns:
        fill_map["main_category"] = "Unknown"
    if "store" in df_meta.columns:
        fill_map["store"] = "Unknown"
    if "average_rating" in df_meta.columns:
        fill_map["average_rating"] = 0.0

    df_meta_filled = df_meta.fillna(fill_map)

    if "price" in df_meta_filled.columns:
        df_meta_clean = df_meta_filled.withColumn(
            "price",
            regexp_replace(col("price"), "[$,]", "").cast(DoubleType())
        )
        # === SỬA ĐỔI: Thêm cột cờ cho price ===
        # Cột này sẽ có giá trị True nếu price bị thiếu, và False nếu có giá trị.
        df_meta_clean = df_meta_clean.withColumn(
            "is_price_missing",
            when(col("price").isNull(), True).otherwise(False)
        )
    else:
        df_meta_clean = df_meta_filled
     
    if "parent_asin" in df_meta_clean.columns:
        df_meta_clean = df_meta_clean.filter(col("parent_asin").isNotNull())

    print("\nSau khi làm sạch:")
    check_nulls(df_meta_clean, name="df_meta_clean")
    
    if "price" in df_meta_clean.columns:
        total_rows = df_meta_clean.count()
        null_price_count = df_meta_clean.filter(col("price").isNull()).count()
        # Tính toán null_ratio chỉ khi total_rows > 0 để tránh lỗi chia cho 0
        if total_rows > 0:
            null_ratio = null_price_count / total_rows * 100
            print(f"\n💰 Tổng số dòng: {total_rows}")
            print(f"💸 Số dòng thiếu price: {null_price_count} ({null_ratio:.2f}%)")
        else:
            print("\nKhông có dữ liệu để tính toán tỷ lệ thiếu price.")

    # Mô tả thêm cho các giá trị không null
    df_meta_clean.filter(col("price").isNotNull()).select("price").describe().show()

    cols_to_describe = [c for c in ("price", "average_rating") if c in df_meta_clean.columns]
    if cols_to_describe:
        print("\nMô tả thống kê cho các cột:", cols_to_describe)
        df_meta_clean.select(*cols_to_describe).describe().show()

    return df_meta_clean


def basic_eda(df):
    """Thực hiện EDA cơ bản cho ratings"""
    print("\n===== Schema =====")
    df.printSchema()
    check_nulls(df)

    print("\n===== Top sản phẩm theo lượt review =====")
    df.groupBy('asin').agg(count('*').alias('review_count')).orderBy(desc('review_count')).show(10, truncate=False)

    print("\n===== Top user theo lượt review =====")
    df.groupBy('user_id').agg(count('*').alias('review_count')).orderBy(desc('review_count')).show(10, truncate=False)

    distinct_users = df.select('user_id').distinct().count()
    distinct_items = df.select('asin').distinct().count()
    total_interactions = df.count()

    print(f"\n===== Tổng interactions: {total_interactions} =====")
    print(f"Distinct users: {distinct_users}")
    print(f"Distinct items: {distinct_items}")
    print(f"Avg actions / user: {total_interactions / distinct_users:.2f}")
    print(f"Avg actions / item: {total_interactions / distinct_items:.2f}")


def main():
    spark = create_spark_session()

    # Đường dẫn input
    ratings_path = "hdfs:///DATALAKE/MyData/staging/Beauty_and_Personal_Care"
    meta_path = "hdfs:///DATALAKE/MyData/staging/meta_Beauty_and_Personal_Care"

    # Đường dẫn output
    ratings_filtered_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
    meta_clean_path = "hdfs:///DATALAKE/MyData/staging/cleaned/meta_Beauty_and_Personal_Care_cleaned"
    meta_filtered_path = "hdfs:///DATALAKE/MyData/staging/filtered/meta_Beauty_and_Personal_Care_filtered"

    # 1️⃣ Load dữ liệu
    df_ratings = load_parquet(spark, ratings_path)
    df_meta = load_parquet(spark, meta_path)
    print(f"Trước khi lọc, số lượng interactions: {df_ratings.count()}")

    # 2️⃣ Làm sạch metadata (fillna + clean price + bỏ parent_asin null)
    df_meta_clean = clean_meta(df_meta, df_ratings)
    print(f"\n💾 Ghi dữ liệu meta sạch ra HDFS: {meta_clean_path}")
    df_meta_clean.write.mode("overwrite").parquet(meta_clean_path)

    # 3️⃣ Lọc user/item có ít hơn 5 reviews
    df_ratings_filtered = filter_min_reviews(df_ratings, spark, min_reviews=5)
    print(f"Sau khi lọc, số lượng interactions: {df_ratings_filtered.count()}")

    # 4️⃣ Lọc metadata chỉ giữ lại các sản phẩm có trong tập ratings_filtered
    # Dùng asin vì ratings thường không có parent_asin
    if "parent_asin" in df_meta_clean.columns and "parent_asin" in df_ratings_filtered.columns:
        join_col = "parent_asin"
    else:
        join_col = "asin"

    remaining_items = df_ratings_filtered.select(join_col).distinct()
    df_meta_filtered = df_meta_clean.join(remaining_items, on=join_col, how="inner")

    print(f"📦 df_meta_clean: {df_meta_clean.count()} rows")
    print(f"📦 df_meta_filtered: {df_meta_filtered.count()} rows")
    print(f"📦 df_ratings_filtered: {df_ratings_filtered.count()} rows")

    # 5️⃣ Ghi ra HDFS
    print(f"\n💾 Ghi dữ liệu ratings đã lọc ra HDFS: {ratings_filtered_path}")
    df_ratings_filtered.write.mode("overwrite").parquet(ratings_filtered_path)

    print(f"💾 Ghi dữ liệu meta đã lọc ra HDFS: {meta_filtered_path}")
    df_meta_filtered.write.mode("overwrite").parquet(meta_filtered_path)

    # 6️⃣ EDA cho tập ratings đã lọc
    basic_eda(df_ratings_filtered)

    print("\n✅ Đã lưu thành công:")
    print(f" - Meta sạch (clean): {meta_clean_path}")
    print(f" - Meta lọc theo rating: {meta_filtered_path}")
    print(f" - Ratings lọc: {ratings_filtered_path}")

    spark.stop()



if __name__ == "__main__":
    main()