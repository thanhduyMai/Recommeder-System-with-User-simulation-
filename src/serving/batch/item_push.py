# item_push.py (Đã sửa lỗi cú pháp an toàn)
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql import functions as F

def main():
    print("🚀 RETRY PUSHING ITEMS TO ELASTICSEARCH...")

    # [FIX] Dùng ngoặc đơn (...) bao quanh để không cần dùng dấu \ nữa
    spark = (SparkSession.builder
        .appName("Fix_Items_Push")
        .config("spark.es.nodes", "node3.nms.net")
        .config("spark.es.port", "9200")
        .config("spark.es.nodes.wan.only", "true")
        .getOrCreate())

   # Path
    items_path = "hdfs:///DATALAKE/MyData/recsim/catalogs/document_catalog.parquet"

    print(f"📥 Loading Items from: {items_path}")
    try:
        df_items = spark.read.parquet(items_path)
        count = df_items.count()
        print(f"   -> Found {count} items in Parquet.")
        
        if count == 0:
            print("❌ FILE PARQUET RỖNG! Kiểm tra lại file recsim.py đã chạy chưa?")
            return

        # 2. Tự động tìm và convert Vector -> Array
        # Định nghĩa UDF
        to_arr = F.udf(lambda v: v.toArray().tolist() if v is not None else [], ArrayType(FloatType()))

        # Lặp qua các cột để xử lý
        cols_to_select = []
        for col_name, dtype in df_items.dtypes:
            # Nếu là vector, convert sang array
            if 'vector' in dtype.lower() or 'struct' in dtype.lower():
                print(f"   -> Converting vector column '{col_name}' to Array for ES...")
                df_items = df_items.withColumn(col_name, to_arr(col_name))
            
            cols_to_select.append(col_name)

        # 3. Đẩy vào ES
        print("⚡ Pushing to ES Index: 'items' ...")
        df_items.select(*cols_to_select).write.format("org.elasticsearch.spark.sql") \
            .option("es.resource", "items/_doc") \
            .option("es.mapping.id", "product_id") \
            .option("es.write.operation", "upsert") \
            .option("es.batch.size.entries", "1000") \
            .mode("overwrite") \
            .save()
            
        print("✅ SUCCESS: Items pushed correctly!")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    spark.stop()

if __name__ == "__main__":
    main()