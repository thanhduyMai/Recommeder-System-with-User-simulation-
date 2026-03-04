
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql import functions as F

def main():
    print("🚀 RETRY PUSHING ITEMS TO ELASTICSEARCH...")

    # [FIX] Dùng ngoặc đơn (...) bao quanh để không cần dùng dấu \ nữa
    spark = (SparkSession.builder
        .appName("Fix_Recs_Push")
        .config("spark.es.nodes", "node3.nms.net")
        .config("spark.es.port", "9200")
        .config("spark.es.nodes.wan.only", "true")
        .getOrCreate())
	
	# Path
    recs_path = "hdfs:///DATALAKE/MyData/predictions/als_topk.parquet"

    print(f"📥 Loading ALS Recs from: {recs_path}")
    try:
        df_recs = spark.read.parquet(recs_path)
        count = df_recs.count()
        print(f"   -> Found {count} users in DeepFM Parquet.")
        
        # Nếu count > 17000 thì chứng tỏ file HDFS ngon, chỉ là chưa vào hết ES
        
        print("⚡ Pushing to ES Index: 'recommendations' ...")
        df_recs.write.format("org.elasticsearch.spark.sql") \
            .option("es.resource", "recommendation_als/_doc") \
            .option("es.mapping.id", "user_id") \
            .option("es.write.operation", "upsert") \
            .option("es.batch.size.entries", "1000") \
            .mode("overwrite") \
            .save()
            
        print("✅ SUCCESS: ALS Recs pushed correctly!")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    spark.stop()

if __name__ == "__main__":
    main()