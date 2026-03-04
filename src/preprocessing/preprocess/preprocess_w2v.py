import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec, Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT

# ====================================================
# 1. CẤU HÌNH & KHỞI TẠO
# ====================================================
spark = SparkSession.builder \
    .appName("Dual_W2V_Generation") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Paths Input
meta_path = "hdfs:///DATALAKE/MyData/staging/filtered/meta_Beauty_and_Personal_Care_filtered"
reviews_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"

# Paths Output
out_title_path = "hdfs:///DATALAKE/MyData/feature_store/item_title_w2v"


# Hàm xử lý vector lỗi (NaN/Null)
@F.udf(VectorUDT())
def fix_vector_udf(v):
    if v is None or np.isnan(v.toArray()).any():
        return Vectors.dense([0.0] * 64)
    return v

# ====================================================
# 2. HÀM TRAIN W2V CHUẨN HÓA (Dùng chung)
# ====================================================
def train_w2v_pipeline(df, input_col_name, output_vec_name, min_count=5):
    print(f"🚀 Processing column '{input_col_name}'...")
    
    # 1. Tokenize (Chỉ lấy chữ cái)
    tokenizer = RegexTokenizer(inputCol=input_col_name, outputCol="tokens_raw", pattern="\\W+")
    df_tok = tokenizer.transform(df)
    
    # 2. Remove Stopwords
    remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens")
    df_tok = remover.transform(df_tok)
    
    # 3. Filter short text
    df_tok = df_tok.filter(F.size(F.col("tokens")) > 1)
    
    # 4. Train Word2Vec
    print(f"   - Training W2V model for {output_vec_name}...")
    w2v = Word2Vec(
        vectorSize=64, 
        windowSize=5, 
        minCount=min_count, 
        inputCol="tokens", 
        outputCol="vec_raw",
        maxIter=5 # 5 iter là đủ nhanh
    )
    model = w2v.fit(df_tok)
    df_vec = model.transform(df_tok)
    
    # 5. Normalize L2 (BẮT BUỘC để tránh nổ Loss)
    print("   - Normalizing vectors...")
    normalizer = Normalizer(inputCol="vec_raw", outputCol=output_vec_name, p=2.0)
    df_final = normalizer.transform(df_vec)
    
    # 6. Fix NaN
    df_final = df_final.withColumn(output_vec_name, fix_vector_udf(F.col(output_vec_name)))
    
    return df_final

# ====================================================
# 3. LUỒNG 1: XỬ LÝ TITLE (Item Identity)
# ====================================================
print("\n=== STARTING PIPELINE 1: TITLE EMBEDDING ===")
# Load Meta
meta = spark.read.parquet(meta_path).select("parent_asin", "title", "main_category") \
    .dropDuplicates(["parent_asin"]).dropna(subset=["title"])

# Ghép Title + Category để định danh rõ hơn
meta = meta.withColumn("text_identity", F.concat_ws(" ", F.col("main_category"), F.col("title")))

# Train & Get Vectors
df_title_vec = train_w2v_pipeline(meta, "text_identity", "title_vec", min_count=5)

# Save
print(f"💾 Saving Title Vectors to {out_title_path}...")
df_title_vec.select("parent_asin", "title_vec") \
    .write.mode("overwrite").parquet(out_title_path)




print("\n✅ ALL DONE! Created 'title_vec' ")
spark.stop()