# producer_real.py
import socket
import time
import numpy as np
from pyspark.sql import functions as F
import random
from itertools import cycle
from pyspark.sql import SparkSession

# --- CẤU HÌNH ---
HOST = 'localhost'
PORT = 9997  # Giữ nguyên 9997 cho khớp với Consumer
USER_CATALOG_PATH = "hdfs:///DATALAKE/MyData/recsim/catalogs/active_users.parquet"

def get_real_users():
    """Load User ID từ HDFS, tính thêm history_length từ ratings nguồn"""
    print("⏳ Starting Spark to load REAL User IDs from HDFS...")
    spark = SparkSession.builder \
        .appName("Producer_Load_Users") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # 1. Load active_users.parquet (chỉ user_id, như cũ)
        df_active = spark.read.parquet(USER_CATALOG_PATH).select("user_id")
        
        # 2. Load ratings nguồn để tính history_length
        ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
        df_ratings = spark.read.parquet(ratings_path) \
            .filter(F.col("verified_purchase") == True) \
            .groupBy("user_id").agg(
                F.count("parent_asin").alias("history_length")
            )
        
        # 3. LEFT JOIN: Giữ toàn bộ active users, thêm history_length (nếu user không có ratings thì history=0)
        df_combined = df_active.join(df_ratings, "user_id", "left") \
            .na.fill({"history_length": 0})  # Fill null thành 0 nếu cần
        
        # 4. Sắp xếp theo history_length GIẢM DẦN → user active nhất đầu tiên
        df_sorted = df_combined.orderBy(F.col("history_length").desc())
        
        rows = df_sorted.select("user_id").collect()  # Chỉ cần user_id cho producer
        user_ids = [row['user_id'] for row in rows]
        
        print(f"✅ Loaded {len(user_ids)} users, đã sắp xếp theo history_length giảm dần.")
        return user_ids
        
    except Exception as e:
        print(f"❌ Error loading/sorting users: {e}")
        # Fallback: dùng thứ tự ngẫu nhiên nếu lỗi
        fallback_df = spark.read.parquet(USER_CATALOG_PATH).select("user_id")
        user_ids = [row['user_id'] for row in fallback_df.collect()]
        print(f"⚠️  Fallback: dùng thứ tự ngẫu nhiên ({len(user_ids)} users).")
        return user_ids
    finally:
        spark.stop()

def server_program():
    # 1. Lấy dữ liệu thật
    user_ids = get_real_users()
    
    if not user_ids:
        print("❌ No users found. Exiting.")
        return

    # 2. Tạo phân phối Zipfian (Mô phỏng hành vi User thật: ít người active nhiều)
    print("🎲 Generating Zipfian distribution weights...")
    ranks = np.arange(1, len(user_ids) + 1)
    weights = ranks ** (-1.2)
    weights /= weights.sum()

    # 3. Mở Socket Server
    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"\n📡 Producer listening on {HOST}:{PORT}...")
    print("Waiting for Spark Streaming to connect...")
    
    conn, address = server_socket.accept()
    print("✅ Spark Streaming Connected: " + str(address))
    
    try:
        # Pre-generate stream index để tốc độ bắn cực nhanh
        stream_indices = cycle(np.random.choice(len(user_ids), size=100000, p=weights))
        
        while True:
            idx = next(stream_indices)
            user_id = user_ids[idx]
            
            # --- SỬA ĐỔI QUAN TRỌNG TẠI ĐÂY ---
            # 1. Thêm \n rõ ràng
            message = f"{user_id}\n"
            
            # 2. Dùng sendall + encode utf-8 để đảm bảo gửi trọn vẹn
            conn.sendall(message.encode('utf-8'))
            
            # 3. Chậm lại chút để Spark kịp xử lý (0.1 - 0.3s)
            time.sleep(random.uniform(0.1, 0.3)) 
            
            # 4. In ra có flush=True để log hiện ngay lập tức
            print(f"Sent: {user_id:<30}", end="\r", flush=True)
            
    except (BrokenPipeError, ConnectionResetError):
        print("\n❌ Client disconnected.")
    finally:
        conn.close()
        server_socket.close()

if __name__ == '__main__':
    server_program()