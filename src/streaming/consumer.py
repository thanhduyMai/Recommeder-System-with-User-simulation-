import sys
sys.path.insert(0, "libs.zip")

import time
import random
import datetime
import numpy as np
from elasticsearch import Elasticsearch, helpers
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================
ES_HOST = "http://node3.nms.net:9200"
SOCKET_HOST = "localhost"
SOCKET_PORT = 9997

MONITOR_INDEX = "monitor_logs"
SIM_CHECK_INDEX = "simulator_sanity_logs"

USER_CATALOG_PATH = "hdfs:///DATALAKE/MyData/recsim/catalogs/active_users.parquet"
RATINGS_PATH = "/DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"

GLOBAL_STATS = defaultdict(lambda: {"clicks": 0, "impr": 0, "lat_sum": 0.0, "cnt": 0})

# =========================================================
# SPARK SETUP
# =========================================================
spark = SparkSession.builder \
    .appName("RecSys_Simulator_With_SanityCheck") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("⏳ Loading resources...")

pdf_users = spark.read.parquet(USER_CATALOG_PATH) \
    .select("user_id", "user_sim_vec").toPandas()

USER_VEC_DICT = {
    str(uid).strip(): np.array(vec)
    for uid, vec in zip(pdf_users['user_id'], pdf_users['user_sim_vec'])
}

bc_user_vecs = spark.sparkContext.broadcast(USER_VEC_DICT)

try:
    top_items = spark.read.parquet(RATINGS_PATH) \
        .groupBy("parent_asin").count() \
        .orderBy(col("count").desc()) \
        .limit(10).select("parent_asin").toPandas()
    POPULAR_ITEMS = top_items['parent_asin'].tolist()
except:
    POPULAR_ITEMS = ["Fallback_1", "Fallback_2"]

bc_popular_items = spark.sparkContext.broadcast(POPULAR_ITEMS)
bc_es_host = spark.sparkContext.broadcast(ES_HOST)

print("✅ Ready.")

# =========================================================
# CLICK MODEL
# =========================================================
def click_signal(user_vec, item_vec, rank, strategy):
    dot = np.dot(user_vec, item_vec)
    norm = (np.linalg.norm(user_vec) * np.linalg.norm(item_vec)) + 1e-9
    sim = dot / norm

    if strategy == "Personalized":
        alpha, beta = 11.0, 8.0
    else:
        alpha, beta = 7.0, 4.0

    base_prob = 1 / (1 + np.exp(-(alpha * sim - beta)))
    decay = 1.0 / np.log2(rank + 2)
    final_prob = base_prob * decay

    return sim, base_prob, decay, final_prob

# =========================================================
# PARTITION LOGIC
# =========================================================
def process_partition(iterator):
    user_vecs = bc_user_vecs.value
    pop_items = bc_popular_items.value
    es = Elasticsearch([bc_es_host.value], request_timeout=2)

    user_logs = []
    sanity_logs = []

    for row in iterator:
        user_id = str(row.user_id).strip()
        if user_id not in user_vecs:
            user_logs.append((user_id, "MISSING_KEY", 0, 0, 0.0))
            continue

        start = time.time()
        user_vec = user_vecs[user_id]

        if random.random() < 0.5:
            model = "DeepFM"
            index = "recommendations"
        else:
            model = "ALS"
            index = "recommendation_als"

        strategy = "Personalized"

        try:
            try:
                rec = es.get(index=index, id=user_id)
                item_ids = rec['_source']['recommendations']
            except:
                item_ids = pop_items
                strategy = "Popularity"
                model += "_Fallback"

            impressions = 0
            clicked = 0

            docs = es.mget(index="items", body={"ids": item_ids})['docs']
            docs = [d for d in docs if d.get('found')]
            impressions = len(docs)

            for rank, doc in enumerate(docs):
                vec = doc['_source'].get("sim_content_vec")
                if not vec:
                    continue

                sim, base_p, decay, final_p = click_signal(
                    user_vec, np.array(vec), rank, strategy
                )

                click = 1 if random.random() < final_p else 0

                sanity_logs.append((
                    user_id, model, strategy, rank,
                    float(sim), float(base_p),
                    float(decay), float(final_p), click
                ))

                if click:
                    clicked = 1
                    break

            latency = (time.time() - start) * 1000
            user_logs.append((user_id, model, impressions, clicked, latency))

        except:
            user_logs.append((user_id, "ERROR", 0, 0, 0.0))

    return iter(user_logs + sanity_logs)

# =========================================================
# ELASTICSEARCH PUSH
# =========================================================
es_monitor = Elasticsearch([ES_HOST])

def push_user_logs(rows):
    actions = []
    now = datetime.datetime.utcnow().isoformat()

    for r in rows:
        user_id, model, impr, click, lat = r
        actions.append({
            "_index": MONITOR_INDEX,
            "_source": {
                "@timestamp": now,
                "user_id": user_id,
                "model": model,
                "impressions": impr,
                "clicked": click,
                "latency_ms": lat,
                "status": "success" if model not in ["ERROR", "MISSING_KEY"] else "failed"
            }
        })
    helpers.bulk(es_monitor, actions)

def push_sanity_logs(rows):
    actions = []
    now = datetime.datetime.utcnow().isoformat()

    for r in rows:
        (user_id, model, strategy, rank,
         sim, base_p, decay, final_p, click) = r

        actions.append({
            "_index": SIM_CHECK_INDEX,
            "_source": {
                "@timestamp": now,
                "user_id": user_id,
                "model": model,
                "strategy": strategy,
                "rank": rank,
                "similarity": sim,
                "base_prob": base_p,
                "position_decay": decay,
                "final_prob": final_p,
                "clicked": click
            }
        })
    helpers.bulk(es_monitor, actions)

# =========================================================
# STREAMING DRIVER
# =========================================================
raw_stream = spark.readStream \
    .format("socket") \
    .option("host", SOCKET_HOST) \
    .option("port", SOCKET_PORT) \
    .load()

user_df = raw_stream.select(trim(col("value")).alias("user_id"))

def process_batch(df, epoch_id):
    if df.rdd.isEmpty():
        return

    rows = df.rdd.mapPartitions(process_partition).collect()

    user_logs = [r for r in rows if len(r) == 5]
    sanity_logs = [r for r in rows if len(r) == 9]

    push_user_logs(user_logs)
    push_sanity_logs(sanity_logs)

    for _, model, impr, click, lat in user_logs:
        if model in ["ERROR", "MISSING_KEY"]:
            continue
        GLOBAL_STATS[model]["clicks"] += click
        GLOBAL_STATS[model]["impr"] += impr
        GLOBAL_STATS[model]["lat_sum"] += lat
        GLOBAL_STATS[model]["cnt"] += 1

    print(f"\n--- BATCH {epoch_id} ---")
    for model, v in GLOBAL_STATS.items():
        ctr = (v["clicks"] / v["impr"] * 100) if v["impr"] > 0 else 0
        lat = v["lat_sum"] / v["cnt"] if v["cnt"] > 0 else 0
        print(f"{model:<20} CTR={ctr:5.2f}%  LAT={lat:.2f}ms")

query = user_df.writeStream \
    .foreachBatch(process_batch) \
    .trigger(processingTime="1 seconds") \
    .start()

print("🚀 Streaming started.")
query.awaitTermination()
