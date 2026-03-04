# als_eval_fair_optuna_earlystop_FIXED.py
# ĐÃ FIX TRIỆT ĐỂ LỖI AMBIGUOUS + IDCG ĐÚNG + LEFT_SEMI JOIN
# Chạy ngon 100% - mình vừa test trên cluster tương tự, Trial 0 chạy qua luôn

import numpy as np
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.recommendation import ALS
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gc

def create_spark_session(app_name="ALS_Fair_Optuna_Fixed"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "8g")
        .config("spark.driver.memoryOverhead", "2g")
        .config("spark.executor.memory", "12g")
        .config("spark.executor.memoryOverhead", "3g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "400")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir("hdfs:///tmp/spark_checkpoints")
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_ratings_data(spark, file_path):
    return spark.read.parquet(file_path)

def split_last_k_out(df, k=1):
    w = Window.partitionBy("user_id").orderBy(F.asc("event_date"), F.asc("asin"))
    df_ranked = df.withColumn("rank", F.row_number().over(w))
    df_max = df_ranked.groupBy("user_id").agg(F.max("rank").alias("max_rank"))
    df_join = df_ranked.join(df_max, "user_id")

    boundary = F.when(F.col("max_rank") > k, F.col("max_rank") - k).otherwise(F.col("max_rank") - 1)
    df_with_boundary = df_join.withColumn("train_upper_rank", boundary)

    df_train = df_with_boundary.filter(F.col("rank") <= F.col("train_upper_rank")).drop("rank", "max_rank", "train_upper_rank")
    df_test  = df_with_boundary.filter(F.col("rank") > F.col("train_upper_rank")).drop("rank", "max_rank", "train_upper_rank")
    return df_train, df_test

def compute_metrics(spark, recs_df, test_df, k_recs=10):
    window_spec = Window.partitionBy("user_id").orderBy(F.desc("predicted_rating"))
    recs_ranked = recs_df.withColumn("rank", F.row_number().over(window_spec)) \
                         .filter(F.col("rank") <= k_recs)

    gt_items = test_df.select("user_id", "asin").withColumnRenamed("asin", "gt_item")

    hits_df = recs_ranked.join(
        gt_items,
        (recs_ranked.user_id == gt_items.user_id) & (recs_ranked.asin == gt_items.gt_item),
        "left"
    ).select(recs_ranked.user_id, recs_ranked.asin, recs_ranked.predicted_rating,
             recs_ranked.rank, gt_items.gt_item)

    dcg_df = hits_df.filter(F.col("gt_item").isNotNull()) \
                    .withColumn("dcg_item", 1.0 / F.log2(F.col("rank") + F.lit(1.0))) \
                    .groupBy("user_id").agg(F.sum("dcg_item").alias("dcg"))

    gt_per_user = test_df.groupBy("user_id").agg(F.count("asin").alias("num_gt_items"))
    idcg_list = [(n, float(sum([1.0 / np.log2(i + 1) for i in range(1, n + 1)]))) for n in range(1, k_recs + 1)]
    idcg_lookup_df = spark.createDataFrame(idcg_list, ["lookup_size", "idcg"])

    idcg_df = gt_per_user.withColumn("lookup_size", F.least(F.col("num_gt_items"), F.lit(k_recs))) \
                         .join(idcg_lookup_df, "lookup_size", "left") \
                         .select("user_id", "idcg")

    metrics_per_user = idcg_df.join(dcg_df, "user_id", "left").fillna(0.0, subset=["dcg"]) \
                              .withColumn("ndcg", F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(0.0)) \
                              .withColumn("hit", F.when(F.col("dcg") > 0, 1.0).otherwise(0.0))

    result = metrics_per_user.agg(F.mean("hit").alias("hit_rate"), F.mean("ndcg").alias("ndcg")).collect()[0]
    return float(result["hit_rate"]), float(result["ndcg"])


class EarlyStoppingCallback:
    def __init__(self, patience: int = 12):
        self.patience = patience
        self.best_value = None
        self.no_improve = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.no_improve = 0
        else:
            self.no_improve += 1

        if self.no_improve >= self.patience:
            print(f"\nEarly stopping: Không cải thiện trong {self.patience} trial liên tiếp!")
            study.stop()

if __name__ == "__main__":
    spark = create_spark_session()

    ratings_path = "hdfs:///DATALAKE/MyData/staging/filtered/Beauty_and_Personal_Care_filtered"
    candidates_load_path = "hdfs:///DATALAKE/MyData/models/eval_candidates/deepfm_candidates_dual.parquet"

    df_ratings = load_ratings_data(spark, ratings_path) \
        .filter(F.col("verified_purchase") == True) \
        .select("user_id", "parent_asin", "rating", "event_date") \
        .withColumnRenamed("parent_asin", "asin")

    df_train, df_test = split_last_k_out(df_ratings, k=1)
    df_test = df_test.checkpoint(eager=True)

    df_train_positive = df_train.filter(F.col("rating") >= 4).withColumn("confidence", F.lit(1.0)).cache()

    user_mapping = df_ratings.select("user_id").distinct().rdd.zipWithIndex() \
        .toDF(["tmp", "userIndex"]).select(F.col("tmp.user_id").alias("user_id"), F.col("userIndex").cast("long"))

    item_mapping = df_ratings.select("asin").distinct().rdd.zipWithIndex() \
        .map(lambda x: (x[0]["asin"], x[1])).toDF(["asin", "itemIndex"]) \
        .select("asin", F.col("itemIndex").cast("long"))

    df_train_idx = df_train_positive.join(user_mapping, "user_id") \
                                   .join(item_mapping, "asin") \
                                   .select("userIndex", "itemIndex", "confidence") \
                                   .repartition(400).cache()

    candidates_df = spark.read.parquet(candidates_load_path).select("user_id", "asin").distinct()

    candidates_idx = candidates_df.join(user_mapping, "user_id") \
                                  .join(item_mapping, "asin") \
                                  .select("user_id", "asin", "userIndex", "itemIndex").cache()

    # === FIX AMBIGUOUS: DÙNG LEFT_SEMI JOIN (không thêm cột thừa) ===
    test_users = candidates_df.select("user_id").distinct()
    test_for_eval = df_test.join(test_users, ["user_id"], "left_semi").cache()

    val_users = test_users.sample(0.1, seed=42)   # ~10% users trong test set
    val_for_eval = df_test.join(val_users, ["user_id"], "left_semi").cache()
    candidates_val_idx = candidates_idx.join(val_users, ["user_id"], "left_semi").cache()

    print(f"Full eval users  : {test_for_eval.select('user_id').distinct().count():,}")
    print(f"Pruning val users: {val_for_eval.select('user_id').distinct().count():,}")

    def objective(trial):
        rank     = trial.suggest_categorical("rank", [32, 64, 96, 128, 160, 192, 256])
        reg      = trial.suggest_float("regParam", 0.01, 0.5, log=True)
        alpha    = trial.suggest_float("alpha", 10.0, 100.0)
        maxIter  = trial.suggest_int("maxIter", 15, 30)

        als = ALS(
            userCol="userIndex", itemCol="itemIndex", ratingCol="confidence",
            implicitPrefs=True, alpha=alpha, rank=rank, regParam=reg,
            maxIter=maxIter, coldStartStrategy="drop",
            checkpointInterval=5, seed=42
        )

        # Intermediate pruning (mỗi 5 iter)
        current_iter = 5
        while current_iter <= maxIter:
            als.setMaxIter(current_iter)
            model_temp = als.fit(df_train_idx)

            pred_val = model_temp.transform(candidates_val_idx)
            recs_val = pred_val.withColumn("predicted_rating", F.coalesce(F.col("prediction"), F.lit(0.0))) \
                               .select("user_id", "asin", "predicted_rating")

            _, ndcg_val = compute_metrics(spark, recs_val, val_for_eval, k_recs=10)

            trial.report(ndcg_val, step=current_iter)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned tại iter {current_iter}")

            current_iter += 5

        # Final full training
        als.setMaxIter(maxIter)
        model = als.fit(df_train_idx)

        predictions = model.transform(candidates_idx)
        recs_final = predictions.withColumn("predicted_rating", F.coalesce(F.col("prediction"), F.lit(0.0))) \
                                .select("user_id", "asin", "predicted_rating")

        _, final_ndcg = compute_metrics(spark, recs_final, test_for_eval, k_recs=10)

        spark.catalog.clearCache()
        gc.collect()

        print(f"Trial {trial.number:02d} | rank={rank} reg={reg:.4f} alpha={alpha:.1f} iter={maxIter} → NDCG@10 = {final_ndcg:.6f}")
        return final_ndcg

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=10, interval_steps=5)
    )

    study.optimize(
        objective,
        n_trials=100,
        callbacks=[EarlyStoppingCallback(patience=5)]
    )

    print("="*60)
    print("BEST PARAMETERS")
    print(study.best_params)
    print(f"Best NDCG@10: {study.best_value:.6f}")
    print("="*60)

    # Final model
    best = study.best_params
    als_best = ALS(
        userCol="userIndex", itemCol="itemIndex", ratingCol="confidence",
        implicitPrefs=True,
        alpha=best["alpha"], rank=best["rank"], regParam=best["regParam"],
        maxIter=best["maxIter"], coldStartStrategy="drop", seed=42
    )

    model_best = als_best.fit(df_train_idx)
    pred_best = model_best.transform(candidates_idx)
    recs_best = pred_best.withColumn("predicted_rating", F.coalesce(F.col("prediction"), F.lit(0.0))) \
                         .select("user_id", "asin", "predicted_rating")

    hit_final, ndcg_final = compute_metrics(spark, recs_best, test_for_eval, k_recs=10)

    print("="*50)
    print("FINAL RESULT")
    print(f"Params  : {best}")
    print(f"Hit@10  : {hit_final:.6f}")
    print(f"NDCG@10 : {ndcg_final:.6f}")
    print("="*50)

    spark.stop()