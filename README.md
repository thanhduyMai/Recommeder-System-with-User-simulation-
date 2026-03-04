# Distributed Recommender System with User Simulation

## Overview

This project implements a distributed recommender system built on Apache Spark.

It supports:

- ALS training using Spark MLlib
- DeepFM training using BigDL on Spark
- User interaction simulation
- Batch indexing into Elasticsearch
- Streaming processing for real-time events

The system is designed to be executed via spark-submit in a cluster environment.

---

## Technology Stack

- Apache Spark
- BigDL
- Elasticsearch
- PySpark
- Python

---

## Architecture

### Batch Layer
- Data preprocessing
- Model training (ALS / DeepFM)
- Model evaluation
- Indexing recommendations into Elasticsearch

### Streaming Layer
- Real-time event processing
- Incremental updates

### Serving Layer
- Recommendations stored in Elasticsearch
- Optimized for low-latency retrieval

---

## Project Structure



src/  
→ Business logic (preprocessing, models, serving)



SparkSession is initialized inside job scripts under jobs/.

Business logic remains isolated in src/ to ensure clean separation between orchestration and computation.

---

## Execution

All batch and streaming workloads are executed via spark-submit.

Example:

spark-submit [SPARK_OPTIONS] jobs/als_job.py  
spark-submit [SPARK_OPTIONS] jobs/deepfm_job.py  
spark-submit [SPARK_OPTIONS] jobs/stream_job.py  

Cluster configuration (master, deploy mode, executors, memory, cores, packages) is expected to be defined externally depending on the deployment environment.

---

## Resource Management

Spark resource allocation is controlled via spark-submit parameters:

- --master
- --deploy-mode
- --num-executors
- --executor-memory
- --executor-cores
- --packages (for BigDL if required)

This allows flexible deployment across:

- Local mode
- Standalone cluster
- YARN
- Kubernetes

---

## Model Details

ALS:
- Implemented using Spark MLlib
- Distributed matrix factorization

DeepFM:
- Implemented using BigDL
- Distributed deep learning executed across Spark executors
- Suitable for large-scale recommendation tasks

---

## Serving

Model outputs are indexed into Elasticsearch for fast retrieval in downstream applications.

---

## Design Principles

- Clear separation between orchestration (jobs/) and business logic (src/)
- Config-driven architecture (configs/)
- Distributed-first design
- Production-ready Spark job structure
- Extendable to workflow orchestration systems (e.g., Airflow)
