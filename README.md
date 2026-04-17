***

## 🏗️ Data Pipeline Architecture

This repository features a robust, out-of-core data engineering pipeline designed to process real-world e-commerce datasets without memory bottlenecks. Built with Python and DuckDB, the pipeline transforms raw operational data into a bitemporal Master Feature Panel for Machine Learning, prioritizing strict data validation, zero-copy cloud synchronization, and comprehensive observability.

### 🚀 Key Engineering Principles

* **Out-of-Core Processing:** Utilizes DuckDB to execute complex SQL transformations and aggregations strictly on disk, eliminating the `pandas` Out-Of-Memory (OOM) limitations common when scaling up datasets.
* **Zero-Copy Cloud Architecture:** Implements a pass-through loader design, streaming processed artifacts directly to AWS S3 without duplicating multi-gigabyte files on the local disk, drastically reducing I/O operations.
* **Point-in-Time Correctness:** Prevents data leakage using dynamic cutoff dates (`snapshot_date`) to securely separate historical feature engineering from future target horizons (Churn & 180-day LTV).
* **Bitemporal Lineage:** Embeds both event time and system processing time (`ingested_at_utc`) directly into the feature store for robust MLOps auditing, data staleness monitoring, and backfill debugging.

### ⚙️ Pipeline Stages

#### 1. Extractor (`01_extractor`)
* **Cloud Sync:** Uses native `boto3` paginators to securely and recursively sync raw dataset files from the AWS S3 Data Lake.
* **Lazy Schema Inference:** Employs DuckDB (`DESCRIBE SELECT * FROM read_csv_auto()`) to instantly sniff schemas, row counts, and null distributions without loading heavy files into RAM.

#### 2. Validator (`02_validator`)
* **Strict Type Enforcement:** Compares incoming raw data against a version-controlled JSON reference schema mapped to modern database types (e.g., `VARCHAR`, `BIGINT`, `DOUBLE`).
* **Gatekeeper Logic:** Halts the pipeline immediately if upstream data types drift or null thresholds are breached, preventing silent failures from corrupting downstream ML models.

#### 3. Transformer (`03_transformer`)
* **Medallion Architecture:** Constructs local, persistent DuckDB views for Bronze (raw ingestion) and Silver (cleansed/conformed) layers.
* **Gold Layer Generation:** Executes complex, multi-snapshot SQL queries to build the analytical base table, featuring aggregated Recency, Frequency, and Monetary (RFM) metrics, fulfillment delays, and review scores.
* **Direct-to-Parquet Export:** Runs a `COPY ... TO ... (FORMAT PARQUET)` command to write the final Master Panel directly to disk, maintaining a zero-pandas execution path.

#### 4. Loader (`04_loader`)
* **Zero-Copy Pass-Through:** Receives the local file path directly from the Transformer artifact and streams the Parquet file to the AWS S3 Feature Store.
* **Lightweight Telemetry:** Uses `pyarrow` to extract file size and row counts instantly from the Parquet footer metadata, ensuring the final step remains fully out-of-core.

### 📊 Observability & Artifact Tracking

To ensure production-readiness, every stage of the pipeline generates isolated, timestamped `metadata.json` payloads capturing:
* Execution times per component.
* Processed row counts, total columns, and payload sizes.
* Dynamic data profiles (e.g., global churn rates, null distributions).
* Lineage URIs bridging local artifact directories with remote AWS S3 paths.

### 🛠️ Tech Stack
* **Core Compute:** Python 3.x, DuckDB
* **Data Serialization:** PyArrow, Parquet, JSON
* **Cloud Infrastructure:** AWS S3, Boto3
* **Design Patterns:** Object-Oriented Programming (OOP), Dataclasses, Centralized Configuration