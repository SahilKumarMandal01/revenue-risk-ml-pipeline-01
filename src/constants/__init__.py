import os
from pathlib import Path
from typing import List

# ==========================================================
# GLOBAL SYSTEM CONSTANTS
# ==========================================================
ARTIFACT_DIR_NAME: str = "artifacts"
REFERENCE_SCHEMA_FILE_PATH: Path = Path("data_schema/v1/schema.json")

# ==========================================================
# CLOUD & INFRASTRUCTURE CONSTANTS (AWS S3)
# ==========================================================
S3_BUCKET_NAME: str = "revenue-risk-ml-pipeline-01"
S3_RAW_DATA_DIR_NAME: str = "raw_data"
S3_FEATURE_STORE_DIR_NAME: str = "feature_store"
S3_ARTIFACT_DIR_NAME: str = "artifacts"
S3_LOGS_DIR_NAME: str = "logs"

# ==========================================================
# BUSINESS LOGIC & HYPERPARAMETERS
# ==========================================================
# Centralized configuration (replaces hardcoded parameters).
TARGET_DAYS: int = 180
COMPUTE_THREADS: int = 4

# Temporal Snapshot Definitions (Bitemporal Feature Engineering)
SNAPSHOT_DATES: List[str] = [
    "2017-09-01",
    "2017-11-01",
    "2018-01-01",
    "2018-03-01",
]

# Out-of-Time (OOT) Splitting Definitions
TRAIN_SNAPSHOTS: List[str] = ["2017-09-01", "2017-11-01"]
VAL_SNAPSHOT: str = "2018-01-01"
TEST_SNAPSHOT: str = "2018-03-01"

# Target Variable and Metadata Columns (Used to isolate X and y)
TARGET_COLUMN: str = "target_is_churn"
SYSTEM_COLUMNS_TO_DROP: List[str] = [
    "customer_unique_id",
    "snapshot_date",
    "ingested_at_utc",
    "target_180d_ltv"
]

# ==========================================================
# DATA PIPELINE COMPONENT CONSTANTS
# ==========================================================
DATA_PIPELINE_ROOT_DIR_NAME: str = "data_pipeline"

# 01 - Extractor
EXTRACTOR_ROOT_DIR_NAME: str = "01_extractor"
EXTRACTOR_RAW_DATA_DIR_NAME: str = "raw_data"
EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME: str = "raw_data_schema.json"
EXTRACTOR_METADATA_FILE_NAME: str = "metadata.json"

# 02 - Validator
VALIDATOR_ROOT_DIR_NAME: str = "02_validator"
VALIDATOR_REPORT_FILE_NAME: str = "report.json"

# 03 - Transformer
TRANSFORMER_ROOT_DIR_NAME: str = "03_transformer"
TRANSFORMER_METADATA_FILE_NAME: str = "metadata.json"
TRANSFORMER_CACHE_FILE_NAME: str = "transformer_cache.db"

# 04 - Loader
LOADER_ROOT_DIR_NAME: str = "04_loader"
LOADER_METADATA_FILE_NAME: str = "metadata.json"
LOADER_MASTER_PANEL_LOCAL_FILE_NAME: str = "master_panel.parquet"

# ==========================================================
# TRAINING PIPELINE COMPONENT CONSTANTS
# ==========================================================
TRAINING_PIPELINE_ROOT_DIR_NAME: str = "training_pipeline"

# 01 - Data Ingestion
DATA_INGESTION_ROOT_DIR_NAME: str = "01_data_ingestion"
DATA_INGESTION_TRAIN_FILE_NAME: str = "train.parquet"
DATA_INGESTION_VAL_FILE_NAME: str = "val.parquet"
DATA_INGESTION_TEST_FILE_NAME: str = "test.parquet"
DATA_INGESTION_METADATA_FILE_NAME: str = "metadata.json"

# 02 - Data Transformation
DATA_TRANSFORMATION_ROOT_DIR_NAME: str = "02_data_transformation"
DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
DATA_TRANSFORMATION_METADATA_FILE_NAME: str = "metadata.json"

# Feature Matrix (X) and Target Vector (y) transformed artifacts
DATA_TRANSFORMATION_X_TRAIN_FILE_NAME: str = "x_train.parquet"
DATA_TRANSFORMATION_Y_TRAIN_FILE_NAME: str = "y_train.parquet"
DATA_TRANSFORMATION_X_VAL_FILE_NAME: str = "x_val.parquet"
DATA_TRANSFORMATION_Y_VAL_FILE_NAME: str = "y_val.parquet"
DATA_TRANSFORMATION_X_TEST_FILE_NAME: str = "x_test.parquet"
DATA_TRANSFORMATION_Y_TEST_FILE_NAME: str = "y_test.parquet"