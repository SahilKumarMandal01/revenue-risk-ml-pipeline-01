import os
import sys
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
# Ensures Data Scientists can tweak these without altering OOP pipeline code.
TARGET_DAYS: int = 180
SNAPSHOT_DATES: List[str] = [
    "2017-09-01",
    "2017-11-01",
    "2018-01-01",
    "2018-03-01",
]
COMPUTE_THREADS: int = 4

# ==========================================================
# DATA PIPELINE COMPONENT CONSTANTS
# ==========================================================
DATA_PIPELINE_ROOT_DIR_NAME: str = "data_pipeline"

# Extractor
EXTRACTOR_ROOT_DIR_NAME: str = "01_extractor"
EXTRACTOR_RAW_DATA_DIR_NAME: str = "raw_data"
EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME: str = "raw_data_schema.json"
EXTRACTOR_METADATA_FILE_NAME: str = "metadata.json"

# Validator
VALIDATOR_ROOT_DIR_NAME: str = "02_validator"
VALIDATOR_REPORT_FILE_NAME: str = "report.json"

# Transformer
TRANSFORMER_ROOT_DIR_NAME: str = "03_transformer"
TRANSFORMER_METADATA_FILE_NAME: str = "metadata.json"
TRANSFORMER_CACHE_FILE_NAME: str = "transformer_cache.db"

# Loader
LOADER_ROOT_DIR_NAME: str = "04_loader"
LOADER_METADATA_FILE_NAME: str = "metadata.json"
LOADER_MASTER_PANEL_LOCAL_FILE_NAME: str = "master_panel.parquet"