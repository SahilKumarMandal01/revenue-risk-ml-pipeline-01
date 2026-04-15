import os
from pathlib import Path

# ==========================================================
# GLOBAL CONSTANTS
# ==========================================================
ARTIFACT_DIR_NAME: str = Path("artifacts")
REFERENCE_SCHEMA_FILE_PATH: str = Path("data_schema/v1/schema.json")

S3_BUCKET_NAME: str = "revenue-risk-ml-pipeline-01"
S3_RAW_DATA_DIR_NAME: str = "raw_data" 
S3_FEATURE_STORE_DIR_NAME: str = "feature_store"
S3_ARTIFACT_DIR_NAME: str = "artifacts"
S3_LOGS_DIR_NAME: str = "logs"

# ==========================================================
# DATA PIPELINE CONSTANTS
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

# Loader
LOADER_ROOT_DIR_NAME: str = "04_loader"
LOADER_METADATA_FILE_NAME: str = "metadata.json"
LOADER_MASTER_PANEL_LOCAL_FILE_NAME: str = "master_panel.parquet"