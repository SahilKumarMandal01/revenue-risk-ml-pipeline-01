import os
import sys
from datetime import datetime, timezone

from src import constants
from src.custom_exception import CustomException


class DataPipelineConfig:
    """
    Base configuration for the Data Pipeline.
    Responsible for creating the unique run ID and root artifact directory.
    """
    def __init__(self):
        try:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.DATA_PIPELINE_ROOT_DIR_NAME,
                self.run_id
            )
            os.makedirs(self.root_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)


class DataPipelineExtractorConfig:
    """
    Configuration for the Extractor component.
    Defines local directory paths for extracted data and remote S3 paths for ingestion.
    """
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        try:
            # Local directory structures
            self.extractor_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.EXTRACTOR_ROOT_DIR_NAME
            )
            self.raw_data_dir_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_RAW_DATA_DIR_NAME
            )
            self.raw_data_schema_file_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_METADATA_FILE_NAME
            )
            
            # S3 Configurations for extracting raw data
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_raw_data_dir: str = constants.S3_RAW_DATA_DIR_NAME
            self.s3_raw_data_uri: str = f"s3://{self.s3_bucket_name}/{self.s3_raw_data_dir}"

        except Exception as e:
            raise CustomException(e, sys)


class DataPipelineValidatorConfig:
    """
    Configuration for the Validator component.
    Defines paths for validation reports and the reference schema.
    """
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        try:
            self.validator_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.VALIDATOR_ROOT_DIR_NAME
            )
            self.report_file_path: str = os.path.join(
                self.validator_root_dir,
                constants.VALIDATOR_REPORT_FILE_NAME
            )
            self.is_valid = None
            self.reference_schema_file_path: str = str(constants.REFERENCE_SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise CustomException(e, sys)


class DataPipelineTransformerConfig:
    """
    Configuration for the Transformer component.
    Defines paths, core business logic parameters (target days, snapshots), and thread limits.
    """
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        try:
            self.transformer_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.TRANSFORMER_ROOT_DIR_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.transformer_root_dir,
                constants.TRANSFORMER_METADATA_FILE_NAME
            )

            # DuckDB Persistent Storage File (Prevents OOM on large datasets)
            self.duckdb_data_file_path: str = os.path.join(
                self.transformer_root_dir,
                "transformer_cache.db"
            )
            
            # Core Business Logic Configuration
            self.target_days: int = 180
            self.snapshots: list = [
                "2017-09-01",
                "2017-11-01",
                "2018-01-01",
                "2018-03-01",
            ]
            self.threads: int = 4

            os.makedirs(self.transformer_root_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)


class DataPipelineLoaderConfig:
    """
    Configuration for the Loader component.
    Defines local artifact paths and remote S3 URIs for the engineered feature store.
    """
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        try:
            self.loader_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.LOADER_ROOT_DIR_NAME
            )
            
            # Local Paths
            self.master_panel_local_file_path: str = os.path.join(
                self.loader_root_dir,
                constants.LOADER_MASTER_PANEL_LOCAL_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.loader_root_dir,
                constants.LOADER_METADATA_FILE_NAME
            )

            # S3 Paths
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_feature_store_dir: str = constants.S3_FEATURE_STORE_DIR_NAME
            
            # Constructs: s3://revenue-risk-ml-pipeline-01/feature_store/master_panel.parquet
            self.s3_master_panel_uri: str = (
                f"s3://{self.s3_bucket_name}/{self.s3_feature_store_dir}/master_panel.parquet"
            )

            os.makedirs(self.loader_root_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)