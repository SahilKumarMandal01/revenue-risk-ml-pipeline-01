import sys, os
from datetime import datetime, timezone

from src import constants
from src.custom_exception import CustomException

class DataPipelineConfig:
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
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        try:
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
        except Exception as e:
            raise CustomException(e, sys)


class DataPipelineValidatorConfig:
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
            self.reference_schema_file_path: str = constants.REFERENCE_SCHEMA_FILE_PATH
            
        except Exception as e:
            raise CustomException(e, sys)


class DataPipelineTransformerConfig:
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
            self.s3_bucket_name = constants.S3_BUCKET_NAME
            self.s3_feature_store_dir = constants.S3_FEATURE_STORE_DIR_NAME
            
            # Constructs: s3://revenue-risk-ml-pipeline-01/feature_store/master_panel.parquet
            self.s3_master_panel_uri: str = (
                f"s3://{self.s3_bucket_name}/{self.s3_feature_store_dir}/master_panel.parquet"
            )

            os.makedirs(self.loader_root_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)