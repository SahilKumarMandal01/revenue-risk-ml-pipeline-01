import os
import sys
from datetime import datetime, timezone

from src import constants
from src.custom_exception import CustomException
from src.custom_logging import logging


# ==========================================================
# DATA PIPELINE CONFIGURATIONS
# ==========================================================
class DataPipelineConfig:
    """
    Base configuration for the Data Pipeline.
    Responsible for creating the unique run ID and root artifact directory.
    """

    def __init__(self) -> None:
        try:
            self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.DATA_PIPELINE_ROOT_DIR_NAME,
                self.run_id,
            )
            os.makedirs(self.root_dir, exist_ok=True)
            logging.info("DataPipelineConfig initialized. Run ID: %s", self.run_id)

        except Exception as e:
            logging.exception("Error initializing DataPipelineConfig.")
            raise CustomException(e, sys) from e


class DataPipelineExtractorConfig:
    """
    Configuration for the Extractor component.
    Defines local directory paths for extracted data and remote S3 paths for ingestion.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        try:
            self.extractor_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.EXTRACTOR_ROOT_DIR_NAME,
            )
            self.raw_data_dir_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_RAW_DATA_DIR_NAME,
            )
            self.raw_data_schema_file_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_METADATA_FILE_NAME,
            )

            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_raw_data_dir: str = constants.S3_RAW_DATA_DIR_NAME
            self.s3_raw_data_uri: str = f"s3://{self.s3_bucket_name}/{self.s3_raw_data_dir}"

            logging.info("DataPipelineExtractorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing DataPipelineExtractorConfig.")
            raise CustomException(e, sys) from e


class DataPipelineValidatorConfig:
    """
    Configuration for the Validator component.
    Defines paths for validation reports and the reference schema.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        try:
            self.validator_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.VALIDATOR_ROOT_DIR_NAME,
            )
            self.report_file_path: str = os.path.join(
                self.validator_root_dir,
                constants.VALIDATOR_REPORT_FILE_NAME,
            )
            self.is_valid: bool = False
            self.reference_schema_file_path: str = str(constants.REFERENCE_SCHEMA_FILE_PATH)

            logging.info("DataPipelineValidatorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing DataPipelineValidatorConfig.")
            raise CustomException(e, sys) from e


class DataPipelineTransformerConfig:
    """
    Configuration for the Transformer component.
    Defines paths, core business logic parameters (target days, snapshots), and thread limits.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        try:
            self.transformer_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.TRANSFORMER_ROOT_DIR_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.transformer_root_dir,
                constants.TRANSFORMER_METADATA_FILE_NAME,
            )
            self.duckdb_data_file_path: str = os.path.join(
                self.transformer_root_dir,
                constants.TRANSFORMER_CACHE_FILE_NAME,
            )
            self.target_days: int = constants.TARGET_DAYS
            self.snapshots: list = constants.SNAPSHOT_DATES
            self.threads: int = constants.COMPUTE_THREADS

            os.makedirs(self.transformer_root_dir, exist_ok=True)
            logging.info("DataPipelineTransformerConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing DataPipelineTransformerConfig.")
            raise CustomException(e, sys) from e


class DataPipelineLoaderConfig:
    """
    Configuration for the Loader component.
    Defines local artifact paths for metadata and remote S3 URIs for the feature store.
    """

    def __init__(self, data_pipeline_config: DataPipelineConfig) -> None:
        try:
            self.loader_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.LOADER_ROOT_DIR_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.loader_root_dir,
                constants.LOADER_METADATA_FILE_NAME,
            )
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_feature_store_dir: str = constants.S3_FEATURE_STORE_DIR_NAME
            self.s3_master_panel_uri: str = (
                f"s3://{self.s3_bucket_name}/{self.s3_feature_store_dir}/"
                f"{constants.LOADER_MASTER_PANEL_LOCAL_FILE_NAME}"
            )

            os.makedirs(self.loader_root_dir, exist_ok=True)
            logging.info("DataPipelineLoaderConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing DataPipelineLoaderConfig.")
            raise CustomException(e, sys) from e


# ==========================================================
# TRAINING PIPELINE CONFIGURATIONS
# ==========================================================
class TrainingPipelineConfig:
    """
    Base configuration for the Training Pipeline.
    Responsible for creating the unique run ID and root artifact directory.
    """

    def __init__(self) -> None:
        try:
            self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.TRAINING_PIPELINE_ROOT_DIR_NAME,
                self.run_id,
            )
            os.makedirs(self.root_dir, exist_ok=True)
            logging.info("TrainingPipelineConfig initialized. Run ID: %s", self.run_id)

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineConfig.")
            raise CustomException(e, sys) from e


class TrainingPipelineDataIngestionConfig:
    """
    Configuration for the Training Pipeline Data Ingestion component.
    Defines S3 source URIs, local artifact paths for the splits, and OOT parameters.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.data_ingestion_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.DATA_INGESTION_ROOT_DIR_NAME,
            )
            
            # Local artifact paths
            self.train_data_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_TRAIN_FILE_NAME
            )
            self.val_data_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_VAL_FILE_NAME
            )
            self.test_data_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_TEST_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.data_ingestion_root_dir, constants.DATA_INGESTION_METADATA_FILE_NAME
            )

            # S3 remote source for the bitemporal master panel
            self.s3_master_panel_uri: str = (
                f"s3://{constants.S3_BUCKET_NAME}/{constants.S3_FEATURE_STORE_DIR_NAME}/"
                f"{constants.LOADER_MASTER_PANEL_LOCAL_FILE_NAME}"
            )

            # Splitting configuration based on bitemporal snapshots
            self.train_snapshots: list = constants.TRAIN_SNAPSHOTS
            self.val_snapshot: str = constants.VAL_SNAPSHOT
            self.test_snapshot: str = constants.TEST_SNAPSHOT

            os.makedirs(self.data_ingestion_root_dir, exist_ok=True)
            logging.info("TrainingPipelineDataIngestionConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineDataIngestionConfig.")
            raise CustomException(e, sys) from e


class TrainingPipelineDataTransformationConfig:
    """
    Configuration for the Training Pipeline Data Transformation component.
    Defines output paths for the preprocessor artifact, transformed datasets, 
    and schema validation rules.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.data_transformation_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.DATA_TRANSFORMATION_ROOT_DIR_NAME,
            )
            
            # Local artifact paths for the serialized preprocessor and metadata
            self.preprocessor_file_path: str = os.path.join(
                self.data_transformation_root_dir,
                constants.DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.data_transformation_root_dir,
                constants.DATA_TRANSFORMATION_METADATA_FILE_NAME,
            )

            # Transformed Feature Matrix (X) and Target Vector (y) paths
            self.x_train_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_X_TRAIN_FILE_NAME
            )
            self.y_train_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_Y_TRAIN_FILE_NAME
            )
            self.x_val_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_X_VAL_FILE_NAME
            )
            self.y_val_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_Y_VAL_FILE_NAME
            )
            self.x_test_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_X_TEST_FILE_NAME
            )
            self.y_test_file_path: str = os.path.join(
                self.data_transformation_root_dir, constants.DATA_TRANSFORMATION_Y_TEST_FILE_NAME
            )

            # Feature schema and processing constants
            self.target_column: str = constants.TARGET_COLUMN
            self.columns_to_drop: list = constants.SYSTEM_COLUMNS_TO_DROP

            os.makedirs(self.data_transformation_root_dir, exist_ok=True)
            logging.info("TrainingPipelineDataTransformationConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineDataTransformationConfig.")
            raise CustomException(e, sys) from e