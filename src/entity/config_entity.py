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
            self.schema_file_path: str = os.path.join(
                self.data_transformation_root_dir,
                constants.DATA_TRANSFORMATION_SCHEMA_FILE_NAME
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


class TrainingPipelineModelTrainerConfig:
    """
    Configuration for the Training Pipeline Model Trainer component.
    Defines output paths for the calibrated model, SHAP summaries, and MLflow metadata.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.model_trainer_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.MODEL_TRAINER_ROOT_DIR_NAME,
            )
            
            self.model_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                constants.MODEL_TRAINER_MODEL_FILE_NAME,
            )
            self.shap_summary_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                constants.MODEL_TRAINER_SHAP_SUMMARY_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.model_trainer_root_dir,
                constants.MODEL_TRAINER_METADATA_FILE_NAME,
            )

            self.mlflow_experiment_name: str = constants.MODEL_TRAINER_MLFLOW_EXPERIMENT_NAME

            os.makedirs(self.model_trainer_root_dir, exist_ok=True)
            logging.info("TrainingPipelineModelTrainerConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineModelTrainerConfig.")
            raise CustomException(e, sys) from e


class TrainingPipelineModelEvaluationConfig:
    """
    Configuration for the Training Pipeline Model Evaluation component.
    Defines output paths for the evaluation report and gating thresholds.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.model_evaluation_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.MODEL_EVALUATION_ROOT_DIR_NAME,
            )
            
            self.report_file_path: str = os.path.join(
                self.model_evaluation_root_dir,
                constants.MODEL_EVALUATION_REPORT_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.model_evaluation_root_dir,
                constants.MODEL_EVALUATION_METADATA_FILE_NAME,
            )

            # Business and Hysteresis Thresholds
            self.min_eroi_threshold: float = constants.MODEL_EVALUATION_MIN_EROI_THRESHOLD
            self.eroi_hysteresis_margin: float = constants.MODEL_EVALUATION_EROI_HYSTERESIS_MARGIN

            os.makedirs(self.model_evaluation_root_dir, exist_ok=True)
            logging.info("TrainingPipelineModelEvaluationConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineModelEvaluationConfig.")
            raise CustomException(e, sys) from e


class TrainingPipelineModelRegistryConfig:
    """
    Configuration for the Training Pipeline Model Registry component.
    Defines S3 URIs for immutable artifact storage and the mutable production pointer.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig) -> None:
        try:
            self.model_registry_root_dir: str = os.path.join(
                training_pipeline_config.root_dir,
                constants.MODEL_REGISTRY_ROOT_DIR_NAME,
            )
            
            self.metadata_file_path: str = os.path.join(
                self.model_registry_root_dir,
                constants.MODEL_REGISTRY_METADATA_FILE_NAME,
            )

            # S3 Registry Configurations
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_registry_base_uri: str = f"s3://{self.s3_bucket_name}/{constants.S3_MODEL_REGISTRY_DIR_NAME}"
            
            self.s3_models_dir_uri: str = f"{self.s3_registry_base_uri}/{constants.S3_MODEL_REGISTRY_MODELS_DIR}"
            self.s3_state_dir_uri: str = f"{self.s3_registry_base_uri}/{constants.S3_MODEL_REGISTRY_STATE_DIR}"
            self.s3_pointer_file_uri: str = f"{self.s3_state_dir_uri}/{constants.S3_MODEL_REGISTRY_POINTER_FILE_NAME}"

            os.makedirs(self.model_registry_root_dir, exist_ok=True)
            logging.info("TrainingPipelineModelRegistryConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing TrainingPipelineModelRegistryConfig.")
            raise CustomException(e, sys) from e


# ==========================================================
# INFERENCE PIPELINE CONFIGURATIONS
# ==========================================================
class InferencePipelineConfig:
    """
    Base configuration for the Inference Pipeline.
    Responsible for creating the unique run ID and root artifact directory.
    """

    def __init__(self) -> None:
        try:
            self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.INFERENCE_PIPELINE_ROOT_DIR_NAME,
                self.run_id,
            )
            os.makedirs(self.root_dir, exist_ok=True)
            logging.info("InferencePipelineConfig initialized. Run ID: %s", self.run_id)

        except Exception as e:
            logging.exception("Error initializing InferencePipelineConfig.")
            raise CustomException(e, sys) from e


class InferenceModelLoaderConfig:
    """
    Configuration for the Inference Pipeline Current Production Model Loader component.
    Defines S3 URIs for retrieving the active production pointer and local paths 
    to safely stash the downloaded immutable model and schema artifacts.
    """

    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            # Root directory for this specific component
            self.model_loader_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_MODEL_LOADER_ROOT_DIR_NAME,
            )

            # Local scratchpad paths for the downloaded assets
            self.model_file_path: str = os.path.join(
                self.model_loader_root_dir,
                constants.INFERENCE_MODEL_LOADER_MODEL_FILE_NAME,
            )
            self.schema_file_path: str = os.path.join(
                self.model_loader_root_dir,
                constants.INFERENCE_MODEL_LOADER_SCHEMA_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.model_loader_root_dir,
                constants.INFERENCE_MODEL_LOADER_METADATA_FILE_NAME,
            )

            # S3 Registry Configurations
            # Aligns precisely with the Phase 2 (Model Registry) atomic state pointer
            self.s3_bucket_name: str = constants.S3_BUCKET_NAME
            self.s3_registry_base_uri: str = f"s3://{self.s3_bucket_name}/{constants.S3_MODEL_REGISTRY_DIR_NAME}"
            self.s3_pointer_uri: str = f"{self.s3_registry_base_uri}/model_state.json"

            # Pre-create the directory structure for safe local I/O
            os.makedirs(self.model_loader_root_dir, exist_ok=True)
            logging.info("InferenceModelLoaderConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing InferenceModelLoaderConfig.")
            raise CustomException(e, sys) from e


class InferenceInputFeatureMatrixBuilderConfig:
    """
    Configuration for the Input Feature Matrix Builder component.
    Defines S3 Data Lake URIs, DuckDB parameters, and local persistence paths
    for the generated inference feature matrix.
    """
    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            # Component Root Directory
            self.feature_matrix_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_FEATURE_MATRIX_BUILDER_ROOT_DIR_NAME,
            )
            
            # Local Artifact Paths
            self.feature_matrix_file_path: str = os.path.join(
                self.feature_matrix_root_dir, constants.INFERENCE_FEATURE_MATRIX_FILE_NAME
            )
            self.schema_file_path: str = os.path.join(
                self.feature_matrix_root_dir, constants.INFERENCE_FEATURE_MATRIX_SCHEMA_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.feature_matrix_root_dir, constants.INFERENCE_FEATURE_MATRIX_METADATA_FILE_NAME
            )

            # Upstream S3 Data Lake Location
            self.s3_data_lake_uri: str = f"s3://{constants.S3_CUSTOMER_DATABASE_NAME}/{constants.S3_DATA_LAKE_BRONZE_DIR_NAME}"

            # Snapshot Logic (Scoring Population Temporal Bound)
            # The pipeline runs on Day T, scoring data up to T-1 (Yesterday).
            # We set snapshot_date to exactly 00:00:00 of the execution day.
            # The SharedFeatureGenerator uses strictly "< snapshot_date".
            self.snapshot_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            os.makedirs(self.feature_matrix_root_dir, exist_ok=True)
            logging.info(
                "InferenceInputFeatureMatrixBuilderConfig initialized. Snapshot date anchor: %s", 
                self.snapshot_date
            )

        except Exception as e:
            logging.exception("Error initializing InferenceInputFeatureMatrixBuilderConfig.")
            raise CustomException(e, sys) from e


class InferenceValidatorConfig:
    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            # Root directory for this specific component
            self.validator_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_VALIDATOR_ROOT_DIR_NAME,
            )

            self.report_file_path: str = os.path.join(
                self.validator_root_dir,
                constants.INFERENCE_VALIDATOR_REPORT_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.validator_root_dir,
                constants.INFERENCE_VALIDATOR_METADATA_FILE_NAME
            )

            # Pre-create the directory structure for safe local I/O
            os.makedirs(self.validator_root_dir, exist_ok=True)
            logging.info("InferenceValidatorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing InferenceValidatorConfig.")
            raise CustomException(e, sys) from e


class InferenceReportGeneratorConfig:
    """
    Configuration for the Inference Publisher (Report Generator) component.
    Defines the probability threshold for churn classification and output 
    paths for the CSV, Parquet, and JSON artifacts.
    """
    def __init__(self, inference_pipeline_config: InferencePipelineConfig) -> None:
        try:
            self.run_id: str = inference_pipeline_config.run_id
            self.report_generator_root_dir: str = os.path.join(
                inference_pipeline_config.root_dir,
                constants.INFERENCE_REPORT_GENERATOR_ROOT_DIR_NAME,
            )

            self.csv_report_path: str = os.path.join(
                self.report_generator_root_dir,
                constants.INFERENCE_REPORT_GENERATOR_CSV_FILE_NAME,
            )
            self.telemetry_log_path: str = os.path.join(
                self.report_generator_root_dir,
                constants.INFERENCE_REPORT_GENERATOR_TELEMETRY_FILE_NAME,
            )
            self.metadata_file_path: str = os.path.join(
                self.report_generator_root_dir,
                constants.INFERENCE_REPORT_GENERATOR_METADATA_FILE_NAME,
            )
            
            self.probability_threshold: float = constants.INFERENCE_REPORT_GENERATOR_PROBABILITY_THRESHOLD

            os.makedirs(self.report_generator_root_dir, exist_ok=True)
            logging.info("InferenceReportGeneratorConfig initialized.")

        except Exception as e:
            logging.exception("Error initializing InferenceReportGeneratorConfig.")
            raise CustomException(e, sys) from e