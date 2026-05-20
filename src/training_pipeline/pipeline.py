import sys

from src.entity.config_entity import (
    TrainingPipelineConfig,
    TrainingPipelineDataIngestionConfig,
    TrainingPipelineDataTransformationConfig,
    TrainingPipelineModelTrainerConfig,
    TrainingPipelineModelEvaluationConfig,
    TrainingPipelineModelRegistryConfig,
)
from src.entity.artifact_entity import (
    TrainingPipelineDataIngestionArtifact,
    TrainingPipelineDataTransformationArtifact,
    TrainingPipelineModelTrainerArtifact,
    TrainingPipelineModelEvaluationArtifact,
    TrainingPipelineModelRegistryArtifact,
)
from src.training_pipeline.components.data_ingestion import DataIngestion
from src.training_pipeline.components.data_transformation import DataTransformation
from src.training_pipeline.components.model_trainer import ModelTrainer
from src.training_pipeline.components.model_evaluation import ModelEvaluation
from src.training_pipeline.components.model_registry import ModelRegistry

from src.cloud.s3_operations import S3Sync
from src import constants
from src.custom_exception import CustomException
from src.custom_logging import logging


class TrainingPipeline:
    """
    Orchestrates the complete Continuous Training (CT) pipeline.
    
    Responsibilities:
    - Sequentially trigger Data Ingestion, Data Transformation, Model Training, 
      Model Evaluation, and Model Registry components.
    - Pass necessary immutable artifacts safely between component boundaries.
    - Implement a strict gatekeeping mechanism utilizing the evaluation approval status.
    - Sync generated execution artifacts safely to AWS S3 for traceability.
    """

    def __init__(self) -> None:
        try:
            logging.info("Initializing Training Pipeline orchestration.")
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            logging.exception("Failed to initialize Training Pipeline.")
            raise CustomException(e, sys) from e

    def _run_data_ingestion(self) -> TrainingPipelineDataIngestionArtifact:
        try:
            logging.info(">>> Starting Phase 1: Data Ingestion")
            config = TrainingPipelineDataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(config=config)
            artifact = data_ingestion.run()
            logging.info("<<< Phase 1: Data Ingestion completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Data Ingestion phase failed.")
            raise CustomException(e, sys) from e

    def _run_data_transformation(
        self, ingestion_artifact: TrainingPipelineDataIngestionArtifact
    ) -> TrainingPipelineDataTransformationArtifact:
        try:
            logging.info(">>> Starting Phase 2: Data Transformation")
            config = TrainingPipelineDataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(
                config=config, ingestion_artifact=ingestion_artifact
            )
            artifact = data_transformation.run()
            logging.info("<<< Phase 2: Data Transformation completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Data Transformation phase failed.")
            raise CustomException(e, sys) from e

    def _run_model_trainer(
        self, transformation_artifact: TrainingPipelineDataTransformationArtifact
    ) -> TrainingPipelineModelTrainerArtifact:
        try:
            logging.info(">>> Starting Phase 3: Model Training")
            config = TrainingPipelineModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(
                config=config, transformation_artifact=transformation_artifact
            )
            artifact = model_trainer.run()
            logging.info("<<< Phase 3: Model Training completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Model Training phase failed.")
            raise CustomException(e, sys) from e

    def _run_model_evaluation(
        self,
        trainer_artifact: TrainingPipelineModelTrainerArtifact,
        ingestion_artifact: TrainingPipelineDataIngestionArtifact,
    ) -> TrainingPipelineModelEvaluationArtifact:
        try:
            logging.info(">>> Starting Phase 4: Model Evaluation")
            config = TrainingPipelineModelEvaluationConfig(self.training_pipeline_config)
            model_evaluation = ModelEvaluation(
                config=config,
                trainer_artifact=trainer_artifact,
                ingestion_artifact=ingestion_artifact,
            )
            artifact = model_evaluation.run()
            logging.info("<<< Phase 4: Model Evaluation completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Model Evaluation phase failed.")
            raise CustomException(e, sys) from e

    def _run_model_registry(
        self,
        trainer_artifact: TrainingPipelineModelTrainerArtifact,
        evaluation_artifact: TrainingPipelineModelEvaluationArtifact,
        transformation_artifact: TrainingPipelineDataTransformationArtifact,
    ) -> TrainingPipelineModelRegistryArtifact:
        try:
            logging.info(">>> Starting Phase 5: Model Registry")
            config = TrainingPipelineModelRegistryConfig(self.training_pipeline_config)
            model_registry = ModelRegistry(
                config=config,
                transformation_artifact=transformation_artifact,
                trainer_artifact=trainer_artifact,
                evaluation_artifact=evaluation_artifact,
            )
            artifact = model_registry.run()
            logging.info("<<< Phase 5: Model Registry completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Model Registry phase failed.")
            raise CustomException(e, sys) from e

    def _sync_artifacts(self) -> None:
        try:
            logging.info(">>> Starting Phase 6: Artifact Sync (S3)")

            S3Sync().sync_folder_to_s3(
                folder=constants.ARTIFACT_DIR_NAME,
                aws_bucket_url=(
                    f"s3://{constants.S3_BUCKET_NAME}/"
                    f"{constants.ARTIFACT_DIR_NAME}/"
                    f"{constants.TRAINING_PIPELINE_ROOT_DIR_NAME}"
                ),
            )

            logging.info("<<< Phase 6: Artifact Sync completed successfully.\n")

        except Exception as e:
            logging.exception("Artifact Sync phase failed.")
            raise CustomException(e, sys) from e

    def run(self) -> None:
        """
        Executes the main entry point logic for the Continuous Training pipeline.
        """
        try:
            logging.info("===================================================")
            logging.info("STARTING CONTINUOUS TRAINING (CT) PIPELINE")
            logging.info("===================================================")

            ingestion_artifact = self._run_data_ingestion()

            transformation_artifact = self._run_data_transformation(ingestion_artifact)

            trainer_artifact = self._run_model_trainer(transformation_artifact)

            evaluation_artifact = self._run_model_evaluation(
                trainer_artifact, ingestion_artifact
            )

            # Strict Deployment Gatekeeper
            if getattr(evaluation_artifact, "approval_status", False):
                logging.info("Gate Check Passed: Proceeding to Model Registry & Deployment.")
                self._run_model_registry(
                    trainer_artifact=trainer_artifact, 
                    evaluation_artifact=evaluation_artifact,
                    transformation_artifact=transformation_artifact
                )
            else:
                logging.warning(
                    "Gate Check Failed: Challenger model rejected. "
                    "Skipping deployment to Model Registry."
                )

            # Persist local execution traces regardless of approval status
            self._sync_artifacts()

            logging.info("===================================================")
            logging.info("CONTINUOUS TRAINING PIPELINE EXECUTION COMPLETED")
            logging.info("===================================================")

        except Exception as e:
            logging.exception("Critical Failure in Continuous Training Pipeline execution.")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run()
    except Exception:
        logging.critical(
            "Pipeline execution terminated due to an error.", exc_info=True
        )
        sys.exit(1)