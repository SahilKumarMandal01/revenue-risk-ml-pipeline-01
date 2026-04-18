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

from src.custom_exception import CustomException
from src.custom_logging import logging


class TrainingPipeline:
    """
    Orchestrates the complete Continuous Training (CT) pipeline, including:
    - Memory-safe Data Ingestion (Out-Of-Time Splitting)
    - Stateful Data Transformation (Schema Enforcement)
    - Cost-Aware Model Training & Calibration (Optuna & XGBoost)
    - Slice-based Model Evaluation & Duel (Champion vs. Challenger)
    - Zero-Downtime S3 Model Registry & Deployment
    """

    def __init__(self) -> None:
        """Initializes the base configuration for the Training Pipeline."""
        try:
            logging.info("Initializing Training Pipeline orchestration.")
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            logging.exception("Failed to initialize Training Pipeline.")
            raise CustomException(e, sys) from e

    def _run_data_ingestion(self) -> TrainingPipelineDataIngestionArtifact:
        """Executes the Data Ingestion component."""
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
        """Executes the Data Transformation component."""
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
        """Executes the Model Trainer component."""
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
        """Executes the Model Evaluation component."""
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
    ) -> TrainingPipelineModelRegistryArtifact:
        """Executes the Model Registry component."""
        try:
            logging.info(">>> Starting Phase 5: Model Registry")
            config = TrainingPipelineModelRegistryConfig(self.training_pipeline_config)
            model_registry = ModelRegistry(
                config=config,
                trainer_artifact=trainer_artifact,
                evaluation_artifact=evaluation_artifact,
            )
            artifact = model_registry.run()
            logging.info("<<< Phase 5: Model Registry completed successfully.\n")
            return artifact
        except Exception as e:
            logging.exception("Model Registry phase failed.")
            raise CustomException(e, sys) from e

    def run(self) -> None:
        """
        Executes the end-to-end Continuous Training (CT) pipeline.
        Manages the strict lifecycle and promotion gating of the machine learning model.
        """
        try:
            logging.info("===================================================")
            logging.info("STARTING CONTINUOUS TRAINING (CT) PIPELINE")
            logging.info("===================================================")

            # Phase 1: Data Ingestion (OOT Splitting)
            ingestion_artifact = self._run_data_ingestion()

            # Phase 2: Data Transformation (Schema Enforcement)
            transformation_artifact = self._run_data_transformation(ingestion_artifact)

            # Phase 3: Model Training & Calibration
            trainer_artifact = self._run_model_trainer(transformation_artifact)

            # Phase 4: Model Evaluation (Champion vs Challenger Duel)
            evaluation_artifact = self._run_model_evaluation(
                trainer_artifact, ingestion_artifact
            )

            # Phase 5: Model Registry (The Promotion Gate)
            if evaluation_artifact.approval_status:
                logging.info("Gate Check Passed: Proceeding to Model Registry & Deployment.")
                self._run_model_registry(trainer_artifact, evaluation_artifact)
            else:
                logging.warning(
                    "Gate Check Failed: Challenger model rejected. "
                    "Skipping deployment to Model Registry."
                )

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
    except Exception as exc:
        logging.critical("Pipeline execution terminated due to an error.", exc_info=True)
        sys.exit(1)