import os
import sys

from dotenv import load_dotenv

from src import constants
from src.cloud.s3_operations import S3Sync
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.entity.config_entity import (
    InferenceInputFeatureMatrixBuilderConfig,
    InferenceModelLoaderConfig,
    InferencePipelineConfig,
    InferenceReportGeneratorConfig,
    InferenceReportPublisherConfig,
    InferenceValidatorConfig,
)
from src.inference_pipeline.components.input_feature_matrix_builder import (
    InputFeatureMatrixBuilder,
)
from src.inference_pipeline.components.model_loader import ModelLoader
from src.inference_pipeline.components.report_generator import ReportGenerator
from src.inference_pipeline.components.report_publisher import ReportPublisher
from src.inference_pipeline.components.validator import InferenceValidator

load_dotenv()


class InferencePipeline:
    """
    Orchestrates the complete inference workflow.

    Responsibilities:
    - Load the inference model.
    - Build the input feature matrix.
    - Validate generated features.
    - Generate inference reports.
    - Publish generated reports.
    - Sync generated artifacts to AWS S3.
    """

    def __init__(self) -> None:
        """
        Initialize inference pipeline configuration.
        """
        try:
            logging.info("Initializing Inference Pipeline.")
            self.pipeline_config = InferencePipelineConfig()
        except Exception as e:
            logging.exception("Failed to initialize Inference Pipeline.")
            raise CustomException(e, sys) from e

    def _run_model_loader(self):
        """
        Execute the model loading phase.

        Returns:
            Model loader artifact.
        """
        try:
            logging.info(">>> Starting Phase 1: Model Loading")

            model_loader_config = InferenceModelLoaderConfig(
                self.pipeline_config
            )

            model_loader = ModelLoader(model_loader_config)
            model_loader_artifact = model_loader.run()

            logging.info(
                "<<< Phase 1: Model Loading completed successfully.\n"
            )

            print(model_loader_artifact)
            return model_loader_artifact

        except Exception as e:
            logging.exception("Model Loading phase failed.")
            raise CustomException(e, sys) from e

    def _run_feature_builder(self):
        """
        Execute the input feature matrix builder phase.

        Returns:
            Feature builder artifact.
        """
        try:
            logging.info(">>> Starting Phase 2: Input Feature Matrix Builder")

            feature_builder_config = (
                InferenceInputFeatureMatrixBuilderConfig(
                    self.pipeline_config
                )
            )

            feature_builder = InputFeatureMatrixBuilder(
                feature_builder_config
            )

            feature_builder_artifact = feature_builder.run()

            logging.info(
                "<<< Phase 2: Input Feature Matrix Builder completed "
                "successfully.\n"
            )

            print(feature_builder_artifact)
            return feature_builder_artifact

        except Exception as e:
            logging.exception("Input Feature Matrix Builder phase failed.")
            raise CustomException(e, sys) from e

    def _run_feature_validator(
        self,
        model_loader_artifact,
        feature_builder_artifact,
    ):
        """
        Execute the feature validation phase.

        Args:
            model_loader_artifact: Output artifact from model loading.
            feature_builder_artifact: Output artifact from feature building.

        Returns:
            Feature validator artifact.
        """
        try:
            logging.info(">>> Starting Phase 3: Feature Validation")

            feature_validator_config = InferenceValidatorConfig(
                self.pipeline_config
            )

            feature_validator = InferenceValidator(
                feature_validator_config
            )

            feature_validator_artifact = feature_validator.run(
                model_loader_artifact,
                feature_builder_artifact,
            )

            logging.info(
                "<<< Phase 3: Feature Validation completed successfully.\n"
            )

            print(feature_validator_artifact)
            return feature_validator_artifact

        except Exception as e:
            logging.exception("Feature Validation phase failed.")
            raise CustomException(e, sys) from e

    def _run_report_generator(
        self,
        model_loader_artifact,
        feature_builder_artifact,
    ):
        """
        Execute the report generation phase.

        Args:
            model_loader_artifact: Output artifact from model loading.
            feature_builder_artifact: Output artifact from feature building.

        Returns:
            Report generator artifact.
        """
        try:
            logging.info(">>> Starting Phase 4: Report Generation")

            report_generator_config = InferenceReportGeneratorConfig(
                self.pipeline_config
            )

            report_generator = ReportGenerator(
                report_generator_config,
                model_loader_artifact,
                feature_builder_artifact,
            )

            report_generator_artifact = report_generator.run()

            logging.info(
                "<<< Phase 4: Report Generation completed successfully.\n"
            )

            print(report_generator_artifact)
            return report_generator_artifact

        except Exception as e:
            logging.exception("Report Generation phase failed.")
            raise CustomException(e, sys) from e

    def _run_report_publisher(self, report_generator_artifact):
        """
        Execute the report publishing phase.

        Args:
            report_generator_artifact: Output artifact from report generation.

        Returns:
            Report publisher artifact.
        """
        try:
            logging.info(">>> Starting Phase 5: Report Publishing")

            report_publisher_config = InferenceReportPublisherConfig(
                self.pipeline_config
            )

            report_publisher = ReportPublisher(
                report_publisher_config,
                report_generator_artifact,
            )

            report_publisher_artifact = report_publisher.run()

            logging.info(
                "<<< Phase 5: Report Publishing completed successfully.\n"
            )

            print(report_publisher_artifact)
            return report_publisher_artifact

        except Exception as e:
            logging.exception("Report Publishing phase failed.")
            raise CustomException(e, sys) from e

    def _sync_artifacts(self) -> None:
        """
        Synchronize local inference artifacts to AWS S3.
        """
        try:
            logging.info(">>> Starting Phase 6: Artifact Sync (S3)")

            S3Sync().sync_folder_to_s3(
                folder=constants.ARTIFACT_DIR_NAME,
                aws_bucket_url=(
                    f"s3://{constants.S3_BUCKET_NAME}/"
                    f"{constants.ARTIFACT_DIR_NAME}/"
                    f"{constants.INFERENCE_PIPELINE_ROOT_DIR_NAME}"
                ),
            )

            logging.info(
                "<<< Phase 6: Artifact Sync completed successfully.\n"
            )

        except Exception as e:
            logging.exception("Artifact Sync phase failed.")
            raise CustomException(e, sys) from e

    def run(self) -> None:
        """
        Execute the complete inference pipeline.
        """
        try:
            logging.info("===================================================")
            logging.info("STARTING INFERENCE PIPELINE")
            logging.info("===================================================")

            model_loader_artifact = self._run_model_loader()

            feature_builder_artifact = self._run_feature_builder()

            feature_validator_artifact = self._run_feature_validator(
                model_loader_artifact,
                feature_builder_artifact,
            )

            if feature_validator_artifact.is_valid:
                report_generator_artifact = self._run_report_generator(
                    model_loader_artifact,
                    feature_builder_artifact,
                )

                self._run_report_publisher(
                    report_generator_artifact
                )

                self._sync_artifacts()

            logging.info("===================================================")
            logging.info("INFERENCE PIPELINE EXECUTION COMPLETED")
            logging.info("===================================================")

        except Exception as e:
            logging.exception(
                "Critical Failure in Inference Pipeline execution."
            )
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        pipeline = InferencePipeline()
        pipeline.run()

    except Exception:
        logging.critical(
            "Inference Pipeline execution terminated due to an error.",
            exc_info=True,
        )
        sys.exit(1)