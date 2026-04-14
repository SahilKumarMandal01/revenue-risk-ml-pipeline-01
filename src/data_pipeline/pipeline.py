import sys

from src.entity.config_entity import (
    DataPipelineConfig,
    DataPipelineExtractorConfig,
    DataPipelineValidatorConfig,
    DataPipelineTransformerConfig,
    DataPipelineLoaderConfig,
)

from src.data_pipeline.components.extractor import Extractor
from src.data_pipeline.components.validator import Validator
from src.data_pipeline.components.transformer import Transformer
from src.data_pipeline.components.loader import Loader

from src.cloud.s3_operations import S3Sync
from src import constants
from src.custom_exception import CustomException
from src.custom_logging import logging


class DataPipeline:
    """
    Orchestrates the complete data pipeline including extraction,
    validation, transformation, loading, and artifact syncing.
    """

    def __init__(self) -> None:
        """Initialize pipeline configuration."""
        self.config = DataPipelineConfig()
        self.logger = logging

    def run(self) -> None:
        """Execute the full data pipeline."""
        try:
            self.logger.info("Starting data pipeline execution.\n")

            extractor_artifact = self._run_extractor()
            validator_artifact = self._run_validator(extractor_artifact)

            if validator_artifact.is_valid:
                transformer_artifact = self._run_transformer(extractor_artifact)
                loader_artifact = self._run_loader(transformer_artifact)
                self._sync_artifacts()

            else:
                self.logger.warning("Validation failed. Skipping further stages.")

            self.logger.info("Data pipeline execution completed successfully.")

        except Exception as exc:
            self.logger.error("Data pipeline execution failed.", exc_info=True)
            raise CustomException(exc, sys) from exc

    def _run_extractor(self):
        """Run the extractor stage."""
        try:
            config = DataPipelineExtractorConfig(self.config)
            extractor = Extractor(config)
            artifact = extractor.run()

            return artifact

        except Exception as exc:
            self.logger.error("Extractor stage failed.", exc_info=True)
            raise CustomException(exc, sys) from exc

    def _run_validator(self, extractor_artifact):
        """Run the validator stage."""
        try:
            config = DataPipelineValidatorConfig(self.config)
            validator = Validator(config, extractor_artifact)
            artifact = validator.run()

            return artifact

        except Exception as exc:
            self.logger.error("Validator stage failed.", exc_info=True)
            raise CustomException(exc, sys) from exc

    def _run_transformer(self, extractor_artifact):
        """Run the transformer stage."""
        try:
            config = DataPipelineTransformerConfig(self.config)
            transformer = Transformer(config, extractor_artifact)
            artifact = transformer.run()

            return artifact

        except Exception as exc:
            self.logger.error("Transformer stage failed.", exc_info=True)
            raise CustomException(exc, sys) from exc

    def _run_loader(self, transformer_artifact):
        """Run the loader stage."""
        try:
            config = DataPipelineLoaderConfig(self.config)
            loader = Loader(config, transformer_artifact)
            artifact = loader.run()
            
            return artifact

        except Exception as exc:
            self.logger.error("Loader stage failed.", exc_info=True)
            raise CustomException(exc, sys) from exc

    def _sync_artifacts(self) -> None:
        """Sync artifacts to S3."""
        try:
            self.logger.info("Syncing artifacts to S3.")

            S3Sync().sync_folder_to_s3(
                folder=constants.ARTIFACT_DIR_NAME,
                aws_bucket_url=f"s3://{constants.S3_BUCKET_NAME}/{constants.ARTIFACT_DIR_NAME}",
            )

            self.logger.info("Artifacts synced to S3 successfully.\n")

        except Exception as exc:
            self.logger.error("S3 sync failed.", exc_info=True)
            raise CustomException(exc, sys) from exc


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()