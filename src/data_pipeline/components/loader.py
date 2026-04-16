import os
import sys
import time
import shutil
from datetime import datetime, timezone
from typing import Dict, Any

import pyarrow.parquet as pq

from src.entity.config_entity import DataPipelineLoaderConfig
from src.entity.artifact_entity import (
    DataPipelineTransformerArtifact,
    DataPipelineLoaderArtifact,
)
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.utils.main_utils import write_json_file
from src.cloud.s3_operations import S3Sync


class Loader:
    """
    Loader component for persisting the Master Feature Panel to AWS S3.

    Responsibilities:
    - Receive the out-of-core generated Parquet file path from Transformer.
    - Transfer the Parquet file to the Loader's artifact directory to preserve lineage.
    - Upload the Parquet file to the AWS S3 Feature Store using native boto3.
    - Extract Parquet metadata (row counts) without loading data into memory.
    - Generate observability telemetry and storage metadata.
    """

    def __init__(
        self,
        config: DataPipelineLoaderConfig,
        transformer_artifact: DataPipelineTransformerArtifact,
    ) -> None:
        """
        Initializes Loader with required configuration and artifacts.
        """
        try:
            self.config: DataPipelineLoaderConfig = config
            self.transformer_artifact: DataPipelineTransformerArtifact = transformer_artifact
            
            self.source_parquet_path: str = self.transformer_artifact.transformed_data_file_path
            self.s3_sync: S3Sync = S3Sync()

            logging.info("Loader initialized successfully.")

        except Exception as e:
            logging.exception("Error during Loader initialization.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataPipelineLoaderArtifact:
        """
        Executes the data loading and S3 upload process completely out-of-core.

        Returns:
            DataPipelineLoaderArtifact: Details of local and remote file paths.
        """
        try:
            logging.info("Starting Data Loader pipeline (Local to S3).")
            start_time: float = time.time()

            # 1. Transfer to Loader Artifact Directory (Preserve Lineage boundaries)
            self._transfer_artifact_locally()

            # 2. Upload to AWS S3 via boto3
            self._upload_to_s3()

            # 3. Extract lightweight metrics and generate metadata
            execution_time: float = round(time.time() - start_time, 2)
            self._generate_metadata(execution_time)

            # 4. Package Artifact
            artifact = DataPipelineLoaderArtifact(
                local_file_path=self.config.master_panel_local_file_path,
                s3_file_uri=self.config.s3_master_panel_uri,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Loader artifact created successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Error during Loader run.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # FILE OPERATIONS
    # ==========================================================
    def _transfer_artifact_locally(self) -> None:
        """Copies the Parquet file from Transformer to Loader artifact directory."""
        try:
            logging.info(
                "Transferring Master Panel locally from %s to %s",
                self.source_parquet_path,
                self.config.master_panel_local_file_path,
            )
            shutil.copy2(
                self.source_parquet_path, 
                self.config.master_panel_local_file_path
            )
        except Exception as e:
            logging.exception("Failed to transfer artifact locally.")
            raise CustomException(e, sys) from e

    def _upload_to_s3(self) -> None:
        """Uploads the local Parquet file to the AWS S3 Feature Store."""
        try:
            logging.info(
                "Uploading Master Panel to S3 URI: %s", 
                self.config.s3_master_panel_uri
            )
            self.s3_sync.upload_file(
                local_path=self.config.master_panel_local_file_path,
                s3_uri=self.config.s3_master_panel_uri,
            )
        except Exception as e:
            logging.exception("Failed to upload Master Panel to S3.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY
    # ==========================================================
    def _generate_metadata(self, execution_time: float) -> None:
        """
        Extracts file metrics (size, row count) directly from Parquet metadata 
        to prevent Out-Of-Memory (OOM) issues, and saves JSON observability data.
        """
        try:
            logging.info("Generating Loader telemetry and metadata...")

            # Get file size
            file_size_bytes: int = os.path.getsize(self.config.master_panel_local_file_path)
            file_size_mb: float = round(file_size_bytes / (1024 * 1024), 2)

            # Efficiently read total rows from Parquet footer without loading data into RAM
            parquet_metadata = pq.read_metadata(self.config.master_panel_local_file_path)
            total_rows: int = parquet_metadata.num_rows

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Loader",
                "execution_time_seconds": execution_time,
                "storage": {
                    "format": "parquet",
                    "compression": "snappy",
                    "file_size_mb": file_size_mb,
                    "total_rows_saved": total_rows,
                },
                "lineage": {
                    "local_path": self.config.master_panel_local_file_path,
                    "s3_uri": self.config.s3_master_panel_uri,
                    "bucket": self.config.s3_bucket_name,
                    "feature_store_prefix": self.config.s3_feature_store_dir,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            
            logging.info(
                "Loader metadata saved. Total size uploaded: %s MB (%s rows).", 
                file_size_mb, 
                total_rows
            )

        except Exception as e:
            logging.exception("Failed to generate Loader metadata.")
            raise CustomException(e, sys) from e