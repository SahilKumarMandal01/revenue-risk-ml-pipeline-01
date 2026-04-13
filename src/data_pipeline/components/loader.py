import os
import sys
import time

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
    - Receive in-memory Master Panel from Transformer.
    - Serialize DataFrame locally as a compressed Parquet file.
    - Upload the Parquet file to the AWS S3 Feature Store.
    - Generate observability telemetry and storage metadata.
    """

    def __init__(
        self,
        config: DataPipelineLoaderConfig,
        transformer_artifact: DataPipelineTransformerArtifact,
    ):
        try:
            self.config = config
            self.transformer_artifact = transformer_artifact
            self.df = self.transformer_artifact.master_panel_df
            self.s3_sync = S3Sync()
            
        except Exception as e:
            raise CustomException(e, sys)

    def run(self) -> DataPipelineLoaderArtifact:
        """Executes the data loading and S3 upload process."""
        try:
            logging.info("Starting Data Loader pipeline (Local to S3).")
            start_time = time.time()

            # 1. Save locally as Parquet
            logging.info(f"Saving Master Panel locally to: {self.config.master_panel_local_file_path}")
            self.df.to_parquet(
                self.config.master_panel_local_file_path, 
                engine="pyarrow", 
                compression="snappy",
                index=False
            )

            # 2. Upload to AWS S3
            logging.info(f"Uploading Master Panel to S3 URI: {self.config.s3_master_panel_uri}")
            self.s3_sync.upload_file(
                local_path=self.config.master_panel_local_file_path,
                s3_uri=self.config.s3_master_panel_uri
            )

            # 3. Get File Metrics for Telemetry
            file_size_bytes = os.path.getsize(self.config.master_panel_local_file_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            execution_time = round(time.time() - start_time, 2)

            # 4. Generate Metadata
            metadata = {
                "pipeline_stage": "Loader",
                "execution_time_seconds": execution_time,
                "storage": {
                    "format": "parquet",
                    "compression": "snappy",
                    "file_size_mb": file_size_mb,
                    "total_rows_saved": int(self.df.shape[0]),
                },
                "lineage": {
                    "local_path": self.config.master_panel_local_file_path,
                    "s3_uri": self.config.s3_master_panel_uri,
                    "bucket": self.config.s3_bucket_name,
                    "feature_store_prefix": self.config.s3_feature_store_dir
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }

            write_json_file(self.config.metadata_file_path, metadata)
            logging.info(f"Loader metadata saved. Total size uploaded: {file_size_mb} MB.")

            # 5. Package Artifact
            artifact = DataPipelineLoaderArtifact(
                local_file_path=self.config.master_panel_local_file_path,
                s3_file_uri=self.config.s3_master_panel_uri,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info(f"Loader artifact created successfully: {artifact}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)