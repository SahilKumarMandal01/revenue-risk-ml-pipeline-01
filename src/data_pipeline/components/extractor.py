import os
import sys
from typing import Dict, Any, List

import pandas as pd

from src.entity.config_entity import DataPipelineExtractorConfig
from src.entity.artifact_entity import DataPipelineExtractorArtifact
from src.utils.main_utils import write_json_file
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.cloud.s3_operations import S3Sync


class Extractor:
    """
    Extractor component for downloading and preparing raw datasets.

    Responsibilities:
    - Sync raw dataset files directly from AWS S3 Data Lake
    - Store raw files in the standardized local artifact directory
    - Generate schema for each dataset dynamically
    - Generate metadata for pipeline observability and lineage
    """

    def __init__(self, config: DataPipelineExtractorConfig):
        """
        Initializes Extractor with configuration and S3 sync utility.
        """
        try:
            self.config = config
            self.s3_sync = S3Sync()

            # Ensure local raw data directory exists
            os.makedirs(self.config.raw_data_dir_path, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataPipelineExtractorArtifact:
        """
        Executes the main extraction pipeline.

        Returns:
            DataPipelineExtractorArtifact: Details of the extracted data paths and metadata.
        """
        try:
            logging.info("Starting data extraction pipeline (S3 to Local)")

            # Step 1: Sync raw files from S3 to local directory
            self._sync_data_from_s3()

            # Step 2: Collect downloaded files
            data_files = self._collect_downloaded_files()

            # Step 3: Map file paths to dataset names
            stored_files_map = self._map_stored_files(data_files)

            # Step 4: Generate structural schema for validation
            schema_info = self._generate_schema(stored_files_map)

            # Step 5: Persist schema definition
            write_json_file(
                file_path=self.config.raw_data_schema_file_path,
                content=schema_info
            )

            # Step 6: Generate and persist execution metadata
            self._generate_metadata(stored_files_map, schema_info)

            # Package and return Artifact
            artifact = DataPipelineExtractorArtifact(
                raw_data_dir_path=self.config.raw_data_dir_path,
                raw_data_schema_file_path=self.config.raw_data_schema_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info(f"Extraction completed successfully: {artifact}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # AWS S3 DATA SYNC
    # ==========================================================
    def _sync_data_from_s3(self) -> None:
        """
        Synchronizes the remote S3 raw data bucket with the local directory.
        """
        try:
            logging.info(
                f"Syncing raw data from {self.config.s3_raw_data_uri} "
                f"to local directory {self.config.raw_data_dir_path}"
            )
            self.s3_sync.sync_folder_from_s3(
                folder=self.config.raw_data_dir_path,
                aws_bucket_url=self.config.s3_raw_data_uri
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # FILE DISCOVERY & MAPPING
    # ==========================================================
    def _collect_downloaded_files(self) -> List[str]:
        """
        Traverses the local raw data directory to find downloaded dataset files.

        Returns:
            List[str]: A list of absolute file paths.
        """
        try:
            logging.info("Collecting downloaded files from local raw data directory")
            data_files = []

            for root, _, files in os.walk(self.config.raw_data_dir_path):
                for file in files:
                    # Support both parquet and legacy csv formats
                    if file.endswith(".parquet") or file.endswith(".csv"):
                        data_files.append(os.path.join(root, file))

            if not data_files:
                raise ValueError(f"No valid data files found in {self.config.raw_data_dir_path}")

            logging.info(f"Found {len(data_files)} data files")
            return data_files

        except Exception as e:
            raise CustomException(e, sys)

    def _map_stored_files(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Maps clean dataset names (keys) to their local file paths (values).

        Args:
            file_paths (List[str]): List of absolute file paths.

        Returns:
            Dict[str, str]: Mapping of dataset names to file paths.
        """
        try:
            logging.info("Mapping stored raw data files")
            stored_files = {}

            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                # Remove known extensions to derive the base dataset name
                dataset_name = file_name.replace(".parquet", "").replace(".csv", "")
                
                stored_files[dataset_name] = file_path
                logging.info(f"Mapped dataset: {dataset_name} -> {file_name}")

            return stored_files

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # SCHEMA GENERATION
    # ==========================================================
    def _generate_schema(self, stored_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Reads local dataset files into pandas DataFrames to infer column names, 
        data types, dimensions, and missing value counts.

        Args:
            stored_files (Dict[str, str]): Mapping of dataset names to file paths.

        Returns:
            Dict[str, Any]: Nested dictionary defining the schema for all datasets.
        """
        try:
            logging.info("Generating schema for datasets dynamically")
            schema = {}

            for name, path in stored_files.items():
                # Lazy load DataFrame depending on extension
                if path.endswith(".parquet"):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, low_memory=False)

                schema[name] = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "num_rows": int(df.shape[0]),
                    "num_columns": int(df.shape[1]),
                    "missing_values": df.isnull().sum().to_dict(),
                }

                logging.info(f"Schema successfully generated for table: {name}")

            return schema

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # METADATA GENERATION
    # ==========================================================
    def _generate_metadata(self, stored_files: Dict[str, str], schema: Dict[str, Any]) -> None:
        """
        Generates and persists execution metadata (observability data).

        Args:
            stored_files (Dict[str, str]): Mapping of dataset names to file paths.
            schema (Dict[str, Any]): The generated schema payload.
        """
        try:
            logging.info("Generating extractor metadata and telemetry")

            total_rows = sum(s["num_rows"] for s in schema.values())

            metadata = {
                "dataset_name": "Brazilian Olist E-commerce",
                "num_tables": len(stored_files),
                "total_rows": total_rows,
                "tables": list(stored_files.keys()),
                "ingestion_timestamp": pd.Timestamp.utcnow().isoformat(),
                "data_source": "aws_s3",
                "data_version": "latest",
            }

            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata
            )

            logging.info(f"Extractor metadata securely saved at: {self.config.metadata_file_path}")

        except Exception as e:
            raise CustomException(e, sys)