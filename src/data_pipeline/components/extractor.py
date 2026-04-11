import os
import sys
import json
import shutil
from typing import Dict, Any

import pandas as pd
import kagglehub

from src.entity.config_entity import DataPipelineExtractorConfig
from src.entity.artifact_entity import DataPipelineExtractorArtifact
from src.utils.main_utils import save_csv_file, write_json_file
from src.custom_exception import CustomException
from src.custom_logging import logging


class Extractor:
    """
    Extractor component for downloading and preparing raw datasets.

    Responsibilities:
    - Download dataset from KaggleHub
    - Store raw CSV files in standardized directory
    - Generate schema for each dataset
    - Generate metadata for observability
    """

    def __init__(self, config: DataPipelineExtractorConfig):
        try:
            self.config = config

            os.makedirs(self.config.raw_data_dir_path, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataPipelineExtractorArtifact:
        """
        Main execution method.

        Returns:
            DataPipelineExtractorArtifact
        """
        try:
            logging.info("Starting data extraction pipeline")

            dataset_path = self._download_dataset()
            csv_files = self._collect_csv_files(dataset_path)

            processed_files = self._store_raw_data(csv_files)
            schema_info = self._generate_schema(processed_files)

            write_json_file(
                file_path=self.config.raw_data_schema_file_path,
                content=schema_info
            )
            self._generate_metadata(processed_files, schema_info)

            artifact = DataPipelineExtractorArtifact(
                raw_data_dir_path=self.config.raw_data_dir_path,
                raw_data_schema_file_path=self.config.raw_data_schema_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info(f"Extraction completed successfully:{artifact}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # DOWNLOAD
    # ==========================================================
    def _download_dataset(self) -> str:
        """
        Download dataset using kagglehub.

        Returns:
            Path to downloaded dataset
        """
        try:
            logging.info("Downloading Olist dataset from KaggleHub...")

            dataset_path = kagglehub.dataset_download(
                "olistbr/brazilian-ecommerce"
            )

            logging.info(f"Dataset downloaded at: {dataset_path}")
            return dataset_path

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # FILE DISCOVERY
    # ==========================================================
    def _collect_csv_files(self, dataset_path: str) -> list:
        """
        Collect all CSV files from dataset directory.

        Returns:
            List of file paths
        """
        try:
            logging.info("Collecting CSV files from dataset directory")

            csv_files = []
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))

            if not csv_files:
                raise ValueError("No CSV files found in dataset")

            logging.info(f"Found {len(csv_files)} CSV files")
            return csv_files

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # STORE RAW DATA
    # ==========================================================
    def _store_raw_data(self, csv_files: list) -> Dict[str, str]:
        """
        Copy CSV files into raw data directory.

        Returns:
            Mapping of dataset_name -> stored_path
        """
        try:
            logging.info("Storing raw data files")

            stored_files = {}

            for file_path in csv_files:
                file_name = os.path.basename(file_path)
                destination_path = os.path.join(
                    self.config.raw_data_dir_path, file_name
                )

                shutil.copy(file_path, destination_path)

                dataset_name = file_name.replace(".csv", "")
                stored_files[dataset_name] = destination_path

                logging.info(f"Stored: {file_name}")

            return stored_files

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # SCHEMA GENERATION
    # ==========================================================
    def _generate_schema(self, stored_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate schema for each dataset.

        Returns:
            Schema dictionary
        """
        try:
            logging.info("Generating schema for datasets")

            schema = {}

            for name, path in stored_files.items():
                df = pd.read_csv(path)

                schema[name] = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "num_rows": int(df.shape[0]),
                    "num_columns": int(df.shape[1]),
                    "missing_values": df.isnull().sum().to_dict(),
                }

                logging.info(f"Schema generated for: {name}")

            return schema

        except Exception as e:
            raise CustomException(e, sys)


    # ==========================================================
    # METADATA GENERATION
    # ==========================================================
    def _generate_metadata(
        self,
        stored_files: Dict[str, str],
        schema: Dict[str, Any],
    ) -> None:
        """
        Generate metadata for observability.

        Includes:
        - dataset version
        - number of tables
        - total rows
        - ingestion timestamp
        """
        try:
            logging.info("Generating metadata")

            total_rows = sum(s["num_rows"] for s in schema.values())

            metadata = {
                "dataset_name": "Brazilian Olist E-commerce",
                "num_tables": len(stored_files),
                "total_rows": total_rows,
                "tables": list(stored_files.keys()),
                "ingestion_timestamp": pd.Timestamp.utcnow().isoformat(),
                "data_source": "kagglehub",
                "data_version": "latest",
            }

            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata
            )

            logging.info(
                f"Metadata saved at: {self.config.metadata_file_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)