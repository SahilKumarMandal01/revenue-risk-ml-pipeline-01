import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any

import duckdb
import pyarrow.parquet as pq

from src.entity.config_entity import InferenceFeatureMatrixBuilderConfig
from src.entity.artifact_entity import InferenceFeatureMatrixBuilderArtifact
from src.shared.shared_feature import SharedFeatureGenerator
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.utils.main_utils import write_json_file


class InputFeatureMatrixBuilder:
    """
    Input Feature Matrix Builder for the Inference Pipeline.

    Responsibilities:
    - Acts as the Executive Chef: takes raw data ingredients and perfectly matches
      the model's expected mathematical recipe using `SharedFeatureGenerator`.
    - Mathematically prevents training-serving skew by utilizing the exact same
      SQL definition used during the Continuous Training (CT) pipeline phase.
    - Utilizes DuckDB's out-of-core `COPY` execution to stream aggregated features
      directly to disk, preventing OOM (Out Of Memory) container crashes.
    """

    def __init__(self, config: InferenceFeatureMatrixBuilderConfig) -> None:
        """
        Initializes the Input Feature Matrix Builder component.

        Args:
            config (InferenceFeatureMatrixBuilderConfig): Component configuration.
        """
        try:
            self.config = config
            logging.info("Inference Pipeline: Input Feature Matrix Builder initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Input Feature Matrix Builder.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> InferenceFeatureMatrixBuilderArtifact:
        """
        Executes the feature generation process for batch inference.

        Returns:
            InferenceFeatureMatrixBuilderArtifact: Artifact containing the localized
                                                   feature matrix Parquet file path.
        """
        try:
            logging.info("Starting Input Feature Matrix Builder execution.")
            start_time = time.time()

            # 1. Define Temporal Anchor (Current UTC Time)
            snapshot_date = self._get_snapshot_date()

            # 2. Execute SQL and Stream out-of-core to Disk
            self._execute_and_stream(snapshot_date)

            # 3. Validate Output and Profile Dataset
            row_count = self._validate_and_extract_row_count()

            if row_count == 0:
                logging.warning(
                    "Feature Matrix generated 0 rows. Check upstream Data Engineering SLA "
                    "or ensure customers exist prior to snapshot_date: %s", snapshot_date
                )

            # 4. Generate Telemetry Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(snapshot_date, row_count, execution_time)

            # 5. Package Artifact
            artifact = InferenceFeatureMatrixBuilderArtifact(
                feature_matrix_file_path=self.config.feature_matrix_file_path,
                snapshot_date=snapshot_date,
                row_count=row_count,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info("Input Feature Matrix Builder execution completed successfully: %s", artifact)
            
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Matrix Builder run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # INTERNAL PROCESSING METHODS
    # ==========================================================
    def _get_snapshot_date(self) -> str:
        """
        Captures the exact moment of pipeline execution to bound temporal feature logic
        and strictly prevent target leakage into the future.
        """
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        logging.info("Calculated temporal snapshot bounds: %s", current_time)
        return current_time

    def _execute_and_stream(self, snapshot_date: str) -> None:
        """
        Fetches the shared SQL logic and executes it using DuckDB.
        Instead of bringing data into pandas/Python memory, it uses DuckDB's native
        `COPY` command to stream the computed matrix directly into a Parquet file.
        """
        try:
            # Initialize the shared logic module targeting the Bronze/Silver layer
            feature_generator = SharedFeatureGenerator(
                data_dir=self.config.company_data_dir,
                is_partitioned=self.config.is_partitioned,
            )

            # Retrieve the raw SQL SELECT statement
            base_sql_query = feature_generator.get_feature_query(snapshot_date=snapshot_date)

            # Wrap in DuckDB's native out-of-core Parquet writer
            # This is critical for OOM-safe enterprise batch processing.
            streaming_query = (
                f"COPY ({base_sql_query}) "
                f"TO '{self.config.feature_matrix_file_path}' (FORMAT PARQUET);"
            )

            logging.info("Executing DuckDB out-of-core streaming query...")

            # Execute via an ephemeral in-memory DuckDB connection
            with duckdb.connect() as con:
                con.execute(streaming_query)

            logging.info(
                "Successfully streamed Feature Matrix to: %s", 
                self.config.feature_matrix_file_path
            )

        except duckdb.IOException as duck_exc:
            # Specific trap for missing upstream Hive partitions/files
            logging.error(
                "DuckDB I/O Error: Upstream files missing. Did the ETL pipeline run? "
                f"Path checked: {self.config.company_data_dir}"
            )
            raise CustomException(duck_exc, sys) from duck_exc

        except Exception as e:
            logging.exception("Failed to execute and stream the feature matrix.")
            raise CustomException(e, sys) from e

    def _validate_and_extract_row_count(self) -> int:
        """
        Uses PyArrow to read the Parquet footer metadata.
        This provides O(1) time complexity row counting without loading data into memory.
        """
        try:
            if not os.path.exists(self.config.feature_matrix_file_path):
                raise FileNotFoundError(
                    f"Expected output matrix not found at {self.config.feature_matrix_file_path}"
                )

            parquet_metadata = pq.read_metadata(self.config.feature_matrix_file_path)
            row_count = parquet_metadata.num_rows

            logging.info("Matrix Validation Success: Generated %d rows.", row_count)
            return row_count

        except Exception as e:
            logging.exception("Failed to read Parquet metadata for row counting.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(self, snapshot_date: str, row_count: int, execution_time: float) -> None:
        """
        Generates telemetry metadata ensuring complete lineage between
        the generated data snapshot and the execution environment.
        """
        try:
            logging.info("Generating Matrix Builder observability metadata.")

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Inference Input Feature Matrix Builder",
                "execution_time_seconds": execution_time,
                "data_lineage": {
                    "source_directory": self.config.company_data_dir,
                    "is_hive_partitioned": self.config.is_partitioned,
                    "snapshot_date": snapshot_date,
                },
                "output_profile": {
                    "row_count": row_count,
                    "output_format": "parquet",
                    "file_path": self.config.feature_matrix_file_path,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            logging.info(
                "Matrix Builder metadata securely saved at: %s", 
                self.config.metadata_file_path
            )

        except Exception as e:
            logging.exception("Failed to generate Matrix Builder metadata.")
            raise CustomException(e, sys) from e