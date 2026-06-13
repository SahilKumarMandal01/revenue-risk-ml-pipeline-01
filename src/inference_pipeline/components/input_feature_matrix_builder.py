# src/inference_pipeline/components/input_feature_matrix_builder.py

import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import duckdb

from src.entity.config_entity import InferenceInputFeatureMatrixBuilderConfig
from src.entity.artifact_entity import InferenceInputFeatureMatrixBuilderArtifact
from src.shared.shared_feature import SharedFeatureGenerator
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.utils.main_utils import write_json_file


class InputFeatureMatrixBuilder:
    """
    Input Feature Matrix Builder for the Inference Pipeline.

    Responsibilities:
    - Establish an out-of-core DuckDB connection directly to the S3 Data Lake.
    - Utilize the `SharedFeatureGenerator` to mathematically enforce zero training-serving skew.
    - Execute a federated query across Hive-partitioned S3 directories (Predicate Pushdown).
    - Materialize the active customer scoring population up to the execution snapshot bound.
    - Export the final feature matrix directly to a local Parquet file.
    - Generate strict schema definitions and observability metadata.
    """

    def __init__(self, config: InferenceInputFeatureMatrixBuilderConfig) -> None:
        """
        Initializes the Input Feature Matrix Builder component.
        """
        try:
            self.config = config
            
            # Initialize the Shared Feature Generator configured for Hive-Partitioned S3 Data
            self.feature_generator = SharedFeatureGenerator(
                data_dir=self.config.s3_data_lake_uri,
                is_partitioned=True
            )

            logging.info("Inference Pipeline: Input Feature Matrix Builder initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Input Feature Matrix Builder.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> InferenceInputFeatureMatrixBuilderArtifact:
        """
        Executes the feature engineering pipeline directly against the S3 Data Lake.

        Returns:
            InferenceInputFeatureMatrixBuilderArtifact: Local paths to the generated matrix, schema, and metadata.
        """
        try:
            logging.info("Starting Input Feature Matrix Builder execution.")
            start_time = time.time()

            # 1. Establish DuckDB Connection with S3 Capabilities
            con = self._initialize_duckdb_s3_connection()

            try:
                # 2. Build and export the Feature Matrix
                self._build_and_export_feature_matrix(con)

                # 3. Generate Schema Definition
                schema_info = self._generate_schema(con)

                # 4. Generate Telemetry & Metadata
                execution_time = round(time.time() - start_time, 2)
                self._generate_metadata(con, schema_info, execution_time)

            finally:
                # Ensure graceful connection closure to prevent memory leaks
                con.close()
                logging.debug("DuckDB S3 connection safely closed.")

            # 5. Package Artifact
            artifact = InferenceInputFeatureMatrixBuilderArtifact(
                feature_matrix_file_path=self.config.feature_matrix_file_path,
                schema_file_path=self.config.schema_file_path,
                metadata_file_path=self.config.metadata_file_path,
                snapshot_date=self.config.snapshot_date
            )

            logging.info("Input Feature Matrix generation completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Critical Failure: Input Feature Matrix Builder run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA ENGINE CONFIGURATION
    # ==========================================================
    def _initialize_duckdb_s3_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Initializes an ephemeral DuckDB connection configured with the `aws` extension
        to securely resolve authentication chains natively from the operating environment.
        """
        try:
            logging.info("Initializing DuckDB with native AWS credential auto-discovery.")
            con = duckdb.connect(database=":memory:")
            
            # Install and load extensions required for secure S3 file access
            con.execute("INSTALL httpfs;")
            con.execute("LOAD httpfs;")
            con.execute("INSTALL aws;")
            con.execute("LOAD aws;")

            # Query system configuration parameters
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            con.execute(f"SET s3_region='{aws_region}';")

            # Instruct DuckDB to automatically evaluate environment keys, configuration profiles, 
            # and local credential managers using the standard AWS provider chain.
            con.execute("CALL load_aws_credentials();")
            logging.info("DuckDB AWS credential chain auto-loaded successfully.")

            # Optimize memory and thread utilization for inference
            con.execute("PRAGMA threads=4;")
            con.execute("PRAGMA memory_limit='4GB';")

            return con

        except Exception as e:
            logging.exception("Failed to initialize DuckDB S3 connection using native chain auto-discovery.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # FEATURE MATRIX GENERATION
    # ==========================================================
    def _build_and_export_feature_matrix(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Retrieves the exact unified SQL feature logic from the SharedFeatureGenerator,
        executes it against the Hive-partitioned S3 Lake, and streams the output 
        directly into a local Snappy-compressed Parquet file.
        """
        try:
            logging.info("Acquiring point-in-time SQL logic from SharedFeatureGenerator.")
            sql_query = self.feature_generator.get_feature_query(self.config.snapshot_date)

            logging.info(
                "Streaming remote feature aggregations directly to local Parquet: %s",
                self.config.feature_matrix_file_path
            )

            # Zero-copy stream to local disk (Bypassing Pandas Memory)
            copy_query = f"""
                COPY ({sql_query}) 
                TO '{self.config.feature_matrix_file_path}' 
                (FORMAT PARQUET, COMPRESSION 'snappy');
            """
            
            con.execute(copy_query)
            logging.info("Feature Matrix materialized and saved successfully.")

        except Exception as e:
            logging.exception("Failed to compile or execute the feature generation query.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # SCHEMA & METADATA GENERATION
    # ==========================================================
    def _generate_schema(self, con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
        """
        Dynamically infers the exact physical schema of the generated inference matrix 
        to ensure data contracts match the Model Loader's expectations.
        """
        try:
            logging.info("Inferring structural schema from the generated feature matrix.")
            
            describe_query = f"DESCRIBE SELECT * FROM read_parquet('{self.config.feature_matrix_file_path}')"
            describe_res = con.execute(describe_query).fetchall()

            columns = [row[0] for row in describe_res]
            dtypes = {row[0]: row[1] for row in describe_res}

            schema_info = {
                "columns": columns,
                "dtypes": dtypes,
                "num_columns": len(columns),
                "snapshot_date": self.config.snapshot_date
            }

            write_json_file(file_path=self.config.schema_file_path, content=schema_info)
            logging.info("Schema definition saved to %s", self.config.schema_file_path)

            return schema_info

        except Exception as e:
            logging.exception("Failed to dynamically generate schema definition.")
            raise CustomException(e, sys) from e

    def _generate_metadata(
        self, 
        con: duckdb.DuckDBPyConnection, 
        schema_info: Dict[str, Any], 
        execution_time: float
    ) -> None:
        """
        Generates rich operational telemetry including source lineage, row counts, 
        and timing metrics for downstream observability and drift detection.
        """
        try:
            logging.info("Compiling telemetry and metadata for the Inference Builder.")

            # Compute actual scoring population size
            count_query = f"SELECT COUNT(*) FROM read_parquet('{self.config.feature_matrix_file_path}')"
            total_rows_res = con.execute(count_query).fetchone()
            total_rows = total_rows_res[0] if total_rows_res else 0

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Inference Feature Matrix Builder",
                "execution_time_seconds": execution_time,
                "data_provenance": {
                    "s3_data_lake_source": self.config.s3_data_lake_uri,
                    "storage_layout": "Hive Partitioned (year/month/day)",
                    "snapshot_date_anchor": self.config.snapshot_date,
                },
                "scoring_population": {
                    "total_eligible_customers": total_rows,
                    "eligibility_rule": "At least one successful historical order prior to snapshot date.",
                },
                "feature_engineering": {
                    "strategy": "SharedFeatureGenerator (Zero Training-Serving Skew)",
                    "total_features_generated": schema_info["num_columns"],
                    "features_list": schema_info["columns"],
                },
                "artifacts": {
                    "feature_matrix_path": self.config.feature_matrix_file_path,
                    "schema_path": self.config.schema_file_path
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            logging.info(
                "Input Feature Matrix Builder metadata securely saved at: %s", 
                self.config.metadata_file_path
            )

        except Exception as e:
            logging.exception("Failed to generate and save component metadata.")
            raise CustomException(e, sys) from e