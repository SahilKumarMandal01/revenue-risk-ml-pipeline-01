import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

import duckdb

from src.entity.config_entity import DataPipelineTransformerConfig
from src.entity.artifact_entity import (
    DataPipelineExtractorArtifact,
    DataPipelineTransformerArtifact,
)
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.utils.main_utils import write_json_file


class Transformer:
    """
    Transformer component for generating the Master Feature Panel.

    Responsibilities:
    - Initialize persistent DuckDB engine to prevent memory overflow.
    - Build Bronze (raw), Silver (cleansed), and Gold (feature) layers.
    - Generate Point-in-Time (OOT) panel data across dynamically defined snapshots.
    - Implement bitemporal lineage tracking (snapshot_date and ingested_at_utc).
    - Export Master Panel directly to Parquet (Zero-Pandas Out-of-Core Processing).
    - Produce observability metadata (execution time, churn rate, row counts).
    """

    def __init__(
        self,
        config: DataPipelineTransformerConfig,
        extractor_artifact: DataPipelineExtractorArtifact,
    ) -> None:
        """
        Initializes Transformer with required configuration and artifacts.
        """
        try:
            self.config: DataPipelineTransformerConfig = config
            self.extractor_artifact: DataPipelineExtractorArtifact = extractor_artifact
            self.raw_data_dir: str = self.extractor_artifact.raw_data_dir_path

            # Define output path for the out-of-core Parquet export
            self.transformed_data_file_path: str = os.path.join(
                self.config.transformer_root_dir, "master_panel.parquet"
            )

            logging.debug(
                "Transformer initialized with raw_data_dir=%s", self.raw_data_dir
            )

        except Exception as e:
            logging.exception("Error during Transformer initialization.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> DataPipelineTransformerArtifact:
        """
        Executes the SQL transformation pipeline out-of-core.

        Returns:
            DataPipelineTransformerArtifact: Artifact containing the path to the engineered feature panel.
        """
        try:
            logging.info("Starting Data Transformation pipeline via DuckDB.")
            start_time: float = time.time()

            con: duckdb.DuckDBPyConnection = self._initialize_duckdb()

            try:
                self._build_bronze_layer(con)
                self._build_silver_layer(con)

                self._generate_master_panel(con)

                execution_time: float = round(time.time() - start_time, 2)
                self._generate_metadata(con, execution_time)

                artifact = DataPipelineTransformerArtifact(
                    transformed_data_file_path=self.transformed_data_file_path,
                    metadata_file_path=self.config.metadata_file_path,
                )

                logging.info(
                    "Transformer pipeline completed in %ss.", execution_time
                )
                logging.info("Transformer artifact created: %s", artifact)

                return artifact

            finally:
                con.close()
                logging.debug("DuckDB connection closed.")

        except Exception as e:
            logging.exception("Error during Transformer run.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATABASE INITIALIZATION
    # ==========================================================
    def _initialize_duckdb(self) -> duckdb.DuckDBPyConnection:
        """Initializes DuckDB connection with persistent local storage and optimized settings."""
        try:
            db_path = self.config.duckdb_data_file_path

            # Wipe the local database from previous runs to avoid state pollution
            if os.path.exists(db_path):
                os.remove(db_path)
                logging.debug("Cleared previous DuckDB persistent storage at %s", db_path)

            logging.info(
                "Initializing DuckDB persistently at %s with %s threads.",
                db_path,
                self.config.threads,
            )

            con: duckdb.DuckDBPyConnection = duckdb.connect(database=db_path)
            con.execute(f"PRAGMA threads={self.config.threads}")
            return con

        except Exception as e:
            logging.exception("Failed to initialize DuckDB.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # BRONZE LAYER (RAW INGESTION)
    # ==========================================================
    def _build_bronze_layer(self, con: duckdb.DuckDBPyConnection) -> None:
        """Creates Bronze layer views dynamically supporting Parquet or CSV files."""
        try:
            logging.info("Building Bronze Layer (Raw Views)...")

            tables: Dict[str, str] = {
                "bronze_customers": "olist_customers_dataset",
                "bronze_orders": "olist_orders_dataset",
                "bronze_items": "olist_order_items_dataset",
                "bronze_payments": "olist_order_payments_dataset",
                "bronze_reviews": "olist_order_reviews_dataset",
            }

            for view_name, base_name in tables.items():
                parquet_path = os.path.join(self.raw_data_dir, f"{base_name}.parquet")
                csv_path = os.path.join(self.raw_data_dir, f"{base_name}.csv")

                if os.path.exists(parquet_path):
                    file_path = parquet_path
                    reader = "read_parquet"
                elif os.path.exists(csv_path):
                    file_path = csv_path
                    reader = "read_csv_auto"
                else:
                    raise FileNotFoundError(
                        f"Missing raw data file for dataset: {base_name}"
                    )

                logging.debug("Creating view %s from %s", view_name, file_path)

                con.execute(
                    f"""
                    CREATE VIEW {view_name} AS
                    SELECT * FROM {reader}('{file_path}')
                    """
                )

            logging.info("Bronze Layer built successfully.")

        except Exception as e:
            logging.exception("Error building Bronze layer.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # SILVER LAYER (CONFORMING & CLEANSING)
    # ==========================================================
    def _build_silver_layer(self, con: duckdb.DuckDBPyConnection) -> None:
        """Creates Silver layer with cleaned, typed, and aggregated tables."""
        try:
            logging.info("Building Silver Layer (Conformed Tables)...")

            con.execute(
                """
                CREATE VIEW silver_payments AS
                SELECT
                    order_id,
                    SUM(payment_value) AS total_payment_value,
                    AVG(payment_installments) AS avg_installments
                FROM bronze_payments
                GROUP BY order_id;

                CREATE VIEW silver_items AS
                SELECT
                    order_id,
                    COUNT(order_item_id) AS total_items,
                    SUM(freight_value) AS total_freight_value,
                    SUM(price) AS total_item_price
                FROM bronze_items
                GROUP BY order_id;

                CREATE VIEW silver_reviews AS
                SELECT
                    order_id,
                    MIN(review_creation_date) AS review_date,
                    AVG(review_score) AS avg_review_score
                FROM bronze_reviews
                GROUP BY order_id;

                CREATE VIEW silver_enriched_orders AS
                SELECT
                    o.order_id,
                    c.customer_unique_id,
                    c.customer_state,
                    o.order_status,
                    TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
                    TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS est_delivery_ts,
                    TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) AS act_delivery_ts,
                    COALESCE(p.total_payment_value, 0) AS payment_value,
                    COALESCE(p.avg_installments, 1) AS installments,
                    COALESCE(i.total_freight_value, 0) AS freight_value,
                    r.review_date,
                    r.avg_review_score
                FROM bronze_orders o
                JOIN bronze_customers c
                    ON o.customer_id = c.customer_id
                LEFT JOIN silver_payments p
                    ON o.order_id = p.order_id
                LEFT JOIN silver_items i
                    ON o.order_id = i.order_id
                LEFT JOIN silver_reviews r
                    ON o.order_id = r.order_id;
                """
            )

            logging.info("Silver Layer built successfully.")

        except Exception as e:
            logging.exception("Error building Silver layer.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # GOLD LAYER (POINT-IN-TIME FEATURE ENGINEERING)
    # ==========================================================
    def _build_loyalist_abt_query(self, cutoff_date: str) -> str:
        """Generates SQL query for Gold Layer Analytical Base Table with bitemporal lineage."""
        return f"""
        SELECT
            hc.customer_unique_id,
            f.customer_state,
            f.recency_days,
            f.tenure_days,
            f.frequency,
            f.monetary_total,
            f.aov,
            f.max_delivery_delay_days,
            f.has_undelivered_order,
            f.total_canceled_orders,
            f.freight_burden_ratio,
            f.avg_installments,
            f.imputed_review_score,
            f.had_terrible_review,
            COALESCE(t.future_ltv, 0.0) AS target_180d_ltv,
            CASE WHEN t.future_orders > 0 THEN 0 ELSE 1 END AS target_is_churn,
            '{cutoff_date}' AS snapshot_date,
            CURRENT_TIMESTAMP AT TIME ZONE 'UTC' AS ingested_at_utc
        FROM (
            SELECT customer_unique_id
            FROM silver_enriched_orders
            WHERE purchase_ts < TIMESTAMP '{cutoff_date}'
            GROUP BY customer_unique_id
            HAVING COUNT(DISTINCT order_id) >= 2
        ) hc
        JOIN (
            SELECT
                customer_unique_id,
                MAX(customer_state) AS customer_state,
                DATE_DIFF('day', MAX(purchase_ts), TIMESTAMP '{cutoff_date}') AS recency_days,
                DATE_DIFF('day', MIN(purchase_ts), TIMESTAMP '{cutoff_date}') AS tenure_days,
                COUNT(DISTINCT order_id) AS frequency,
                SUM(payment_value) AS monetary_total,
                SUM(payment_value) / COUNT(DISTINCT order_id) AS aov,
                MAX(DATE_DIFF('day', est_delivery_ts, act_delivery_ts)) AS max_delivery_delay_days,
                MAX(CASE WHEN act_delivery_ts IS NULL THEN 1 ELSE 0 END) AS has_undelivered_order,
                SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS total_canceled_orders,
                CASE
                    WHEN SUM(payment_value) > 0
                    THEN SUM(freight_value) / SUM(payment_value)
                    ELSE 0
                END AS freight_burden_ratio,
                AVG(installments) AS avg_installments,
                COALESCE(AVG(avg_review_score), 3.0) AS imputed_review_score,
                MAX(CASE WHEN avg_review_score <= 2 THEN 1 ELSE 0 END) AS had_terrible_review
            FROM silver_enriched_orders
            WHERE purchase_ts < TIMESTAMP '{cutoff_date}'
            GROUP BY customer_unique_id
        ) f USING(customer_unique_id)
        LEFT JOIN (
            SELECT
                customer_unique_id,
                COUNT(DISTINCT order_id) AS future_orders,
                SUM(payment_value) AS future_ltv
            FROM silver_enriched_orders
            WHERE purchase_ts >= TIMESTAMP '{cutoff_date}'
              AND purchase_ts < TIMESTAMP '{cutoff_date}' + INTERVAL {self.config.target_days} DAY
              AND order_status IN ('delivered', 'shipped')
            GROUP BY customer_unique_id
        ) t USING(customer_unique_id)
        """

    def _generate_master_panel(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Generates master panel by iterating over snapshots and directly 
        exporting to Parquet to prevent memory bottlenecks.
        """
        try:
            logging.info(
                "Generating Master Panel across %s dynamic snapshots...",
                len(self.config.snapshots),
            )

            # Construct a massive UNION ALL query to execute out-of-core
            snapshot_queries = [
                self._build_loyalist_abt_query(date) for date in self.config.snapshots
            ]
            union_query = " UNION ALL ".join(snapshot_queries)

            # Create internal table
            con.execute(f"CREATE TABLE master_panel AS {union_query}")

            logging.info("Master Panel built in DuckDB. Exporting to Parquet...")

            # Direct to Parquet export (Zero-Pandas operation)
            con.execute(
                f"""
                COPY (SELECT * FROM master_panel) 
                TO '{self.transformed_data_file_path}' 
                (FORMAT PARQUET, COMPRESSION 'snappy');
                """
            )

            logging.info("Successfully exported Master Panel to Parquet.")

        except Exception as e:
            logging.exception("Error generating master panel.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY
    # ==========================================================
    def _generate_metadata(
        self, con: duckdb.DuckDBPyConnection, execution_time: float
    ) -> None:
        """Generates lineage and observability metrics directly via DuckDB SQL."""
        try:
            logging.info("Calculating Transformer telemetry via SQL...")

            # Calculate metrics via SQL to keep memory footprint low
            total_rows = con.execute("SELECT COUNT(*) FROM master_panel").fetchone()[0]
            
            # Fetch Schema columns to calculate dimensions and nulls
            describe_res = con.execute("DESCRIBE master_panel").fetchall()
            columns = [col[0] for col in describe_res]
            total_columns = len(columns)

            churn_rate_res = con.execute(
                "SELECT AVG(target_is_churn) * 100 FROM master_panel"
            ).fetchone()[0]
            churn_rate = float(churn_rate_res) if churn_rate_res is not None else 0.0

            # Calculate Nulls using dynamic SQL
            null_queries = [
                f"CAST(SUM(CASE WHEN \"{c}\" IS NULL THEN 1 ELSE 0 END) AS INTEGER)" 
                for c in columns
            ]
            null_counts_query = f"SELECT {', '.join(null_queries)} FROM master_panel"
            null_counts_res = con.execute(null_counts_query).fetchone()
            
            null_counts = dict(zip(columns, null_counts_res))

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Transformer",
                "execution_time_seconds": execution_time,
                "data_profiles": {
                    "total_rows": total_rows,
                    "total_columns": total_columns,
                    "snapshots_processed": self.config.snapshots,
                    "target_days_window": self.config.target_days,
                },
                "business_metrics": {
                    "global_churn_rate_percentage": round(churn_rate, 2),
                },
                "schema": {
                    "features": columns,
                    "null_counts": {k: v for k, v in null_counts.items() if v > 0},
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata,
            )

            logging.info(
                "Transformer metadata saved at: %s",
                self.config.metadata_file_path,
            )

        except Exception as e:
            logging.exception("Error generating metadata.")
            raise CustomException(e, sys) from e