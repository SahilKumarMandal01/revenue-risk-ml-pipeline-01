# import os
# import sys
# import uuid
# import random
# import argparse
# import shutil
# from datetime import datetime, timedelta
# from typing import Dict, List, Tuple, Any

# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# try:
#     from src.custom_exception import CustomException
#     from src.custom_logging import logging
# except ImportError:
#     import logging
#     logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
#     class CustomException(Exception):
#         pass


# class SyntheticDataGeneratorConfig:
#     """
#     Configuration dataclass for the Synthetic Data Generator.
#     """
#     def __init__(
#         self,
#         target_date: str,
#         num_customers: int,
#         num_orders: int,
#         output_base_dir: str = "company_data/silver"
#     ) -> None:
#         self.target_date = datetime.strptime(target_date, "%Y-%m-%d")
#         self.num_customers = num_customers
#         self.num_orders = num_orders
#         self.output_base_dir = output_base_dir
#         self.states = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "CE", "PE", "DF"]


# class SyntheticDataGenerator:
#     """
#     Generates structured, schema-compliant synthetic data and persists it
#     as Hive-partitioned Parquet files (year/month/day) with strict atomic overwrite.
#     """

#     def __init__(self, config: SyntheticDataGeneratorConfig) -> None:
#         self.config = config
#         logging.info("Synthetic Data Generator initialized for target date: %s", config.target_date.date())

#     def _generate_core_entities(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """
#         Generates the raw data lists for Customers, Orders, and Payments.
#         Ensures relational integrity across the three datasets.
#         """
#         logging.info("Generating base entities: %d unique customers...", self.config.num_customers)
        
#         global_customers = [str(uuid.uuid4()) for _ in range(self.config.num_customers)]
        
#         orders_data: List[Dict[str, Any]] = []
#         customers_data: List[Dict[str, Any]] = []
#         payments_data: List[Dict[str, Any]] = []

#         logging.info("Generating %d orders and simulating purchasing behavior...", self.config.num_orders)
        
#         for _ in range(self.config.num_orders):
#             order_id = str(uuid.uuid4())
#             transactional_customer_id = str(uuid.uuid4())
#             global_customer_id = random.choice(global_customers)
            
#             # Temporal Logic: Generates transactions leading up to the target date
#             days_ago = random.randint(0, 365)
#             seconds_ago = random.randint(0, 86400) if days_ago > 0 else random.randint(0, 43200)
#             purchase_ts = self.config.target_date - timedelta(days=days_ago, seconds=seconds_ago)
            
#             # Delivery Logic
#             est_delivery_days = random.randint(5, 15)
#             est_delivery_ts = purchase_ts + timedelta(days=est_delivery_days)
            
#             # Status & Actual Delivery
#             is_canceled = random.random() < 0.05
#             if is_canceled:
#                 order_status = "canceled"
#                 act_delivery_ts = None
#             else:
#                 order_status = "delivered"
#                 act_delivery_days = random.randint(3, 25)
#                 act_delivery_ts = purchase_ts + timedelta(days=act_delivery_days)

#             # Partitioning Keys
#             year = str(purchase_ts.year)
#             month = f"{purchase_ts.month:02d}"
#             day = f"{purchase_ts.day:02d}"

#             orders_data.append({
#                 "order_id": order_id,
#                 "customer_id": transactional_customer_id,
#                 "order_purchase_timestamp": purchase_ts.strftime("%Y-%m-%d %H:%M:%S"),
#                 "order_estimated_delivery_date": est_delivery_ts.strftime("%Y-%m-%d %H:%M:%S"),
#                 "order_delivered_customer_date": act_delivery_ts.strftime("%Y-%m-%d %H:%M:%S") if act_delivery_ts else None,
#                 "order_status": order_status,
#                 "year": year,
#                 "month": month,
#                 "day": day
#             })

#             customers_data.append({
#                 "customer_id": transactional_customer_id,
#                 "customer_unique_id": global_customer_id,
#                 "customer_state": random.choice(self.config.states),
#                 "year": year,
#                 "month": month,
#                 "day": day
#             })

#             num_payments = random.randint(1, 3)
#             base_value = round(random.uniform(10.0, 500.0), 2)
#             for _ in range(num_payments):
#                 payments_data.append({
#                     "order_id": order_id,
#                     "payment_value": round(base_value / num_payments, 2),
#                     "year": year,
#                     "month": month,
#                     "day": day
#                 })

#         return pd.DataFrame(orders_data), pd.DataFrame(customers_data), pd.DataFrame(payments_data)

#     def _atomic_partition_cleanup(self, df: pd.DataFrame, table_name: str) -> None:
#         """
#         Identifies all unique year/month/day partitions in the generated DataFrame
#         and completely removes those directories to ensure idempotency (no duplicates).
#         """
#         output_path = os.path.join(self.config.output_base_dir, table_name)
#         unique_partitions = df[['year', 'month', 'day']].drop_duplicates()

#         for _, row in unique_partitions.iterrows():
#             partition_dir = os.path.join(
#                 output_path, 
#                 f"year={row['year']}", 
#                 f"month={row['month']}", 
#                 f"day={row['day']}"
#             )
#             if os.path.exists(partition_dir):
#                 shutil.rmtree(partition_dir)
#                 logging.debug("Cleaned existing partition for atomic overwrite: %s", partition_dir)

#     def _write_hive_partitioned_parquet(self, df: pd.DataFrame, table_name: str) -> None:
#         """
#         Cleans existing partitions atomically, then writes the DataFrame to disk 
#         using Hive-style partitioning (year=.../month=.../day=...).
#         """
#         try:
#             self._atomic_partition_cleanup(df, table_name)
            
#             output_path = os.path.join(self.config.output_base_dir, table_name)
#             table = pa.Table.from_pandas(df)

#             pq.write_to_dataset(
#                 table,
#                 root_path=output_path,
#                 partition_cols=['year', 'month', 'day'],
#                 compression='snappy',
#                 existing_data_behavior='overwrite_or_ignore'
#             )
#             logging.info("Successfully wrote %s partitioned by year/month/day to %s", table_name, output_path)

#         except Exception as e:
#             logging.exception("Failed to write partitioned Parquet dataset for table: %s", table_name)
#             raise CustomException(e, sys) from e

#     def run(self) -> None:
#         """
#         Executes the data generation and export pipeline.
#         """
#         try:
#             logging.info("===================================================")
#             logging.info("STARTING SYNTHETIC DATA GENERATION (ETL SIMULATION)")
#             logging.info("===================================================")

#             df_orders, df_customers, df_payments = self._generate_core_entities()

#             self._write_hive_partitioned_parquet(df_orders, "olist_orders_dataset")
#             self._write_hive_partitioned_parquet(df_customers, "olist_customers_dataset")
#             self._write_hive_partitioned_parquet(df_payments, "olist_order_payments_dataset")

#             logging.info("===================================================")
#             logging.info("SYNTHETIC DATA GENERATION COMPLETED SUCCESSFULLY")
#             logging.info("===================================================")

#         except Exception as e:
#             logging.critical("Synthetic Data Generator terminated due to an error.", exc_info=True)
#             sys.exit(1)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate synthetic transaction data for ML pipeline testing.")
#     parser.add_argument("--target-date", type=str, required=True, help="Anchor date for execution (YYYY-MM-DD)")
#     parser.add_argument("--num-customers", type=int, default=5000, help="Number of unique global customers")
#     parser.add_argument("--num-orders", type=int, default=15000, help="Number of total orders to simulate")
#     parser.add_argument("--output-dir", type=str, default="company_data/silver", help="Root directory for output Parquet files")
    
#     args = parser.parse_args()

#     try:
#         config = SyntheticDataGeneratorConfig(
#             target_date=args.target_date,
#             num_customers=args.num_customers,
#             num_orders=args.num_orders,
#             output_base_dir=args.output_dir
#         )
#         generator = SyntheticDataGenerator(config)
#         generator.run()
#     except Exception as e:
#         logging.error("Initialization failed: %s", str(e))
#         sys.exit(1)







"""
Cloud-Native Synthetic Data Generator for Enterprise ML Testing.

This script simulates the upstream Data Engineering ETL process. It generates
realistic customer, order, and payment records matching the exact schema expected
by downstream ML systems, and streams them directly into an Amazon S3 Data Lake
using strict Hive-style partitioning (year/month/day).

Requirements:
    pip install pandas pyarrow s3fs python-dotenv

Usage:
    python synthetic_data_generator.py --target-date 2026-05-22 --num-customers 50 --num-orders 150
"""

import os
import sys
import uuid
import random
import argparse
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from dotenv import load_dotenv

# Load AWS Credentials securely from .env file
load_dotenv()

try:
    from src.custom_exception import CustomException
    from src.custom_logging import logging
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    class CustomException(Exception):
        pass


class SyntheticDataGeneratorConfig:
    """
    Configuration dataclass for the S3-Native Synthetic Data Generator.
    """
    def __init__(
        self,
        target_date: str,
        num_customers: int,
        num_orders: int,
        output_base_dir: str = "s3://company-central-data-lake/silver"
    ) -> None:
        self.target_date = datetime.strptime(target_date, "%Y-%m-%d")
        self.num_customers = num_customers
        self.num_orders = num_orders
        
        # Enforce S3 URI prefix for cloud-native execution
        if not output_base_dir.startswith("s3://"):
            logging.warning("Output directory does not start with s3://. Defaulting to S3 behavior.")
        self.output_base_dir = output_base_dir
        
        self.states = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "CE", "PE", "DF"]

        # Validate AWS environment variables
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            logging.warning("AWS credentials missing from environment. s3fs will attempt IAM Role fallback.")


class SyntheticDataGenerator:
    """
    Generates structured, schema-compliant synthetic data and persists it
    directly to AWS S3 as Hive-partitioned Parquet files with atomic overwrite.
    """

    def __init__(self, config: SyntheticDataGeneratorConfig) -> None:
        self.config = config
        
        try:
            # Initialize S3 File System wrapper
            self.s3_fs = s3fs.S3FileSystem()
            logging.info("S3 File System connected successfully.")
        except Exception as e:
            logging.error("Failed to initialize S3 FileSystem. Check AWS credentials.")
            raise CustomException(e, sys) from e

        logging.info("Synthetic Data Generator initialized for target date: %s", config.target_date.date())

    def _generate_core_entities(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates the raw data lists for Customers, Orders, and Payments.
        Ensures relational integrity across the three datasets.
        """
        logging.info("Generating base entities: %d unique customers...", self.config.num_customers)
        
        global_customers = [str(uuid.uuid4()) for _ in range(self.config.num_customers)]
        
        orders_data: List[Dict[str, Any]] = []
        customers_data: List[Dict[str, Any]] = []
        payments_data: List[Dict[str, Any]] = []

        logging.info("Generating %d orders and simulating purchasing behavior...", self.config.num_orders)
        
        for _ in range(self.config.num_orders):
            order_id = str(uuid.uuid4())
            transactional_customer_id = str(uuid.uuid4())
            global_customer_id = random.choice(global_customers)
            
            # Temporal Logic: Generates transactions leading up to the target date
            days_ago = random.randint(0, 365)
            seconds_ago = random.randint(0, 86400) if days_ago > 0 else random.randint(0, 43200)
            purchase_ts = self.config.target_date - timedelta(days=days_ago, seconds=seconds_ago)
            
            # Delivery Logic
            est_delivery_days = random.randint(5, 15)
            est_delivery_ts = purchase_ts + timedelta(days=est_delivery_days)
            
            # Status & Actual Delivery
            is_canceled = random.random() < 0.05
            if is_canceled:
                order_status = "canceled"
                act_delivery_ts = None
            else:
                order_status = "delivered"
                act_delivery_days = random.randint(3, 25)
                act_delivery_ts = purchase_ts + timedelta(days=act_delivery_days)

            # Partitioning Keys
            year = str(purchase_ts.year)
            month = f"{purchase_ts.month:02d}"
            day = f"{purchase_ts.day:02d}"

            orders_data.append({
                "order_id": order_id,
                "customer_id": transactional_customer_id,
                "order_purchase_timestamp": purchase_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "order_estimated_delivery_date": est_delivery_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "order_delivered_customer_date": act_delivery_ts.strftime("%Y-%m-%d %H:%M:%S") if act_delivery_ts else None,
                "order_status": order_status,
                "year": year,
                "month": month,
                "day": day
            })

            customers_data.append({
                "customer_id": transactional_customer_id,
                "customer_unique_id": global_customer_id,
                "customer_state": random.choice(self.config.states),
                "year": year,
                "month": month,
                "day": day
            })

            num_payments = random.randint(1, 3)
            base_value = round(random.uniform(10.0, 500.0), 2)
            for _ in range(num_payments):
                payments_data.append({
                    "order_id": order_id,
                    "payment_value": round(base_value / num_payments, 2),
                    "year": year,
                    "month": month,
                    "day": day
                })

        return pd.DataFrame(orders_data), pd.DataFrame(customers_data), pd.DataFrame(payments_data)

    def _atomic_partition_cleanup_s3(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Identifies all unique year/month/day partitions in the generated DataFrame
        and completely removes those directories in S3 to ensure idempotency.
        """
        output_path = f"{self.config.output_base_dir.rstrip('/')}/{table_name}"
        unique_partitions = df[['year', 'month', 'day']].drop_duplicates()

        for _, row in unique_partitions.iterrows():
            # Build the exact S3 partition prefix
            partition_prefix = f"{output_path}/year={row['year']}/month={row['month']}/day={row['day']}"
            
            if self.s3_fs.exists(partition_prefix):
                # Recursively delete the S3 prefix (atomic clean)
                self.s3_fs.rm(partition_prefix, recursive=True)
                logging.debug("Cleaned existing S3 partition for atomic overwrite: %s", partition_prefix)

    def _write_hive_partitioned_parquet(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Cleans existing partitions atomically in S3, then streams the DataFrame directly 
        to the S3 bucket using Hive-style partitioning (year=.../month=.../day=...).
        """
        try:
            # 1. Atomic Clean
            self._atomic_partition_cleanup_s3(df, table_name)
            
            output_path = f"{self.config.output_base_dir.rstrip('/')}/{table_name}"
            
            # 2. Memory Optimization: Convert to PyArrow Table immediately
            table = pa.Table.from_pandas(df)

            # 3. Stream to S3 Native
            # PyArrow seamlessly writes to S3 when provided with the 's3://' scheme
            # and authenticates automatically via the underlying AWS credentials.
            pq.write_to_dataset(
                table,
                root_path=output_path,
                partition_cols=['year', 'month', 'day'],
                compression='snappy',
                existing_data_behavior='overwrite_or_ignore'
            )
            logging.info("Successfully streamed %s partitioned by year/month/day to %s", table_name, output_path)

        except Exception as e:
            logging.exception("Failed to write partitioned Parquet dataset to S3 for table: %s", table_name)
            raise CustomException(e, sys) from e

    def run(self) -> None:
        """
        Executes the data generation and S3 streaming pipeline.
        Manages memory aggressively for low-resource environments.
        """
        try:
            logging.info("===================================================")
            logging.info("STARTING S3-NATIVE SYNTHETIC DATA GENERATION (ETL)")
            logging.info("===================================================")

            df_orders, df_customers, df_payments = self._generate_core_entities()

            # Write and instantly free memory (critical for Codespaces)
            self._write_hive_partitioned_parquet(df_orders, "olist_orders_dataset")
            del df_orders
            gc.collect()

            self._write_hive_partitioned_parquet(df_customers, "olist_customers_dataset")
            del df_customers
            gc.collect()

            self._write_hive_partitioned_parquet(df_payments, "olist_order_payments_dataset")
            del df_payments
            gc.collect()

            logging.info("===================================================")
            logging.info("S3-NATIVE DATA GENERATION COMPLETED SUCCESSFULLY")
            logging.info("===================================================")

        except Exception as e:
            logging.critical("Synthetic Data Generator terminated due to an error.", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data to AWS S3 for ML pipeline testing.")
    parser.add_argument("--target-date", type=str, required=True, help="Anchor date for execution (YYYY-MM-DD)")
    parser.add_argument("--num-customers", type=int, default=50, help="Number of unique global customers")
    parser.add_argument("--num-orders", type=int, default=150, help="Number of total orders to simulate")
    parser.add_argument(
        "--s3-output-dir", 
        type=str, 
        default="s3://company-central-data-lake/silver", 
        help="S3 root directory for output Parquet files"
    )
    
    args = parser.parse_args()

    try:
        config = SyntheticDataGeneratorConfig(
            target_date=args.target_date,
            num_customers=args.num_customers,
            num_orders=args.num_orders,
            output_base_dir=args.s3_output_dir
        )
        generator = SyntheticDataGenerator(config)
        generator.run()
    except Exception as e:
        logging.error("Initialization failed: %s", str(e))
        sys.exit(1)