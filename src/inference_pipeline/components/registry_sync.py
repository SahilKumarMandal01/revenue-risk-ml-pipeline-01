import os
import sys
import uuid
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

from src.entity.config_entity import InferencePipelineRegistrySyncConfig
from src.entity.artifact_entity import InferencePipelineRegistrySyncArtifact
from src.cloud.s3_operations import S3Sync
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.utils.main_utils import write_json_file


class RegistrySync:
    """
    Stage 1 Component: Registry Sync (The Context Initializer)

    Responsibilities:
    - Act as the entry point for the Inference Pipeline DAG.
    - Generate an idempotent, universally unique identifier (UUID) for the execution run.
    - Fetch the mutable `production_champion.json` state pointer from the S3 Vault.
    - Validate the Model Registry schema contract.
    - Download the physical serialized Mega-Pipeline (`model.pkl`) to the local container disk.
    - Generate standard observability metadata.
    """

    def __init__(self, config: InferencePipelineRegistrySyncConfig):
        try:
            self.config = config
            self.s3_sync = S3Sync()
            
            # Create isolated component directory
            os.makedirs(self.config.registry_sync_root_dir, exist_ok=True)
            logging.info("Inference Pipeline: Registry Sync component initialized.")
            
        except Exception as e:
            logging.exception("Failed to initialize Registry Sync component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> InferencePipelineRegistrySyncArtifact:
        """
        Executes the context initialization and model fetching process.
        """
        try:
            logging.info("Starting Inference Stage 1: Registry Sync.")
            start_time = time.time()

            # 1. Generate Global Run Identity
            inference_run_uuid = self._generate_run_uuid()

            # 2. Fetch and Parse the Production Pointer
            champion_run_id, s3_model_path = self._fetch_production_pointer()

            # 3. Download the Physical Model
            self._download_champion_model(s3_model_path)

            # 4. Generate Enterprise-Grade Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(inference_run_uuid, champion_run_id, s3_model_path, execution_time)

            # 5. Package and Return Artifacts for downstream DAG steps
            artifact = InferencePipelineRegistrySyncArtifact(
                inference_run_uuid=inference_run_uuid,
                champion_run_id=champion_run_id,
                model_file_path=self.config.downloaded_model_file_path,
                metadata_file_path=self.config.metadata_file_path
            )

            logging.info("Registry Sync completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Registry Sync run failed. Inference pipeline aborted.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # CORE LOGIC OPERATIONS
    # ==========================================================
    def _generate_run_uuid(self) -> str:
        """
        Generates a URL-safe, highly unique ID for idempotency in downstream file writing.
        """
        run_uuid = uuid.uuid4().hex
        logging.info(f"Generated Global Inference Run UUID: {run_uuid}")
        return run_uuid

    def _fetch_production_pointer(self) -> Tuple[str, str]:
        """
        Downloads the lightweight JSON pointer from S3, validates the schema version,
        and extracts the routing logic.
        """
        try:
            logging.info(f"Fetching Champion pointer from: {self.config.s3_pointer_file_uri}")
            local_tmp_pointer_path = os.path.join(self.config.registry_sync_root_dir, "tmp_pointer.json")

            self.s3_sync.download_file(
                s3_uri=self.config.s3_pointer_file_uri, 
                local_path=local_tmp_pointer_path
            )

            with open(local_tmp_pointer_path, "r") as f:
                pointer_data = json.load(f)

            # Fail Loudly: API Contract Validation
            schema_version = pointer_data.get("schema_version")
            if schema_version != 1:
                raise ValueError(f"Incompatible pointer schema version: {schema_version}. Expected v1.")

            champion_run_id = pointer_data.get("champion_run_id")
            s3_model_path = pointer_data.get("s3_model_path")

            if not champion_run_id or not s3_model_path:
                raise KeyError("Pointer file is missing critical routing keys ('champion_run_id' or 's3_model_path').")

            logging.info(f"Successfully resolved Champion Run ID: {champion_run_id}")

            # Clean up temp file to prevent container bloat
            os.remove(local_tmp_pointer_path)

            return champion_run_id, s3_model_path

        except Exception as e:
            logging.exception("Failed to fetch or parse the production pointer.")
            raise CustomException(e, sys) from e

    def _download_champion_model(self, s3_model_path: str) -> None:
        """
        Pulls the actual serialized model from the S3 Vault to the local disk.
        """
        try:
            logging.info(f"Downloading Champion model from Vault: {s3_model_path}")
            self.s3_sync.download_file(
                s3_uri=s3_model_path,
                local_path=self.config.downloaded_model_file_path
            )
            logging.info(f"Model successfully saved to: {self.config.downloaded_model_file_path}")

        except Exception as e:
            logging.exception("Failed to download the physical model artifact.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(self, inference_uuid: str, champion_id: str, s3_path: str, execution_time: float) -> None:
        """
        Generates standard telemetry metadata for the pipeline component.
        """
        try:
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Inference: Registry Sync",
                "execution_time_seconds": execution_time,
                "context": {
                    "inference_run_uuid": inference_uuid,
                    "champion_run_id_loaded": champion_id,
                    "model_vault_source": s3_path
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)

        except Exception as e:
            logging.exception("Failed to generate registry sync metadata.")
            raise CustomException(e, sys) from e