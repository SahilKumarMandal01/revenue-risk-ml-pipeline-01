import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any

from src.entity.config_entity import TrainingPipelineModelRegistryConfig
from src.entity.artifact_entity import (
    TrainingPipelineModelTrainerArtifact,
    TrainingPipelineModelEvaluationArtifact,
    TrainingPipelineModelRegistryArtifact,
)
from src.cloud.s3_operations import S3Sync
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.utils.main_utils import write_json_file


class ModelRegistry:
    """
    Model Registry component for the Training Pipeline.

    Responsibilities:
    - Act as the deployment execution layer.
    - Check the approval status from the Model Evaluation component.
    - Terminate safely if the Challenger model was rejected.
    - If approved, bundle the model, hyperparameters, and evaluation report.
    - Upload the bundle to a permanent, immutable S3 folder (The Vault).
    - Update the mutable production pointer file in S3 (Zero-Downtime Deployment).
    - Maintain an automated rollback pointer (Linked List approach) and schema versioning.
    """

    def __init__(
        self,
        config: TrainingPipelineModelRegistryConfig,
        trainer_artifact: TrainingPipelineModelTrainerArtifact,
        evaluation_artifact: TrainingPipelineModelEvaluationArtifact,
    ) -> None:
        """
        Initializes the Model Registry component.
        """
        try:
            self.config = config
            self.trainer_artifact = trainer_artifact
            self.evaluation_artifact = evaluation_artifact
            self.s3_sync = S3Sync()

            os.makedirs(self.config.model_registry_root_dir, exist_ok=True)
            logging.info("Training Pipeline: Model Registry component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Model Registry component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> TrainingPipelineModelRegistryArtifact:
        """
        Executes the registry and deployment pipeline based on evaluation gating.
        """
        try:
            logging.info("Starting Model Registry component.")
            start_time = time.time()

            # 1. The Gatekeeper Check
            approval_status = self.evaluation_artifact.approval_status

            previous_run_id = "None"
            
            if not approval_status:
                logging.warning("Challenger model was REJECTED. Halting deployment process.")
                s3_model_uri = "N/A"
                deployment_status = False
            else:
                logging.info("Challenger model APPROVED. Initiating production deployment.")
                
                # 2. Extract Deployment Metrics
                metrics = self._extract_evaluation_metrics()
                
                # Fetch rollback pointer before overwriting state
                previous_run_id = self._get_previous_champion_run_id()
                
                # 3. Register Immutable Artifacts and Update Pointer
                s3_model_uri = self._register_model(metrics, previous_run_id)
                deployment_status = True

            # 4. Generate Enterprise-Grade Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(
                deployment_status=deployment_status, 
                s3_model_uri=s3_model_uri, 
                execution_time=execution_time,
                previous_run_id=previous_run_id
            )

            # 5. Package Artifact
            artifact = TrainingPipelineModelRegistryArtifact(
                s3_model_uri=s3_model_uri,
                metadata_file_path=self.config.metadata_file_path,
                deployment_status=deployment_status,
            )

            logging.info("Model Registry execution completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Model Registry run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # METRICS EXTRACTION & STATE RETRIEVAL
    # ==========================================================
    def _extract_evaluation_metrics(self) -> Dict[str, float]:
        """
        Extracts key business and statistical metrics from the evaluation report
        to populate the production pointer file.
        """
        try:
            with open(self.evaluation_artifact.report_file_path, "r") as f:
                report_data = json.load(f)

            challenger_metrics = report_data.get("challenger_metrics", {})
            
            return {
                "eroi": challenger_metrics.get("eroi", 0.0),
                "log_loss": challenger_metrics.get("log_loss", 0.0)
            }

        except Exception as e:
            logging.exception("Failed to extract evaluation metrics.")
            raise CustomException(e, sys) from e

    def _get_previous_champion_run_id(self) -> str:
        """
        Retrieves the run ID of the currently deployed Champion model.
        This forms the 'Linked List' allowing for instantaneous 1-click rollbacks.
        """
        try:
            local_tmp_path = os.path.join(self.config.model_registry_root_dir, "tmp_pointer_for_rollback.json")
            try:
                self.s3_sync.download_file(self.config.s3_pointer_file_uri, local_tmp_path)
                with open(local_tmp_path, "r") as f:
                    pointer_data = json.load(f)
                os.remove(local_tmp_path)
                
                previous_run_id = pointer_data.get("champion_run_id", "None")
                logging.info(f"Retrieved previous Champion Run ID for rollback state: {previous_run_id}")
                return previous_run_id
                
            except Exception:
                logging.info("No previous pointer file found in S3. Assuming Cold Start (No rollback available).")
                return "None"
                
        except Exception as e:
            logging.warning("Could not establish rollback pointer due to a system error. Defaulting to 'None'.")
            return "None"

    # ==========================================================
    # CORE DEPLOYMENT LOGIC (S3 INTERACTIONS)
    # ==========================================================
    def _register_model(self, metrics: Dict[str, float], previous_run_id: str) -> str:
        """
        Handles the dual-phase deployment: archiving the immutable bundle 
        and updating the mutable state pointer.
        """
        try:
            # 1. Generate Unique Deployment ID
            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            s3_run_dir_uri = f"{self.config.s3_models_dir_uri}/{run_id}"
            logging.info("Generated deployment Run ID: %s", run_id)

            # 2. Upload Immutable Artifacts (The Vault)
            logging.info("Uploading immutable artifact bundle to S3 Vault: %s", s3_run_dir_uri)
            
            self.s3_sync.upload_file(
                local_path=self.trainer_artifact.model_file_path,
                s3_uri=f"{s3_run_dir_uri}/model.pkl"
            )
            self.s3_sync.upload_file(
                local_path=self.trainer_artifact.metadata_file_path,
                s3_uri=f"{s3_run_dir_uri}/trainer_metadata.json"
            )
            self.s3_sync.upload_file(
                local_path=self.evaluation_artifact.report_file_path,
                s3_uri=f"{s3_run_dir_uri}/evaluation_report.json"
            )

            # 3. Update the Production Pointer (Zero-Downtime Deployment)
            self._update_production_pointer(run_id, previous_run_id, metrics)

            logging.info("Deployment successful. Model promoted to Champion.")
            return s3_run_dir_uri

        except Exception as e:
            logging.exception("Failed to register model and update deployment pointer.")
            raise CustomException(e, sys) from e

    def _update_production_pointer(self, run_id: str, previous_run_id: str, metrics: Dict[str, float]) -> None:
        """
        Creates and uploads the lightweight state JSON file that downstream 
        Inference APIs will query to locate the active Champion model.
        """
        try:
            logging.info("Updating mutable production pointer in S3.")
            
            pointer_payload = {
                "schema_version": 1,  # Hardcoded contract version for downstream API validation
                "champion_run_id": run_id,
                "previous_champion_run_id": previous_run_id,
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                "eroi_baseline": metrics["eroi"],
                "log_loss_baseline": metrics["log_loss"],
                "s3_model_path": f"{self.config.s3_models_dir_uri}/{run_id}/model.pkl"
            }

            local_pointer_path = os.path.join(
                self.config.model_registry_root_dir, "production_champion.json"
            )
            
            write_json_file(file_path=local_pointer_path, content=pointer_payload)

            self.s3_sync.upload_file(
                local_path=local_pointer_path,
                s3_uri=self.config.s3_pointer_file_uri
            )
            
            logging.info("Production pointer successfully updated at: %s", self.config.s3_pointer_file_uri)

        except Exception as e:
            logging.exception("Failed to update the production pointer file.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(
        self, 
        deployment_status: bool, 
        s3_model_uri: str, 
        execution_time: float, 
        previous_run_id: str
    ) -> None:
        """
        Generates standard telemetry metadata for the pipeline component,
        including rollback state preservation.
        """
        try:
            metadata: Dict[str, Any] = {
                "pipeline_stage": "Model Registry & Deployment",
                "execution_time_seconds": execution_time,
                "deployment_executed": deployment_status,
                "s3_vault_uri": s3_model_uri,
                "state_management": {
                    "rollback_pointer_preserved": previous_run_id != "None",
                    "previous_champion": previous_run_id,
                    "schema_version": 1
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            write_json_file(file_path=self.config.metadata_file_path, content=metadata)

        except Exception as e:
            logging.exception("Failed to generate registry metadata.")
            raise CustomException(e, sys) from e