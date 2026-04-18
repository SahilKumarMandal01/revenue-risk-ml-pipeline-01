from dataclasses import dataclass


# ==========================================================
# DATA PIPELINE ARTIFACTS
# ==========================================================

@dataclass(frozen=True)
class DataPipelineExtractorArtifact:
    """
    Artifact containing paths for the Extractor component outputs.
    """
    raw_data_dir_path: str
    raw_data_schema_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineExtractorArtifact(\n"
            f"  raw_data_dir_path = {self.raw_data_dir_path}\n"
            f"  raw_data_schema_file_path = {self.raw_data_schema_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataPipelineValidatorArtifact:
    """
    Artifact containing the validation report path and boolean status.
    """
    report_file_path: str
    is_valid: bool

    def __str__(self) -> str:
        return (
            "\nDataPipelineValidatorArtifact(\n"
            f"  report_file_path = {self.report_file_path}\n"
            f"  is_valid = {self.is_valid}\n"
            ")"
        )


@dataclass(frozen=True)
class DataPipelineTransformerArtifact:
    """
    Artifact containing paths for the Transformer component outputs.
    Contains the path to the definitive out-of-core generated Parquet file.
    """
    transformed_data_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineTransformerArtifact(\n"
            f"  transformed_data_file_path = {self.transformed_data_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class DataPipelineLoaderArtifact:
    """
    Artifact containing remote (S3) path for the exported feature store
    and local path for the loader's telemetry metadata.
    """
    s3_file_uri: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineLoaderArtifact(\n"
            f"  s3_file_uri = {self.s3_file_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


# ==========================================================
# TRAINING PIPELINE ARTIFACTS
# ==========================================================

@dataclass(frozen=True)
class TrainingPipelineDataIngestionArtifact:
    """
    Artifact containing paths for the Out-Of-Time (OOT) temporal data splits.
    """
    train_data_path: str
    val_data_path: str
    test_data_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineDataIngestionArtifact(\n"
            f"  train_data_path = {self.train_data_path}\n"
            f"  val_data_path = {self.val_data_path}\n"
            f"  test_data_path = {self.test_data_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class TrainingPipelineDataTransformationArtifact:
    """
    Artifact containing paths for the serialized preprocessor, schema metadata,
    and the fully transformed Feature Matrices (X) and Target Vectors (y) ready for model training.
    """
    preprocessor_file_path: str
    metadata_file_path: str
    x_train_file_path: str
    y_train_file_path: str
    x_val_file_path: str
    y_val_file_path: str
    x_test_file_path: str
    y_test_file_path: str

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineDataTransformationArtifact(\n"
            f"  preprocessor_file_path = {self.preprocessor_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  x_train_file_path = {self.x_train_file_path}\n"
            f"  y_train_file_path = {self.y_train_file_path}\n"
            f"  x_val_file_path = {self.x_val_file_path}\n"
            f"  y_val_file_path = {self.y_val_file_path}\n"
            f"  x_test_file_path = {self.x_test_file_path}\n"
            f"  y_test_file_path = {self.y_test_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class TrainingPipelineModelTrainerArtifact:
    """
    Artifact containing the path to the definitive Scikit-Learn Mega-Pipeline (model.pkl),
    the SHAP global summary plot, and the MLflow run metadata.
    """
    model_file_path: str
    shap_summary_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineModelTrainerArtifact(\n"
            f"  model_file_path = {self.model_file_path}\n"
            f"  shap_summary_file_path = {self.shap_summary_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")"
        )


@dataclass(frozen=True)
class TrainingPipelineModelEvaluationArtifact:
    """
    Artifact containing the evaluation report (Champion vs. Challenger metrics),
    component metadata, and the critical deployment gating boolean.
    """
    report_file_path: str
    metadata_file_path: str
    approval_status: bool

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineModelEvaluationArtifact(\n"
            f"  report_file_path = {self.report_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  approval_status = {self.approval_status}\n"
            ")"
        )


@dataclass(frozen=True)
class TrainingPipelineModelRegistryArtifact:
    """
    Artifact containing the remote S3 URI of the newly registered model bundle,
    local component metadata, and the final state of the deployment action.
    """
    s3_model_uri: str
    metadata_file_path: str
    deployment_status: bool

    def __str__(self) -> str:
        return (
            "\nTrainingPipelineModelRegistryArtifact(\n"
            f"  s3_model_uri = {self.s3_model_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  deployment_status = {self.deployment_status}\n"
            ")"
        )