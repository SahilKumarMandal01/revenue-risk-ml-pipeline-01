from dataclasses import dataclass


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
            ")\n"
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
            ")\n"
        )


@dataclass(frozen=True)
class DataPipelineTransformerArtifact:
    """
    Artifact containing paths for the Transformer component outputs.
    Note: master_panel_df has been removed to prevent Out-Of-Memory (OOM) 
    errors, adhering to the out-of-core DuckDB processing strategy.
    """
    transformed_data_file_path: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineTransformerArtifact(\n"
            f"  transformed_data_file_path = {self.transformed_data_file_path}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")\n"
        )


@dataclass(frozen=True)
class DataPipelineLoaderArtifact:
    """
    Artifact containing local and remote (S3) paths for the exported feature store.
    """
    local_file_path: str
    s3_file_uri: str
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineLoaderArtifact(\n"
            f"  local_file_path = {self.local_file_path}\n"
            f"  s3_file_uri = {self.s3_file_uri}\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            ")\n"
        )