from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class DataPipelineExtractorArtifact:
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
    master_panel_df: pd.DataFrame
    metadata_file_path: str

    def __str__(self) -> str:
        return (
            "\nDataPipelineTransformerArtifact(\n"
            f"  metadata_file_path = {self.metadata_file_path}\n"
            f"  master_panel_shape = {self.master_panel_df.shape}\n"
            ")\n"
        )


@dataclass(frozen=True)
class DataPipelineLoaderArtifact:
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