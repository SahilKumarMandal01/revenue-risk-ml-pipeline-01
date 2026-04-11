from dataclasses import dataclass


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
            ")"
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
            ")"
        )

