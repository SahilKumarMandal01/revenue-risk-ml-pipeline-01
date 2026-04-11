import sys, os
from datetime import datetime, timezone

from src import constants
from src.custom_exception import CustomException

class DataPipelineConfig:
    def __init__(self):
        try:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            self.root_dir: str = os.path.join(
                constants.ARTIFACT_DIR_NAME,
                constants.DATA_PIPELINE_ROOT_DIR_NAME,
                self.run_id
            )
            os.makedirs(self.root_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)


class DataPipelineExtractorConfig:
    def __init__(self, data_pipeline_config: DataPipelineConfig):
        try:
            self.extractor_root_dir: str = os.path.join(
                data_pipeline_config.root_dir,
                constants.EXTRACTOR_ROOT_DIR_NAME
            )
            self.raw_data_dir_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_RAW_DATA_DIR_NAME
            )
            self.raw_data_schema_file_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_RAW_DATA_SCHEMA_FILE_NAME
            )
            self.metadata_file_path: str = os.path.join(
                self.extractor_root_dir,
                constants.EXTRACTOR_METADATA_FILE_NAME
            )
        except Exception as e:
            raise CustomException(e, sys)