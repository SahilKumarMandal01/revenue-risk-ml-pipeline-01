import os, sys

from src.entity.config_entity import (
    TrainingPipelineConfig,
    TrainingPipelineDataIngestionConfig,
    TrainingPipelineDataTransformationConfig,
)

from src.training_pipeline.components.data_ingestion import DataIngestion
from src.training_pipeline.components.data_transformation import DataTransformation

from src.custom_exception import CustomException
from src.custom_logging import logging


if __name__ == "__main__":
    try:
        pipeline_config = TrainingPipelineConfig()

        # step 1: data ingestion
        ingestion_config = TrainingPipelineDataIngestionConfig(pipeline_config)
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.run()
        print(ingestion_artifact)

        # step 2: data tansformation
        transformation_config = TrainingPipelineDataTransformationConfig(pipeline_config)
        transformation = DataTransformation(
            transformation_config,
            ingestion_artifact
        )
        transformation_artifact = transformation.run()
        print(transformation_artifact)
        
    except Exception as e:
        raise CustomException(e, sys)