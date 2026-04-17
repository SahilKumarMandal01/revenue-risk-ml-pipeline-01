import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import polars as pl
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.entity.config_entity import TrainingPipelineDataTransformationConfig
from src.entity.artifact_entity import (
    TrainingPipelineDataIngestionArtifact,
    TrainingPipelineDataTransformationArtifact,
)
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.utils.main_utils import write_json_file


# ==========================================================
# CUSTOM STATEFUL TRANSFORMERS
# ==========================================================
class CategoricalSchemaEnforcer(BaseEstimator, TransformerMixin):
    """
    Stateful Scikit-Learn transformer to strictly enforce Pandas categorical types.
    
    Why this is required over a stateless function:
    XGBoost relies on the underlying integer codes of Pandas categories. If a validation
    set is missing a category present in the train set, a stateless conversion will misalign
    the integer codes, causing the model to learn and predict incorrect patterns silently.
    
    This transformer learns the exact categorical schema (including unique classes) during 
    the `.fit()` stage on the training data, and applies those exact mappings during `.transform()`.
    Unknown categories in validation/test sets are safely coerced to NaN, which XGBoost 
    handles natively.
    """

    def __init__(self) -> None:
        self.schema_: Dict[str, pd.CategoricalDtype] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoricalSchemaEnforcer":
        """
        Learns the exact categorical mappings from the training feature matrix.
        """
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "string":
                # Extract unique categories and create a strict Pandas CategoricalDtype
                unique_categories = X[col].dropna().unique()
                self.schema_[col] = pd.CategoricalDtype(
                    categories=unique_categories, ordered=False
                )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """
        Applies the learned categorical mappings to the incoming feature matrix.
        """
        X_out = X.copy()
        for col, cat_dtype in self.schema_.items():
            if col in X_out.columns:
                # astype with a strict CategoricalDtype aligns the internal integer codes
                # perfectly with the training set. Unknown classes become NaN.
                X_out[col] = X_out[col].astype(cat_dtype)
        return X_out


class DataTransformation:
    """
    Data Transformation component for the Training Pipeline.

    Responsibilities:
    - Act as a Strict Schema Enforcer rather than a heavy math processor.
    - Isolate features (X) from the target variable (y) and system metadata for ALL data splits.
    - Build and fit a stateful categorical preprocessor on the training data.
    - Apply the transformation safely to the Validation and Test sets to ensure schema alignment.
    - Serialize the Scikit-Learn Pipeline as `preprocessor.pkl`.
    - Save the fully transformed X and y arrays as Parquet files for the Model Training stage.
    """

    def __init__(
        self,
        config: TrainingPipelineDataTransformationConfig,
        ingestion_artifact: TrainingPipelineDataIngestionArtifact,
    ) -> None:
        """
        Initializes the Data Transformation component.
        """
        try:
            self.config = config
            self.ingestion_artifact = ingestion_artifact

            os.makedirs(self.config.data_transformation_root_dir, exist_ok=True)
            logging.info("Training Pipeline: Data Transformation component initialized.")

        except Exception as e:
            logging.exception("Failed to initialize Data Transformation component.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> TrainingPipelineDataTransformationArtifact:
        """
        Executes the data transformation pipeline across all temporal splits.
        """
        try:
            logging.info("Starting Data Transformation (Stateful Schema Enforcement).")
            start_time = time.time()

            # 1. Load data from the ingestion artifacts
            train_df = self._load_data(self.ingestion_artifact.train_data_path)
            val_df = self._load_data(self.ingestion_artifact.val_data_path)
            test_df = self._load_data(self.ingestion_artifact.test_data_path)

            # 2. Isolate Features (X) and Target (y)
            X_train, y_train = self._isolate_features_and_target(train_df, "Train")
            X_val, y_val = self._isolate_features_and_target(val_df, "Validation")
            X_test, y_test = self._isolate_features_and_target(test_df, "Test")

            # 3. Build and Fit Pipeline exclusively on Training data
            logging.info("Fitting stateful categorical preprocessor on Train split.")
            preprocessor_pipeline = Pipeline(
                steps=[("schema_enforcer", CategoricalSchemaEnforcer())]
            )
            X_train_transformed = preprocessor_pipeline.fit_transform(X_train)

            # 4. Transform Validation and Test data safely
            logging.info("Applying learned transformations to Validation and Test splits.")
            X_val_transformed = preprocessor_pipeline.transform(X_val)
            X_test_transformed = preprocessor_pipeline.transform(X_test)

            # 5. Serialize Artifacts to Disk
            self._serialize_preprocessor(preprocessor_pipeline)
            self._save_transformed_datasets(
                X_train_transformed, y_train,
                X_val_transformed, y_val,
                X_test_transformed, y_test
            )

            # 6. Generate Observability Metadata
            execution_time = round(time.time() - start_time, 2)
            self._generate_metadata(X_train_transformed, execution_time)

            # 7. Package and Return Artifact
            artifact = TrainingPipelineDataTransformationArtifact(
                preprocessor_file_path=self.config.preprocessor_file_path,
                metadata_file_path=self.config.metadata_file_path,
                x_train_file_path=self.config.x_train_file_path,
                y_train_file_path=self.config.y_train_file_path,
                x_val_file_path=self.config.x_val_file_path,
                y_val_file_path=self.config.y_val_file_path,
                x_test_file_path=self.config.x_test_file_path,
                y_test_file_path=self.config.y_test_file_path,
            )

            logging.info("Data Transformation completed successfully: %s", artifact)
            return artifact

        except Exception as e:
            logging.exception("Data Transformation run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA PROCESSING OPERATIONS
    # ==========================================================
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads a Parquet file into a Pandas DataFrame using Polars for fast I/O.
        """
        try:
            return pl.read_parquet(file_path).to_pandas()
        except Exception as e:
            logging.exception("Failed to load data from %s", file_path)
            raise CustomException(e, sys) from e

    def _isolate_features_and_target(self, df: pd.DataFrame, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separates features (X) from the target (y) and drops system metadata.
        """
        try:
            logging.debug("Isolating features and target for %s split.", split_name)
            
            # Extract target vector as a DataFrame for easy Parquet serialization later
            if self.config.target_column not in df.columns:
                raise ValueError(f"Target column '{self.config.target_column}' missing in {split_name} split.")
            y = df[[self.config.target_column]].copy()

            # Drop target and metadata columns to create feature matrix
            cols_to_drop = self.config.columns_to_drop + [self.config.target_column]
            cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
            
            X = df.drop(columns=cols_to_drop_existing)
            return X, y

        except Exception as e:
            logging.exception("Failed to isolate features for %s split.", split_name)
            raise CustomException(e, sys) from e

    # ==========================================================
    # ARTIFACT SERIALIZATION
    # ==========================================================
    def _serialize_preprocessor(self, pipeline: Pipeline) -> None:
        """
        Saves the fitted Scikit-Learn pipeline to disk via Joblib.
        """
        try:
            logging.info("Serializing preprocessor pipeline to %s", self.config.preprocessor_file_path)
            joblib.dump(pipeline, self.config.preprocessor_file_path)
        except Exception as e:
            logging.exception("Failed to serialize the preprocessor.")
            raise CustomException(e, sys) from e

    def _save_transformed_datasets(
        self,
        X_train: pd.DataFrame, y_train: pd.DataFrame,
        X_val: pd.DataFrame, y_val: pd.DataFrame,
        X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> None:
        """
        Saves the processed X and y DataFrames as Parquet files.
        """
        try:
            logging.info("Saving transformed feature matrices and target vectors to disk.")
            
            X_train.to_parquet(self.config.x_train_file_path, index=False)
            y_train.to_parquet(self.config.y_train_file_path, index=False)
            
            X_val.to_parquet(self.config.x_val_file_path, index=False)
            y_val.to_parquet(self.config.y_val_file_path, index=False)
            
            X_test.to_parquet(self.config.x_test_file_path, index=False)
            y_test.to_parquet(self.config.y_test_file_path, index=False)
            
        except Exception as e:
            logging.exception("Failed to save transformed datasets.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY
    # ==========================================================
    def _generate_metadata(self, X_train: pd.DataFrame, execution_time: float) -> None:
        """
        Generates observability telemetry detailing the strict API contract.
        """
        try:
            logging.info("Generating Data Transformation observability metadata.")

            input_features = X_train.columns.tolist()
            # In Pandas, checking dtype == 'category' is the safest way to find categories post-transformation
            categorical_columns = [col for col in input_features if X_train[col].dtype.name == 'category']
            numerical_columns = [col for col in input_features if col not in categorical_columns]

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Training Data Transformation",
                "execution_time_seconds": execution_time,
                "strategy": "Stateful Schema Enforcement (XGBoost Native Support)",
                "api_contract": {
                    "total_features": len(input_features),
                    "input_features": input_features,
                    "categorical_columns": categorical_columns,
                    "numerical_columns": numerical_columns,
                },
                "dropped_system_columns": self.config.columns_to_drop,
                "target_column": self.config.target_column,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_json_file(file_path=self.config.metadata_file_path, content=metadata)
            
        except Exception as e:
            logging.exception("Failed to generate metadata.")
            raise CustomException(e, sys) from e