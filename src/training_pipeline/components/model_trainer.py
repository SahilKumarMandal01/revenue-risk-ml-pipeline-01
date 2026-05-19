# import os
# import sys
# import time
# import warnings
# import platform
# from datetime import datetime, timezone
# from typing import Dict, Any, Tuple

# import pandas as pd
# import numpy as np
# import joblib
# import mlflow
# import optuna
# import shap
# import matplotlib.pyplot as plt

# import xgboost
# import sklearn
# from xgboost import XGBClassifier
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import log_loss

# from src import constants
# from src.entity.config_entity import TrainingPipelineModelTrainerConfig
# from src.entity.artifact_entity import (
#     TrainingPipelineDataTransformationArtifact,
#     TrainingPipelineModelTrainerArtifact,
# )
# from src.custom_exception import CustomException
# from src.custom_logging import logging
# from src.utils.main_utils import write_json_file

# warnings.filterwarnings("ignore")


# class ModelTrainer:
#     """
#     Model Trainer component for the Training Pipeline.

#     Responsibilities:
#     - Load transformed datasets and the stateful schema enforcer (preprocessor).
#     - Perform cost-aware hyperparameter tuning using Optuna with XGBoost pruning.
#     - Calibrate the best XGBoost model using Isotonic Regression to output true probabilities.
#     - Assemble a self-contained Scikit-Learn "Mega-Pipeline" for deployment.
#     - Generate global business explainability plots using SHAP.
#     - Log experiments, parameters, metrics, models strictly via MLflow.
#     - Generate FAANG-grade observability metadata including data provenance and environment state.
#     """

#     def __init__(
#         self,
#         config: TrainingPipelineModelTrainerConfig,
#         transformation_artifact: TrainingPipelineDataTransformationArtifact,
#     ) -> None:
#         """
#         Initializes the Model Trainer component.
#         """
#         try:
#             self.config = config
#             self.transformation_artifact = transformation_artifact

#             os.makedirs(self.config.model_trainer_root_dir, exist_ok=True)
#             logging.info("Training Pipeline: Model Trainer component initialized.")

#         except Exception as e:
#             logging.exception("Failed to initialize Model Trainer component.")
#             raise CustomException(e, sys) from e

#     # ==========================================================
#     # PUBLIC ENTRYPOINT
#     # ==========================================================
#     def run(self) -> TrainingPipelineModelTrainerArtifact:
#         """
#         Executes the model training, calibration, and explainability pipeline.
#         """
#         try:
#             logging.info("Starting Model Training Pipeline.")
#             start_time = time.time()

#             # 1. Load Parquet Data and Preprocessor
#             X_train, y_train = self._load_data(
#                 self.transformation_artifact.x_train_file_path,
#                 self.transformation_artifact.y_train_file_path,
#             )
#             X_val, y_val = self._load_data(
#                 self.transformation_artifact.x_val_file_path,
#                 self.transformation_artifact.y_val_file_path,
#             )
#             preprocessor = joblib.load(self.transformation_artifact.preprocessor_file_path)

#             # Extract Data Provenance Shapes
#             train_rows, num_features = X_train.shape
#             val_rows, _ = X_val.shape

#             # Calculate Class Imbalance Weight dynamically
#             imbalance_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
#             logging.info(f"Calculated scale_pos_weight: {imbalance_weight:.2f}")

#             # 2. Setup MLflow Tracking Context
#             mlflow.set_experiment(self.config.mlflow_experiment_name)
            
#             with mlflow.start_run(run_name=f"XGB_Isotonic_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}") as run:
#                 run_id = run.info.run_id
#                 logging.info(f"MLflow Run ID: {run_id}")

#                 mlflow.log_param("imbalance_weight", imbalance_weight)

#                 # 3. Cost-Aware Hyperparameter Tuning (Optuna)
#                 best_params = self._optimize_hyperparameters(
#                     X_train, y_train, X_val, y_val, imbalance_weight
#                 )
#                 mlflow.log_params(best_params)

#                 # 4. Train Base Model & Calibrate
#                 calibrated_model, base_xgb = self._train_and_calibrate(
#                     X_train, y_train, best_params, imbalance_weight
#                 )

#                 # 5. Assemble Mega-Pipeline
#                 mega_pipeline = Pipeline([
#                     ("schema_enforcer", preprocessor),
#                     ("model", calibrated_model)
#                 ])

#                 # 6. Business Explainability (SHAP)
#                 self._generate_shap_summary(base_xgb, X_val)

#                 # 7. Serialize Artifacts & Log to MLflow
#                 joblib.dump(mega_pipeline, self.config.model_file_path)
#                 mlflow.sklearn.log_model(mega_pipeline, "mega_pipeline_model")
#                 mlflow.log_artifact(self.config.shap_summary_file_path, "explainability")

#                 # 8. Generate Enterprise-Grade Metadata
#                 execution_time = round(time.time() - start_time, 2)
#                 self._generate_metadata(
#                     best_params=best_params, 
#                     imbalance_weight=imbalance_weight, 
#                     run_id=run_id, 
#                     execution_time=execution_time,
#                     train_rows=train_rows,
#                     val_rows=val_rows,
#                     num_features=num_features
#                 )

#             # 9. Package Artifact
#             artifact = TrainingPipelineModelTrainerArtifact(
#                 model_file_path=self.config.model_file_path,
#                 shap_summary_file_path=self.config.shap_summary_file_path,
#                 metadata_file_path=self.config.metadata_file_path,
#             )

#             logging.info("Model Training completed successfully: %s", artifact)
#             return artifact

#         except Exception as e:
#             logging.exception("Model Training run failed.")
#             raise CustomException(e, sys) from e

#     # ==========================================================
#     # DATA LOADING
#     # ==========================================================
#     def _load_data(self, x_path: str, y_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
#         """
#         Loads feature matrices and target vectors from Parquet files.
#         """
#         try:
#             X = pd.read_parquet(x_path)
#             y = pd.read_parquet(y_path).values.ravel()  # Flatten for Scikit-Learn/XGBoost
#             return X, y
#         except Exception as e:
#             logging.exception("Failed to load training datasets.")
#             raise CustomException(e, sys) from e

#     # ==========================================================
#     # HYPERPARAMETER TUNING (OPTUNA)
#     # ==========================================================
#     def _optimize_hyperparameters(
#         self,
#         X_train: pd.DataFrame,
#         y_train: np.ndarray,
#         X_val: pd.DataFrame,
#         y_val: np.ndarray,
#         imbalance_weight: float
#     ) -> Dict[str, Any]:
#         """
#         Runs Optuna study with XGBoostPruningCallback to aggressively terminate 
#         underperforming trials and save compute resources.
#         """
#         try:
#             logging.info("Starting Optuna hyperparameter optimization.")

#             def objective(trial: optuna.Trial) -> float:
#                 params = {
#                     "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
#                     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
#                     "max_depth": trial.suggest_int("max_depth", 3, 7),
#                     "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#                     "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#                     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#                     "gamma": trial.suggest_float("gamma", 0.0, 5.0),
#                     "tree_method": "hist",
#                     "enable_categorical": True,
#                     "scale_pos_weight": imbalance_weight,
#                     "objective": "binary:logistic",
#                     "eval_metric": "logloss",
#                     "random_state": 42,
#                     "n_jobs": -1
#                 }

#                 pruning_callback = optuna.integration.XGBoostPruningCallback(
#                     trial, "validation_0-logloss"
#                 )

#                 model = XGBClassifier(**params, callbacks=[pruning_callback])
                
#                 # Fit with evaluation set for early stopping and pruning
#                 model.fit(
#                     X_train, y_train,
#                     eval_set=[(X_val, y_val)],
#                     verbose=False
#                 )

#                 preds = model.predict_proba(X_val)[:, 1]
#                 return log_loss(y_val, preds)

#             study = optuna.create_study(direction="minimize", study_name="XGBoost_Churn")
#             study.optimize(objective, n_trials=30)  # Capped for compute efficiency

#             logging.info("Optuna optimization completed. Best Log Loss: {:.4f}".format(study.best_value))
#             return study.best_params

#         except Exception as e:
#             logging.exception("Failed during Optuna optimization.")
#             raise CustomException(e, sys) from e

#     # ==========================================================
#     # MODEL TRAINING & CALIBRATION
#     # ==========================================================
#     def _train_and_calibrate(
#         self, 
#         X_train: pd.DataFrame, 
#         y_train: np.ndarray, 
#         best_params: Dict[str, Any], 
#         imbalance_weight: float
#     ) -> Tuple[CalibratedClassifierCV, XGBClassifier]:
#         """
#         Trains the final base XGBoost model and wraps it in Isotonic calibration 
#         to ensure outputs represent true financial probabilities.
#         """
#         try:
#             logging.info("Training base XGBoost model with optimal parameters.")
            
#             # 1. Initialize and train Base XGBoost
#             base_xgb = XGBClassifier(
#                 **best_params,
#                 tree_method="hist",
#                 enable_categorical=True,
#                 scale_pos_weight=imbalance_weight,
#                 objective="binary:logistic",
#                 random_state=42,
#                 n_jobs=-1
#             )
#             base_xgb.fit(X_train, y_train)

#             # 2. Calibrate using Isotonic Regression (5-fold CV)
#             logging.info("Applying Isotonic Probability Calibration via CalibratedClassifierCV.")
#             calibrated_model = CalibratedClassifierCV(
#                 estimator=base_xgb, 
#                 method="isotonic", 
#                 cv=5
#             )
#             calibrated_model.fit(X_train, y_train)

#             return calibrated_model, base_xgb

#         except Exception as e:
#             logging.exception("Failed to train and calibrate model.")
#             raise CustomException(e, sys) from e

#     # ==========================================================
#     # BUSINESS EXPLAINABILITY (SHAP)
#     # ==========================================================
#     def _generate_shap_summary(self, model: XGBClassifier, X_val: pd.DataFrame) -> None:
#         """
#         Generates global feature importance explanations to align model behavior 
#         with business intuition.
#         """
#         try:
#             logging.info("Generating SHAP feature importance summary.")
            
#             # Use a representative sample of Validation data for computational speed
#             sample_size = min(2000, len(X_val))
#             X_sample = X_val.sample(n=sample_size, random_state=42)

#             explainer = shap.TreeExplainer(model)
#             shap_values = explainer.shap_values(X_sample)

#             plt.figure(figsize=(10, 8))
#             shap.summary_plot(shap_values, X_sample, show=False)
#             plt.tight_layout()
            
#             plt.savefig(self.config.shap_summary_file_path, dpi=300)
#             plt.close()
            
#             logging.info("SHAP summary plot saved successfully.")

#         except Exception as e:
#             logging.exception("Failed to generate SHAP summary.")
#             raise CustomException(e, sys) from e

#     # ==========================================================
#     # OBSERVABILITY METADATA
#     # ==========================================================
#     def _generate_metadata(
#         self, 
#         best_params: Dict[str, Any], 
#         imbalance_weight: float, 
#         run_id: str, 
#         execution_time: float,
#         train_rows: int,
#         val_rows: int,
#         num_features: int
#     ) -> None:
#         """
#         Generates FAANG-level telemetry payload for downstream component auditing,
#         including crucial data provenance and environment states.
#         """
#         try:
#             logging.info("Generating Model Trainer observability metadata.")

#             metadata: Dict[str, Any] = {
#                 "pipeline_stage": "Model Training & Calibration",
#                 "execution_time_seconds": execution_time,
#                 "data_provenance": {
#                     "train_rows": train_rows,
#                     "val_rows": val_rows,
#                     "feature_count": num_features,
#                     "train_snapshots_used": constants.TRAIN_SNAPSHOTS,
#                     "val_snapshot_used": constants.VAL_SNAPSHOT
#                 },
#                 "environment_state": {
#                     "python_version": platform.python_version(),
#                     "xgboost_version": xgboost.__version__,
#                     "scikit_learn_version": sklearn.__version__
#                 },
#                 "mlflow_tracking": {
#                     "experiment_name": self.config.mlflow_experiment_name,
#                     "run_id": run_id
#                 },
#                 "model_architecture": {
#                     "base_estimator": "XGBClassifier (Hist Gradient Boosting)",
#                     "calibration_method": "Isotonic Regression",
#                     "calibration_cv_folds": 5,
#                     "scale_pos_weight": imbalance_weight
#                 },
#                 "optimal_hyperparameters": best_params,
#                 "timestamp": datetime.now(timezone.utc).isoformat()
#             }

#             write_json_file(file_path=self.config.metadata_file_path, content=metadata)

#         except Exception as e:
#             logging.exception("Failed to generate metadata.")
#             raise CustomException(e, sys) from e




import os
import sys
import time
import warnings
import platform
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import constants
from src.custom_exception import CustomException
from src.custom_logging import logging
from src.entity.artifact_entity import (
    TrainingPipelineDataTransformationArtifact,
    TrainingPipelineModelTrainerArtifact,
)
from src.entity.config_entity import TrainingPipelineModelTrainerConfig
from src.utils.main_utils import write_json_file

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Model Trainer component for the Training Pipeline.

    Responsibilities:
    - Load transformed datasets and the stateful schema enforcer (preprocessor).
    - Perform cost-aware hyperparameter tuning using Optuna with XGBoost pruning.
    - Calibrate the best XGBoost model using Isotonic Regression to output true probabilities.
    - Assemble a self-contained Scikit-Learn deployment pipeline.
    - Generate global business explainability plots using SHAP.
    - Generate observability metadata including data provenance and environment state.
    """

    def __init__(
        self,
        config: TrainingPipelineModelTrainerConfig,
        transformation_artifact: TrainingPipelineDataTransformationArtifact,
    ) -> None:
        """
        Initialize the Model Trainer component.
        """
        try:
            self.config = config
            self.transformation_artifact = transformation_artifact

            os.makedirs(self.config.model_trainer_root_dir, exist_ok=True)

            logging.info(
                "Training Pipeline: Model Trainer component initialized."
            )

        except Exception as e:
            logging.exception(
                "Failed to initialize Model Trainer component."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # PUBLIC ENTRYPOINT
    # ==========================================================
    def run(self) -> TrainingPipelineModelTrainerArtifact:
        """
        Execute the model training, calibration, and explainability pipeline.
        """
        try:
            logging.info("Starting Model Training Pipeline.")
            start_time = time.time()

            # 1. Load Parquet Data and Preprocessor
            X_train, y_train = self._load_data(
                self.transformation_artifact.x_train_file_path,
                self.transformation_artifact.y_train_file_path,
            )

            X_val, y_val = self._load_data(
                self.transformation_artifact.x_val_file_path,
                self.transformation_artifact.y_val_file_path,
            )

            preprocessor = joblib.load(
                self.transformation_artifact.preprocessor_file_path
            )

            # Extract data provenance shapes
            train_rows, num_features = X_train.shape
            val_rows, _ = X_val.shape

            # Calculate class imbalance weight dynamically
            imbalance_weight = float(
                (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            )

            logging.info(
                "Calculated scale_pos_weight: %.2f",
                imbalance_weight,
            )

            # 2. Hyperparameter Tuning (Optuna)
            best_params = self._optimize_hyperparameters(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                imbalance_weight=imbalance_weight,
            )

            # 3. Train Base Model & Calibrate
            calibrated_model, base_xgb = self._train_and_calibrate(
                X_train=X_train,
                y_train=y_train,
                best_params=best_params,
                imbalance_weight=imbalance_weight,
            )

            # 4. Assemble Deployment Pipeline
            mega_pipeline = Pipeline(
                steps=[
                    ("schema_enforcer", preprocessor),
                    ("model", calibrated_model),
                ]
            )

            # 5. Business Explainability (SHAP)
            self._generate_shap_summary(
                model=base_xgb,
                X_val=X_val,
            )

            # 6. Serialize Artifacts
            joblib.dump(
                mega_pipeline,
                self.config.model_file_path,
            )

            # 7. Generate Metadata
            execution_time = round(time.time() - start_time, 2)

            self._generate_metadata(
                best_params=best_params,
                imbalance_weight=imbalance_weight,
                execution_time=execution_time,
                train_rows=train_rows,
                val_rows=val_rows,
                num_features=num_features,
            )

            # 8. Package Artifact
            artifact = TrainingPipelineModelTrainerArtifact(
                model_file_path=self.config.model_file_path,
                shap_summary_file_path=self.config.shap_summary_file_path,
                metadata_file_path=self.config.metadata_file_path,
            )

            logging.info(
                "Model Training completed successfully: %s",
                artifact,
            )

            return artifact

        except Exception as e:
            logging.exception("Model Training run failed.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # DATA LOADING
    # ==========================================================
    def _load_data(
        self,
        x_path: str,
        y_path: str,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load feature matrices and target vectors from Parquet files.
        """
        try:
            X = pd.read_parquet(x_path)

            y = pd.read_parquet(y_path).values.ravel()

            return X, y

        except Exception as e:
            logging.exception("Failed to load training datasets.")
            raise CustomException(e, sys) from e

    # ==========================================================
    # HYPERPARAMETER TUNING (OPTUNA)
    # ==========================================================
    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        imbalance_weight: float,
    ) -> Dict[str, Any]:
        """
        Run Optuna study with XGBoost pruning to aggressively terminate
        underperforming trials and save compute resources.
        """
        try:
            logging.info(
                "Starting Optuna hyperparameter optimization."
            )

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        100,
                        500,
                        step=50,
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.1,
                        log=True,
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        3,
                        7,
                    ),
                    "min_child_weight": trial.suggest_int(
                        "min_child_weight",
                        1,
                        10,
                    ),
                    "subsample": trial.suggest_float(
                        "subsample",
                        0.6,
                        1.0,
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.6,
                        1.0,
                    ),
                    "gamma": trial.suggest_float(
                        "gamma",
                        0.0,
                        5.0,
                    ),
                    "tree_method": "hist",
                    "enable_categorical": True,
                    "scale_pos_weight": imbalance_weight,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": 42,
                    "n_jobs": -1,
                }

                pruning_callback = (
                    optuna.integration.XGBoostPruningCallback(
                        trial,
                        "validation_0-logloss",
                    )
                )

                model = XGBClassifier(
                    **params,
                    callbacks=[pruning_callback],
                )

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                predictions = model.predict_proba(X_val)[:, 1]

                return log_loss(y_val, predictions)

            study = optuna.create_study(
                direction="minimize",
                study_name="XGBoost_Churn",
            )

            study.optimize(
                objective,
                n_trials=30,
            )

            logging.info(
                "Optuna optimization completed. "
                "Best Log Loss: %.4f",
                study.best_value,
            )

            return study.best_params

        except Exception as e:
            logging.exception(
                "Failed during Optuna optimization."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # MODEL TRAINING & CALIBRATION
    # ==========================================================
    def _train_and_calibrate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        best_params: Dict[str, Any],
        imbalance_weight: float,
    ) -> Tuple[CalibratedClassifierCV, XGBClassifier]:
        """
        Train the final XGBoost model and apply Isotonic calibration
        to improve probability estimates.
        """
        try:
            logging.info(
                "Training base XGBoost model with optimal parameters."
            )

            # Initialize and train base XGBoost model
            base_xgb = XGBClassifier(
                **best_params,
                tree_method="hist",
                enable_categorical=True,
                scale_pos_weight=imbalance_weight,
                objective="binary:logistic",
                random_state=42,
                n_jobs=-1,
            )

            base_xgb.fit(X_train, y_train)

            # Apply Isotonic Regression calibration
            logging.info(
                "Applying Isotonic Probability Calibration."
            )

            calibrated_model = CalibratedClassifierCV(
                estimator=base_xgb,
                method="isotonic",
                cv=5,
            )

            calibrated_model.fit(X_train, y_train)

            return calibrated_model, base_xgb

        except Exception as e:
            logging.exception(
                "Failed to train and calibrate model."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # BUSINESS EXPLAINABILITY (SHAP)
    # ==========================================================
    def _generate_shap_summary(
        self,
        model: XGBClassifier,
        X_val: pd.DataFrame,
    ) -> None:
        """
        Generate global feature importance explanations
        aligned with business intuition.
        """
        try:
            logging.info(
                "Generating SHAP feature importance summary."
            )

            # Use representative validation sample for speed
            sample_size = min(2000, len(X_val))

            X_sample = X_val.sample(
                n=sample_size,
                random_state=42,
            )

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(10, 8))

            shap.summary_plot(
                shap_values,
                X_sample,
                show=False,
            )

            plt.tight_layout()

            plt.savefig(
                self.config.shap_summary_file_path,
                dpi=300,
            )

            plt.close()

            logging.info(
                "SHAP summary plot saved successfully."
            )

        except Exception as e:
            logging.exception(
                "Failed to generate SHAP summary."
            )
            raise CustomException(e, sys) from e

    # ==========================================================
    # OBSERVABILITY METADATA
    # ==========================================================
    def _generate_metadata(
        self,
        best_params: Dict[str, Any],
        imbalance_weight: float,
        execution_time: float,
        train_rows: int,
        val_rows: int,
        num_features: int,
    ) -> None:
        """
        Generate telemetry metadata for downstream auditing,
        reproducibility, and observability.
        """
        try:
            logging.info(
                "Generating Model Trainer observability metadata."
            )

            metadata: Dict[str, Any] = {
                "pipeline_stage": "Model Training & Calibration",
                "execution_time_seconds": execution_time,
                "data_provenance": {
                    "train_rows": train_rows,
                    "val_rows": val_rows,
                    "feature_count": num_features,
                    "train_snapshots_used": (
                        constants.TRAIN_SNAPSHOTS
                    ),
                    "val_snapshot_used": (
                        constants.VAL_SNAPSHOT
                    ),
                },
                "environment_state": {
                    "python_version": (
                        platform.python_version()
                    ),
                    "xgboost_version": (
                        xgboost.__version__
                    ),
                    "scikit_learn_version": (
                        sklearn.__version__
                    ),
                },
                "model_architecture": {
                    "base_estimator": (
                        "XGBClassifier "
                        "(Hist Gradient Boosting)"
                    ),
                    "calibration_method": (
                        "Isotonic Regression"
                    ),
                    "calibration_cv_folds": 5,
                    "scale_pos_weight": (
                        imbalance_weight
                    ),
                },
                "optimal_hyperparameters": best_params,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            write_json_file(
                file_path=self.config.metadata_file_path,
                content=metadata,
            )

        except Exception as e:
            logging.exception(
                "Failed to generate metadata."
            )
            raise CustomException(e, sys) from e