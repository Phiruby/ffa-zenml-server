from typing_extensions import Annotated

import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from zenml import ArtifactConfig, log_artifact_metadata, step, get_step_context
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from zenml.logger import get_logger
import numpy as np 

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Have you configured an MLflow experiment tracker in your stack? Seems like you haven't. Check out https://docs.zenml.io/stack-components/experiment-trackers/mlflow"
    )

@step(enable_cache=False, experiment_tracker=experiment_tracker.name) 
def model_trainer(
    dataset: pd.DataFrame,
    model: ClassifierMixin,
    target_col: str,
    model_name: str
) -> Annotated[ClassifierMixin, ArtifactConfig(name="model", is_model_artifact=True)]:
    """Trains a model using the specified algorithm and saves the model to disk."""
    logger.info("Starting model training process...")

    mlflow.sklearn.autolog()

    logger.info("Flattening training data before fitting...")
    X = np.vstack(dataset["images"].values)  # Shape: (n_samples, n_features)
    y = dataset[target_col]

    logger.info("Training model...")
    model.fit(
        X, y
    )
    logger.info("Finished training! Registering on MLFlow")

    mlflow_register_model_step.entrypoint(
        model,
        name=model_name,
    )

    # add to mlflow metadata
    model_registry = Client().active_stack.model_registry
    if model_registry:
        version = model_registry.get_latest_model_version(name=model_name, stage=None)
        if version:
            try:
                model_ = get_step_context().model
                model_.log_metadata({"model_registry_version": version.version})

                logger.info("MODEL VERSION IS: {}".format(version.version))
            except Exception as e:
                logger.warning(f"Failed to log metadata to MLFlow while training the model: {e}")
    
    logger.info("Returning the model")
    return model