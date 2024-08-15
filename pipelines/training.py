from typing import List, Optional, Any, Dict
import random
import pandas as pd 
from zenml.enums import ModelStages
# from steps import (
#     download_images,
#     get_image_batch_np_array,
#     feature_extractor_step,
#     hp_select_best_model,
#     hp_tuning_single_search,
#     model_trainer,
#     model_evaluator,
#     compute_performance_metrics_on_current_data,
#     promote_with_metric_compare
# )
from steps.etl.data_loader import download_images
from steps.etl.get_image_np_array import get_image_batch_np_array 
from steps.etl.extract_features import feature_extractor_step
from steps.etl.get_train_test_data import get_train_test_data
from steps.hp_tuning.hp_search import hp_tuning_single_search
from steps.hp_tuning.hp_select_best_model import hp_tuning_select_best_model
from steps.training.model_trainer import model_trainer
from steps.training.model_evaluator import model_evaluator
from steps.promotion.promote_with_metrics import promote_with_metric_compare
from steps.promotion.compute_performance_metrics import compute_performance_metrics_on_current_data

from zenml import Model, pipeline
from zenml.logger import get_logger
from zenml.artifacts.external_artifact import ExternalArtifact
logger = get_logger(__name__)
@pipeline(model=Model(
        name="train_food_pipeline",
        license="Apache",
        description="Show case Model Control Plane.",
    ))
def training_pipeline():
    # --- ETL --- #
    dataset_path = download_images()

    train_df_uri, test_df_uri = get_image_batch_np_array(dataset_path)
    # validation_images, validation_labels = get_image_batch_np_array(validation_path)

    # --- Feature Extractor --- #
    training_features, testing_features = feature_extractor_step(
        train_df_uri,
        test_df_uri,
    )
    
    # training_data = pd.DataFrame({"images": training_features, "labels": training_data["labels"]})
    # testing_data = pd.DataFrame({"images": testing_features, "labels": testing_data["labels"]})
    training_data, testing_data = get_train_test_data(training_features, testing_features, train_df_uri, test_df_uri)
    # --- Hyperparameter Tuner --- #
    step_name = "hp_tuning_search"
    best_model = hp_tuning_single_search(
        "sklearn.linear_model",
        "LogisticRegression",
        search_grid={
            "penalty": ["l2", "l1"],
            "solver": ["saga"]
        },
        dataset_trn=training_data,
        dataset_tst=testing_data,
        target="labels",
        id=step_name
    )
    # best_model = hp_tuning_select_best_model([step_name], after=[step_name])

    # --- Model Trainer --- #
    model = model_trainer(
        dataset=training_data,
        model=best_model,
        target_col="labels",
        model_name="food_classifier_lr",
    )

    # --- Model Evaluator --- #
    model_evaluator(
        model=model,
        dataset_trn=training_data,
        dataset_tst=testing_data,
        target="labels",
    )

    # --- Promote in model registry --- #
    latest_metric,current_metric = compute_performance_metrics_on_current_data(
        dataset_tst=testing_data,
        target_env="development",
        after=["model_evaluator"]
    )

    promote_with_metric_compare(
        latest_metric=latest_metric,
        current_metric=current_metric,
        mlflow_model_name="food_classifier_lr",
        target_env=ModelStages.STAGING,
    )