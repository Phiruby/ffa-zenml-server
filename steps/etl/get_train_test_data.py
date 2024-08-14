
from typing import List, Optional, Tuple
from typing_extensions import Annotated
from zenml.logger import get_logger
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from zenml import step
import os
from PIL import Image
import numpy as np 

logger = get_logger(__name__)

@step(enable_cache=True)
def get_train_test_data(
    training_features: pd.Series,
    test_features: pd.Series, 
    train_image_batch: pd.DataFrame,
    test_image_batch: pd.DataFrame
    ) -> Tuple[Annotated[pd.DataFrame, "train_data"], Annotated[pd.DataFrame, "test_data"]]:
    
    train_labels, test_labels = train_image_batch["labels"], test_image_batch["labels"]
    logger.info("Converting training data to pd dataframe...")

    training_data = pd.DataFrame({"images": training_features, "labels": train_labels})
    
    logger.info("Converting testing data to pd dataframe...")
    testing_data = pd.DataFrame({"images": test_features, "labels": test_labels})
    return training_data, testing_data