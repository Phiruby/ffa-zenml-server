
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
from zenml.io import fileio

logger = get_logger(__name__)

@step(enable_cache=True)
def get_train_test_data(
    training_features: pd.Series,
    test_features: pd.Series, 
    train_image_batch_uri: str,
    test_image_batch_uri: str
    ) -> Tuple[Annotated[pd.DataFrame, "train_data"], Annotated[pd.DataFrame, "test_data"]]:
    
    logger.info("Reading train and test images uri before getting training and testing data with extracted features...")
    with fileio.open(train_image_batch_uri, 'rb') as f:
        train_file = np.load(train_image_batch_uri, allow_pickle=True)
        train_image_batch = train_file["images"]
        train_image_labels = train_file["labels"]
    
    with fileio.open(test_image_batch_uri, 'rb') as f:
        test_file = np.load(test_image_batch_uri, allow_pickle=True)
        test_image_batch = test_file["images"]
        test_image_labels = test_file["labels"]

    logger.info("Loaded train and test images uri before getting training and testing data with extracted features...")
    logger.info("Feature shapes are: %s, %s", str(training_features.values.shape), str(test_features.values.shape))
    logger.info("Sample feature shapes for train and test data: %s, %s", str(len(training_features.values[0])),
                 str(len(test_features.values[0])))
    train_labels, test_labels = train_image_labels, test_image_labels
    logger.info("Converting training data to pd dataframe...")

    training_data = pd.DataFrame({"images": training_features.values, "labels": train_labels})
    
    logger.info("Converting testing data to pd dataframe...")
    testing_data = pd.DataFrame({"images": test_features.values, "labels": test_labels})

    logger.info("Sample data frame shape of feature: "+str(len(training_data["images"][0])))
    return training_data, testing_data
