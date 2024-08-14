from typing import Tuple

import kaggle as kg
from sklearn.datasets import load_breast_cancer
from typing_extensions import Annotated
from zenml import step
import os 
from zenml.logger import get_logger

logger = get_logger(__name__)
@step
def download_images(path: str = "../../data") -> None:
    '''
    Downloads the Foods-5k dataset from Kaggle into the specified path.
    ---
    Args
        path: str = "../../data" - The path to download the dataset to
    '''
    kg.api.authenticate() # make sure to set KAGGLE_USERNAME and KAGGLE_KEY env variables
    
    data_dir = path
    dataset_path = f"{data_dir}/food5k"
    if not os.path.exists(dataset_path):
        logger.info("Downloading food5k dataset...")
        kg.api.dataset_download_files(dataset = "binhminhs10/food5k", path = data_dir, unzip = True)
        logger.info("Downloaded food5k dataset.")
    else:
        logger.info("food5k dataset already exists. Skipping download.")
    # kaggle datasets download -d binhminhs10/food5k
