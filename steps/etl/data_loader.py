from typing import Tuple

import kaggle as kg
from sklearn.datasets import load_breast_cancer
from typing_extensions import Annotated
from zenml import step
import os 
from zenml.logger import get_logger

logger = get_logger(__name__)
@step
def download_images(
    path: str = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
    ) -> Annotated[str, "dataset_path"]:
    '''
    Downloads the Foods-5k dataset from Kaggle into the specified path.
    ---
    Args
        path: str = "../../data" - The path to download the dataset to
    '''
    kg.api.authenticate() # make sure to set KAGGLE_USERNAME and KAGGLE_KEY env variables
    
    data_dir = path
    # dataset_path = f"{data_dir}/Food-5K"
    dataset_path = os.path.join(data_dir, 'Food-5K')

    logger.info("Will save the images in "+str(dataset_path))

    if not os.path.exists(dataset_path):
        logger.info("Downloading food5k dataset...")
        kg.api.dataset_download_files(dataset = "binhminhs10/food5k", path = data_dir, unzip = True)
        logger.info("Downloaded food5k dataset.")
    else:
        logger.info("food5k dataset already exists. Skipping download.")
    # kaggle datasets download -d binhminhs10/food5k
    # train_path, eval_path, val_path = os.path.join(dataset_path, 'training'), os.path.join(dataset_path, 'evaluation'), os.path.join(dataset_path, 'validation')
    return dataset_path
