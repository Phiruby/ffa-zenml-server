
from typing import List, Optional, Tuple, DefaultDict, Dict
from typing_extensions import Annotated
from zenml.logger import get_logger
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from zenml import step
import os
from PIL import Image
import numpy as np 
from zenml.materializers import PandasMaterializer

logger = get_logger(__name__)

@step(enable_cache=True)
def get_image_batch_np_array(
    food_data_path: str
    ) -> Tuple[
        Annotated[str, "train_np_img"], 
        Annotated[str, "test_np_img"]
        ]:
    '''
    Returns the training data as a pd Dataframe of np array.
    ---
    Args
        food_data_path: str - The path to the food 5k dataset (should contain training and evaluation folders)
    ---
    Returns
        A tuple of the training and test data as URIs (use np.load(uri.path) to load the data)
    '''
    # get the training path
    logger.info("Getting image np arrays from path "+str(food_data_path)+"...")
    
    training_images_path = os.path.join(food_data_path, 'training')
    testing_images_path = os.path.join(food_data_path, 'evaluation')
    dirs = []
    for root, dirs_, files in os.walk('/'):
        dirs += [os.path.join(root, d) for d in dirs_]
    logger.info("All directories in the machine are: %s", str(dirs))

    train_df = convert_to_np_array(training_images_path)
    test_df = convert_to_np_array(testing_images_path)
    logger.info("Got image np arrays... Now returning train and test data frames")
    logger.info("Types are (train and test df respectively) "+str(type(train_df))+" And "+str(type(test_df)))

    train_path = os.path.join(food_data_path, "train_image_np_array")
    test_path = os.path.join(food_data_path, "test_image_np_array")

    np.savez(train_path, images=train_df['images'].values, labels=train_df['labels'].values)
    np.savez(test_path, images=test_df['images'].values, labels=test_df['labels'].values)

    return train_path+'.npz', test_path+'.npz'
    

def convert_to_np_array(images_folder_path: str) -> pd.DataFrame:
    '''
    Returns the training data as a pd DataFrame of np arrays.
    ---
    Args:
        images_folder_path: str - The path that consists of all images to convert to np arrays.
    ---
    Returns:
        A pandas DataFrame containing the image data:
            images: list of np.array, where each array represents the RGB image (height, width, channels)
            labels: list of int, where each int is the label of the image
    '''
    logger.info("Getting image np arrays from path "+str(images_folder_path)+"...")

    images_list = []
    labels_list = []
    num_files = len(os.listdir(images_folder_path))
    for idx, filename in enumerate(os.listdir(images_folder_path)):
        if idx % 200 == 0:
            logger.info(f"Converting image {idx+1}/{num_files} ({(idx+1)/num_files*100:.2f}%): {filename}")
        img = Image.open(os.path.join(images_folder_path, filename)).convert('RGB')
        img_np = np.array(img)  # Preserve the RGB channels
        images_list.append(img_np)
        
        # Assign a label based on filename
        if filename.startswith('0'):
            labels_list.append(0)
        elif filename.startswith('1'):
            labels_list.append(1)
    
    # Create a pandas DataFrame
    df = pd.DataFrame({"images": images_list, "labels": labels_list})
    return df
