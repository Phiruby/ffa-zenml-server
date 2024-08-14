from typing import List, Optional, Tuple
from typing_extensions import Annotated

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from zenml import step
import os
from PIL import Image
import numpy as np 

def get_batch_np_array(food_data_path: str) -> Annotated[pd.DataFrame, "image_np_array"]:
    '''
    Returns the training data as a pd Dataframe of np array.
    ---
    Args
        food_data_path: str - The path that consist of all images to convert to np array
    ---
    Returns
        A pandas DataFrame containing the training data.
            images: list of np.array, where the np,array is the image (the np.array is flattened)
            labels: list of int, where the int is the label of the image
    '''
    # get the training path
    training_images_path = food_data_path
    # loop through all images in training_images_path and get their corresponding np array
    images_list = []
    labels_list = []
    for filename in os.listdir(training_images_path):
        img = Image.open(os.path.join(training_images_path, filename))
        img_np = np.array(img)
        img_np = img_np.flatten()
        images_list.append(img_np)
        # for each image, add a label
        if filename.startswith('0'):
            labels_list.append(0)
        elif filename.startswith('1'):
            labels_list.append(1)
    
    # create a pandas dataframe
    df = pd.DataFrame({"images": images_list, "labels": labels_list})
    return df
