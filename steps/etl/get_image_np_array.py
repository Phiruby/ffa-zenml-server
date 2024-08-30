
import io
from typing import List, Optional, Tuple, DefaultDict, Dict
from typing_extensions import Annotated
from zenml.logger import get_logger
import pandas as pd
from zenml import step
import os
from PIL import Image
import numpy as np 
from zenml.client import Client
import boto3

logger = get_logger(__name__)

@step(enable_cache=False)
def get_image_batch_np_array(
    food_data_path: str,
    bucket_name: str = "foodforall-zenml-artifact-store"
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

    artifact_store = Client().active_stack.artifact_store
    if artifact_store and artifact_store.flavor == "s3": # if s3 artifact store, save in s3
        logger.info("S3 artifact store found. Reading Images from S3.")
        
        train_path = convert_to_np_array_s3(bucket_name, food_data_path, 'training')
        test_path = convert_to_np_array_s3(bucket_name, food_data_path, 'evaluation')
        logger.info("Finished converting images from S3 to np arrays.")
        # Save the npz files to S3
        s3_upload_path = "s3://{}/{}".format(bucket_name, "image_np_arrays")
        logger.info("Uploading npz file to S3. S3 upload path is "+str(s3_upload_path))
    
    else: 
        training_images_path = os.path.join(food_data_path, 'training')
        testing_images_path = os.path.join(food_data_path, 'evaluation')
        dirs = []
        for root, dirs_, files in os.walk('/'):
            dirs += [os.path.join(root, d) for d in dirs_]
        logger.info("All directories in the machine are: %s", str(dirs))

        train_df = convert_to_np_array(training_images_path)
        test_df = convert_to_np_array(testing_images_path)
        logger.info("Got image np arrays... Now returning train and test data frames")

        train_path = os.path.join(food_data_path, "train_image_np_array.npz")
        test_path = os.path.join(food_data_path, "test_image_np_array.npz")

        np.savez(train_path, images=train_df['images'].values, labels=train_df['labels'].values)
        np.savez(test_path, images=test_df['images'].values, labels=test_df['labels'].values)

    return train_path, test_path
        

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

def convert_to_np_array_s3(bucket_name: str, bucket_path: str, subfolder: str, batch_size: int = 32) -> str:
    '''
    Process S3 images in batches.
    Returns the URI to npz file.
    ---
    Args:
        bucket_name: str - The name of the S3 bucket containing the images.
        bucket_path: str - The path in the S3 bucket where the train / test / val folders are located.
        subfolder: str - The subfolder within the bucket where the images are stored (eg: training / testing / validation folders).
    ---
    Returns:
        A pandas DataFrame containing the image data:
            images: list of np.array, where each array represents the RGB image (height, width, channels)
            labels: list of int, where each int is the label of the image
    '''
    s3_client = boto3.client('s3')
    if bucket_path.startswith('s3://'):
        bucket_path = bucket_path[5:]
        bucket_path = bucket_path[bucket_path.find('/')+1:]

    prefix = os.path.join(bucket_path, subfolder) + '/'

    s3_output_key = f"{bucket_path}/{subfolder}_image_np_array.npz"
    s3_output_path = f"s3://{bucket_name}/{s3_output_key}"
    

    all_files = []
    continuation_token = None

    while True:
        list_params = {
            'Bucket': bucket_name,
            'Prefix': prefix
        }
        if continuation_token:
            list_params['ContinuationToken'] = continuation_token

        s3_objects = s3_client.list_objects_v2(**list_params)
        
        if 'Contents' not in s3_objects:
            break

        all_files.extend([obj['Key'] for obj in s3_objects['Contents'] if obj['Key'].endswith(('.jpg', '.png', '.jpeg'))])

        if not s3_objects.get('IsTruncated'):
            break
        continuation_token = s3_objects['NextContinuationToken']

    total_batches = len(all_files) // batch_size + (1 if len(all_files) % batch_size else 0)

    all_images = []
    all_labels = []

    for batch in range(total_batches):
        logger.info(f"Processing batch {batch+1}/{total_batches}")
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(all_files))
        
        batch_files = all_files[start_idx:end_idx]
        images_list = []
        labels_list = []

        for s3_key in batch_files:
            try:
                s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                img_data = s3_response['Body'].read()
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                img_np = np.array(img)
                images_list.append(img_np)
                labels_list.append(0 if os.path.basename(s3_key).startswith('0') else 1)
            except Exception as e:
                logger.error(f"Error processing file {s3_key}: {str(e)}")

        all_images.extend(images_list)
        all_labels.extend(labels_list)

    logger.info("Saving processed data to NPZ file in S3")
    
    # Save the NPZ file to S3
    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, images=np.array(all_images), labels=np.array(all_labels))
        buffer.seek(0)
        s3_client.put_object(Bucket=bucket_name, Key=s3_output_key, Body=buffer.getvalue())

    logger.info(f"NPZ file saved to {s3_output_path}")

    return s3_output_path