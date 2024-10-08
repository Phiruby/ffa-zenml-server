from typing import Dict, Tuple
from zenml import step
from torchvision import models, transforms
from typing_extensions import Annotated
from PIL import Image
import torch
import pandas as pd
from zenml.logger import get_logger
import numpy as np 
from zenml.io import fileio
from zenml.config.docker_settings import DockerSettings

logger = get_logger(__name__)

# or use "poetry_export"
docker_settings = DockerSettings(
    requirements=["torch==2.4.0", "torchvision==0.19.0"],
    # python_package_installer_args={"default-timeout": 600}
    python_package_installer="uv"

    # requirements="requirements.txt"
)
@step(enable_cache=True, settings={
    "docker": docker_settings
    })
def feature_extractor_step(
    train_images_path: str,
    test_images_path: str
    ) -> Tuple[
        Annotated[pd.Series, "train_features"],
        Annotated[pd.Series, "test_features"]
        ]:
    """Extract features from images using a pre-trained model."""

    logger.info("Reading from the URI before extracting features...")
    with fileio.open(train_images_path, 'rb') as f:
        train_images_np_arrays = np.load(f, allow_pickle=True)["images"]
    with fileio.open(test_images_path, 'rb') as f:
        test_images_np_arrays = np.load(f, allow_pickle=True)["images"]
    
    logger.info("Starting Feature Extraction...")


    logger.info("Getting feature series now...")
    test_feature_series = get_feature_series(test_images_np_arrays)
    train_feature_series = get_feature_series(train_images_np_arrays)

    return train_feature_series, test_feature_series

def get_feature_series(images_np_arrays: np.ndarray) -> pd.Series:
    # Load a pre-trained ResNet model without the final classification layer
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize directly to the input size
        transforms.ToTensor(),  # Convert the image to a tensor
    ])

    features = []
    for idx, image_np in enumerate(images_np_arrays):
        if idx % 100 == 0:
            logger.info(f"Extracting feature {idx+1}/{len(images_np_arrays)} ({(idx+1)/len(images_np_arrays)*100:.2f}%)")

        image_pil = Image.fromarray(image_np)
        # logger.info("Converted image to PIL array")

        input_tensor = preprocess(image_pil).unsqueeze(0)

        # logger.info("Preprocessed image to get the features")
        with torch.no_grad():
            feature = model(input_tensor).flatten().numpy()
        
        # logger.info("Appending the feature")
        features.append(feature.flatten())

    logger.info("Sample shape of the features: %s", str(features[0].shape)+", "+str(features[1].shape))
    logger.info("Final shape of the features: %s", str((len(features), len(features[0]))))
    # Convert features to DataFrame
    feature_df = pd.Series(features)
    
    logger.info("Returning feature series... Shape of the series is %s", str(feature_df.shape))
    return feature_df
