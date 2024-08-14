from typing import Dict, Tuple
from zenml import step
from torchvision import models, transforms
from typing_extensions import Annotated
from PIL import Image
import torch
import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)
@step(enable_cache=True)
def feature_extractor_step(
    train_images_np_arrays: pd.DataFrame,
    test_images_np_arrays: pd.DataFrame
    ) -> Tuple[
        Annotated[pd.Series, "train_features"],
        Annotated[pd.Series, "test_features"]
        ]:
    """Extract features from images using a pre-trained model."""
    
    logger.info("Starting Feature Extraction...")
    train_images_np_arrays = train_images_np_arrays["images"]
    test_images_np_arrays = test_images_np_arrays["images"]

    train_feature_series = get_feature_series(train_images_np_arrays)
    test_feature_series = get_feature_series(test_images_np_arrays)

    return train_feature_series, test_feature_series

def get_feature_series(images_np_arrays: pd.DataFrame) -> pd.Series:
    # Load a pre-trained ResNet model without the final classification layer
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Define image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for idx, image_np in enumerate(images_np_arrays):
        logger.info(f"Extracting feature {idx+1}/{len(images_np_arrays)} ({(idx+1)/len(images_np_arrays)*100:.2f}%)")
        input_tensor = preprocess(image_np).unsqueeze(0)
        with torch.no_grad():
            feature = model(input_tensor).flatten().numpy()
        features.append(feature)

    # Convert features to DataFrame
    feature_df = pd.Series(features)
    
    return feature_df
