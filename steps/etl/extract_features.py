from zenml import step
from torchvision import models, transforms
from typing_extensions import Annotated
from PIL import Image
import torch
import pandas as pd

@step
def feature_extractor_step(images_np_arrays: pd.Series) -> Annotated[pd.Series, "features"]:
    """Extract features from images using a pre-trained model."""
    
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
    for image_np in images_np_arrays:
        input_tensor = preprocess(image_np).unsqueeze(0)
        with torch.no_grad():
            feature = model(input_tensor).flatten().numpy()
        features.append(feature)

    # Convert features to DataFrame
    feature_df = pd.Series(features)
    
    return feature_df
