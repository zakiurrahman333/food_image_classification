import torch
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_effnetb2(num_classes):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1408, num_classes)
    )

    return model.to(device)