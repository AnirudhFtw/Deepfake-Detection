import torch
import torch.nn as nn
from torchvision import models


class DeepfakeResNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )

        # Unfreeze classifier if backbone is frozen
        if freeze_backbone:
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)