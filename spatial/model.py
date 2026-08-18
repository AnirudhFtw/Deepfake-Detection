"""
Spatial (RGB) branch — pretrained-ResNet18 backbone. Operates on the
normalized RGB face crops in processed/spatial/ produced by
pipeline/build_spatial_domain.py.
"""
import torch.nn as nn
from torchvision import models


class DeepfakeResNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )

        if freeze_backbone:
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def embed(self, x):
        """
        Returns the pre-classifier feature vector (pooled ResNet features,
        before the final Linear(2)) — this is the hook the temporal branch
        needs (see temporal/model.py, TemporalAttentionPool) and doesn't
        exist yet in the classification-only forward().
        """
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        return x.flatten(1)
