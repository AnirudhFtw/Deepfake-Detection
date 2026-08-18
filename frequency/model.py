"""
Frequency branch — a small CNN trained from scratch on the block-DCT
representation in processed/frequency/ (see
pipeline/build_frequency_domain.py).

Deliberately not the pretrained ImageNet ResNet18 used for the spatial
branch: ImageNet weights encode natural RGB-image statistics that a DCT
coefficient map doesn't share, so that pretraining isn't a useful prior
here (see tasks/IMPROVEMENT_PLAN.md, Phase 1). The frequency signal is
also lower-dimensional than natural image texture, so a small CNN is
enough and far cheaper to train.
"""
import torch.nn as nn


class DeepfakeFrequencyCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def embed(self, x):
        """Pre-classifier feature vector — same hook as spatial/model.py."""
        x = self.features(x)
        return self.pool(x).flatten(1)
