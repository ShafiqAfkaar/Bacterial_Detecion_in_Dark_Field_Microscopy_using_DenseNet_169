import torch.nn as nn
from torchvision import models

class DenseNet169_Base(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNet169_Base, self).__init__()
        self.backbone = models.densenet169(pretrained=pretrained)
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)