import torch.nn as nn
from torchvision import models


def get_teacher(num_classes=100):
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_student(num_classes=100):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
