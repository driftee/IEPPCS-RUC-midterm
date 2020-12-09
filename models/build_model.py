from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
import torchvision
from models import resnet101


def Resnet101(num_classes, test=False):
    model = resnet101()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model