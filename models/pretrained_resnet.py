import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class PretrainedResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, num_classes=0, arch='R18'):
        super(PretrainedResNet, self).__init__()

        if arch=='R18':
            model = models.resnet18(pretrained=True)
            model.fc = Identity()
        elif arch=='R50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 512)
        
        self.encoder = model
        # Note: torchvision pretrained model is slightly different from ours, 
        # when training CUB, using torchvision model will be more memory efficient

        dim_in = 512
        self.fc = nn.Linear(dim_in, num_classes)
        # self.head = nn.Sequential(
        #         nn.Linear(dim_in, dim_in),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(dim_in, 128)
        #     )
          

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.fc(feat)
        return logits, F.normalize(feat, dim=1)