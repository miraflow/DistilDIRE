import torch.nn as nn
import torchvision.models as TVM
from collections import OrderedDict

import torch

# ------------------------------------------------------------------------------
# Model Definition
class DistilDIRE(torch.nn.Module):
    def __init__(self, device):
        super(DistilDIRE, self).__init__()
    
        # define models
        student = TVM.resnet50()
        self.student_backbone = nn.Sequential(OrderedDict([*(list(student.named_children())[:-2])])) # drop last layer which is classifier
        # extract last classifier head from teacher resnet 
        self.student_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                          nn.Flatten(),
                                          nn.Linear(2048, 1))
        self.student_backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.device = device
        
        
    def forward(self, img, eps):
        img = img.to(self.device)
        eps = eps.to(self.device)
    
        # concat image and noise
        img_tens = torch.cat([img, eps], dim=1)
        
        feature = self.student_backbone(img_tens) 
        logit = self.student_head(feature)
        return {'logit':logit, 'feature':feature}