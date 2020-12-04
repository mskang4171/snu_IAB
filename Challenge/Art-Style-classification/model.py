import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):   
    def __init__(self):
        super(SimpleModel, self).__init__()
        layers = []
        in_channels = 3
        for v in [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        self.cnn_layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 5)
        )
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.pooling(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

