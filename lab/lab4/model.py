import torch
import torch.nn as nn

# Lightweight models as specified in README
from torchvision.models import resnet18, mobilenet_v2

def cifar_resnet18():
    model = resnet18(weights=None)
    # adjust conv to accept 32x32 RGB image
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # remove final linear, the output dim is 512 for resnet18
    model.fc = nn.Identity()
    return model

def cifar_mobilenet_v2():
    model = mobilenet_v2(weights=None)
    # adjust conv to accept 32x32 RGB image
    model.features[0][0].stride = (1, 1)
    # remove final linear, the output dim is 1280 for mobilenet_v2
    model.classifier = nn.Identity()
    return model

class SimCLRModel(nn.Module):
    def __init__(self,
                 encoder: str = 'resnet18',
                 head_hidden_dim = 128,
                 head_use_batchnorm = True,
                 projection_dim = 64,
                 ):
        super().__init__()
        
        if encoder == 'resnet18':
            self.encoder = cifar_resnet18()
            self.feature_dim = 512
        elif encoder == 'mobilenet_v2':
            self.encoder = cifar_mobilenet_v2()
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        head_layers = [
            nn.Linear(self.feature_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, projection_dim),
        ]
        if head_use_batchnorm:
            head_layers.insert(1, nn.BatchNorm1d(head_hidden_dim))
        self.projection_head = nn.Sequential(*head_layers)
        
        self.classifier = nn.Linear(self.feature_dim, 1)
        
    def forward(self, x, mode='projection'):
        features = self.encoder(x)
        if mode == 'projection':
            return self.projection_head(features)
        elif mode == 'classification':
            return self.classifier(features)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
