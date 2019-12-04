
# pretrained pytorch model : http://www.robots.ox.ac.uk/~albanie/pytorch-models.html
# original model : http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

import os
from PIL import Image

import torch
import torch.nn as nn

import torchvision.transforms as transforms

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def VGGFace(model_path=''):

    model_path = os.path.expanduser(model_path)
    assert os.path.isfile(model_path)
    weights = torch.load(model_path)

    model = VGG(make_layers(cfg['D']))
    model.load_state_dict(weights)
    model = model.cuda()
    return model


def extract_feature(feature_extractor, image_path):

    image = Image.open(image_path).convert('RGB')

    image_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ])

    image = image_transform(image)
    image = torch.unsqueeze(image, dim=0)
    image = image.cuda()

    embeddings = feature_extractor(image)
    return embeddings.data.cpu().numpy()
