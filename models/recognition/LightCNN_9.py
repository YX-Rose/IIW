
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import os
from PIL import Image


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class _LightCNN9(nn.Module):
    def __init__(self, num_classes):
        super(_LightCNN9, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(8 * 8 * 128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        feature = x
        out = self.fc2(feature)
        return out, feature


def LightCNN9(pretrained=True, model_path=''):
    netIP = _LightCNN9(num_classes=79077)
    netIP = torch.nn.DataParallel(netIP).cuda()
    if pretrained:
        assert model_path is not None
        pretrained_dict = torch.load(os.path.expanduser(model_path))['state_dict']
        model_dict = netIP.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        netIP.load_state_dict(model_dict)
    return netIP


def convert_color(input_tensor):
    r, g, b = input_tensor[:, 0, :, :], input_tensor[:, 1, :, :], input_tensor[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = torch.unsqueeze(gray, dim=1)
    return gray


def extract_feature(feature_extractor, image_path):
    # read the image from the path and use lightcnn to extract the identity representation
    image = Image.open(image_path).convert('L')
    image_transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(128), transforms.ToTensor()])
    image = image_transform(image)
    image = torch.unsqueeze(image, dim=0)
    image = image.cuda()

    _, embeddings = feature_extractor(image)
    return embeddings.data.cpu().numpy()
