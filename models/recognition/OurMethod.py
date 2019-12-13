import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms

from network.pose_gan import Generator
from PIL import Image

def extract_feature_our(feature_extractor, image_path):
    # read the image from the path and use our model to extract the feature embedding
    image = Image.open(image_path)
    image_transform = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ])
    image = image_transform(image)
    image = torch.unsqueeze(image, dim=0)
    image = image.cuda()

    embeddings, _ = feature_extractor(image)
    return embeddings.data.cpu().numpy()

def OurMethod(model_path=''):
    assert model_path is not None
    generator = Generator()
    # generator = torch.nn.DataParallel(generator).cuda()
    generator.load_state_dict(torch.load(model_path))
    generator.cuda()
    generator.eval()

    return generator

if __name__ == '__main__':
    net = OurMethod(model_path="../../pretrained/OurMethod/model.pth.tar")