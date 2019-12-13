import torch.nn as nn
import os
import torchvision.transforms as transforms


from network.pose_gan import *
from PIL import Image

def extract_feature_myModel(feature_extractor, image_path):
    # read the image from the path and use our model to extract the feature embedding
    image = Image.open(image_path)
    image_emb_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    image_emb = image_emb_transform(image)
    image_emb = torch.unsqueeze(image_emb, dim=0)
    image_emb = image_emb.cuda()

    embeddings, _ = feature_extractor(image_emb)
    return embeddings.data.cpu().numpy()


def myModel(model_path=''):
    generator = Generator()
    generator = torch.nn.DataParallel(generator).cuda()
    assert model_path is not None
    # generator.load_state_dict_gpu(torch.load(g_path))
    generator.load_state_dict(torch.load(model_path))

    return generator

def load_state_dict_gpu(path):
    # state_dict = torch.load(path)

    # if torch.cuda.device_count() > 1:
    #     return state_dict
    # else:
    #     from collections import OrderedDict
    #     # create new OrderedDict that does not contain `module.`
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]  # remove `module.`
    #         new_state_dict[name] = v
    #     # load params
    #     return new_state_dict

    state_dict = torch.load(path)
    from collections import OrderedDict
    # create new OrderedDict that contains `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module" not in k:
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        else:
            name = k  # add `module.`
            new_state_dict[name] = v
    return new_state_dict

