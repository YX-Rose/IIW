

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
import numpy as np

class Generator(nn.Module):
    def __init__(self, ngf=64, input_channel=3, output_channel=3, n_downsampling=4):
        super(Generator, self).__init__()

        self.ngf = ngf
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.n_downsampling = n_downsampling

        self.enc = Encoder(self.ngf, self.input_channel, self.output_channel, self.n_downsampling)
        self.dec = Decoder(self.ngf, self.input_channel, self.output_channel, self.n_downsampling)

        output_layer = []

        output_layer += [nn.ReflectionPad2d(3)]
        output_layer += [nn.Conv2d(ngf, output_channel, kernel_size=7, padding=0)]
        output_layer += [nn.Tanh()]

        self.output_layer = nn.Sequential(*output_layer)

    def forward(self, input):
        # embeddings
        feature_emb = self.enc(input)

        identity_emb = feature_emb[:,:256]

        # feature_map
        x = self.dec(feature_emb)

        rgb_image = self.output_layer(x)
        return feature_emb, rgb_image, identity_emb

class Encoder(nn.Module):
    def __init__(self, ngf, input_channel, output_channel, n_downsampling):
        super(Encoder, self).__init__()
        conv_layers = []
        conv_layers = [
            conv_unit(input_channel, ngf//2),  #B*32*128*128
            conv_unit(ngf//2, ngf),            #B*64*128*128
            conv_unit(ngf, ngf, pooling=True),  #B*64*64*64
        ]

        for i in range(n_downsampling):
            mult = 2 ** i
            conv_layers += [
                conv_unit(ngf * mult, ngf * mult * 2),  # B*128*64*64
                conv_unit(ngf * mult * 2, ngf * mult * 2, pooling=True),  # B*128*32*32
            ]
#B*32*128*128   B*64*128*128   B*64*64*64   B*128*64*64  B*128*32*32   B*256*32*32   B*256*16*16
#B*512*16*16    B*512*8*8      B*1024*8*8   B*1024*4*4

        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(1024*4*4, 256 + 50)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.reshape(-1, 1024*4*4)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, ngf, input_channel, output_channel, n_downsampling):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256 + 50, 1024 * 4 * 4)
        # ngf * mult, int(ngf * mult / 2)
        Fconv_layers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Fconv_layers += [
                Fconv_unit(ngf * mult, ngf * mult, unsampling=True),
                Fconv_unit(ngf * mult, int(ngf * mult / 2)),
            ]

        Fconv_layers += [
            Fconv_unit(ngf, ngf, unsampling=True),
        ]
#B*1024*8*8   B*512*8*8   B*512*16*16   B*256*16*16   B*256*32*32   B*128*32*32   B*128*64*64   B*64*64*64
#B*64*128*128   B*32*128*128   B*3*128*128
        self.Fconv_layers = nn.Sequential(*Fconv_layers)

    def forward(self, input):
        x = self.fc(input)
        x = x.reshape(-1, 1024, 4, 4)
        x = self.Fconv_layers(x)
        return x

class Fconv_unit(nn.Module):
    def __init__(self, in_channels, out_channels, unsampling=False):
        super(Fconv_unit, self).__init__()

        if unsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class conv_unit(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=False):
        super(conv_unit, self).__init__()

        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ReLU(True)])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class View(nn.Module):
    def __init__(self, C, H, W):
        super(View, self).__init__()
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        if self.H is None and self.W is None:
            x = x.view(x.size(0), self.C)
        elif self.W is None:
            x = x.view(x.size(0), self.C, self.H)
        else:
            x = x.view(x.size(0), self.C, self.H, self.W)
        return x


class NLayerDiscriminator(nn.Module):

    def __init__(self, image_size=128, conv_dim=64, repeat_num=6):
        super(NLayerDiscriminator, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)

        # conv1 = list()
        # conv1.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        # conv1.append(nn.AdaptiveAvgPool2d([4, 4]))
        # self.conv1 = nn.Sequential(*conv1)

        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src


class MultiTaskDiscriminator(nn.Module):

    def __init__(self, image_size=128, conv_dim=64, repeat_num=6, c_dim=8):
        super(MultiTaskDiscriminator, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        conv1 = list()
        conv1.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        conv1.append(nn.AdaptiveAvgPool2d([4, 4]))
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


if __name__ == '__main__':

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    x = torch.rand((2, 3, 256, 256))
    model = MultiTaskDiscriminator(image_size=256, repeat_num=4)
    real, attr = model(x)
    # print(real.size())
    # print(attr.size())

    x = torch.rand((2, 3, 128, 128))
    # model = Generator()
    # real = model(x)
    # print(real.size())

    y = torch.rand((2,256))
    conv = Generator()
    inter, real, a = conv(x)
    print(a.size())



