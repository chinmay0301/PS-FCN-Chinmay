import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import utils

def convUnit(batchNorm, cin, cout, kernel_size=3, stride=1, pad=-1, dilation=1):
    pad = (kernel_size - 1) // 2 if pad < 0 else pad
    if batchNorm:
        conv2d  = nn.Conv2d(cin, cout, stride=stride, padding=pad, kernel_size=kernel_size, bias=False, dilation=dilation)
        kaiming_normal_(conv2d.weight)
        batchnorm2d = nn.BatchNorm2d(cout)
        batchnorm2d.weight.data.fill_(1)
        batchnorm2d.bias.data.zero_()
        return nn.Sequential(
                conv2d,
                batchnorm2d,
                nn.LeakyReLU(0.1, inplace=True))
    else:
        conv2d = nn.Conv2d(cin, cout, stride=stride, padding=pad, kernel_size=kernel_size, bias=True)
        kaiming_normal_(conv2d.weight)
        return nn.Sequential(
                conv2d,
                nn.LeakyReLU(0.1, inplace=True))

def deconvUnit(cin, cout):
    conv2dT = nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False)
    kaiming_normal_(conv2dT.weight)
    return nn.Sequential(
            conv2dT,
            nn.LeakyReLU(0.1, inplace=True))

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3):
        super(FeatExtractor, self).__init__()
        self.conv1 = convUnit(batchNorm, c_in, 64,  kernel_size=3, stride=1, pad=1, dilation=1)
        self.conv2 = convUnit(batchNorm, 64,   128, kernel_size=3, stride=2, pad=1)
        self.conv3 = convUnit(batchNorm, 128,  128, kernel_size=3, stride=1, pad=1)
        self.conv4 = convUnit(batchNorm, 128,  256, kernel_size=3, stride=2, pad=1)
        self.conv5 = convUnit(batchNorm, 256,  256, kernel_size=3, stride=1, pad=1)
        self.conv6 = deconvUnit(256, 128)
        self.conv7 = convUnit(batchNorm, 128, 128, kernel_size=3, stride=1, pad=1)

    def forward(self, var):
        out = self.conv1(var)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        feat_vec = self.conv7(out)
        shape= feat_vec.data.shape
        feat_vec   = feat_vec.view(-1)
        return feat_vec, shape

class Regressor(nn.Module):
    def __init__(self, batchNorm=False): 
        super(Regressor, self).__init__()
        self.deconv1 = convUnit(batchNorm, 128, 128,  kernel_size=3, stride=1, pad=1)
        self.deconv2 = convUnit(batchNorm, 128, 128,  kernel_size=3, stride=1, pad=1)
        self.deconv3 = deconvUnit(128, 64)
        self.normal_out = nn.Sequential(nn.Conv2d(64, 3, stride=1, padding=1, kernel_size=3, bias=False),
                                        nn.LeakyReLU(0.1, inplace=True))
    def forward(self, var, shape):
        var = var.view(shape)
        out = self.deconv1(var)
        out = self.deconv2(out)
        out = self.deconv3(out)
        normal = self.normal_out(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class PS_FCN(nn.Module):
    def __init__(self, batchNorm=False, c_in=3):
        super(PS_FCN, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in)
        self.regressor = Regressor(batchNorm)
        self.c_in      = c_in

    def forward(self, var):
        image = torch.split(var[0], 3, 1)
        if len(var) > 1: # Lighting directions present
            light_dir = torch.split(var[1], 3, 1)

        features = []
        for i in range(len(image)):
            inp = image[i] if len(var) == 1 else torch.cat([image[i], light_dir[i]], 1)
            feat_vec, shape = self.featExtractor(inp)
            features.append(feat_vec)
        fused_features, _ = torch.stack(features, 1).max(1)
        normal = self.regressor(fused_features, shape)
        return normal

class PS_FCN_run(nn.Module):
    def __init__(self, batchNorm=False, c_in=3):
        super(PS_FCN_run, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in)
        self.regressor = Regressor(batchNorm)
        self.c_in      = c_in

    def forward(self, var):
        image = torch.split(var[0], 3, 1)
        if len(var) > 1: # Lighting directions present
            light_dir = torch.split(var[1], 3, 1)

        features = torch.Tensor()
        for i in range(len(image)):
            if len(var) == 1:
                inp = image[i] 
            else:
                inp = torch.cat([image[i], light_dir[i]], 1)
            feat_vec, shape = self.featExtractor(inp)
            if i == 0:
                features = feat_vec
            else:
                features, _ = torch.stack([features, feat_vec], 1).max(1)
        fused_features = features
        normal = self.regressor(fused_features, shape)
        return normal
