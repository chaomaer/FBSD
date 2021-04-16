import torch
import torch.nn as nn
from torchvision.models import densenet161, resnet50, resnet101
from config import HyperParams

class FSBM(nn.Module):
    def __init__(self, in_channel, k):
        super(FSBM, self).__init__()
        self.k = k
        self.stripconv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, fm):
        b, c, w, h = fm.shape
        fms = torch.split(fm, w // self.k, dim=2)
        fms_conv = map(self.stripconv, fms)
        fms_pool = list(map(self.avgpool, fms_conv))
        fms_pool = torch.cat(fms_pool, dim=2)
        fms_softmax = torch.softmax(fms_pool, dim=2)  # every parts has one score [B*C*K*1]
        fms_softmax_boost = torch.repeat_interleave(fms_softmax, w // self.k, dim=2)
        alpha = HyperParams['alpha']
        fms_boost = fm + alpha*(fm * fms_softmax_boost)

        beta = HyperParams['beta']
        fms_max = torch.max(fms_softmax, dim=2, keepdim=True)[0]
        fms_softmax_suppress = torch.clamp((fms_softmax < fms_max).float(), min=beta)
        fms_softmax_suppress = torch.repeat_interleave(fms_softmax_suppress, w // self.k, dim=2)
        fms_suppress = fm * fms_softmax_suppress

        return fms_boost, fms_suppress

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.model = list(densenet161(pretrained=True, progress=True).features.children())
        self.block0_2 = nn.Sequential(*self.model[:7])
        self.block3 = nn.Sequential(*self.model[7:9])
        self.block4 = nn.Sequential(*self.model[9:11])

        self.strip1 = FSBM(in_channel=768, k=8)
        self.strip2 = FSBM(in_channel=2112, k=4)
        self.strip3 = FSBM(in_channel=2208, k=2)

    def forward(self, x):
        # [768, 2112, 2208]
        # torch.Size([1, 768, 56, 56]) torch.Size([1, 2112, 28, 28]) torch.Size([1, 2208, 14, 14])
        fm2 = self.block0_2(x)
        fm2_boost, fm2_suppress = self.strip1(fm2)

        fm3 = self.block3(fm2_suppress)
        fm3_boost, fm3_suppess = self.strip2(fm3)

        fm4 = self.block4(fm3_suppess)
        fm4_boost, _ = self.strip3(fm4)

        return fm2_boost, fm3_boost, fm4_boost

    def get_params(self):
        new_layers = list(self.strip1.parameters()) + \
                     list(self.strip2.parameters()) + \
                     list(self.strip3.parameters())
        new_layers_id = list(map(id, new_layers))
        old_layers = filter(lambda p: id(p) not in new_layers_id, self.parameters())
        return new_layers, old_layers

class ResNet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(ResNet, self).__init__()
        if arch == 'resnet50':
            self.model = list(resnet50(pretrained=True, progress=True).children())
        elif arch == 'resnet101':
            self.model = list(resnet101(pretrained=True, progress=True).children())
        self.layer0_2 = nn.Sequential(*self.model[:6])
        self.layer3 = nn.Sequential(*self.model[6:7])
        self.layer4 = nn.Sequential(*self.model[7:8])

        self.strip1 = FSBM(in_channel=512, k=8)
        self.strip2 = FSBM(in_channel=1024, k=4)
        self.strip3 = FSBM(in_channel=2048, k=2)

    def forward(self, x):
        fm2 = self.layer0_2(x)
        fm2_boost, fm2_suppress = self.strip1(fm2)
        fm3 = self.layer3(fm2_suppress)
        fm3_boost, fm3_suppess = self.strip2(fm3)
        fm4 = self.layer4(fm3_suppess)
        fm4_boost, _ = self.strip3(fm4)

        return fm2_boost, fm3_boost, fm4_boost

    def get_params(self):
        new_layers = list(self.strip1.parameters()) + \
                     list(self.strip2.parameters()) + \
                     list(self.strip3.parameters())
        new_layers_id = list(map(id, new_layers))
        old_layers = filter(lambda p: id(p) not in new_layers_id, self.parameters())
        return new_layers, old_layers

if __name__ == '__main__':
    x = torch.randn((2,3, 448, 448))
    model = DenseNet()
    fm2, fm3, fm4 = model(x)
    print(fm2.shape, fm3.shape, fm4.shape)
