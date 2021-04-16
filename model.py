import torch.nn as nn
import torch
import torch.nn.functional as F
from backbone import ResNet, DenseNet 
from config import HyperParams

class TopkPool(nn.Module):
    def __init__(self):
        super(TopkPool, self).__init__()

    def forward(self, x):
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        topkv, _ = x.topk(5, dim=-1)
        return topkv.mean(dim=-1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FDM(nn.Module):
    def __init__(self):
        super(FDM, self).__init__()
        self.factor = round(1.0/(28*28), 3)

    def forward(self, fm1, fm2):
        b, c, w1, h1 = fm1.shape
        _, _, w2, h2 = fm2.shape
        fm1 = fm1.view(b, c, -1) # B*C*S
        fm2 = fm2.view(b, c, -1) # B*C*M

        fm1_t = fm1.permute(0, 2, 1) # B*S*C

        # may not need to normalize
        fm1_t_norm = F.normalize(fm1_t, dim=-1)
        fm2_norm = F.normalize(fm2, dim=1)
        M = -1 * torch.bmm(fm1_t_norm, fm2_norm) # B*S*M

        M_1 = F.softmax(M, dim=1)
        M_2 = F.softmax(M.permute(0, 2, 1), dim=1)
        new_fm2 = torch.bmm(fm1, M_1).view(b, c, w2, h2)
        new_fm1 = torch.bmm(fm2, M_2).view(b, c, w1, h1)

        return self.factor*new_fm1,self.factor* new_fm2

class FBSD(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(FBSD, self).__init__()
        feature_size = 512
        if arch == 'resnet50':
            self.features = ResNet(arch='resnet50')
            chans = [512, 1024, 2048]
        elif arch == 'resnet101':
            self.features = ResNet(arch='resnet101')
            chans = [512, 1024, 2048]
        elif arch == 'densenet161':
            self.features = DenseNet()
            chans = [768, 2112, 2208]

        self.pool = TopkPool()

        part_feature = 1024

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(part_feature* 3),
            nn.Linear(part_feature* 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.conv_block1 = nn.Sequential(
            BasicConv(chans[0], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(chans[1], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(chans[2], feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, part_feature, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(part_feature),
            nn.Linear(part_feature, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, class_num)
        )
        self.inter = FDM()

    def forward(self, x):
        fm1, fm2, fm3 = self.features(x)

        #########################################
        ##### cross-level attention #############
        #########################################

        att1 = self.conv_block1(fm1)
        att2 = self.conv_block2(fm2)
        att3 = self.conv_block3(fm3)

        new_d1_from2, new_d2_from1 = self.inter(att1, att2)  # 1 2
        new_d1_from3, new_d3_from1 = self.inter(att1, att3)  # 1 3
        new_d2_from3, new_d3_from2 = self.inter(att2, att3)  # 2 3

        gamma = HyperParams['gamma']
        att1 = att1 + gamma*(new_d1_from2 + new_d1_from3)
        att2 = att2 + gamma*(new_d2_from1 + new_d2_from3)
        att3 = att3 + gamma*(new_d3_from1 + new_d3_from2)

        xl1 = self.pool(att1)
        xc1 = self.classifier1(xl1)

        xl2 = self.pool(att2)
        xc2 = self.classifier2(xl2)

        xl3 = self.pool(att3)
        xc3 = self.classifier3(xl3)

        x_concat = torch.cat((xl1, xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)

        return xc1, xc2, xc3, x_concat

    def get_params(self):
        new_layers, old_layers = self.features.get_params()
        new_layers += list(self.conv_block1.parameters()) + \
                      list(self.conv_block2.parameters()) + \
                      list(self.conv_block3.parameters()) + \
                      list(self.classifier1.parameters()) + \
                      list(self.classifier2.parameters()) + \
                      list(self.classifier3.parameters()) + \
                      list(self.classifier_concat.parameters())
        return new_layers, old_layers
