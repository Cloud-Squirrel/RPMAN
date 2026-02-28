import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch
from models.ResNet import FCN_res101 as FCN

BN_MOMENTUM = 0.01

"""PAM + ASPP is parallel, and the VGG classification results are used as segmentation weights"""


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# ─────────────────────────────────────────────
#  1. PAM
# ─────────────────────────────────────────────
class PositionalAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionalAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels,      kernel_size=1)
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        queries = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        keys    = self.key_conv(x).view(batch_size, -1, height * width)
        values  = self.value_conv(x).view(batch_size, -1, height * width)

        attention_map    = torch.bmm(queries, keys)
        attention_map    = self.softmax(attention_map)

        attended_values  = torch.bmm(values, attention_map.permute(0, 2, 1))
        attended_values  = attended_values.view(batch_size, -1, height, width)

        return attended_values + x


# ─────────────────────────────────────────────
# 2. VGG classification weights module

#

# Principle:

#  Feed raw input image into VGG16 (remove final classification layer)

#  Output class score of (B, num_classes) after global average pooling + fully connected layer

#  The confidence of each class is obtained by Softmax, which is used as the prior of "which class is more likely"

#  reshape to (B, num_classes, 1, 1) and then multiply with the split logit plot

# -- Apply the corresponding confidence weight to each category channel

# - Let the network focus on the categories that are actually present in the image during segmentation
# ─────────────────────────────────────────────
class VGGClassificationWeightModule(nn.Module):

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(VGGClassificationWeightModule, self).__init__()

        vgg16 = models.vgg16(pretrained=pretrained)
        self.vgg_features = vgg16.features          # print (B, 512, H/32, W/32)

        # Replace the original classification header：Global average pooling → Fully connected → num_classes
        self.gap = nn.AdaptiveAvgPool2d(1)          # (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        feat   = self.vgg_features(x)               # (B, 512, h, w)
        feat   = self.gap(feat).flatten(1)           # (B, 512)
        logits = self.classifier(feat)               # (B, num_classes)
        weight = self.softmax(logits) 
        weight = weight.unsqueeze(-1).unsqueeze(-1)  # (B, num_classes, 1, 1)
        return weight

# ─────────────────────────────────────────────
#  3. ASPP
# ─────────────────────────────────────────────
class ASPPModule(nn.Module):
    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18)):
        super(ASPPModule, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95), nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95), nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95), nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95), nn.ReLU(inplace=True))

        self.branch5_conv  = nn.Conv2d(features, inner_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch5_bn    = nn.BatchNorm2d(inner_features, momentum=0.95)
        self.branch5_relu  = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_features, momentum=0.95), nn.ReLU(inplace=True))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([feat1, feat2, feat3, feat4, global_feature], dim=1)
        return self.conv_cat(feature_cat)


# ─────────────────────────────────────────────
#  4. main network
# ─────────────────────────────────────────────
class RPMAN(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(RPMAN, self).__init__()

        # Backbone
        self.FCN  = FCN(in_channels, num_classes)
        #  PAM
        self.pam  = PositionalAttentionModule(2048)
        self.head = ASPPModule(2048, 64, 512)
        # VGG
        self.vgg_weight = VGGClassificationWeightModule(num_classes=num_classes,
                                                        pretrained=pretrained)

        # Low-level feature processing
        self.low  = nn.Sequential(conv1x1(256, 64),  nn.BatchNorm2d(64),  nn.ReLU())
        self.low1 = nn.Sequential(conv1x1(64,  64),  nn.BatchNorm2d(64),  nn.ReLU())

        # Decoder fusion
        self.fuse  = nn.Sequential(conv3x3(64 + num_classes, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.fuse1 = nn.Sequential(conv3x3(128,              64), nn.BatchNorm2d(64), nn.ReLU())

        # Classification
        self.classifier        = nn.Conv2d(64,   num_classes, kernel_size=1)
        self.classifier_DAMM   = nn.Sequential(conv1x1(2048, 512), nn.BatchNorm2d(512), nn.ReLU())
        self.classifier_aux    = nn.Sequential(conv1x1(512, 64), nn.BatchNorm2d(64), nn.ReLU(),
                                               conv1x1(64, num_classes))

    def forward(self, x):
        x_size = x.size()

        # ── Backbone ──────────────────────────────────
        x0 = self.FCN.layer0(x)          # 1/2
        xp = self.FCN.maxpool(x0)        # 1/4
        x1 = self.FCN.layer1(xp)         # 1/4,  256ch
        xp = self.FCN.layer2(x1)         # 1/8
        xp = self.FCN.layer3(xp)         # 1/16
        xp = self.FCN.layer4(xp)         # 1/16, 2048ch

        x_pam  = self.pam(xp)                           # (B, 2048, h, w)
        x_PAM  = self.classifier_DAMM(x_pam)            # (B, 512,  h, w)
        x_PAM  = self.classifier_aux(x_PAM)             # (B, num_classes, h, w)
        x_aspp = self.head(xp)                          # (B, 512, h, w)
        x_aspp = self.classifier_aux(x_aspp)            # (B, num_classes, h, w)
        aux = x_aspp + x_PAM                            # (B, num_classes, h, w)
        x1   = self.low(x1)                             # (B, 64, h/4, w/4)
        xcat = torch.cat(
            (F.interpolate(aux, x1.size()[2:], mode='bilinear', align_corners=True), x1), dim=1
        )                                               # (B, 64+num_classes, h/4, w/4)
        fuse = self.fuse(xcat)                          # (B, 64, h/4, w/4)
        x0   = self.low1(x0)                            # (B, 64, h/2, w/2)
        xcat = torch.cat(
            (F.interpolate(fuse, x0.size()[2:], mode='bilinear', align_corners=True), x0), dim=1
        )                                               # (B, 128, h/2, w/2)
        fuse = self.fuse1(xcat)                         # (B, 64,  h/2, w/2)
        out = self.classifier(fuse)                     # (B, num_classes, h/2, w/2)
        out = F.interpolate(out, x_size[2:], mode='bilinear', align_corners=True)
                                                        # (B, num_classes, H, W)

        vgg_w = self.vgg_weight(x)                      # (B, num_classes, 1, 1)
        out   = out * vgg_w                             # (B, num_classes, H, W)

        aux = F.interpolate(aux, x_size[2:], mode='bilinear', align_corners=True)
        return out, aux
