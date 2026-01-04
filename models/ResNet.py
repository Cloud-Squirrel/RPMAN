import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import functools
import sys, os

# 打印resnet101网络结构
# resnet_detail = models.resnet101()
# print(resnet_detail.__dict__)

# # 打印resnet152网络结构
# resnet = models.resnet152
# print(resnet.__dict__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN_res50(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super(FCN_res50, self).__init__()
        # 载入预训练的ResNet-50网络
        resnet = models.resnet50(pretrained=True)
        # 新建卷积层，用于将输入图像通道数对齐到ResNet的预训练模型中
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 复制原始卷积层的权重
        newconv1.weight.data[:, :3, :, :].copy_(resnet.conv1.weight.data)
        # 如果输入通道数大于3，则将多余的通道的权重也复制到新建卷积层中
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        # 定义网络各层
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 将ResNet的layer3和layer4的部分卷积层的步长改为1，用于增加输出特征图的分辨率
        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # 定义输出特征图经过的卷积层
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False), #2048-512
                                  nn.BatchNorm2d(128, momentum=0.95),
                                  nn.ReLU())
        # 定义分类器
        self.classifier = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # 获取输入特征图的大小
        x_size = x.size()

        # 前向传播过程
        x = self.layer0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        out = self.classifier(x)

        # 返回分类结果和上采样后的特征图
        return out, F.interpolate(out, x_size[2:], mode='bilinear')


class FCN_res18(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res18, self).__init__()
        # 载入预训练的ResNet-18网络
        resnet = models.resnet18(pretrained)
        # 新建卷积层，用于将输入图像通道数对齐到ResNet的预训练模型中
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 复制原始卷积层的权重
        newconv1.weight.data[:, 0:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels, :, :])
        # 如果输入通道数大于3，则将多余的通道的权重也复制到新建卷积层中
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        # 定义网络各层
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # 将ResNet的layer3和layer4的部分卷积层的步长改为1，用于增加输出特征图的分辨率
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # 定义输出特征图经过的卷积层
        self.head = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(64, momentum=0.95),
                                  nn.ReLU())
        # 定义分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64, momentum=0.95),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 获取输入特征图的大小
        x_size = x.size()

        # 前向传播过程
        x0 = self.layer0(x)
        x = self.maxpool(x0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        x = self.classifier(x)

        # 返回分类结果和上采样后的特征图
        out = F.interpolate(x, x_size[2:], mode='bilinear')
        
        return out


class FCN_res34(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res34, self).__init__()
        # 载入预训练的ResNet-34网络
        resnet = models.resnet34(pretrained=False)
        # 新建卷积层，用于将输入图像通道数对齐到ResNet的预训练模型中
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 复制原始卷积层的权重
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        # 如果输入通道数大于3，则将多余的通道的权重也复制到新建卷积层中
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        # 定义网络各层
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # 将ResNet的layer3和layer4的部分卷积层的步长改为1，用于增加输出特征图的分辨率
        '''
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        '''
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # 定义输出特征图经过的卷积层
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())

        # 定义分类器
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
    
    def forward(self, x):
        # 获取输入特征图的大小
        x_size = x.size()

        # 前向传播过程
        x = self.layer0(x)  # size:1/2
        x = self.maxpool(x)  # size:1/4
        x = self.layer1(x)  # size:1/4
        x = self.layer2(x)  # size:1/8
        x = self.layer3(x)  # size:1/16
        x = self.layer4(x)
        x = self.head(x)
        out = self.classifier(x)

        # 返回分类结果和上采样后的特征图
        return F.interpolate(out, x_size[2:], mode='bilinear')


class FCN_res101(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(FCN_res101, self).__init__()
        # 载入预训练的ResNet-101网络
        resnet = models.resnet101(pretrained)
        # 新建卷积层，用于将输入图像通道数对齐到ResNet的预训练模型中, 前面改通道数，后面参数主要是使图像大小减半
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 复制原始卷积层的权重
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        # 如果输入通道数大于3，则将多余的通道的权重也复制到新建卷积层中
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        # 定义网络各层
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 将ResNet的layer3和layer4的部分卷积层的步长改为1，用于增加输出特征图的分辨率
        """
        这段代码是用于调整ResNet中的某些卷积层和下采样层的步长（stride）的。
        具体地，它首先遍历ResNet的第三个卷积层（layer3）中的所有子模块，
        并检查每个子模块的名称（n）中是否包含'conv1'或'downsample.0'，
        如果包含则将该子模块的步长（stride）设置为(1, 1)。
        将卷积层的步长改为1可以增加输出特征图的分辨率。在深度学习中，卷积层通常用于从输入图像中提取特征。
        通过不断减小特征图的尺寸，模型可以逐步抽象出更高级别的特征，但这也可能会导致信息的丢失。
        当卷积层的步长为2时，每个输出像素都对应于输入图像中的一个2x2的区域。这意味着特征图的尺寸将减小一半。
        这种下采样可以有助于提高模型的计算效率和减少内存需求，但也可能导致信息的丢失。
        通过将卷积层的步长改为1，可以避免特征图的尺寸缩小，从而增加输出特征图的分辨率。
        这有助于保留更多的细节信息，提高模型的性能和精度。但这也会增加计算成本和内存需求，因为特征图的尺寸会增加。
        因此，在选择卷积层的步长时，需要平衡模型的性能和计算效率之间的权衡。
        """
        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n or 'downsample.0' in n:
        #         m.stride = (1, 1)
        #
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # 定义输出特征图经过的卷积层
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        # 定义分类器
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # 获取输入特征图的大小
        x_size = x.size()

        # 前向传播过程
        x = self.layer0(x)  # size:1/2 strid=2
        x = self.maxpool(x)  # size:1/4 strid=2

        x = self.layer1(x)  # size:1/4 strid=1
        """
        ResNet-101中的downsample层执行的时机是在残差块中输入特征图和输出特征图的尺寸不同时。
        (指的是，在每个残差块的两个分支中，它们的尺寸不相同)
        在其他情况下，downsample层不会被执行。"""
        x = self.layer2(x)  # size:1/8 strid=2 downsample层不执行
        x = self.layer3(x)  # size:1/16 strid=2
        x = self.layer4(x)  # size：1/16 strid=2

        x = self.head(x)
        out = self.classifier(x)

        # 返回分类结果和上采样后的特征图
        return F.interpolate(out, x_size[2:], mode='bilinear')


