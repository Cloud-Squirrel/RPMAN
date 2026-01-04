import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
# from osgeo import gdal_array
import cv2

num_classes = 2
DG_COLORMAP = [0, 255]
DG_CLASSES = ['not road', 'road']

DG_MEAN = np.array([166.54, 165.83, 163.82])
DG_STD = np.array([88.52, 87.04, 88.16])
# VAL_DG_MEAN = np.array([104.58, 93.86, 68.07])
# VAL_DG_STD  = np.array([40.09, 32.77, 30.49])

DIRECTION_COLOR = {0: [0, 0, 0], 1: [255, 0, 0], 2: [255, 0, 125], 3: [255, 0, 255], 4: [125, 0, 255], 5: [0, 0, 255],
                   6: [0, 125, 255], 7: [0, 255, 255], 8: [0, 255, 125], 9: [0, 255, 0], 10: [125, 255, 0],
                   11: [255, 255, 0], 12: [255, 125, 0], 13: [255, 255, 255]}

DIRECTION_COLOR_4 = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 0, 255], 3: [0, 255, 0], 4: [255, 255, 0], 5: [255, 255, 255]}

root = 'D:/AI/model/ICHseg/they/RoadExtraction/Data/daxing_road/'
def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im):
    if len(im.shape) == 2:  # 灰度图像
        return (im - DG_MEAN[0]) / DG_STD[0]  # 使用单通道均值和标准差
    elif len(im.shape) == 3:  # 彩色图像
        return (im - DG_MEAN) / DG_STD  # 使用三通道均值和标准差
    else:
        raise ValueError(f"Unsupported image shape: {im.shape}")

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap


def Index2Color(pred):
    pred = pred * 255
    pred = np.asarray(pred, dtype='uint8')
    return pred


def rescale_images(imgs, scale, order):
    for i, im in enumerate(imgs):
        imgs[i] = rescale_image(im, scale, order)
    return imgs


def rescale_image(img, scale=1 / 8, order=0):
    flag = cv2.INTER_NEAREST
    if order == 1:
        flag = cv2.INTER_LINEAR
    elif order == 2:
        flag = cv2.INTER_AREA
    elif order > 2:
        flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)),
                             interpolation=flag)
    return im_rescaled


def get_file_name(mode):
    data_dir = root
    assert mode in ['train', 'val', 'test']
    img_dir = os.path.join(data_dir, mode, 'image')

    data_list = os.listdir(img_dir)
    for vi, it in enumerate(data_list):
        data_list[vi] = it[:-4]
        print(it[:-4])
    return data_list


def read_RSimages(mode, rescale=False, rotate_aug=False):
    data_dir = root
    assert mode in ['train', 'val', 'test']
    img_dir = os.path.join(data_dir, mode, 'image')
    mask_dir = os.path.join(data_dir, mode, 'label')

    data_list = os.listdir(img_dir)
    imgs, labels = [], []
    data_length = int(len(data_list))
    count = 0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        #if (it_ext == '.tif'):
        if (it_ext == '.jpg'):
        # if (it_ext == '.png'):
            img_path = os.path.join(img_dir, it)
            # mask_path = os.path.join(mask_dir, it)
            #mask_path = os.path.join(mask_dir, it_name+'.png')
            mask_path = os.path.join(mask_dir, it_name+'.jpg')
            img = io.imread(img_path)
            label = io.imread(mask_path)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            #label = cv2.COLOR_RGB2GRAY(label)
            label = Color2Index(label)
            #label = Color2Index(io.imread(mask_path))
            #print(label)
            #label = cv2.COLOR_RGB2GRAY(label)
            imgs.append(img)
            labels.append(label)
            count += 1
            if not count % 1000: print('%d/%d images loaded.' % (count, data_length))
            # if count>10: break
    print(str(len(imgs)) + ' ' + mode + ' images' + ' loaded.')
    return imgs, labels


class RS(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.mode = mode
        self.random_flip = random_flip
        self.data, self.labels = read_RSimages(mode)
        self.len = len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.random_flip:
            data, label = transform.rand_flip(data, label)
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        label = torch.from_numpy(label)
        return data, label

    def __len__(self):
        return self.len


