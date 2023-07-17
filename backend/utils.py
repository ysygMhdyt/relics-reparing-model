import numpy as np
import matplotlib as plt
import os
import imageio
import scipy

from PIL import Image, ExifTags
from glob import glob


def get_image(image_path, image_size, is_crop=True):
    return transform(Image.open(image_path), image_size, is_crop)


# 归一化，将图片值转化为[-1,1]，减少计算量
def transform(image, size=64, is_crop=True):
    # 若需要裁剪
    if is_crop:
        image = center_crop(image, size)

    img = np.array(image) / 127.5 - 1.0
    return img


# 中心裁剪为64*64
def center_crop(x, crop_size, resize_size=(64, 64)):
    x = np.array(x)
    # 原始的高和宽
    h, w = x.shape[:2]
    # 中心裁剪后的高和宽
    j = int(round((h - crop_size) / 2))
    i = int(round((w - crop_size) / 2))
    # 裁剪后转化为64*64
    y = np.array(Image.fromarray(x[j:j + crop_size, i:i + crop_size]).resize(resize_size))
    return y


# 生成噪音块
def generate_mask(batch_size=4, image_shape=[64, 64, 3], scale=0.25):
    # 生成全1矩阵 形状为4*64*64*3
    mask = np.ones([batch_size] + image_shape).astype(np.float32)
    # 全0矩阵
    imask = np.zeros([batch_size] + image_shape).astype(np.float32)
    # 取出中间块儿
    x = int(image_shape[0] * scale)
    y = int(image_shape[0] - x)
    # 中间全0，外围全1 用于获取外围部分的图像
    mask[:, x:y, x:y, :] = 0.0
    # 中间全1 外部全0 用于获取生成部分的图像
    imask[:, x:y, x:y, :] = 1.0

    return mask, imask
