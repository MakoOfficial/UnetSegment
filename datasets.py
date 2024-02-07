import os
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
import cv2


def read_image(file_path, image_size=512):
    """读取图片，并统一修改为512x512"""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(np.uint8(img))
    # 开始修改尺寸
    w, h = img.size
    long = max(w, h)
    # 按比例缩放成512
    w, h = int(w / long * image_size), int(h / long * image_size)
    # 压缩并插值
    img = img.resize((w, h), Image.ANTIALIAS)
    # 然后是给短边扩充，使用ImageOps.expand
    delta_w, delta_h = image_size - w, image_size - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    # 转化成np.array
    return ImageOps.expand(img, padding)



class SegmentDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = os.path.join(root_dir, "input")
        self.mask_dir = os.path.join(root_dir, "mask")
        
        self.transform = transforms.ToTensor()  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件
        self.masks = os.listdir(self.mask_dir)
        
        self.list = []
        self.mask = []
        for i in range(len(self.images)):
            self.list.append(self.read_a_pic(i))

        for i in range(len(self.masks)):
            self.mask.append(self.read_a_mask(i))

    def __len__(self):
        return len(self.images)

    def read_a_pic(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        # img = Image.open(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(np.uint8(img))
        return self.transform(img)
    
    def read_a_mask(self, index):
        mask_index = self.masks[index]
        img_path = os.path.join(self.mask_dir, mask_index)
        # img = Image.open(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(np.uint8(img))
        return self.transform(img)

    def __getitem__(self, index):
        return self.list[index], self.mask[index]

    def __repr__(self):
        repr = "(DatasetsForUnet,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  len = %s,\n" % str(self.__len__())
        repr += ")"
        return repr


class EvaluateDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.transform = transforms.ToTensor()  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

        self.list = []
        for i in range(len(self.images)):
            self.list.append(self.read_a_pic(i))

    def __len__(self):
        return len(self.images)

    def read_a_pic(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        # img = Image.open(img_path)
        img = read_image(img_path)
        return self.transform(img)


    def __getitem__(self, index):
        return self.list[index], self.images[index]

    def __repr__(self):
        repr = "(DatasetsForEvaluate,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  len = %s,\n" % str(self.__len__())
        repr += ")"
        return repr


if __name__ == '__main__':
    root_dir = "../archiveMasked/UnetDataset/train"
    dataset = SegmentDataset(root_dir)
    print(dataset)
    et = dataset.__getitem__(1)
    input, mask = et
    print(input.shape)
