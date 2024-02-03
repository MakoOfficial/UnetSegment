import os
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms

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
        img = Image.open(img_path)
        return self.transform(img)
    
    def read_a_mask(self, index):
        mask_index = self.masks[index]
        img_path = os.path.join(self.mask_dir, mask_index)
        img = Image.open(img_path)
        return self.transform(img)

    def __getitem__(self, index):
        return self.list[index], self.mask[index]

    def __repr__(self):
        repr = "(DatasetsForUnet,\n"
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
