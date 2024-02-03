import torch.utils.data as data

import torch.optim as optim
from torchvision import transforms
import time
import torch

from torch.optim.lr_scheduler import StepLR
from datasets import SegmentDataset
from torch import nn
from model import UNet

import torch
import numpy as np
import random


seed = 1#seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False


def pixel_accuracy(output, label, threshold=0.9):
    output = torch.where(output > threshold, torch.tensor(1), torch.tensor(0)).type(torch.FloatTensor)
    correct_pixels = torch.sum(output == label).item()
    total_pixels = label.numel()
    accuracy = correct_pixels / total_pixels
    return correct_pixels, total_pixels


net = UNet().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

loss_func = nn.BCELoss()
EPOCH = 100

dataset = SegmentDataset("../../autodl-tmp/archiveUnet/train")
print(f"train dataset is {dataset}")
sampler = torch.utils.data.RandomSampler(data_source=dataset)
print(f"sampler is {sampler}")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    sampler=sampler,
    drop_last=False
)

valDataset = SegmentDataset("../../autodl-tmp/archiveUnet/test")
print(f"valid dataset is {valDataset}")
valloader = torch.utils.data.DataLoader(
    valDataset,
    batch_size=16,
    shuffle=False,
    drop_last=False
)

for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    net.train()
    total_loss = 0.
    total_length = 0.
    for i, (img, label) in enumerate(dataloader):
        img = img.cuda()
        label = label.cuda()
        img_out = net(img)
        loss = loss_func(img_out, label)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        total_length += len(img)
        optimizer.step()
    scheduler.step()

    net.eval()
    correct = 0.
    total_pix = 0.
    for i, (img, label) in enumerate(valloader):
        img = img.cuda()
        label = label.type(torch.LongTensor).cuda()
        img_out = net(img)
        correct_pixels, total_pixels = pixel_accuracy(img_out, label)
        correct += correct_pixels
        total_pix += total_pixels

    print(f'第{epoch + 1}轮结束, train loss: {round(total_loss / total_length, 2)}, '
          f'accuracy: {(round(correct / total_pix, 4) * 100)}%')

torch.save(net.state_dict(), "./Unet.pth")







