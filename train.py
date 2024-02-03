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

dataset = SegmentDataset("../archiveMasked/UnetDataset/train")
print(f"train dataset is {dataset}")
sampler = torch.utils.data.RandomSampler(data_source=dataset)
print(f"sampler is {sampler}")
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    sampler=sampler,
    drop_last=False
)

valDataset = SegmentDataset('../archiveMasked/UnetDataset/test')
print(f"valid dataset is {valDataset}")
valloader = torch.utils.data.DataLoader(
    valDataset,
    batch_size=4,
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







