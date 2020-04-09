# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from fpn.fpn_head import FPNDecoder
from fpn.resnet import resnet50


class fpn(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50()
        self.head = FPNDecoder(encoder_channels = [2048, 1024, 512, 256])

    def forward(self, input):
        x = self.backbone(input)
        x = self.head(x)
        # x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        print("Use CPU")
        device = torch.device('cpu')

    model = fpn().to(device)
    x = torch.zeros(8, 1, 640, 640).to(device)
    y = model(x)
    print(y.size())
