import torch
from torch import nn
from gym.spaces import Box
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from cbam import CBAMBlock, CBAMBlock_w_cnn
"""
class FeatureExtractor(nn.Module):
    """""" Convolutional Feature Extractor for input images """"""

    def __init__(self, obs_space_shape, num_of_features):
        super(FeatureExtractor, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=obs_space_shape[0], out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.cnn_out_size = np.prod(self.cnn(torch.zeros(size=(1,
                                                               obs_space_shape[0],
                                                               obs_space_shape[1],
                                                               obs_space_shape[2]))).shape)

        self.linear = nn.Linear(int(self.cnn_out_size), num_of_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))

"""
class FeatureExtractor(nn.Module):
    """ Convolutional Feature Extractor for input images """

    def __init__(self, obs_space_shape, num_of_features, nettype='0'):
        super(FeatureExtractor, self).__init__()
        if nettype == '0':
          self.cnn = nn.Sequential(
              nn.Conv2d(in_channels=obs_space_shape[0], out_channels=64, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Flatten()
          )
        if nettype == '1':
          self.cnn = nn.Sequential(
              nn.Conv2d(in_channels=obs_space_shape[0], out_channels=32, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=0),
              nn.ReLU(),
              nn.Flatten()
          )
        if nettype == '2':
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=obs_space_shape[0], out_channels=64, kernel_size=3, stride=1, padding=0),
                CBAMBlock(channel=64,reduction=16,kernel_size=7),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
                CBAMBlock(channel=32,reduction=16,kernel_size=7),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
                CBAMBlock(channel=16,reduction=16,kernel_size=7),
                nn.ReLU(),
                nn.Flatten()
            )
        if nettype == '3':
            self.cnn = nn.Sequential(
                # nn.Conv2d(in_channels=obs_space_shape[0], out_channels=64, kernel_size=3, stride=1, padding=0),
                CBAMBlock_w_cnn(channel=128, reduction=16, kernel_size_att=7, in_channels=obs_space_shape[0],
                                out_channels=128, kernel_size_cnn=3, stride=1),
                nn.ReLU(),
                CBAMBlock_w_cnn(channel=64, reduction=16, kernel_size_att=7, in_channels=128, out_channels=64,
                                kernel_size_cnn=3, stride=1),
                nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
                CBAMBlock_w_cnn(channel=32, reduction=16, kernel_size_att=7, in_channels=64, out_channels=32,
                                kernel_size_cnn=3, stride=1),
                nn.ReLU(),
                # nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
                CBAMBlock_w_cnn(channel=16, reduction=16, kernel_size_att=7, in_channels=32, out_channels=16,
                                kernel_size_cnn=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
        if nettype == '4':
          self.cnn = nn.Sequential(
              nn.Conv2d(in_channels=obs_space_shape[0], out_channels=128, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Flatten()
          )
        if nettype == '5':
          self.cnn = nn.Sequential(
              nn.Conv2d(in_channels=obs_space_shape[0], out_channels=128, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
              nn.ReLU(),
              nn.Flatten()
          )
        self.cnn_out_size = np.prod(self.cnn(torch.zeros(size=(1,
                                                               obs_space_shape[0],
                                                               obs_space_shape[1],
                                                               obs_space_shape[2]))).shape)

        self.linear = nn.Linear(int(self.cnn_out_size), num_of_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))

"""
from torchsummary import summary

print(summary(model, (4,38,58)))"""