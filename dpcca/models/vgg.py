"""=============================================================================
PyTorch implementation of VGG16. 

  References:
    https://github.com/pytorch/vision.git
    https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html
    https://neurohive.io/en/popular-networks/vgg16/
    https://www.quora.com/What-is-the-VGG-neural-network
    https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py  << THIS IS THE CODE I USE BELOW

============================================================================="""

import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):

  def __init__( self, cfg ):
    
      super(VGG, self).__init__()

      # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
      self.conv1_1 = nn.Conv2d(3, 64,    kernel_size=3, padding=1)
      self.conv1_2 = nn.Conv2d(64, 64,   kernel_size=3, padding=1)

      self.conv2_1 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)
      self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

      self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
      self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
      self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

      self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
      self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
      self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

      self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
      self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
      self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

      # max pooling (kernel_size, stride)
      self.pool = nn.MaxPool2d(2, 2)

      # fully conected layers:
      self.fc6 = nn.Linear(7*7*512, 4096)
      self.fc7 = nn.Linear(4096, 4096)
      self.fc8 = nn.Linear(4096, 1000)

  def encode( self, x ):
      x = F.relu(self.conv1_1(x))
      x = F.relu(self.conv1_2(x))
      x = self.pool(x)
      x = F.relu(self.conv2_1(x))
      x = F.relu(self.conv2_2(x))
      x = self.pool(x)
      x = F.relu(self.conv3_1(x))
      x = F.relu(self.conv3_2(x))
      x = F.relu(self.conv3_3(x))
      x = self.pool(x)
      x = F.relu(self.conv4_1(x))
      x = F.relu(self.conv4_2(x))
      x = F.relu(self.conv4_3(x))
      x = self.pool(x)
      x = F.relu(self.conv5_1(x))
      x = F.relu(self.conv5_2(x))
      x = F.relu(self.conv5_3(x))
      x = self.pool(x)
      x = x.view(-1, 7 * 7 * 512)
      x = F.relu(self.fc6(x))
      x = F.dropout(x, 0.5, training=training)
      x = F.relu(self.fc7(x))
      x = F.dropout(x, 0.5, training=training)
      x = self.fc8(x)
      return x
