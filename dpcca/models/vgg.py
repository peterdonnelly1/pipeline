"""=============================================================================
PyTorch implementation of VGG16. 

  References:
    https://github.com/pytorch/vision.git
    https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html
    https://neurohive.io/en/popular-networks/vgg16/
    https://www.quora.com/What-is-the-VGG-neural-network
    https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py  << THIS IS THE CODE I USE BELOW
    https://github.com/prlz77/vgg-face.pytorch/blob/master/models/vgg_face.py << AND ALSO SOME OF THIS !

============================================================================="""

import torch.nn as nn
import torch.nn.functional as F

# at end of convolutional layers: batch * channels * width * height 
# first FC layer requires: batch_size * columns = batch_size * 

DEBUG=1

columns=4*4*512

class VGG(nn.Module):

  def __init__( self, cfg ):
    
      super(VGG, self).__init__()

      # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
      self.conv1_1 = nn.Conv2d(  3,  64, kernel_size=3, stride=1, padding=1)   # output:  64 filters, 3x3 kernel
      self.bnrm1_1 = nn.BatchNorm2d(64)
      self.conv1_2 = nn.Conv2d( 64,  64, kernel_size=3, stride=1, padding=1)   # output:  64 filters, 3x3 kernel
      self.bnrm1_2 = nn.BatchNorm2d(64)

      self.conv2_1 = nn.Conv2d( 64, 128, kernel_size=3, stride=1, padding=1)   # output: 128 filters, 3x3 kernel
      self.bnrm2_1 = nn.BatchNorm2d(128)
      self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)   # output: 128 filters, 3x3 kernel
      self.bnrm2_2 = nn.BatchNorm2d(128)

      self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)   # output: 256 filters, 3x3 kernel
      self.bnrm3_1 = nn.BatchNorm2d(256)
      self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)   # output: 256 filters, 3x3 kernel
      self.bnrm3_2 = nn.BatchNorm2d(256)
      self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)   # output: 256 filters, 3x3 kernel
      self.bnrm3_3 = nn.BatchNorm2d(256)

      self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3x3 kernel
      self.bnrm4_1 = nn.BatchNorm2d(512)
      self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3x3 kernel
      self.bnrm4_2 = nn.BatchNorm2d(512)
      self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3x3 kernel
      self.bnrm4_3 = nn.BatchNorm2d(512)

      self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3x3 kernel
      self.bnrm5_1 = nn.BatchNorm2d(512)
      self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3x3 kernel
      self.bnrm5_2 = nn.BatchNorm2d(512)
      self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3x3 kernel
      self.bnrm5_3 = nn.BatchNorm2d(512)

      # max pooling (kernel_size, stride)
      self.pool = nn.MaxPool2d(2, 2)

      # fully conected layers:
#      self.fc6 = nn.Linear(7*7*512, 4096)
      self.fc6 = nn.Linear(columns, 4096)
      self.fc7 = nn.Linear(4096, 4096)
      self.fc8 = nn.Linear(4096, 3)    # 3 classes only

  def encode( self, x ):

      if DEBUG>0:
        print ( "VGG:            INFO:     encode(): x.size()                                 = {:}".format ( x.size() ) )

      x = F.relu(self.bnrm1_1(self.conv1_1(x)))
      x = F.relu(self.bnrm1_2(self.conv1_2(x)))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.bnrm2_1(self.conv2_1(x)))
      x = F.relu(self.bnrm2_2(self.conv2_2(x)))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.bnrm3_1(self.conv3_1(x)))
      x = F.relu(self.bnrm3_2(self.conv3_2(x)))
      x = F.relu(self.bnrm3_3(self.conv3_3(x)))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.bnrm4_1(self.conv4_1(x)))
      x = F.relu(self.bnrm4_2(self.conv4_2(x)))
      x = F.relu(self.bnrm4_3(self.conv4_3(x)))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.bnrm5_1(self.conv5_1(x)))
      x = F.relu(self.bnrm5_2(self.conv5_2(x)))
      x = F.relu(self.bnrm5_3(self.conv5_3(x)))
      x = F.max_pool2d(x, 2, 2)

      if DEBUG>9:
        print ( "VGG:            INFO:     encode(): after all convolutional layers, x.size() = {:}".format ( x.size() ) )

      x = x.view(x.size(0), -1)

      if DEBUG>9:
        print ( "VGG:            INFO:     encode(): after reshaping, x.size()                = {:}".format ( x.size() ) )

      x = F.relu(self.fc6(x))
#      x = F.dropout(x, 0.5, self.training)
      x = F.relu(self.fc7(x))
#      x = F.dropout(x, 0.5, self.training)
      x = self.fc8(x)

      if DEBUG>9:
        print ( "VGG:            INFO:     encode(): after all fully connected layers         = {:}".format ( x.size() ) )

      return x
