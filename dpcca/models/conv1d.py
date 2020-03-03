# CAUTION: NUMBER OF CLASSES IS HARD CODED!!!
"""=============================================================================
PyTorch implementation of 1d Convolutional Network 

  References:
    https://pytorch.org/docs/stable/nn.html


============================================================================="""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# at end of convolutional layers: batch * channels * width * height 
# first FC layer requires: batch_size * columns = batch_size * 

DEBUG=1

columns=241920

class CONV1D( nn.Module ):

  def __init__( self, cfg ):

      number_of_classes = 9
    
      super(CONV1D, self).__init__()

      # Note - started out with the vgg16 settings, then gradually stripped put layers because not enough memory. Then halved the number of kernels in conv1, conv2 and conv3 to speed it up a little (it was extremely slow with the VGG16 values)
      # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
      self.conv1_1 = nn.Conv1d(  1,  16, kernel_size=3, stride=1, padding=1)   # output:  16 filters, 3 kernel
      self.bnrm1_1 = nn.BatchNorm1d(16)
      self.conv1_2 = nn.Conv1d( 16,  16, kernel_size=3, stride=1, padding=1)   # output:  16 filters, 3 kernel
      self.bnrm1_2 = nn.BatchNorm1d(16)

      self.conv2_1 = nn.Conv1d( 16, 32, kernel_size=3, stride=1, padding=1)   # output: 32 filters, 3 kernel
      self.bnrm2_1 = nn.BatchNorm1d(32)
      self.conv2_2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)   # output: 32 filters, 3 kernel
      self.bnrm2_2 = nn.BatchNorm1d(32)

      self.conv3_1 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)   # output: 64 filters, 3 kernel
      self.bnrm3_1 = nn.BatchNorm1d(64)
      #self.conv3_2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)   # output: 64 filters, 3 kernel
      #self.bnrm3_2 = nn.BatchNorm1d(64)
      #self.conv3_3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)   # output: 64 filters, 3 kernel
      #self.bnrm3_3 = nn.BatchNorm1d(64)

      self.conv4_1 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)   # output: 128 filters, 3 kernel
      self.bnrm4_1 = nn.BatchNorm1d(128)
      #self.conv4_2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3 kernel
      #self.bnrm4_2 = nn.BatchNorm1d(512)
      #self.conv4_3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3 kernel
      #self.bnrm4_3 = nn.BatchNorm1d(512)

      #self.conv5_1 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3 kernel
      #self.bnrm5_1 = nn.BatchNorm1d(512)
      #self.conv5_2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3 kernel
      #self.bnrm5_2 = nn.BatchNorm1d(512)
      #self.conv5_3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)   # output: 512 filters, 3 kernel
      #self.bnrm5_3 = nn.BatchNorm1d(512)
  

      # max pooling (kernel_size, stride)
      self.pool = nn.MaxPool1d(2)

      # fully conected layers:
      #self.fc6 = nn.Linear(columns, 4096)
      #self.fc7 = nn.Linear(4096, 4096)
      self.fc8 = nn.Linear(columns, number_of_classes)

  def forward( self, x ):

      x = x.unsqueeze(1)   # PGD 200301 necessary because I generate single dimensional vector in generate_image.py whereas Torch needs a 2D vector. This will break DPCCA however.

      if DEBUG>9:
        print ( "CONV1D:         INFO:       encode(): x.type()                   = {:}".format ( type(x) )  )
        print ( "CONV1D:         INFO:       encode(): x.size()                   = {:}".format ( x.shape ) )
        if DEBUG>999:
          print ( "CONV1D:         INFO:       encode(): x                   = {:}".format ( x )        )

      x = F.relu(self.bnrm1_1(self.conv1_1(x)))
      x = F.relu(self.bnrm1_2(self.conv1_2(x)))
      x = F.max_pool1d(x, 2, 2)
      x = F.relu(self.bnrm2_1(self.conv2_1(x)))
      x = F.relu(self.bnrm2_2(self.conv2_2(x)))
      x = F.max_pool1d(x, 2, 2)
      x = F.relu(self.bnrm3_1(self.conv3_1(x)))
      #x = F.relu(self.bnrm3_2(self.conv3_2(x)))
      #x = F.relu(self.bnrm3_3(self.conv3_3(x)))
      x = F.max_pool1d(x, 2, 2)
      x = F.relu(self.bnrm4_1(self.conv4_1(x)))
      #x = F.relu(self.bnrm4_2(self.conv4_2(x)))
      #x = F.relu(self.bnrm4_3(self.conv4_3(x)))
      x = F.max_pool1d(x, 2, 2)
      #x = F.relu(self.bnrm5_1(self.conv5_1(x)))
      #x = F.relu(self.bnrm5_2(self.conv5_2(x)))
      #x = F.relu(self.bnrm5_3(self.conv5_3(x)))
      x = F.max_pool1d(x, 2, 2)


      if DEBUG>9:
        print ( "CONV1D:            INFO:       encode(): after all convolutional layers, x.size() = {:}".format ( x.size() ) )

      x = x.view(x.size(0), -1)

      if DEBUG>9:
        print ( "CONV1D:            INFO:       encode(): after reshaping, x.size()                = {:}".format ( x.size() ) )

#      x = F.relu(self.fc6(x))
#      x = F.dropout(x, 0.5, self.training)
#      x = F.relu(self.fc7(x))
#      x = F.dropout(x, 0.5, self.training)
      x = self.fc8(x)

      if DEBUG>9:
        print ( "CONV1D:            INFO:       encode(): after all fully connected layers         = {:}".format ( x.size() ) )

      return x
