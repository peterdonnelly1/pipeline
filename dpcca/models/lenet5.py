"""=============================================================================
PyTorch implementation of LeNet5. See:

    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
============================================================================="""

import torch.nn as nn
import torch.nn.functional as F

DEBUG=0

# ------------------------------------------------------------------------------

class LENET5( nn.Module ):

    def __init__(self, cfg):

        number_of_classes = cfg.IMG_EMBED_DIM

        """Initialize LENET5
        """
        super(LENET5, self).__init__()

        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE

        # in_channels (int)  - Number of channels in the input image
        # out_channels (int) - Number of channels produced by the convolution

        self.conv1 = nn.Conv2d( self.nc, 6,  5 )        # in_channels=1, out_channels=6,  kernel_size=5, (stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv2d(   6,    16,  5 )        # in_channels=6, out_channels=16, kernel_size=5, (stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc1 = nn.Linear( 16*30*30, 256)              # FOR CANCER IMAGES  <- * DIMS OF PRECEEDING LAYER (# KERNELS * KERNELS SIZE * KERNEL SIZE), NUMBER OF SAMPLES PGD 200109 - PARAMETERIZE THIS !!!!!
        self.fc2 = nn.Linear( 256, 84)                    # FOR CANCER IMAGES  <- * PARAMETERIZE THIS !!!!!
        self.fc3 = nn.Linear( 84, number_of_classes )

        # the below are not used since we only encode
        self.fc5 = nn.Linear( number_of_classes, 84)
        self.fc6 = nn.Linear(84, self.nc * self.w * self.w)


# ------------------------------------------------------------------------------

    def forward(self, x):

        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  x.size                          = {:}".format( x.size() ) )

        x = F.pad(x, (2, 2, 2, 2))
        
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  x <padded>.size                 = {:}".format( x.size() ) )

        y = F.relu(self.conv1(x))  # for MNIST: nc x 32 x 32 ---> 6  x 28 x 28
        if DEBUG>0:
          print ( "\nLENET5:         INFO:           encode():  y <relu I>.size               = {:}".format( y.size() ) )

        y = F.max_pool2d(y, 2)     # for MNIST: 6  x 28 x 28 ---> 6  x 14 x 14
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <max_pool2d I>.size           = {:}".format( y.size() ) )

        y = F.relu(self.conv2(y))  # for MNIST: 6  x 14 x 14 ---> 16 x 10 x 10
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <relu 2>.size                 = {:}".format( y.size() ) )

        y = F.max_pool2d(y, 2)     # for MNIST: 16 x 10 x 10 ---> 16 x 5  x 5
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <max_pool2d II>.size           = {:}".format( y.size() ) )

        y = y.view(y.size(0), -1)  # for MNIST: 16 x 5  x 5  ---> 400
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <y.view(y.size(0), -1)>.size   = {:}".format( y.size() ) )

        y = F.relu(self.fc1(y))    # for MNIST: 400          ---> 120
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <F.relu(self.fc1(y)>.size      = {:}".format( y.size() ) )

        y = F.relu(self.fc2(y))    # for MNIST: 120          ---> 84
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <F.relu(self.fc2(y))>.size     = {:}".format( y.size() ) )

        y = self.fc3(y)            # for MNIST:  84          ---> 10
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <self.fc3(y)>.size             = {:}".format( y.size() ) )

        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <encoded version>.size         = {:}".format( y.size() ) )
        
        return y
        
