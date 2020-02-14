"""=============================================================================
PyTorch implementation of LeNet5. See:

    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
============================================================================="""

import torch.nn as nn
import torch.nn.functional as F

DEBUG=0

# ------------------------------------------------------------------------------

class LENET5(nn.Module):

    def __init__(self, cfg):
        """Initialize LENET5
        """
        super(LENET5, self).__init__()

        self.nc = cfg.N_CHANNELS
        self.w  = cfg.IMG_SIZE

        # In-channels, out-channels, kernel size. See `forward()` for dimensionality analysis.
        self.conv1 = nn.Conv2d(self.nc, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear( 16*5*5, 120)              # FOR MNIST ONLY
        self.fc2 = nn.Linear( 120, 84)                  # FOR MNIST ONLY
        self.fc3 = nn.Linear( 84, cfg.IMG_EMBED_DIM)    # FOR MNIST ONLY. SECOND PARAMETER IS THE NUMBER OF CLASSES (i.e. 10)

        #self.fc1 = nn.Linear(16*30*30, 256)          ## <- * DIMS OF PRECEEDING LAYER (# KERNELS * KERNELS SIZE * KERNEL SIZE), NUMBER OF SAMPLES PGD 200109 - PARAMETERIZE THIS !!!!!
        #self.fc2 = nn.Linear(256, 84)                ## <- * PARAMETERIZE THIS !!!!!
        #self.fc3 = nn.Linear(84, cfg.IMG_EMBED_DIM)  ## <- * PARAMETERIZE THIS !!!!!


        # the below are not used since we only encode
        self.fc5 = nn.Linear(cfg.IMG_EMBED_DIM, 84)
        self.fc6 = nn.Linear(84, self.nc * self.w * self.w)




# ------------------------------------------------------------------------------

    def encode(self, x):
		
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  x.size                          = {:}".format( x.size() ) )
	
        x = F.pad(x, (2, 2, 2, 2))
        
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  x <padded>.size                 = {:}".format( x.size() ) )

        y = F.relu(self.conv1(x))  # nc x 32 x 32 ---> 6  x 28 x 28
        if DEBUG>0:
          print ( "\nLENET5:         INFO:           encode():  y <relu I>.size                 = {:}".format( y.size() ) )

        y = F.max_pool2d(y, 2)     # 6  x 28 x 28 ---> 6  x 14 x 14
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <max_pool2d I>.size           = {:}".format( y.size() ) )

        y = F.relu(self.conv2(y))  # 6  x 14 x 14 ---> 16 x 10 x 10
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <relu 2>.size                 = {:}".format( y.size() ) )

        y = F.max_pool2d(y, 2)     # 16 x 10 x 10 ---> 16 x 5  x 5
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <max_pool2d II>.size           = {:}".format( y.size() ) )

        y = y.view(y.size(0), -1)  # 16 x 5  x 5  ---> 400
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <y.view(y.size(0), -1)>.size   = {:}".format( y.size() ) )

        y = F.relu(self.fc1(y))    # 400          ---> 120
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <F.relu(self.fc1(y)>.size      = {:}".format( y.size() ) )

        y = F.relu(self.fc2(y))    # 120          ---> 57
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <F.relu(self.fc2(y))>.size     = {:}".format( y.size() ) )

        y = self.fc3(y)            # 57           ---> k
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <self.fc3(y)>.size             = {:}".format( y.size() ) )

        if DEBUG>0:
          print ( "LENET5:         INFO:           encode(): y <encoded version>.size         = {:}".format( y.size() ) )
        
        return y
        
        
# ------------------------------------------------------------------------------

    def decode(self, z):
        y = F.relu(self.fc5(z))
        y = F.relu(self.fc6(y))
        y = y.view(-1, self.nc, self.w, self.w)
        return y

# ------------------------------------------------------------------------------

    def forward(self, x):
        """Perform forward pass on neural network.
        """
        x = self.encode(x)
        x = self.decode(x)

        return x
