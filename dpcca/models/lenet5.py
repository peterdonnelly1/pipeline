"""=============================================================================
PyTorch implementation of LeNet5. See:

    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
============================================================================="""

import torch.nn as nn
import torch.nn.functional as F

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
AUREOLIN='\033[38;2;253;238;0m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
ASPARAGUS='\033[38;2;135;169;107m'
CHARTREUSE='\033[38;2;223;255;0m'
COQUELICOT='\033[38;2;255;56;0m'
COTTON_CANDY='\033[38;2;255;188;217m'
HOT_PINK='\033[38;2;255;105;180m'
CAMEL='\033[38;2;193;154;107m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
ARYLIDE='\033[38;2;233;214;107m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
BITTER_SWEET='\033[38;2;254;111;94m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;255;103;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
CARRIBEAN_GREEN='\033[38;2;0;204;153m'
PALE_GREEN='\033[32m'
GREY_BACKGROUND='\033[48;2;60;60;60m'


BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'

FAIL    = 0
SUCCESS = 1

DEBUG   = 0

# ------------------------------------------------------------------------------

class LENET5( nn.Module ):

    def __init__( self, cfg, args, n_classes, tile_size ):

        """Initialize LENET5
        """
        
        if DEBUG>1:
          print ( f"AEDCECCAE_3:    INFO:         __init__:  n_classes  = {MIKADO}{n_classes}{RESET}", flush=True     )         
          print ( f"AEDCECCAE_3:    INFO:         __init__:  tile_size  = {MIKADO}{tile_size}{RESET}", flush=True     )         
        
        super(LENET5, self).__init__()

        nc = 3
        w  = tile_size

        # in_channels (int)  - Number of channels in the input image
        # out_channels (int) - Number of channels produced by the convolution

        self.conv1 = nn.Conv2d( nc, 6,  5 )               # in_channels=1, out_channels=6,  kernel_size=5, (stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv2d(   6,    16,  5 )          # in_channels=6, out_channels=16, kernel_size=5, (stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        # ~ self.fc1 = nn.Linear( 16*30*30, 256)              # FOR CANCER IMAGES  <- * DIMS OF PRECEEDING LAYER (# KERNELS * KERNELS SIZE * KERNEL SIZE), NUMBER OF SAMPLES PGD 200109 - PARAMETERIZE THIS !!!!!
        self.fc1 = nn.Linear( 576, 256)              # FOR CANCER IMAGES  <- * DIMS OF PRECEEDING LAYER (# KERNELS * KERNELS SIZE * KERNEL SIZE), NUMBER OF SAMPLES PGD 200109 - PARAMETERIZE THIS !!!!!
        self.fc2 = nn.Linear( 256, 84)                    # FOR CANCER IMAGES  <- * PARAMETERIZE THIS !!!!!
        self.fc3 = nn.Linear( 84, n_classes )

        # the below are not used since we only encode
        self.fc5 = nn.Linear( n_classes, 84)
        self.fc6 = nn.Linear(84, nc * w * w)


# ------------------------------------------------------------------------------

    def forward(self, x, batch_fnames):

        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  x.size                          = {:}".format( x.size() ) )

        x = F.pad(x, (2, 2, 2, 2))
        
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  x <padded>.size                 = {:}".format( x.size() ) )

        y = F.relu(self.conv1(x))  # for MNIST: nc x 32 x 32 ---> 6  x 28 x 28
        if DEBUG>0:
          print ( "\nLENET5:         INFO:           encode():  y <relu I>.size                 = {:}".format( y.size() ) )

        y = F.max_pool2d(y, 2)     # for MNIST: 6  x 28 x 28 ---> 6  x 14 x 14
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <max_pool2d I>.size           = {:}".format( y.size() ) )

        y = F.relu(self.conv2(y))  # for MNIST: 6  x 14 x 14 ---> 16 x 10 x 10
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <relu 2>.size                 = {:}".format( y.size() ) )

        y = F.max_pool2d(y, 2)     # for MNIST: 16 x 10 x 10 ---> 16 x 5  x 5
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <max_pool2d II>.size          = {:}".format( y.size() ) )

        y = y.view(y.size(0), -1)  # for MNIST: 16 x 5  x 5  ---> 400
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <y.view(y.size(0), -1)>.size  = {:}".format( y.size() ) )

        y = F.relu(self.fc1(y))    # for MNIST: 400          ---> 120
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <F.relu(self.fc1(y)>.size     = {:}".format( y.size() ) )

        y = F.relu(self.fc2(y))    # for MNIST: 120          ---> 84
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <F.relu(self.fc2(y))>.size    = {:}".format( y.size() ) )

        y = self.fc3(y)            # for MNIST:  84          ---> 10
        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <self.fc3(y)>.size            = {:}".format( y.size() ) )

        if DEBUG>0:
          print ( "LENET5:         INFO:           encode():  y <encoded version>.size        = {:}".format( y.size() ) )
        
        return y, 0
        
