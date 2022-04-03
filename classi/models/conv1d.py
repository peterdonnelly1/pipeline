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

WHITE           ='\033[37;1m'
PURPLE          ='\033[35;1m'
DIM_WHITE       ='\033[37;2m'
CYAN            ='\033[36;1m'
PALE_RED        ='\033[31m'
PALE_GREEN      ='\033[32m'
BLACK           ='\033[38;2;0;0;0m' 
AUREOLIN        ='\033[38;2;253;238;0m' 
DULL_WHITE      ='\033[38;2;140;140;140m'
MIKADO          ='\033[38;2;255;196;12m'
AZURE           ='\033[38;2;0;127;255m'
AMETHYST        ='\033[38;2;153;102;204m'
ASPARAGUS       ='\033[38;2;135;169;107m'
CHARTREUSE      ='\033[38;2;223;255;0m'
COQUELICOT      ='\033[38;2;255;56;0m'
COTTON_CANDY    ='\033[38;2;255;188;217m'
HOT_PINK        ='\033[38;2;255;105;180m'
CAMEL           ='\033[38;2;193;154;107m'
MAGENTA         ='\033[38;2;255;0;255m'
YELLOW          ='\033[38;2;255;255;0m'
DULL_YELLOW     ='\033[38;2;179;179;0m'
ARYLIDE         ='\033[38;2;233;214;107m'
BLEU            ='\033[38;2;49;140;231m'
DULL_BLUE       ='\033[38;2;0;102;204m'
RED             ='\033[38;2;255;0;0m'
PINK            ='\033[38;2;255;192;203m'
BITTER_SWEET    ='\033[38;2;254;111;94m'
DARK_RED        ='\033[38;2;120;0;0m'
ORANGE          ='\033[38;2;255;103;0m'
PALE_ORANGE     ='\033[38;2;127;63;0m'
GOLD            ='\033[38;2;255;215;0m'
GREEN           ='\033[38;2;19;136;8m'
BRIGHT_GREEN    ='\033[38;2;102;255;0m'
CARRIBEAN_GREEN ='\033[38;2;0;204;153m'
GREY_BACKGROUND ='\033[48;2;60;60;60m'


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

DEBUG   = 99
# ~ columns=241920
columns=120960
columns=15120
columns=3776

class CONV1D( nn.Module ):

  def __init__(  self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 ):

    number_of_classes = 9
  
    super(CONV1D, self).__init__()

    # Note - started out with the vgg16 settings, then gradually stripped put layers because not enough memory. Then halved the number of kernels in conv1, conv2 and conv3 to speed it up a little (it was extremely slow with the VGG16 values)
    # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
    
    self.conv1_1 = nn.Conv1d(  1,  16, kernel_size=3, stride=1, padding=1)   # output: 16 filters, 3 kernel
    self.bnrm1_1 = nn.BatchNorm1d(16)
    self.conv1_2 = nn.Conv1d( 16,  16, kernel_size=3, stride=1, padding=1)   # output: 16 filters, 3 kernel
    self.bnrm1_2 = nn.BatchNorm1d(16)

    # ~ self.conv2_1 = nn.Conv1d( 16, 32,  kernel_size=3, stride=1, padding=1)   # output: 32 filters, 3 kernel
    # ~ self.bnrm2_1 = nn.BatchNorm1d(32)
    # ~ self.conv2_2 = nn.Conv1d(32, 32,   kernel_size=3, stride=1, padding=1)   # output: 32 filters, 3 kernel
    # ~ self.bnrm2_2 = nn.BatchNorm1d(32)

    # ~ self.conv3_1 = nn.Conv1d(32, 64,   kernel_size=3, stride=1, padding=1)   # output: 64 filters, 3 kernel
    # ~ self.bnrm3_1 = nn.BatchNorm1d(64)
    #self.conv3_2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)    # output: 64 filters, 3 kernel
    #self.bnrm3_2 = nn.BatchNorm1d(64)
    #self.conv3_3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)    # output: 64 filters, 3 kernel
    #self.bnrm3_3 = nn.BatchNorm1d(64)

    # ~ self.conv4_1 = nn.Conv1d(64, 128,  kernel_size=3, stride=1, padding=1)   # output: 128 filters, 3 kernel
    # ~ self.bnrm4_1 = nn.BatchNorm1d(128)
    #self.conv4_2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)  # output: 512 filters, 3 kernel
    #self.bnrm4_2 = nn.BatchNorm1d(512)
    #self.conv4_3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)  # output: 512 filters, 3 kernel
    #self.bnrm4_3 = nn.BatchNorm1d(512)

    #self.conv5_1 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)  # output: 512 filters, 3 kernel
    #self.bnrm5_1 = nn.BatchNorm1d(512)
    #self.conv5_2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)  # output: 512 filters, 3 kernel
    #self.bnrm5_2 = nn.BatchNorm1d(512)
    #self.conv5_3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)  # output: 512 filters, 3 kernel
    #self.bnrm5_3 = nn.BatchNorm1d(512)


    # max pooling (kernel_size, stride)
    self.pool = nn.MaxPool1d(2)

    # fully conected layers:
    #self.fc6 = nn.Linear(columns, 4096)
    #self.fc7 = nn.Linear(4096, 4096)
    self.fc8 = nn.Linear(columns, number_of_classes)


  def forward( self, x, gpu  ):

    if DEBUG>8:
      print ( f"{MIKADO}CONV1D:         INFO:       encode(): at input,                                x.size() = {x.shape}{RESET}", flush=True )
    if DEBUG>999:
      print ( f"CONV1D:         INFO:       encode(): x.type()                   = {type(x)}", flush=True )  
      print ( f"CONV1D:         INFO:       encode(): x                   = {x}", flush=True )

    x = F.relu(self.bnrm1_1(self.conv1_1(x)))
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after conv1_1,                           x.size() = {x.size()}", flush=True )
    x = F.relu(self.bnrm1_2(self.conv1_2(x)))
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after conv1_2,                           x.size() = {x.size()}", flush=True )
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )    
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True ) 
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    x = F.max_pool1d(x, 2, 2)
    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    # ~ x = F.relu(self.bnrm2_1(self.conv2_1(x)))
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after conv2_1,                           x.size() = {x.size()}", flush=True ) 
    # ~ x = F.relu(self.bnrm2_2(self.conv2_2(x)))
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after conv2_2,                           x.size() = {x.size()}", flush=True ) 
    # ~ x = F.max_pool1d(x, 2, 2)
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    # ~ x = F.relu(self.bnrm3_1(self.conv3_1(x)))
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after conv3_1,                           x.size() = {x.size()}", flush=True )  
    #x = F.relu(self.bnrm3_2(self.conv3_2(x)))
    #x = F.relu(self.bnrm3_3(self.conv3_3(x)))
    # ~ x = F.max_pool1d(x, 2, 2)
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True ) 
    # ~ x = F.relu(self.bnrm4_1(self.conv4_1(x)))
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after conv4_1,                           x.size() = {x.size()}", flush=True ) 
    #x = F.relu(self.bnrm4_2(self.conv4_2(x)))
    #x = F.relu(self.bnrm4_3(self.conv4_3(x)))
    # ~ x = F.max_pool1d(x, 2, 2)
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  
    #x = F.relu(self.bnrm5_1(self.conv5_1(x)))
    #x = F.relu(self.bnrm5_2(self.conv5_2(x)))
    #x = F.relu(self.bnrm5_3(self.conv5_3(x)))
    # ~ x = F.max_pool1d(x, 2, 2)
    # ~ if DEBUG>8:
      # ~ print ( f"CONV1D:         INFO:       encode(): after max_pool1d,                        x.size() = {x.size()}", flush=True )  


    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after all convolutional layers,          x.size() = {x.size()}", flush=True )

    x = x.view(x.size(0), -1)

    if DEBUG>8:
      print ( f"CONV1D:         INFO:       encode(): after reshaping,                         x.size() = {x.size()}", flush=True )

#      x = F.relu(self.fc6(x))
#      x = F.dropout(x, 0.5, self.training)
#      x = F.relu(self.fc7(x))
#      x = F.dropout(x, 0.5, self.training)
    x = self.fc8(x)

    if DEBUG>8:
      print ( f"{AZURE}CONV1D:         INFO:       encode(): after all fully connected layers,        x.size() = {x.size()}{RESET}", flush=True )

    return x, 0
