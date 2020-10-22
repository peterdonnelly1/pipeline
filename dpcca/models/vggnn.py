# CAUTION: NUMBER OF CLASSES IS HARD CODED!!!

"""=============================================================================
PyTorch implementation of VGG

  References:
    [1] Karen Simonyan, Andrew Zisserman - Very Deep Convolutional Networks for Large-Scale Image Recognition - https://arxiv.org/abs/1409.1556v6
    [2] https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py

============================================================================="""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
ARYLIDE='\033[38;2;233;214;107m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
PALE_GREEN='\033[32m'

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
DEBUG=1


configs = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGNN( nn.Module ):

    def __init__(self, cfg, n_classes, tile_size, features, num_class=0):

        super().__init__()

        self.features = features

        if DEBUG>0:
          print( f"VGGNN:          INFO:   n_classes =  {MIKADO}{n_classes}{RESET}" )

        if DEBUG>9:
          print ( "VGGNN:          INFO:       at \033[35;1m __init__()\033[m: number of classes = \033[36;1m{:}\033[m".format( n_classes ))

        if DEBUG>99:
          print ( "features = {:}".format ( self.features ) )

        

        first_fc_width=int(tile_size**2/2)                                                                  # PGD 200428 - first_fc_width was previously a hard wired value which meant could not use for diffferent tile sizes
        
        # ~ self.classifier = nn.Sequential(

            # ~ nn.Linear(first_fc_width, 4096),                                                               # PGD 200428: 2048 is correct for tile size=64;  8192 is correct for tile size=128;  32768 is correct for tile size=256;
            # ~ nn.ReLU(inplace=True),
            # ~ nn.Dropout(),
            # ~ nn.Linear(4096, 4096),
            # ~ nn.ReLU(inplace=True),
            # ~ nn.Dropout(),
            # ~ nn.Linear(4096, n_classes)
 
        # ~ )

        if DEBUG>99:
          print ( "classifier = {:}".format ( self.classifier ) )

        self.fc1 = nn.Linear(first_fc_width, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)

        self.Dropout = nn.Dropout()


    def forward(self, x, batch_fnames):

        if DEBUG>9:
          print ( "VGGNN:          INFO:     forward(): type(x)                                       = {:}".format ( type(x) ) )
          print ( "VGGNN:          INFO:     forward(): x.size()                                      = {:}".format ( x.size() ) )

        if DEBUG>99:
          print ( f"VGGNN:          INFO:     forward(): contents of batch_fnames = {MAGENTA}{batch_fnames}{RESET}" )
          batch_fnames_npy = batch_fnames.numpy()                                                # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details
          np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
          print ( f"VGGNN:          INFO:     forward():       batch_fnames_npy.shape      = {batch_fnames_npy.shape:}" )        
          print ( f"VGGNN:          INFO:     forward():       batch_fnames_npy            = {batch_fnames_npy:}"       )
#          fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"
          fq_link = f"/home/peter/git/pipeline/dataset/{batch_fnames_npy[0]}.fqln"
          np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
          print ( f"VGGNN:          INFO:     forward():       fq_link                     = {PINK}{fq_link:}{RESET}"                )
          print ( f"VGGNN:          INFO:     forward():       file fq_link points to      = {PINK}{os.readlink(fq_link)}{RESET}"    )

          
        output = self.features(x)  

        if DEBUG>9:
          print ( "VGGNN:          INFO:     forward(): after all convolutional layers, output.size() = {:}".format ( x.size() ) )

        output = output.view(output.size()[0], -1)

        if DEBUG>9:
          print ( "VGGNN:          INFO:     forward(): after reshaping, output.size()                = {:}".format ( output.size() ) )

#        output = self.classifier(output)
  
        if DEBUG>9:
          print ( "VGG:            INFO:     encode(): after reshaping, output.size()                = {:}".format ( output.size() ) )
  
        output = self.fc1(output)
        output = F.relu(output)
        output = self.Dropout(output)        
        output = self.fc2(output)
        output = F.relu(output)
        output = self.Dropout(output)
        output = self.fc3(output)


        if DEBUG>9 :
          print ( "VGGNN:          INFO:     forward(): after all fully connected layers              = {:}".format ( output.size() ) )
    
        return output



def make_layers(configs, batch_norm=False):

    layers = []

    input_channel = 3

    for l in configs:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def vgg11_bn(cfg, n_classes, tile_size ):

      return VGGNN( cfg, n_classes, tile_size, make_layers( configs['A'], batch_norm=True) )

def vgg13_bn(cfg, n_classes, tile_size ):

      return VGGNN( cfg, n_classes, tile_size, make_layers( configs['B'], batch_norm=True) )

def vgg16_bn(cfg, n_classes, tile_size ):
  
      return VGGNN( cfg, n_classes, tile_size, make_layers( configs['D'], batch_norm=True) )

def vgg19_bn(cfg, n_classes, tile_size ):

      return VGGNN( cfg, n_classes, tile_size, make_layers( configs['E'], batch_norm=True) )
