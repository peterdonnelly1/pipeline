"""=============================================================================
PyTorch implementation of VGG

  References:
    [1] Karen Simonyan, Andrew Zisserman - Very Deep Convolutional Networks for Large-Scale Image Recognition - https://arxiv.org/abs/1409.1556v6
    [2] https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py

============================================================================="""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants  import *


DEBUG=1


configs = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

counter=0
        
class VGGNN( nn.Module ):                                                                                  

    def __init__( self, cfg, args, n_classes, tile_size, features, num_class=0 ):                          # featues = make_layers( configs['X'] )      where X = A or B or D or E

        super().__init__()

        self.features = features                                                                           # features = make_layers( configs['X'] = torch.nn.Sequential(*layers)

        if DEBUG>0:
          print ( f"VGGNN:          INFO:   {CYAN}__init__(){RESET}: number of classes = {MIKADO}{n_classes}{RESET}" )        

        if DEBUG>2:
          print ( f"VGGNN:          INFO:   {CYAN}__init__(){RESET}:          features = \n{MIKADO}{self.features}{RESET}" )
 
        first_fc_width=int(tile_size**2/2)                                                                 # PGD 200428 - first_fc_width was previously a hard wired value which meant it could not use for diffferent tile sizes

        if DEBUG>99:
          print ( f"VGGNN:          INFO:   {CYAN}__init__(){RESET}:    first_fc_width = {MIKADO}{first_fc_width}{RESET}" )        

        
        # ~ self.classifier = nn.Sequential(

            # ~ nn.Linear(first_fc_width, 4096),                                                           # PGD 200428: 2048 is correct for tile size=64;  8192 is correct for tile size=128;  32768 is correct for tile size=256;
            # ~ nn.ReLU(inplace=True),
            # ~ nn.Dropout(),
            # ~ nn.Linear(4096, 4096),
            # ~ nn.ReLU(inplace=True),
            # ~ nn.Dropout(),
            # ~ nn.Linear(4096, n_classes)
 
        # ~ )

        if 0<=tile_size<32:
          print( f"{RED}VGGNN:           FATAL:  for the VGG models '{CYAN}TILE_SIZE{RESET}{RED}' (corresponding to python argument '{CYAN}--tile_size{RESET}{RED}') is not permitted to be less than {MIKADO}32{RESET}", flush=True)
          print( f"{RED}VGGNN:           FATAL: ... halting now{RESET}" )
          sys.exit(0)
        
        first_fc_width = int((tile_size/32)**2 * 512)
        
           
        self.fc1 = nn.Linear(first_fc_width, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)

        self.Dropout = nn.Dropout()



    def forward(self, x, batch_fnames):                                                                    # y1, embedding = model.image_net.forward( x, batch_fnames ) = model.VGGNN.forward( x, batch_fnames )

        global counter
        
        if DEBUG>99:
          print ( f"VGGNN:          INFO:     forward(): type(x)                                     = {MIKADO}{type(x)}{RESET}",    flush=True  )
          print ( f"VGGNN:          INFO:     forward(): type(self)                                  = {MIKADO}{type(self)}{RESET}", flush=True   )


        if DEBUG>10:
          print ( f"VGGNN:          INFO:     forward(): contents of batch_fnames = {MAGENTA}{batch_fnames}{RESET}" )
          batch_fnames_npy = batch_fnames.numpy()                                                          # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details
          np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
          print ( f"VGGNN:          INFO:     forward():       batch_fnames_npy.shape      = {batch_fnames_npy.shape:}" )        
          print ( f"VGGNN:          INFO:     forward():       batch_fnames_npy            = {batch_fnames_npy:}"       )
#          fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"
          fq_link = f"/home/peter/git/pipeline/working_data/{batch_fnames_npy[0]}.fqln"
          np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
          print ( f"VGGNN:          INFO:     forward():       fq_link                     = {PINK}{fq_link:}{RESET}"                )
          print ( f"VGGNN:          INFO:     forward():       file fq_link points to      = {PINK}{os.readlink(fq_link)}{RESET}"    )

        if DEBUG>10:
          print ( f"VGGNN:          INFO:     forward(): before all convolutional layers, x.size      = {BITTER_SWEET}{x.size()}{RESET}{CLEAR_LINE}" )
          
        output = self.features(x)        # features = make_layers( configs['X'] = torch.nn.Sequential(*layers) => output = self.features(x) = VGGNN.features(x)
        
        if DEBUG>10:
          print ( f" {MIKADO}self.type={type(self)}{RESET}" )
        

        if DEBUG>10:
          print ( f"VGGNN:          INFO:     forward(): after  all convolutional layers, x.size      = {MIKADO}{x.size()}{RESET}{CLEAR_LINE}" )

        output = output.view(output.size()[0], -1)

        if DEBUG>10:
          print ( f"VGGNN:          INFO:     forward(): after  flatenning,               x.size      = {BLEU}{output.size()}{RESET}{CLEAR_LINE}" )

#       output = self.classifier(output)
    
        output = self.fc1(output)
        output = F.relu(output)
        if DEBUG>10:
          print ( f"VGGNN:          INFO:     forward(): after  F.relu(self.fc1(output)), output.size = {BLEU}{output.size()}{RESET}{CLEAR_LINE}" )
        output = self.Dropout(output)
        
        output = self.fc2(output)
        embedding = output
        output = F.relu(output)
        if DEBUG>10:
          print ( f"VGGNN:          INFO:     forward(): after  F.relu(self.fc2(output)), output.size = {BLEU}{output.size()}{RESET}{CLEAR_LINE}"            )
        if DEBUG>88:
          print ( f"VGGNN:          INFO:     forward(): x[:,0:20]                                   = {BLEU}{x[:,0:20]}{RESET}{CLEAR_LINE}"                 )

        output = self.Dropout(output)
        
        output = self.fc3(output)
        if DEBUG>10:
          print ( f"VGGNN:          INFO:     forward(): after  F.relu(self.fc3(output)), output.size = {BLEU}{output.size()}{RESET}{CLEAR_LINE}"            )
          print ( f"VGGNN:          INFO:     forward(): after  all fully connected layers, x.size    = {CARRIBEAN_GREEN}{output.size()}{RESET}{CLEAR_LINE}" )
          print ( f"VGGNN:          INFO:     forward(): counter                                      = {MIKADO}{counter}{RESET}{CLEAR_LINE}"                )


        if DEBUG>0:
          print ( f"VGGNN:          INFO:     forward():                                  output.size = {BLEU}{output.size()}{RESET}{CLEAR_LINE}"            )

        return output, embedding



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
    
    return nn.Sequential(*layers)                                                        # returns a method to do the convolutional layer processing for the chosen VGGNN (11, 13, 16, 19) 


def vgg11_bn( cfg, args, n_classes, tile_size ):

      return VGGNN( cfg, args, n_classes, tile_size, make_layers( configs['A'], batch_norm=True) )

def vgg13_bn( cfg,args, n_classes, tile_size ):

      return VGGNN( cfg, args, n_classes, tile_size, make_layers( configs['B'], batch_norm=True) )

def vgg16_bn( cfg,args, n_classes, tile_size ):
  
      return VGGNN( cfg, args, n_classes, tile_size, make_layers( configs['D'], batch_norm=True) )

def vgg19_bn( cfg,args, n_classes, tile_size ):

      return VGGNN( cfg, args, n_classes, tile_size, make_layers( configs['E'], batch_norm=True) )
