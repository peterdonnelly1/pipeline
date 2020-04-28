# CAUTION: NUMBER OF CLASSES IS HARD CODED!!!

"""=============================================================================
PyTorch implementation of VGG

  References:
    [1] Karen Simonyan, Andrew Zisserman - Very Deep Convolutional Networks for Large-Scale Image Recognition - https://arxiv.org/abs/1409.1556v6
    [2] https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py

============================================================================="""

import torch
import torch.nn as nn

DEBUG=1

configs = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGNN( nn.Module ):

    def __init__(self, cfg, tile_size, features, num_class=0):

        super().__init__()

        self.features = features

        number_of_classes = cfg.IMG_EMBED_DIM

        if DEBUG>9:
          print ( "VGGNN:          INFO:       at \033[35;1m __init__()\033[m: number of classes = \033[36;1m{:}\033[m".format( number_of_classes ))

        if DEBUG>99:
          print ( "features = {:}".format ( self.features ) )

        

        first_fc_width=int(tile_size**2/2)                                                                  # PGD 200428 - first_fc_width was previously a hard wired value which meant could not use for diffferent tile sizes
        
        self.classifier = nn.Sequential(

            nn.Linear(first_fc_width, 4096),                                                               # PGD 200428: 2048 is correct for tile size=64;  8192 is correct for tile size=128;  32768 is correct for tile size=256;
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, number_of_classes)
 
        )

        if DEBUG>99:
          print ( "classifier = {:}".format ( self.classifier ) )

    def forward(self, x):

        if DEBUG>9:
          print ( "VGGNN:          INFO:     encode(): type(x)                                       = {:}".format ( type(x) ) )
          print ( "VGGNN:          INFO:     encode(): x.size()                                      = {:}".format ( x.size() ) )
  
        #x=x.contiguous()   # attempt to fix "RuntimeError: max_pool2d_with_indices_out_cuda_frame failed with error code 0" which didn't work. See: https://github.com/pytorch/pytorch/issues/33988
        
        output = self.features(x)  

        if DEBUG>9:
          print ( "VGGNN:          INFO:     encode(): after all convolutional layers, output.size() = {:}".format ( output.size() ) )

        output = output.view(output.size()[0], -1)

        if DEBUG>9:
          print ( "VGGNN:          INFO:     encode(): after reshaping, output.size()                = {:}".format ( output.size() ) )

        output = self.classifier(output)

        if DEBUG>9 :
          print ( "VGGNN:          INFO:     encode(): after all fully connected layers              = {:}".format ( output.size() ) )
    
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

def vgg11_bn(cfg, tile_size ):

      return VGGNN( cfg, tile_size, make_layers( configs['A'], batch_norm=True) )

def vgg13_bn(cfg, tile_size ):

      return VGGNN( cfg, tile_size, make_layers( configs['B'], batch_norm=True) )

def vgg16_bn(cfg, tile_size ):
  
      return VGGNN( cfg, tile_size, make_layers( configs['D'], batch_norm=True) )

def vgg19_bn(cfg, tile_size ):

      return VGGNN( cfg, tile_size, make_layers( configs['E'], batch_norm=True) )
