"""=============================================================================
simple 3-layer 3D convolutional autoencoder
============================================================================="""

import torch
from  torch import nn
from  torch import sigmoid
from  torch import relu
from  torch import tanh
import numpy as np

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

DEBUG=0

# ------------------------------------------------------------------------------
class AE3LAYERCONV2D( nn.Module ):

  def __init__(  self, cfg, n_classes, tile_size  ):

    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:    at {CYAN} __init__(){RESET}" )
    
    super(AE3LAYERCONV2D, self).__init__()
           
    self.f1       = nn.Conv2d( in_channels=3,   out_channels=36, kernel_size=3, stride=2, padding=1 )               
    self.f2       = nn.Conv2d( in_channels=36,  out_channels=144, kernel_size=3, stride=2, padding=1 )               
    self.f3       = nn.Conv2d( in_channels=144, out_channels=6, kernel_size=7                      ) 
                           
    self.r1       = nn.ConvTranspose2d( in_channels=6, out_channels=144, kernel_size=7)      
    self.r2       = nn.ConvTranspose2d( in_channels=144, out_channels=36, kernel_size=3,  stride=2, padding=1, output_padding=1)
    self.r3       = nn.ConvTranspose2d( in_channels=36, out_channels=3,  kernel_size=3,  stride=2, padding=1, output_padding=1)  
  
# ------------------------------------------------------------------------------
  def encode(self, x, gpu, encoder_activation ):
   
    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:       encode(): x.shape   = {WHITE}{x.shape}{RESET}", flush=True   ) 

    if DEBUG>99:
      print ( f"AE3LAYERCONV2D: INFO:       encode(): x  = {CYAN}{x}{RESET}", flush=True   ) 
        
    x =  relu(self.f1(x))
    x =  relu(self.f2(x))
    z =  relu(self.f3(x))
  
    if DEBUG>0:
      print ( f"AE3LAYERCONV2D: INFO:       encode(): z.shape   = {ARYLIDE}{z.shape}{RESET}", flush=True   ) 
      
    return z
          
  # ------------------------------------------------------------------------------
  
  def decode( self, z ):
    
    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:       decode(): z.shape   = {CYAN}{z.shape}{RESET}", flush=True   ) 
    
    x =  relu( self.r1(z) )
    x =  relu( self.r2(x) )
    x =  self.r3(x)       
  
    if DEBUG>9: 
      print ( f"AE3LAYERCONV2D: INFO:       decode(): x.shape   = {CYAN}{x.shape}{RESET}", flush=True   ) 
    
    return x
  
  # ------------------------------------------------------------------------------
  
  def forward( self, x, gpu, encoder_activation ):
  
    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:       forward(): x.shape  = {CYAN}{x.shape}{RESET}", flush=True   ) 
    
    z = self.encode( x, gpu, encoder_activation)
#    z = self.encode( x.view(-1, self.input_dim), gpu, encoder_activation)
  
    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:       forward(): z.shape  = {CYAN}{z.shape}{RESET}", flush=True   ) 
    
    x2r = self.decode(z)
    
    return x2r, 0, 0
  
