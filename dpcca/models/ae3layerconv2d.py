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

DEBUG=1

# ------------------------------------------------------------------------------
class AE3LAYERCONV2D( nn.Module ):

  def __init__(  self, cfg, n_classes, tile_size  ):

    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:    at {CYAN} __init__(){RESET}" )
    
    super(AE3LAYERCONV2D, self).__init__()
           
    self.f1       = nn.Conv2d( in_channels=3,   out_channels=36,  kernel_size=3, stride=2, padding=1                              )
    self.f2       = nn.Conv2d( in_channels=36,  out_channels=144, kernel_size=3, stride=2, padding=1                              )
    self.f3       = nn.Conv2d( in_channels=144, out_channels=3,   kernel_size=7                                                   ) 

    self.r1       = nn.ConvTranspose2d( in_channels=3, out_channels=144,  kernel_size=7                                           )
    self.r2       = nn.ConvTranspose2d( in_channels=144, out_channels=36, kernel_size=3,  stride=2, padding=1, output_padding=1   )
    self.r3       = nn.ConvTranspose2d( in_channels=36, out_channels=3,   kernel_size=3,  stride=2, padding=1, output_padding=1   )  
  
    
    
# ------------------------------------------------------------------------------
  def encode(self, x, gpu, encoder_activation ):
   
    if DEBUG>1:
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
  
    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:       forward(): x.shape  = {CYAN}{x.shape}{RESET}", flush=True   ) 
    
    
    x = self.gaussian_noise( x )
    
    z = self.encode( x, gpu, encoder_activation)

    # ~ z = self.encode( x, gpu, encoder_activation)
#    z = self.encode( x.view(-1, self.input_dim), gpu, encoder_activation)
  
    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:       forward(): z.shape  = {CYAN}{z.shape}{RESET}", flush=True   ) 
    
    x2r = self.decode(z)
    
    return x2r, 0, 0
    
  
  # ------------------------------------------------------------------------------
  # HELPER FUNCTIONS
  # ------------------------------------------------------------------------------  
  
  # Methods to add noise are from https://dropsofai.com/convolutional-denoising-autoencoders-for-image-noise-reduction/
  # By Kartik Chaudhary | November 10, 2020


  def gaussian_noise( self, x):
  
      if DEBUG>9:
        print ( f"AE3LAYERCONV2D: INFO:           gaussian_noise():  x.shape       = {DULL_YELLOW}{x.shape}{RESET}", flush=True   ) 
      
      var = 0.1
      
      noise = (var**0.5)*torch.randn( x.shape).cuda()
      noisy_x = x + noise


      if DEBUG>9:
        print ( f"AE3LAYERCONV2D: INFO:           gaussian_noise():  noisy_x.shape = {DULL_YELLOW}{noisy_x.shape}{RESET}", flush=True   )  
      
      return noisy_x
   
   
  def salt_and_pepper_noise( self, x):
      ratio = 0.9
      amount = 0.1
      noisy = np.copy(x)
       
      salt_count = np.ceil(amount * x.size * ratio)
      coords = [np.random.randint(0, i - 1, int(salt_count)) for i in x.shape]
      noisy[coords] = 1
   
      pepper_count = np.ceil(amount* x.size * (1. - ratio))
      coords = [np.random.randint(0, i - 1, int(pepper_count)) for i in x.shape]
      noisy[coords] = 0
      return noisy
   
  def poisson_noise( self, x):
      vals = len(np.unique(x))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(x * vals) / float(vals)
      return noisy
   
  def speckle_noise( self, x):
      r,c = x.shape
      speckle = np.random.randn(r,c)
      speckle = speckle.reshape(r,c)        
      noisy = x + x * speckle
      return noisy    
   
  def add_noise( self, x):
      p = np.random.random()
      if p <= 0.25:
          #print("Guassian")
          noisy = guassian_noise( self, x)
      elif p <= 0.5:
          #print("SnP")
          noisy = salt_and_pepper_noise( self, x)
      elif p <= 0.75:
          #print("Poison")
          noisy = poisson_noise( self, x)
      else:
          #print("speckle")
          noisy = speckle_noise( self, x)
      return noisy
