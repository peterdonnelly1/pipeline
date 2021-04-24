"""=============================================================================
simple 3-layer 3D convolutional autoencoder
============================================================================="""

import torch
from  torch import nn
from  torch import sigmoid
from  torch import relu
from  torch import tanh
import numpy as np

from skimage.util import random_noise

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
    
    
    # ~ x = self.gaussian_noise( x )
    # ~ x = self.s_and_p_noise( x )
    # ~ x = self.poisson_noise( x )
    # ~ x = self.speckle_noise( x )
    
    
    # ~ x = self.add_noise( x )
    
    
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
      
      mean = 0
      var  = 0.1
      
      npy_noise = np.float32(random_noise( x.cpu(), mode='gaussian', mean=mean, var=var, clip=True))
      noise     = torch.tensor( npy_noise )  
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AE3LAYERCONV2D: INFO:           type(x)                         (gaussian)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(noise)                     (gaussian)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           x.size()                        (gaussian)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AE3LAYERCONV2D: INFO:           noisy_x.size()                  (gaussian)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x
   
   
  def s_and_p_noise( self, x):

      amount = 0.5
          
      noise = torch.tensor(random_noise( x.cpu().numpy(), mode='s&p', salt_vs_pepper=amount, clip=True) )  
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AE3LAYERCONV2D: INFO:           type(x)                         (s_and_p)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(noise)                     (s_and_p)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           x.size()                        (s_and_p)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AE3LAYERCONV2D: INFO:           noisy_x.size()                  (s_and_p)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x
   
   
  def poisson_noise( self, x):
      
      npy_noise = np.float32(random_noise( x.cpu(), mode='poisson', clip=True))                    # for poisson, random_noise returns float64 for some reasons. Have to convert because tensors use single precision
      noise     = torch.tensor( npy_noise )     
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AE3LAYERCONV2D: INFO:           type(x)                         (poisson)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(npy_noise)                 (poisson)      = {DULL_YELLOW}{type(npy_noise) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(x.cpu().numpy()[0,0,0,0])  (poisson)      = {DULL_YELLOW}{type(x.cpu().numpy()[0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(npy_noise[0,0,0,0])        (poisson)      = {DULL_YELLOW}{type(npy_noise      [0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(noise)                     (poisson)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           x.size()                        (poisson)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AE3LAYERCONV2D: INFO:           noisy_x.size()                  (poisson)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x


  def speckle_noise( self, x):

      npy_noise = np.float32(random_noise( x.cpu(), mode='speckle', mean=0, var=0.05, clip=True))          # for speckle, random_noise returns float64 for some reasons. Have to convert because tensors use single precision    
      noise     = torch.tensor( npy_noise )     
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AE3LAYERCONV2D: INFO:           type(x)                         (speckle)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(npy_noise)                 (speckle)      = {DULL_YELLOW}{type(npy_noise) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(x.cpu().numpy()[0,0,0,0])  (speckle)      = {DULL_YELLOW}{type(x.cpu().numpy()[0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(npy_noise[0,0,0,0])        (speckle)      = {DULL_YELLOW}{type(npy_noise      [0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           type(noise)                     (speckle)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AE3LAYERCONV2D: INFO:           x.size()                        (speckle)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AE3LAYERCONV2D: INFO:           noisy_x.size()                  (speckle)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )   
      
      return noisy_x
      
   
  def add_noise( self, x):
      p = np.random.random()
      if p <= 0.25:
          noisy_x = self.gaussian_noise( x )
      elif p <= 0.5:
          noisy_x = self.s_and_p_noise( x )
      elif p <= 0.75:
          noisy_x = self.poisson_noise( x )
      else:
          noisy_x = self.speckle_noise( x )
      return noisy_x
