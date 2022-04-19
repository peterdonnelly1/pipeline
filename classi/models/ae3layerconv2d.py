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

from constants  import *

DEBUG=0

# ------------------------------------------------------------------------------
class AE3LAYERCONV2D( nn.Module ):

  def __init__(  self, cfg, args, n_classes, tile_size  ):

    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:    at {CYAN} __init__(){RESET}" )
    
    super(AE3LAYERCONV2D, self).__init__()

    input_shape     =  [ tile_size, tile_size, 3, args.batch_size[0] ]
    filters         =  [ 36, 144, 3 ]
    kernel_default  =  3
    kernel_final    =  7
    padding_default =  1
    padding_final   =  0
    stride_default  =  2
    stride_final    =  1
    out_pad         =  1
    bias            =  True

    num_layers = len(filters)    

    self.f1 = nn.Conv2d         ( input_shape[2], filters[0],     kernel_default,  stride_default,  padding=padding_default  )        
    self.f2 = nn.Conv2d         ( filters[0],     filters[1],     kernel_default,  stride_default,  padding=padding_default  )
    self.f3 = nn.Conv2d         ( filters[1],     filters[2],     kernel_final,    stride_final,    padding=padding_final    )

    self.r1 = nn.ConvTranspose2d( filters[2],     filters[1],     kernel_final,    stride_final,    padding=padding_final,                            )        
    self.r2 = nn.ConvTranspose2d( filters[1],     filters[0],     kernel_default,  stride_default,  padding=padding_default,  output_padding=out_pad  )
    self.r3 = nn.ConvTranspose2d( filters[0],     input_shape[2], kernel_default,  stride_default,  padding=padding_default,  output_padding=out_pad  )
               
    # ~ self.f1       = nn.Conv2d         ( in_channels=3,   out_channels=36,  kernel_size=3, stride_default=2, padding=1                     )
    # ~ self.f2       = nn.Conv2d         ( in_channels=36,  out_channels=144, kernel_size=3, stride_default=2, padding=1                     )
    # ~ self.f3       = nn.Conv2d         ( in_channels=144, out_channels=3,   kernel_size=7                                          ) 

    # ~ self.r1       = nn.ConvTranspose2d( in_channels=3,   out_channels=144, kernel_size=7                                          )
    # ~ self.r2       = nn.ConvTranspose2d( in_channels=144, out_channels=36,  kernel_size=3, stride_default=2, padding=1, output_padding=1   )
    # ~ self.r3       = nn.ConvTranspose2d( in_channels=36,  out_channels=3,   kernel_size=3, stride_default=2, padding=1, output_padding=1   )  
    

    size_after_conv= [None] * num_layers

    size_after_conv[0] = int ( ( ( tile_size - kernel_default + 2*padding_default ) / stride_default ) + 1 )                            # first convolutional layer is special

    print ( f"AE3LAYERCONV2D: INFO:         __init__():                                  size_after_conv[1] = {CARRIBEAN_GREEN}{size_after_conv[0]}{RESET}",                        flush=True     )
      
    for layer in range ( 1, num_layers-1 ):                                                                                     # all hidden layers are the same
  
      size_after_conv[layer] = int ( ( ( size_after_conv[layer-1] - kernel_default + 2*padding_default ) / stride_default ) + 1 )
      
      print ( f"AE3LAYERCONV2D: INFO:         __init__():                                  size_after_conv[{layer+1}] = {CARRIBEAN_GREEN}{size_after_conv[layer]}{RESET}",          flush=True     )


    size_after_conv[num_layers-1] = int ( ( ( size_after_conv[layer] - kernel_final   + 2*padding_final ) / stride_final ) + 1 )        # final convolutional layer is special
    
    print ( f"AE3LAYERCONV2D: INFO:         __init__():                                  size_after_conv[{num_layers}] = {CARRIBEAN_GREEN}{size_after_conv[num_layers-1]}{RESET}",  flush=True     )  

    print ( f"AE3LAYERCONV2D: INFO:         __init__():                             summary of output sizes = {CARRIBEAN_GREEN}{size_after_conv}{RESET}",  flush=True                              )      
  
    final_conv_output_size = size_after_conv[num_layers-1]
    first_fc_layer_size    = final_conv_output_size * final_conv_output_size * filters[len(filters)-1]
    print ( f"AE3LAYERCONV2D: INFO:         __init__():   hence required size of first fully connected layer = {MIKADO}{final_conv_output_size}{RESET} * {MIKADO}{final_conv_output_size}{RESET} * {MIKADO}{filters[num_layers-1]}{RESET} = {CARRIBEAN_GREEN}{first_fc_layer_size}{RESET}", flush=True     )   

    self.embedding   = nn.Linear( first_fc_layer_size, args.embedding_dimensions[0],   bias=bias)




# ------------------------------------------------------------------------------
  def encode_no_x_view(self, x, gpu, args ):                                                 # used for training
   
    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {DULL_YELLOW}{x.size()}{RESET}", flush=True   ) 

    if DEBUG>99:
      print ( f"AE3LAYERCONV2D: INFO:       encode(): x  = {CYAN}{x}{RESET}",                        flush=True   ) 
        
    x =  relu(self.f1(x))

    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {PINK}{x.size()}{RESET}",        flush=True   )     
    
    
    x =  relu(self.f2(x))

    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {ARYLIDE}{x.size()}{RESET}",     flush=True   ) 

    z =  relu(self.f3(x))
  
    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  z.size() = {BLEU}{x.size()}{RESET}",        flush=True   ) 
            
    return z
          
# ------------------------------------------------------------------------------
  def encode(self, x, gpu, args ):                                                           # used to generate the embeddings
   
    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {DULL_YELLOW}{x.size()}{RESET}", flush=True   ) 

    if DEBUG>99:
      print ( f"AE3LAYERCONV2D: INFO:       encode(): x  = {CYAN}{x}{RESET}",                        flush=True   ) 
        
    x =  relu(self.f1(x))

    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {PINK}{x.size()}{RESET}",        flush=True   )     
    
    
    x =  relu(self.f2(x))

    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {ARYLIDE}{x.size()}{RESET}",     flush=True   ) 

    x =  relu(self.f3(x))
  
    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {BLEU}{x.size()}{RESET}",        flush=True   ) 
      
    x = x.view(x.size(0), -1)                                                                              # flatten

    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  x.size() = {BLEU}{x.size()}{RESET}",        flush=True   ) 
 
    z = self.embedding(x)
  
    if DEBUG>0:
      print ( f"AE3LAYERCONV2D: INFO:         encode():  z.size() after   embedding      = {BRIGHT_GREEN}{z.size()}{RESET}", flush=True   )  
            
    return z
          
  # ------------------------------------------------------------------------------
  
  def decode( self, z ):
    
    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:         decode():  z.size() = {CAMEL}{z.size()}{RESET}", flush=True     )  
    
    x =  relu( self.r1(z) )
    x =  relu( self.r2(x) )
    x =  self.r3(x)       
  
    if DEBUG>1: 
      print ( f"AE3LAYERCONV2D: INFO:         decode():  x.size() = {HOT_PINK}{x.size()}{RESET}", flush=True  )  
    
    return x
  
  # ------------------------------------------------------------------------------
  
  def forward(  self, x, gpu, args ):
  
    if DEBUG>1:
      print ( f"AE3LAYERCONV2D: INFO:       forward():  x.size()  = {AMETHYST}{x.size()}{RESET}", flush=True  )  
    
    
    # ~ x = self.gaussian_noise( x )
    # ~ x = self.s_and_p_noise( x )
    # ~ x = self.poisson_noise( x )
    # ~ x = self.speckle_noise( x )
    
    
    if args.ae_add_noise=='True':

      if DEBUG>9:
        print ( f"{BOLD}{RED}AE3LAYERCONV2D: INFO:       forward():   NOISE IS BEING ADDED{RESET}", flush=True   ) 

      x = self.add_noise( x )
      
    
    z = self.encode_no_x_view( x, gpu, args)

  
    if DEBUG>9:
      print ( f"AE3LAYERCONV2D: INFO:       forward(): z.size()   = {ASPARAGUS}{x.size()}{RESET}", flush=True   )  
    
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
