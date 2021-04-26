"""============================================================================================================================================================
Deep Convolutional Autoencoder

  based on paper:
    "Deep Clustering with Convolutional Autoencoders" - Xifeng Guo1, Xinwang Liu1, En Zhu1, and Jianping Yin2
  
  this pytorch implementation by:
    "Michał Nazarczuk - PhD Student at Imperial College London, Intelligent Systems and Networks research group"
    
Note: implemenation used only a forward function - had to split it out into encode/decode to make it compatible with my existing overall software framework

============================================================================================================================================================="""

import copy
import torch
from   torch import nn  
from   torch import sigmoid
from   torch import relu
from   torch import tanh

import  numpy as np

from skimage.util import random_noise

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

DEBUG=0

# ------------------------------------------------------------------------------

# Clustering layer definition (see DCEC article for equations)
class ClusteringLayer(nn.Module):
  
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
        
# ------------------------------------------------------------------------------
class AEDCECCAE_3( nn.Module ):

  def __init__(  self, cfg, args, n_classes, tile_size  ):

    if DEBUG>1:
      print ( f"AEDCECCAE_3:    INFO:         __init__():  args.batch_size  = {MIKADO}{args.batch_size[0]}{RESET}", flush=True     ) 
    
    super(AEDCECCAE_3   , self).__init__()
   
    input_shape   =  [ tile_size, tile_size, 3, args.batch_size[0] ]
    num_clusters  =  10
    filters       =  [32, 64, 128]
    leaky         =  True
    neg_slope     =  0.01
    activations   =  False
    bias          =  True
  
    self.activations  = activations
    self.pretrained   = False
    self.num_clusters = num_clusters
    self.input_shape  = input_shape
    self.filters      = filters

    if DEBUG>1:
      print ( f"AEDCECCAE_3:    INFO:         __init__():  input_shape        = {MIKADO}{input_shape}{RESET}", flush=True     ) 
      
      
    self.conv1 = nn.Conv2d(input_shape[2], filters[0],  5, stride=2,  padding=2, bias=bias)
    
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=neg_slope)
    else:
      self.relu = nn.ReLU(inplace=False)
        
    self.conv2       = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
    
    self.conv3       = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)

    lin_features_len = ((input_shape[0]//2//2-1) // 2) * ((input_shape[0]//2//2-1) // 2) * filters[2]
    
    self.embedding   = nn.Linear(lin_features_len, num_clusters, bias=bias)
    self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)

    out_pad          = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
    self.deconv3     = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)

    out_pad          = 1 if input_shape[0] // 2 % 2 == 0 else 0
    self.deconv2     = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)

    out_pad         = 1 if input_shape[0] % 2 == 0 else 0
    self.deconv1    = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
    
    # ~ self.clustering = ClusteringLayer(num_clusters, num_clusters)

    # ReLU copies for graph representation in tensorboard
    self.relu1_1 = copy.deepcopy(self.relu)
    self.relu2_1 = copy.deepcopy(self.relu)
    self.relu3_1 = copy.deepcopy(self.relu)
    self.relu1_2 = copy.deepcopy(self.relu)
    self.relu2_2 = copy.deepcopy(self.relu)
    self.relu3_2 = copy.deepcopy(self.relu)

    self.sig  = nn.Sigmoid()
    self.tanh = nn.Tanh()
    
    

# ------------------------------------------------------------------------------
  def encode(self, x, gpu, encoder_activation ):
   

    x = self.conv1(x)
    x = self.relu1_1(x)
    if DEBUG>1:
      print ( f"AEDCECCAE_3:    INFO:         encode():  x.size() after conv1/relu1  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.conv2(x)
    x = self.relu2_1(x)
    if DEBUG>1:
      print ( f"AEDCECCAE_3:    INFO:         encode():  x.size() after conv2/relu2  = {ARYLIDE}{x.size()}{RESET}", flush=True   ) 
      
    x = self.conv3(x)
    if self.activations:
        x = self.sig(x)
    else:
        x = self.relu3_1(x)
        
    if DEBUG>1:
      print ( f"AEDCECCAE_3:    INFO:         encode():  x.size() after conv3/relu3  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 


    x = x.view(x.size(0), -1)

    if DEBUG>1:
      print ( f"AEDCECCAE_3:    INFO:         encode():  x.size() after  x.view      = {ARYLIDE}{x.size()}{RESET}", flush=True     )     
    
    z = self.embedding(x)
    
  
    if DEBUG>0:
      print ( f"AEDCECCAE_3:    INFO:         encode():  z.size() after embedding    = {BRIGHT_GREEN}{z.size()}{RESET}", flush=True   ) 
      
    return z
          
  # ------------------------------------------------------------------------------
  
  def decode( self, z ):
    
    x = self.deembedding(z)

    if DEBUG>9:
      print ( f"AEDCECCAE_3:    INFO:         decode():  z.size() after deembedding  = {CAMEL}{x.size()}{RESET}", flush=True   ) 
      
      
    x = self.relu1_2(x)

    x = x.view(x.size(0), self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))

    if DEBUG>9:
      print ( f"AEDCECCAE_3:    INFO:         decode():  x.size() after x.view       = {CAMEL}{x.size()}{RESET}", flush=True   ) 
      
    x = self.deconv3(x)
    x = self.relu2_2(x)

    if DEBUG>9:
      print ( f"AEDCECCAE_3:    INFO:         decode():  x.size() after deconv3/relu = {CAMEL}{x.size()}{RESET}", flush=True   ) 
      
    x = self.deconv2(x)
    x = self.relu3_2(x)

    if DEBUG>9:
      print ( f"AEDCECCAE_3:    INFO:         decode():  x.size() after deconv2/relu = {CAMEL}{x.size()}{RESET}", flush=True   ) 
      
      
    x = self.deconv1(x)
    if self.activations:
        x = self.tanh(x)
  
    if DEBUG>9: 
      print ( f"AEDCECCAE_3:    INFO:         decode():  x.size() after deconv1/actv = {CAMEL}{x.size()}{RESET}", flush=True   ) 
    
    return x
  
  # ------------------------------------------------------------------------------
  
  def forward( self, args, x, gpu, encoder_activation ):
  
    if DEBUG>9:
      print ( f"\nAEDCECCAE_3:    INFO:       forward():   x.size() before encode      = {BOLD}{RED}{x.size()}{RESET}", flush=True   ) 

        
    # ~ x = self.gaussian_noise( x )
    # ~ x = self.s_and_p_noise( x )
    # ~ x = self.poisson_noise( x )
    # ~ x = self.speckle_noise( x )

    if args.ae_add_noise=='True':

      if DEBUG>9:
        print ( f"{BOLD}{RED}AEDCECCAE_3:    INFO:       forward():   NOISE IS BEING ADDED{RESET}", flush=True   ) 

      x = self.add_noise( x )
    
    z = self.encode( x, gpu, encoder_activation)    
    
    # ~ clustering_out = self.clustering(z)

    if DEBUG>9:
      print ( f"AEDCECCAE_3:    INFO:       forward():   z.size() before decode      = {BRIGHT_GREEN}{z.size()}{RESET}", flush=True   ) 

    x = self.decode(z)

    if DEBUG>9:
      print ( f"AEDCECCAE_3:    INFO:       forward():   x.size() after decode       = {BOLD}{RED}{x.size()}{RESET}", flush=True   ) 

    # ~ return x, clustering_out
    return x, 0, 0



  # ------------------------------------------------------------------------------
  # HELPER FUNCTIONS
  # ------------------------------------------------------------------------------  
  
  # Methods to add noise are from https://dropsofai.com/convolutional-denoising-autoencoders-for-image-noise-reduction/
  # By Kartik Chaudhary | November 10, 2020


  def gaussian_noise( self, x):
  
      if DEBUG>9:
        print ( f"AEDCECCAE_3:    INFO:           gaussian_noise():  x.size()       = {DULL_YELLOW}{x.size()}{RESET}", flush=True   ) 
      
      mean = 0
      var  = 0.1
      
      npy_noise = np.float32(random_noise( x.cpu(), mode='gaussian', mean=mean, var=var, clip=True))
      noise     = torch.tensor( npy_noise )  
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_3:    INFO:           type(x)                         (gaussian)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(noise)                     (gaussian)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           x.size()                        (gaussian)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_3:    INFO:           noisy_x.size()                  (gaussian)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x
   
   
  def s_and_p_noise( self, x):

      amount = 0.5
          
      noise = torch.tensor(random_noise( x.cpu().numpy(), mode='s&p', salt_vs_pepper=amount, clip=True) )  
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_3:    INFO:           type(x)                         (s_and_p)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(noise)                     (s_and_p)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           x.size()                        (s_and_p)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_3:    INFO:           noisy_x.size()                  (s_and_p)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x
   
   
  def poisson_noise( self, x):
      
      npy_noise = np.float32(random_noise( x.cpu(), mode='poisson', clip=True))                    # for poisson, random_noise returns float64 for some reasons. Have to convert because tensors use single precision
      noise     = torch.tensor( npy_noise )     
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_3:    INFO:           type(x)                         (poisson)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(npy_noise)                 (poisson)      = {DULL_YELLOW}{type(npy_noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(x.cpu().numpy()[0,0,0,0])  (poisson)      = {DULL_YELLOW}{type(x.cpu().numpy()[0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(npy_noise[0,0,0,0])        (poisson)      = {DULL_YELLOW}{type(npy_noise      [0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(noise)                     (poisson)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           x.size()                        (poisson)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_3:    INFO:           noisy_x.size()                  (poisson)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x


  def speckle_noise( self, x):

      npy_noise = np.float32(random_noise( x.cpu(), mode='speckle', mean=0, var=0.05, clip=True))          # for speckle, random_noise returns float64 for some reasons. Have to convert because tensors use single precision    
      noise     = torch.tensor( npy_noise )     
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_3:    INFO:           type(x)                         (speckle)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(npy_noise)                 (speckle)      = {DULL_YELLOW}{type(npy_noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(x.cpu().numpy()[0,0,0,0])  (speckle)      = {DULL_YELLOW}{type(x.cpu().numpy()[0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(npy_noise[0,0,0,0])        (speckle)      = {DULL_YELLOW}{type(npy_noise      [0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           type(noise)                     (speckle)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_3:    INFO:           x.size()                        (speckle)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_3:    INFO:           noisy_x.size()                  (speckle)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )   
      
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