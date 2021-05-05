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
class AEDCECCAE_5( nn.Module ):

  def __init__(  self, cfg, args, n_classes, tile_size  ):

    if DEBUG>0:
      print ( f"AEDCECCAE_5:    INFO:         __init__():  args.batch_size  = {MIKADO}{args.batch_size[0]}{RESET}", flush=True     ) 
    
    super(AEDCECCAE_5   , self).__init__()
   
    input_shape   =  [ tile_size, tile_size, 3, args.batch_size[0] ]
    # ~ num_clusters  =  7                                                                                 # = number of features output 
    num_clusters  =  args.gene_embed_dim[0]                                                                # = number of features output
    filters       =  [ 32, 64, 128, 256, 512] ]
    leaky           =  True
    kernel_default  =  5
    kernel_final    =  3
    padding_default =  2
    padding_final   =  0
    stride          =  2
    neg_slope       =  0.01
    activations     =  False
    bias            =  True
  
    self.activations  = activations
    self.pretrained   = False
    self.num_clusters = num_clusters
    self.input_shape  = input_shape
    self.filters      = filters

    num_layers = len(filters)
    
    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         __init__():  input_shape       = {MIKADO}{input_shape}{RESET}",     flush=True     )
      print ( f"AEDCECCAE_5:    INFO:         __init__():  filters           = {MIKADO}{filters}{RESET}",         flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  num_layers        = {MIKADO}{num_layers}{RESET}",      flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  num_clusters      = {MIKADO}{num_clusters}{RESET}",    flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  kernel_default    = {MIKADO}{kernel_default}{RESET}",  flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  kernel_final      = {MIKADO}{kernel_final}{RESET}",    flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  padding           = {MIKADO}{kernel_final}{RESET}",    flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  padding_final     = {MIKADO}{padding_final}{RESET}",   flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  neg_slope         = {MIKADO}{neg_slope}{RESET}",       flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  bias              = {MIKADO}{bias}{RESET}",            flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  activations       = {MIKADO}{activations}{RESET}",     flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  leaky             = {MIKADO}{leaky}{RESET}",           flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  pretrained        = {MIKADO}{self.pretrained}{RESET}", flush=True     ) 
      
    
    # For development only: calculate size of the outputs of each successive convolutional layer. Needed to to determine size required for first linear layer following convolutional layers. Formula is output width/height =((W-K+2*P )/S)+1
    if DEBUG>0:
      
      size_after_conv= [None] * num_layers
  
      size_after_conv[0] = int ( ( ( tile_size - kernel_default + 2*padding_default ) / stride ) + 1 )                            # first convolutional layer is special
  
      print ( f"AEDCECCAE_5:    INFO:         __init__():                                  size_after_conv[1] = {CARRIBEAN_GREEN}{size_after_conv[0]}{RESET}",                      flush=True     )
        
      for layer in range ( 1, num_layers-1 ):                                                                                     # all hidden layers are the same
    
        size_after_conv[layer] = int ( ( ( size_after_conv[layer-1] - kernel_default + 2*padding_default ) / stride ) + 1 )
        
        print ( f"AEDCECCAE_5:    INFO:         __init__():                                  size_after_conv[{layer+1}] = {CARRIBEAN_GREEN}{size_after_conv[layer]}{RESET}",        flush=True     )
  
  
      size_after_conv[num_layers-1] = int ( ( ( size_after_conv[layer] - kernel_final   + 2*padding_final ) / stride ) + 1 )        # final convolutional layer is special
      
      print ( f"AEDCECCAE_5:    INFO:         __init__():                                  size_after_conv[{num_layers}] = {CARRIBEAN_GREEN}{size_after_conv[num_layers-1]}{RESET}",  flush=True     )  

      print ( f"AEDCECCAE_5:    INFO:         __init__():                             summary of output sizes = {CARRIBEAN_GREEN}{size_after_conv}{RESET}",  flush=True     )      
    
      final_conv_output_size = size_after_conv[num_layers-1]
      first_fc_layer_size    = final_conv_output_size * final_conv_output_size * filters[len(filters)-1]
      print ( f"AEDCECCAE_5:    INFO:         __init__():   hence required size of first fully connected layer = {CARRIBEAN_GREEN}{first_fc_layer_size}{RESET}",  flush=True     )      
    
    
    
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=neg_slope)
    else:
      self.relu = nn.ReLU(inplace=False)

    self.conv1 = nn.Conv2d( input_shape[2], filters[0],  kernel_default, stride,  padding_default,  bias=bias)        
    self.conv2 = nn.Conv2d( filters[0],     filters[1],  kernel_default, stride,  padding_default,  bias=bias)
    self.conv3 = nn.Conv2d( filters[1],     filters[2],  kernel_default, stride,  padding_default,  bias=bias)
    self.conv4 = nn.Conv2d( filters[2],     filters[3],  kernel_default, stride,  padding_default,  bias=bias)
    self.conv5 = nn.Conv2d( filters[3],     filters[4],  kernel_final,   stride,  padding_final,    bias=bias)

    factor = (( input_shape[0] // 2 // 2 // 2 // 2 )-1 ) // 2 
    
    lin_features_len =  factor  *  factor  *  filters[num_layers-1]
    
    # ~ lin_features_len = ( ((input_shape[0] // 2 // 2 // 2 // 2 ) -1) // 2 ) * ( (input_shape[0] // 2 // 2 // 2 // 2-1) // 2) * filters[4]

    if DEBUG>0:
      print ( f"AEDCECCAE_5:    INFO:         __init__():  factor           = {MIKADO}{ factor }{RESET}",   flush=True     ) 
      print ( f"AEDCECCAE_5:    INFO:         __init__():  lin_features_len = factor * factor  * filters[4] = {MIKADO}{ factor }{RESET} * {MIKADO}{factor  }{RESET} * {MIKADO}{filters[4]}{RESET} = {MIKADO}{lin_features_len}{RESET}",   flush=True     ) 
     
    self.embedding   = nn.Linear( lin_features_len, num_clusters,   bias=bias)


    
    self.deembedding = nn.Linear( num_clusters,   lin_features_len, bias=bias)

    out_pad      = 1 if input_shape[0] // 2 // 2 // 2 // 2 % 2 == 0 else 0 
    self.deconv5 = nn.ConvTranspose2d( filters[4], filters[3],     kernel_final,   stride, padding_final,   output_padding=out_pad, bias=bias)
    
    out_pad      = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
    self.deconv4 = nn.ConvTranspose2d( filters[3], filters[2],     kernel_default, stride, padding_default, output_padding=out_pad, bias=bias)

    out_pad      = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
    self.deconv3 = nn.ConvTranspose2d( filters[2], filters[1],     kernel_default, stride, padding_default, output_padding=out_pad, bias=bias)

    out_pad      = 1 if input_shape[0] // 2 % 2 == 0 else 0
    self.deconv2 = nn.ConvTranspose2d( filters[1], filters[0],     kernel_default, stride, padding_default, output_padding=out_pad, bias=bias)

    out_pad      = 1 if input_shape[0] % 2 == 0 else 0
    self.deconv1 = nn.ConvTranspose2d( filters[0], input_shape[2], kernel_default, stride, padding_default, output_padding=out_pad, bias=bias)
    
    
    
    
    # ~ self.clustering = ClusteringLayer(num_clusters, num_clusters)

    # ReLU copies for graph representation in tensorboard
    self.relu1_1 = copy.deepcopy(self.relu)
    self.relu2_1 = copy.deepcopy(self.relu)
    self.relu3_1 = copy.deepcopy(self.relu)
    self.relu4_1 = copy.deepcopy(self.relu)
    self.relu5_1 = copy.deepcopy(self.relu)
    self.relu1_2 = copy.deepcopy(self.relu)
    self.relu2_2 = copy.deepcopy(self.relu)
    self.relu3_2 = copy.deepcopy(self.relu)
    self.relu4_2 = copy.deepcopy(self.relu)
    self.relu5_2 = copy.deepcopy(self.relu)

    self.sig  = nn.Sigmoid()
    self.tanh = nn.Tanh()
    
    

# ------------------------------------------------------------------------------
  def encode(self, x, gpu, encoder_activation ):
   

    x = self.conv1  (x)
    x = self.relu1_1(x)
    
    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after   conv1/relu1_1  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.conv2  (x)
    x = self.relu2_1(x)
    
    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after   conv2/relu2_1  = {ARYLIDE}{x.size()}{RESET}", flush=True   ) 
 
    x = self.conv3  (x)
    x = self.relu3_1(x)
    
    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after   conv3/relu3_1  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.conv4  (x)
    x = self.relu4_1(x)
    
    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after   conv4/relu4_1  = {ARYLIDE}{x.size()}{RESET}", flush=True   )  
 
    x = self.conv5  (x)

    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after   conv5          = {ARYLIDE}{x.size()}{RESET}", flush=True   ) 
    
    if self.activations:
      
        if DEBUG>1:
          print ( f"AEDCECCAE_5:    INFO:         encode():  applying sigmoid", flush=True     )
            
        x = self.sig(x)

    else:
      
        x = self.relu5_1(x)

        if DEBUG>99:
          print ( f"AEDCECCAE_5:    INFO:         encode():  applying relu5", flush=True     )    
        
    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after   conv5/relu5    = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 

    x = x.view(x.size(0), -1)                                                                              # flatten

    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after   x.view         = {ARYLIDE}{x.size()}{RESET}", flush=True     )     
    
    z = self.embedding(x)
  
    if DEBUG>0:
      print ( f"AEDCECCAE_5:    INFO:         encode():  z.size() after   embedding      = {BRIGHT_GREEN}{z.size()}{RESET}", flush=True   ) 
      
    return z
          
  # ------------------------------------------------------------------------------
  
  def decode( self, z ):
    
    x = self.deembedding(z)

    if DEBUG>9:
      print ( f"AEDCECCAE_5:    INFO:         decode():  z.size() after deembedding      = {CAMEL}{x.size()}{RESET}", flush=True   ) 
      
    x = self.relu5_2(x)
    x = x.view(x.size(0), self.filters[4], ((self.input_shape[0]//2//2//2//2-1) // 2), ((self.input_shape[0]//2//2//2//2-1) // 2))

    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after x.view/relu4_2   = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.deconv5(x)
    x = self.relu4_2(x)

    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after deconv5/relu4_2  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.deconv4(x)
    x = self.relu3_2(x)

    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after deconv4/relu3_2  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.deconv3(x)
    x = self.relu2_2(x)

    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after deconv3/relu2_2  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.deconv2(x)
    x = self.relu1_2(x)

    if DEBUG>1:
      print ( f"AEDCECCAE_5:    INFO:         encode():  x.size() after deconv2/relu1_2  = {ARYLIDE}{x.size()}{RESET}", flush=True     ) 
      
    x = self.deconv1(x)
    
    if self.activations:
      x = self.tanh(x)
  
    if DEBUG>9: 
      print ( f"AEDCECCAE_5:    INFO:         decode():  x.size() after deconv1/actv     = {CAMEL}{x.size()}{RESET}", flush=True   ) 
    
    return x
  


  # ------------------------------------------------------------------------------
  
  def forward( self, args, x, gpu, encoder_activation ):
  
    if DEBUG>9:
      print ( f"\nAEDCECCAE_5:    INFO:       forward():   x.size() before encode          = {BOLD}{RED}{x.size()}{RESET}", flush=True   ) 


# MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE 

    if ( args.just_test!= 'True' ) & ( args.peer_noise_perunit != 0 ):

      if DEBUG>0:
        print ( f"{ORANGE}AEDCECCAE_5:    INFO:       forward():   NOISE IS BEING ADDED{RESET}", flush=True   )       
      
      x = self.add_peer_noise( x, args.peer_noise_perunit  )                                                  # then add peer noise to this batch of images

# MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE MOD TO ADD PEER NOISE 


    z = self.encode( x, gpu, encoder_activation)    
    
    # ~ clustering_out = self.clustering(z)

    if DEBUG>9:
      print ( f"AEDCECCAE_5:    INFO:       forward():   z.size() before decode          = {BRIGHT_GREEN}{z.size()}{RESET}", flush=True   ) 

    x = self.decode(z)

    if DEBUG>9:
      print ( f"AEDCECCAE_5:    INFO:       forward():   x.size() after decode           = {BOLD}{RED}{x.size()}{RESET}\n", flush=True   ) 

    # ~ return x, clustering_out
    return x, 0, 0



  # ------------------------------------------------------------------------------
  # HELPER FUNCTIONS
  # ------------------------------------------------------------------------------  
  
  # Methods to add noise are from https://dropsofai.com/convolutional-denoising-autoencoders-for-image-noise-reduction/
  # By Kartik Chaudhary | November 10, 2020
  
  # except for add_peer_noise() which is by me 29 April 2021

  def add_peer_noise( self, images, peer_noise_perunit ):
  
    if DEBUG>3:
      print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()                   type( images)       = {CARRIBEAN_GREEN}{   type( images)  }{RESET}", flush=True   )
      print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()                   images.size         = {CARRIBEAN_GREEN}{    images.size() }{RESET}", flush=True   )
      
    images_NPY  = images.cpu().numpy()
  
    if DEBUG>3:
      print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()                   type( images_NPY)   = {COTTON_CANDY}{   type( images_NPY) }{RESET}", flush=True   )
      print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()                   images_NPY.shape    = {COTTON_CANDY}{    images_NPY.shape }{RESET}", flush=True   )
  
  
    for i in range( 0, images_NPY.shape[0]-1 ):
   
      target = np.random.randint( 0, images_NPY.shape[0]-1 )
  
      if DEBUG>3:
        print ( f"\nAEDCECCAE_5:    INFO:    add_peer_noise()   about to add {MIKADO}{peer_noise_perunit*100}{RESET} % 'peer noise' {BOLD}from{RESET} image {MIKADO}{target:^4d}{RESET} in the current batch {BOLD}to{RESET} image {MIKADO}{i:^5d}{RESET} in the current batch.",        flush=True        )
        
      if DEBUG>3:
        print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()   images_NPY     [{BLEU}{i:5d}{RESET}] = {BLEU}{images_NPY[i,0,0,0:-1]}{RESET} ",       flush=True        )
  
      if DEBUG>3:
        print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()   image          [{BLEU}{i:5d}{RESET}] = {BLEU}{images_NPY[target,0,0,0:-1]}{RESET} ",                    flush=True   )
        
      images_NPY[i,:,:,:] =  images_NPY[i,:,:,:] + peer_noise_perunit * images_NPY[target,:,:,:]
  
      max_value = np.amax( images_NPY[i,:,:,:] )
      images_NPY[i,:,:,:] = images_NPY[i,:,:,:] / max_value * 255
  
      images_NPY = np.around( images_NPY, decimals=0, out=None)
  
      if DEBUG>3:
        print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()   images_NPY_NORM[{BITTER_SWEET}{i:5d}{RESET}] = {BITTER_SWEET}{images_NPY[i,0,0,0:-1]}{RESET} ",    flush=True   )
        print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()   max_value                                    = {BITTER_SWEET}{max_value:.0f}{RESET} ",                      flush=True   )
  
    images_TORCH = torch.from_numpy (images_NPY ).cuda()
  
    if DEBUG>3:
      print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()                   type( images_TORCH)   = {BITTER_SWEET}{   type( images_TORCH) }{RESET}", flush=True   )
      print ( f"AEDCECCAE_5:    INFO:    add_peer_noise()                   images_TORCH.size     = {BITTER_SWEET}{    images_TORCH.size()}{RESET}", flush=True   )
    
    return images_TORCH

  # ------------------------------------------------------------------------------  
  
  def gaussian_noise( self, x):
  
      if DEBUG>9:
        print ( f"AEDCECCAE_5:    INFO:           gaussian_noise():  x.size()       = {DULL_YELLOW}{x.size()}{RESET}", flush=True   ) 
      
      mean = 0
      var  = 0.1
      
      npy_noise = np.float32( random_noise( x.cpu(), mode='gaussian', mean=mean, var=var, clip=True))
      noise     = torch.tensor( npy_noise )  
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_5:    INFO:           type(x)                         (gaussian)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(noise)                     (gaussian)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           x.size()                        (gaussian)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_5:    INFO:           noisy_x.size()                  (gaussian)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x
   
  # ------------------------------------------------------------------------------  
     
  def s_and_p_noise( self, x):

      amount = 0.5
          
      noise = torch.tensor( random_noise( x.cpu().numpy(), mode='s&p', salt_vs_pepper=amount, clip=True) )  
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_5:    INFO:           type(x)                         (s_and_p)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(noise)                     (s_and_p)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           x.size()                        (s_and_p)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_5:    INFO:           noisy_x.size()                  (s_and_p)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x
   

  # ------------------------------------------------------------------------------  
     
  def poisson_noise( self, x):
      
      npy_noise = np.float32(random_noise( x.cpu(), mode='poisson', clip=True))                    # for poisson, random_noise returns float64 for some reasons. Have to convert because tensors use single precision
      noise     = torch.tensor( npy_noise )     
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_5:    INFO:           type(x)                         (poisson)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(npy_noise)                 (poisson)      = {DULL_YELLOW}{type(npy_noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(x.cpu().numpy()[0,0,0,0])  (poisson)      = {DULL_YELLOW}{type(x.cpu().numpy()[0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(npy_noise[0,0,0,0])        (poisson)      = {DULL_YELLOW}{type(npy_noise      [0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(noise)                     (poisson)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           x.size()                        (poisson)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_5:    INFO:           noisy_x.size()                  (poisson)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )  
      
      return noisy_x


  # ------------------------------------------------------------------------------  
  
  def speckle_noise( self, x):

      npy_noise = np.float32(random_noise( x.cpu(), mode='speckle', mean=0, var=0.05, clip=True))          # for speckle, random_noise returns float64 for some reasons. Have to convert because tensors use single precision    
      noise     = torch.tensor( npy_noise )     
      noisy_x   = x + noise.cuda()

      if DEBUG>0:
        print ( f"AEDCECCAE_5:    INFO:           type(x)                         (speckle)      = {DULL_YELLOW}{type(x) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(npy_noise)                 (speckle)      = {DULL_YELLOW}{type(npy_noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(x.cpu().numpy()[0,0,0,0])  (speckle)      = {DULL_YELLOW}{type(x.cpu().numpy()[0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(npy_noise[0,0,0,0])        (speckle)      = {DULL_YELLOW}{type(npy_noise      [0,0,0,0]) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           type(noise)                     (speckle)      = {DULL_YELLOW}{type(noise) }{RESET}", flush=True   )  
        print ( f"AEDCECCAE_5:    INFO:           x.size()                        (speckle)      = {DULL_YELLOW}{x.size()}{RESET}", flush=True        )  
        print ( f"AEDCECCAE_5:    INFO:           noisy_x.size()                  (speckle)      = {DULL_YELLOW}{noisy_x.size()}{RESET}", flush=True   )   
      
      return noisy_x
      
  # ------------------------------------------------------------------------------  
     
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
