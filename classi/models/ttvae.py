"""================================================================================================================================
TTVAE Autoencoder
Slightly adapted version of "MethylNet" :
    "METHYLNET: AN AUTOMATED AND MODULAR DEEP LEARNING APPROACH FOR DNA METHYLATION ANALYSIS"
By:
     Joshua J. Levy1,2* , Alexander J. Titus3, Curtis L. Petersen1,2,4, Youdinghuan Chen1,2, Lucas A. Salas2 and Brock C. Christensen2,5
Source:
     https://github.com/Christensen-Lab-Dartmouth/MethylNet
==================================================================================================================================="""

import copy
import inspect
import numpy as np

import torch
from   torch import nn
from   torch import sigmoid
from   torch import relu
from   torch import tanh
from   torch.nn import functional as F


#from methylnet.plotter     import * 
from torch.autograd        import Variable                                                                      # reinstate later
from sklearn.preprocessing import LabelEncoder
#from pymethylprocess.visualizations import umap_embed, plotly_plot

RANDOM_SEED=42
np.random.seed   (RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

        
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
ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
RESET='\033[m'

UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=9999

# ------------------------------------------------------------------------------

class TTVAE( nn.Module) :
  
  
  """Pytorch NN Module housing VAE with fully connected layers and customizable topology.

  Parameters
  ----------
  n_input : type
    Number of input CpGs
  n_latent : type
    Size of latent embeddings.
  hidden_layer_encoder_topology : type
    List, length of list contains number of hidden layers for encoder, and each element is number of neurons, mirrored for decoder.
  cuda : type
    GPU?

  Attributes
  ----------
  cuda_on : type
    GPU?
  pre_latent_topology : type
    Hidden layer topology for encoder.
  post_latent_topology : type
    Mirrored hidden layer topology for decoder.
  encoder_layers : list
    Encoder pytorch layers.
  encoder : type
    Encoder layers wrapped into pytorch module.
  z_mean : type
    Linear layer from last encoder layer to mean layer.
  z_var : type
    Linear layer from last encoder layer to var layer.
  z_develop : type
    Linear layer connecting sampled latent embedding to first layer decoder.
  decoder_layers : type
    Decoder layers wrapped into pytorch module.
  output_layer : type
    Linear layer connecting last decoder layer to output layer, which is same size as input..
  decoder : type
    Wraps decoder_layers and output_layers into Sequential module.
  n_input
  n_latent

  """

  torch.set_printoptions( edgeitems = 6     )
  torch.set_printoptions( linewidth = 250   )
  torch.set_printoptions( precision = 2     )
  torch.set_printoptions( sci_mode  = False )


  def __init__( self, cfg, args, gpu, rank, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2    ):
    
#    cuda=False
    cuda=True # PGD
    
    if DEBUG>9:
      print ( f"TTVAE:          INFO:    at {MIKADO} __init__(){RESET}" )
    
    super( TTVAE, self).__init__()

    hidden_layer_encoder_topology =  args.hidden_layer_encoder_topology
    n_input                       =  n_genes
    self.n_input                  =  n_genes
    n_latent                      =  embedding_dimensions
    self.n_latent                 =  embedding_dimensions
    self.cuda_on                  =  cuda
    self.pre_latent_topology      =  [n_input]  + (hidden_layer_encoder_topology       if hidden_layer_encoder_topology else [])  # layer before the output (latent layer)
    self.post_latent_topology     =  [n_latent] + (hidden_layer_encoder_topology[::-1] if hidden_layer_encoder_topology else [])  # layer after output
    self.encoder_layers           = []

    if DEBUG>9:
      print ( f"TTVAE:          INFO:  pre_latent_topology           = {MIKADO}{self.pre_latent_topology}{RESET}",       flush=True   )
    if len(self.pre_latent_topology)>1:                                                                    # if more than one pre-latent layer is defined, then establish those layers
      for i in range(len(self.pre_latent_topology)-1):
        layer = nn.Linear( self.pre_latent_topology[i], self.pre_latent_topology[i+1] )                    # add another linear later with dimensions derived from hidden_layer_encoder_topology vector
        torch.nn.init.xavier_uniform_(layer.weight)                                                        # specify Xavier initialization
        self.encoder_layers.append( nn.Sequential( layer,nn.ReLU() )  )

    self.encoder        = nn.Sequential( *self.encoder_layers ) if self.encoder_layers else nn.Dropout( p=0.0 )
    if DEBUG>9:
      print ( f"TTVAE:          INFO:    encoder_layers = \n {MIKADO}{self.encoder_layers}{RESET}", flush=True   )
    self.z_mean         = nn.Sequential( nn.Linear( self.pre_latent_topology[-1], n_latent ), 
                          nn.BatchNorm1d( n_latent )                                       )               # "Learned means"  (BatchNorm1d "Applies Batch Normalization over a 2D or 3D input")
    if DEBUG>9: 
      print ( f"{MIKADO}{self.z_mean}{RESET}",     flush=True   )
    self.z_var          = nn.Sequential( nn.Linear( self.pre_latent_topology[-1], n_latent ), 
                          nn.BatchNorm1d( n_latent )                                       )               # "Learned vars"   (BatchNorm1d "Applies Batch Normalization over a 2D or 3D input")
    if DEBUG>9: 
      print ( f"{MIKADO}{self.z_var}{RESET}",      flush=True   )
    self.z_develop      = nn.Linear    (   n_latent, self.pre_latent_topology[-1]   )                      # layer connecting sampled latent embedding to first layer decoder.
    if DEBUG>9: 
      print ( f"{MIKADO}{self.z_develop}{RESET}",  flush=True   )

    self.decoder_layers = []      
    if len(self.post_latent_topology)>1:                                                                   # i.e. if more than one post-latent layer is defined, then establish those layers
      for i in range(len(self.post_latent_topology)-1):
        layer           = nn.Linear(self.post_latent_topology[i],self.post_latent_topology[i+1])
        torch.nn.init.xavier_uniform_(layer.weight)                                                        # specify Xavier initialization
        self.decoder_layers.append(nn.Sequential(layer,nn.ReLU()))
    self.decoder_layers = nn.Sequential(*self.decoder_layers)
    self.output_layer   = nn.Sequential(nn.Linear( self.post_latent_topology[-1], n_input ))
    if self.decoder_layers:
      self.decoder = nn.Sequential(*[self.decoder_layers,self.output_layer])
    else:
      self.decoder = self.output_layer

    if DEBUG>9: 
      print ( f"{MIKADO}{self.decoder_layers}{RESET}", flush=True   )
      print ( f"{MIKADO}{self.output_layer}{RESET}",   flush=True   )


  def encode( self, x, gpu, encoder_activation ):
    """Encode input into latent representation.

    Parameters
    ----------
    x : type
      Input methylation data.

    Returns
    -------
    torch.tensor
      Learned mean vector of embeddings.
    torch.tensor
      Learned variance of learned mean embeddings.
    """

    if DEBUG>99:
      cuda_check = x.is_cuda
      if cuda_check:
        get_cuda_device = x.get_device()
      print ( f"TTVAE:          INFO:         {BRIGHT_GREEN}DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! encode(): x.get_device() = {MIKADO}{get_cuda_device}{RESET}",    flush=True   )
    
    
    x = self.encoder(x)


    if DEBUG>9:
      print ( f"TTVAE:          INFO:         encode(): x_latent.shape = {MIKADO}{x.shape}{RESET}",    flush=True   )
    if DEBUG>9:      
      print ( f"TTVAE:          INFO:         encode(): x_latent (from which means and vars will be taken      = \n{MIKADO}{x}{RESET}",        flush=True   ) 

    mean = self.z_mean(x)                                                                                  # z_mean is the name of the array holding the means of x, it doesn't contain the sampling logic
    var =  self.z_var (x)                                                                                  # z_var  is the name of the array holding the vars  of x,  it doesn't contain the sampling logic

    if DEBUG>9:
      print ( f"TTVAE:          INFO:         encode(): mean.shape     = {MIKADO}{mean.shape}{RESET}", flush=True   ) 
      print ( f"TTVAE:          INFO:         encode(): var.shape      = {MIKADO}{var.shape}{RESET}",  flush=True   )       

    if DEBUG>9:
      print ( f"TTVAE:          INFO:         encode(): mean           = \n{MIKADO}{mean}{RESET}",     flush=True   ) 
      print ( f"TTVAE:          INFO:         encode(): var            = \n{MIKADO}{var}{RESET}",      flush=True   ) 


    return mean, var



  def decode(self, z):
    """Decode latent embeddings back into reconstructed input.

    Parameters
    ----------
    z : type
      Reparameterized latent embedding.

    Returns
    -------
    torch.tensor
      Reconstructed input.

    """

    out = self.decoder(z)

    return out


  def sample_z(self, gpu, mean, logvar):
    
    """Sample latent embeddings, reparameterize by adding noise to embedding.

    Parameters
    ----------
    mean : type
      Learned mean vector of embeddings             (due to Batch Normalization)
    logvar : type
      Learned variance of learned mean embeddings.  (due to Batch Normalization)

    Returns
    -------
    torch.tensor
      Mean + noise, reparameterization trick.

    """
    stddev = torch.exp(0.5 * logvar)
    
    noise  = Variable(torch.randn(stddev.size()))                                                          # define pytorch variable called 'noise'
    
    if self.cuda_on:
      noise=noise.cuda(gpu)
    
    if not self.training:
      noise  = 0.
      stddev = 0.
    
    return (noise * stddev) + mean




  def forward( self, x, gpu, encoder_activation ):
    
    """Return reconstructed output, mean and variance of embeddings.
    """

    if DEBUG>9:
      print ( f"TTVAE:          INFO:       forward() about to take a single encode/decode step" )
    
    mean, logvar = self.encode( x, gpu, encoder_activation )
    
    if DEBUG>9:
      print ( f"TTVAE:          INFO:       forward(): mean.shape    = {MIKADO}{mean.shape}{RESET}",    flush=True   ) 
      print ( f"TTVAE:          INFO:       forward(): logvar.shape  = {MIKADO}{logvar.shape}{RESET}",  flush=True   )   

    z = self.sample_z( gpu, mean, logvar )                                                                      # apply 'sample_z' method (defined above) to mean and logvar

    if DEBUG>9:
      print ( f"TTVAE:          INFO:         forward(): samples.shape = {PINK}{z.shape}{RESET}",     flush=True   )
    if DEBUG>9:      
      print ( f"TTVAE:          INFO:         forward(): samples (from which inputs will be reconstructed)    = \n{PINK}{z}{RESET}",         flush=True   )     

    x2r = self.decode(z)                                                                                   # apply 'decode'   method (defined above) to the z samples

    return x2r, mean, logvar



  def get_latent_z(self, x):
    
    """Encode X into reparameterized latent representation.

    Parameters
    ----------
    x : type
      Input methylation data.

    Returns
    -------
    torch.tensor
      Latent embeddings.

    """
    mean, logvar = self.encode(x)
    
    return self.sample_z(mean, logvar)


  def forward_predict(self, x):
    
    """Forward pass from input to reconstructed input."""
    return self.get_latent_z(x)





#----------------------------------------------------------------------------------------------------------

def vae_loss( x2r, x2, mean, logvar, loss_func, epoch, kl_warm_up=0, beta=1. ):
  
  """Function to calculate VAE Loss, Reconstruction Loss + Beta KLLoss.

  Parameters
  ----------
  x2r : torch.tensor
    Reconstructed x2r from autoencoder
    
  x2 : torch.tensor
    Original input data.
  mean : type
    Learned mean tensor for each sample point
  logvar : type
    Variation around that mean sample point, learned from reparameterization.
  loss_func : type
    Loss function for reconstruction loss, MSE or BCE.
  epoch : type
    Epoch of training.
  kl_warm_up : type
    Number of epochs until fully utilizing KLLoss, begin saving models here.
  beta : type
    Weighting for KLLoss.

  Returns
  -------
  torch.tensor
    Total loss
  torch.tensor
    Recon loss
  torch.tensor
    KL loss

  """
  if type(x2r) != type([]):
    x2r = [x2r]

  if DEBUG>999:
    print ( f"TTVAE:          INFO:      ttvae(): loss_func = {MIKADO}{loss_func}{RESET}" )     
    print ( f"TTVAE:          INFO:      ttvae(): type(x2)  = {MIKADO}{type(x2)}{RESET}" ) 
    print ( f"TTVAE:          INFO:      ttvae(): type(x2r) = {MIKADO}{type(x2r)}{RESET}" ) 
      
  reconstruction_loss = sum( [loss_func(out, x2) for out in x2r] )
  
  kl_loss             = torch.mean(0.5 * torch.sum( torch.exp(logvar) + mean**2 - 1. - logvar, 1) )
  kl_loss            *= beta
  if epoch < kl_warm_up:
    kl_loss *= np.clip(epoch/kl_warm_up, 0.0, 1.0 )
    
  total_loss = reconstruction_loss+kl_loss
  
  return total_loss, reconstruction_loss, kl_loss


