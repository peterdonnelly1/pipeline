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

import torch
from  torch import nn
from  torch import sigmoid
from  torch import relu
from  torch import tanh
import numpy as np

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
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
BLUE='\033[38;2;0;0;255m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
ORANGE='\033[38;2;255;127;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;0;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

DEBUG=1

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


  def __init__( self, cfg, encoder_activation, nn_dense_dropout_1, nn_dense_dropout_2  ):
    
    
    hidden_layer_encoder_topology=[100,100,100]
#    cuda=False
    cuda=True # PGD
    
    if DEBUG>0:
      print ( "TTVAE:          INFO:    at \033[35;1m __init__()\033[m" )
    
    super(TTVAE, self).__init__()

    n_input                   = cfg.N_GENES
    self.n_input              = cfg.N_GENES
    n_latent                  = cfg.GENE_EMBED_DIM
    self.n_latent             = cfg.GENE_EMBED_DIM
    self.cuda_on              = cuda
    self.pre_latent_topology  = [n_input]+(hidden_layer_encoder_topology if hidden_layer_encoder_topology else [])
    self.post_latent_topology = [n_latent]+(hidden_layer_encoder_topology[::-1] if hidden_layer_encoder_topology else [])
    self.encoder_layers       = []

    if len(self.pre_latent_topology)>1:
      for i in range(len(self.pre_latent_topology)-1):
        layer = nn.Linear(self.pre_latent_topology[i],self.pre_latent_topology[i+1])
        torch.nn.init.xavier_uniform_(layer.weight)
        self.encoder_layers.append(nn.Sequential(layer,nn.ReLU()))
    self.encoder = nn.Sequential(*self.encoder_layers) if self.encoder_layers else nn.Dropout(p=0.)
    self.z_mean = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent))
    self.z_var = nn.Sequential(nn.Linear(self.pre_latent_topology[-1],n_latent),nn.BatchNorm1d(n_latent))
    self.z_develop = nn.Linear(n_latent,self.pre_latent_topology[-1])
    self.decoder_layers = []
    if len(self.post_latent_topology)>1:
      for i in range(len(self.post_latent_topology)-1):
        layer = nn.Linear(self.post_latent_topology[i],self.post_latent_topology[i+1])
        torch.nn.init.xavier_uniform_(layer.weight)
        self.decoder_layers.append(nn.Sequential(layer,nn.ReLU()))
    self.decoder_layers = nn.Sequential(*self.decoder_layers)
    self.output_layer = nn.Sequential(nn.Linear(self.post_latent_topology[-1],n_input),nn.Sigmoid())
    if self.decoder_layers:
      self.decoder = nn.Sequential(*[self.decoder_layers,self.output_layer])
    else:
      self.decoder = self.output_layer


  def sample_z(self, mean, logvar):
    """Sample latent embeddings, reparameterize by adding noise to embedding.

    Parameters
    ----------
    mean : type
      Learned mean vector of embeddings.
    logvar : type
      Learned variance of learned mean embeddings.

    Returns
    -------
    torch.tensor
      Mean + noise, reparameterization trick.

    """
    stddev = torch.exp(0.5 * logvar)
    noise = Variable(torch.randn(stddev.size()))
    if self.cuda_on:
      noise=noise.cuda()
    if not self.training:
      noise = 0.
      stddev = 0.
    return (noise * stddev) + mean


  def encode(self, x, encoder_activation ):
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
    x = self.encoder(x)


    if DEBUG>9:
      print ( f"TTVAE:          INFO:         encode(): x.shape      = {CYAN}{x.shape}{RESET}", flush=True   ) 

    #print(x.size())
    #x = x.view(x.size(0), -1)
    mean = self.z_mean(x)
    var = self.z_var(x)
    #print('mean',mean.size())

    if DEBUG>9:
      print ( f"TTVAE:          INFO:         encode(): mean.shape   = {CYAN}{mean.shape}{RESET}", flush=True   ) 
      print ( f"TTVAE:          INFO:         encode(): var.shape    = {CYAN}{var.shape}{RESET}",  flush=True   )       

    if DEBUG>99:
      print ( f"TTVAE:          INFO:         encode(): mean         = \n{CYAN}{mean}{RESET}", flush=True   ) 
      print ( f"TTVAE:          INFO:         encode(): var          = \n{CYAN}{var}{RESET}",  flush=True   ) 


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
    #out = self.z_develop(z)
    #print('out',out.size())
    #out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
    out = self.decoder(z)
    #print(out)
    return out


  def forward(self, x, encoder_activation):
    """Return reconstructed output, mean and variance of embeddings.
    """

    if DEBUG>9:
      print ( f"TTVAE:          INFO:       forward() about to take a single encode/decode step" )
    
    mean, logvar = self.encode(x, encoder_activation)
    
    if DEBUG>9:
      print ( f"TTVAE:          INFO:       forward(): mean.shape    = {CYAN}{mean.shape}{RESET}",    flush=True   ) 
      print ( f"TTVAE:          INFO:       forward(): logvar.shape  = {CYAN}{logvar.shape}{RESET}",  flush=True   )   

    z = self.sample_z(mean, logvar)

    out = self.decode(z)

    return out, mean, logvar


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

def vae_loss(output, inputs, mean, logvar, loss_func, epoch, kl_warm_up=0, beta=1.):
  """Function to calculate VAE Loss, Reconstruction Loss + Beta KLLoss.

  Parameters
  ----------
  output : torch.tensor
    Reconstructed output from autoencoder.
  input : torch.tensor
    Original input data.
  mean : type
    Learned mean tensor for each sample point.
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
  if type(output) != type([]):
    output = [output]
  recon_loss = sum([loss_func(out, inputs) for out in output])
  kl_loss = torch.mean(0.5 * torch.sum(
    torch.exp(logvar) + mean**2 - 1. - logvar, 1))
  kl_loss *= beta
  if epoch < kl_warm_up:
    kl_loss *= np.clip(epoch/kl_warm_up,0.,1.)
  #print(recon_loss,kl_loss)
  return recon_loss + kl_loss, recon_loss, kl_loss