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
from  torch import nn
from  torch import sigmoid
from  torch import relu
from  torch import tanh

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
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
RESET='\033[m'

UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1

# ------------------------------------------------------------------------------

class AEDEEPDENSE( nn.Module) :
  
  
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

  pre_latent_topology : type
    Hidden layer topology for encoder.
  post_latent_topology : type
    Mirrored hidden layer topology for decoder.
  encoder_layers : list
    Encoder pytorch layers.
  encoder : type
    Encoder layers wrapped into pytorch module.
  decoder_layers : type
    Decoder layers wrapped into pytorch module.
  output_layer : type
    Linear layer connecting last decoder layer to output layer, which is same size as input..
  decoder : type
    Wraps decoder_layers and output_layers into Sequential module.
  n_input

  """


  def __init__( self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2   ):
    
    if DEBUG>99:
      print ( f"AEDEEPDENSE:    INFO:    at {MIKADO} __init__(){RESET}" )
    
    super(AEDEEPDENSE, self).__init__()

    hidden_layer_encoder_topology =  args.hidden_layer_encoder_topology

    n_input                       = n_genes
    self.n_input                  = n_genes
    self.pre_latent_topology      = [n_input]  + (hidden_layer_encoder_topology       if hidden_layer_encoder_topology else [])  # layer before the output (latent layer)
    self.post_latent_topology     = (hidden_layer_encoder_topology[::-1] + [n_input]  if hidden_layer_encoder_topology else [])  # layer after output
    if DEBUG>0:
      print ( f"AEDEEPDENSE:    INFO:  pre_latent_topology           = {CYAN}{self.pre_latent_topology}{RESET}",       flush=True   )
      print ( f"AEDEEPDENSE:    INFO:  post_latent_topology          = {CYAN}{self.post_latent_topology}{RESET}",       flush=True   )

    self.encoder_layers       = []
    if len(self.pre_latent_topology)>1:                                                                    # if more than one pre-latent layer is defined, then establish those layers
      for i in range(len(self.pre_latent_topology)-1):
        layer = nn.Linear( self.pre_latent_topology[i], self.pre_latent_topology[i+1] )                    # add another linear later with dimensions derived from hidden_layer_encoder_topology vector
        torch.nn.init.xavier_uniform_(layer.weight)                                                        # specify Xavier initialization
        self.encoder_layers.append(nn.Sequential(layer,nn.ReLU()))

    self.encoder        = nn.Sequential( *self.encoder_layers ) if self.encoder_layers else nn.Dropout( p=0.0 )
    
    if DEBUG>0  :
      print ( f"AEDEEPDENSE:    INFO:    encoder_layers = \n {CYAN}{self.encoder_layers}{RESET}", flush=True   )

    self.decoder_layers = []
    if len(self.post_latent_topology)>1:                                                                   # i.e. if more than one post-latent layer is defined, then establish those layers
      for i in range(len(self.post_latent_topology)-1):
        layer           = nn.Linear(self.post_latent_topology[i],self.post_latent_topology[i+1])
        torch.nn.init.xavier_uniform_(layer.weight)
        self.decoder_layers.append(nn.Sequential(layer,nn.ReLU()))
 
    self.decoder        = nn.Sequential( *self.decoder_layers )
    if DEBUG>0: 
      print ( f"AEDEEPDENSE:    INFO:    decoder_layers              = \n {CYAN}{self.decoder_layers}{RESET}", flush=True   )



  def encode(self, x, gpu, encoder_activation ):
    """Encode input into latent representation.

    Parameters
    ----------
    x : type
      input vectors

    Returns
    -------
    torch.tensor
      latent embedding

    """

    z = self.encoder(x)

    if DEBUG>9:
      print ( f"AEDEEPDENSE:    INFO:         encode(): z.shape = {CYAN}{z.shape}{RESET}",    flush=True   )
    if DEBUG>9:      
      print ( f"AEDEEPDENSE:    INFO:         encode(): z       = \n{CYAN}{x}{RESET}",        flush=True   ) 

    return z



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



  def forward( self, x, gpu, encoder_activation ):
    
    """Return reconstructed output, mean and variance of embeddings.
    """

    if DEBUG>9:
      print ( f"AEDEEPDENSE:    INFO:       forward() about to take a single encode/decode step" )
    
    z = self.encode(x, gpu, encoder_activation)
    
    x2r = self.decode(z)                                                                                   # apply 'decode'   method (defined above) to the z samples

    return x2r, 0, 0

