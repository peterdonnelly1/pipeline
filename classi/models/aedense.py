"""=============================================================================
Dense autoencoder with dropout
============================================================================="""

import torch
import numpy as np

from  torch import nn
from  torch import sigmoid
from  torch import relu
from  torch import tanh

from constants  import *

DEBUG=1

# ------------------------------------------------------------------------------

class AEDENSE(nn.Module):

  def __init__( self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2   ):

    if DEBUG>6:
      print ( f"AEDENSE:        INFO:       __init__: n_genes                = {CYAN}{n_genes}{RESET}",               flush=True   ) 
      print ( f"AEDENSE:        INFO:       __init__: embedding_dimensions   = {CYAN}{embedding_dimensions}{RESET}",  flush=True   ) 
      print ( f"AEDENSE:        INFO:       __init__: hidden_layer_neurons   = {CYAN}{hidden_layer_neurons}{RESET}",  flush=True   ) 
    
    super(AEDENSE, self).__init__()
    
    self.input_dim        =  n_genes
    hidden_layer_neurons  =  hidden_layer_neurons
        
    self.fc1       = nn.Linear(self.input_dim,       hidden_layer_neurons)               
    self.fc4       = nn.Linear(hidden_layer_neurons, embedding_dimensions)                                 # produces the embedding that we will return as z

    self.rc1       = nn.Linear(embedding_dimensions, hidden_layer_neurons)                             
    self.rc4       = nn.Linear(hidden_layer_neurons, self.input_dim)

    self.dropout_1 = nn.Dropout(p=nn_dense_dropout_1)     
  
# ------------------------------------------------------------------------------

  def encode(self, x, gpu, encoder_activation ):
     
    if DEBUG>99:
      print ( f"AEDENSE:        INFO:       encode(): x.shape   = {MIKADO}{x.shape}{RESET}", flush=True   ) 

    # ~ if encoder_activation=='none':
      # ~ z =  self.fc1(x)
    if encoder_activation=='sigmoid':
      z =  sigmoid(self.fc1(x))
    if encoder_activation=='tanh':
      z =  tanh(self.fc1(x))
    if encoder_activation=='relu':
      z =  relu(self.fc1(x))
    else:
      z =  self.fc1(x)
      
    z =  self.dropout_1(z)
    z =  self.fc4(z)

    if DEBUG>99:
      print ( f"AEDENSE:        INFO:       encode(): z.shape   = {MIKADO}{z.shape}{RESET}", flush=True   ) 
      
    return z
    
# ------------------------------------------------------------------------------

  def decode(self, z):
    
    if DEBUG>99:
      print ( f"AEDENSE:        INFO:       decode(): z.shape   = {MIKADO}{z.shape}{RESET}", flush=True         ) 
    
    x =  self.rc1(z)
    x =  self.rc4(x)        

    if DEBUG>99:
      print ( f"AEDENSE:        INFO:       decode(): x.shape   = {MIKADO}{x.shape}{RESET}", flush=True   ) 
    
    return x

# ------------------------------------------------------------------------------

  def forward( self, x, gpu, encoder_activation ):

    if DEBUG>99:
      print ( f"AEDENSE:        INFO:    forward():   x.shape   = {MIKADO}{x.shape}{RESET}", flush=True             ) 
    
    z   = self.encode( x.view(-1, self.input_dim), gpu, encoder_activation)

    if DEBUG>99:
      print ( f"AEDENSE:        INFO:    forward():   z.shape   = {MIKADO}{z.shape}{RESET}", flush=True             ) 
    
    x2r = self.decode(z)
    
    return x2r, 0, 0
