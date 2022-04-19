"""================================================================================================================================
Basically the encoder portion of the AEDEEPDENSE model
==================================================================================================================================="""

import copy
import inspect
import numpy as np

import torch
from  torch import nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
        
from constants  import *

DEBUG   = 1

# ------------------------------------------------------------------------------

class DEEPDENSE( nn.Module) :
  
  
  """Pytorch DENSE encoder with fully connected layers and customizable topology.

  """

  def __init__( self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2  ):
    
    super(DEEPDENSE, self).__init__()

    hidden_layer_encoder_topology =  args.hidden_layer_encoder_topology
    n_input                       = n_genes
    self.pre_latent_topology      = [n_input]  + (hidden_layer_encoder_topology  if hidden_layer_encoder_topology else [])  # layer before the output (latent layer)
    
    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:      pre_latent_topology           = {CYAN}{self.pre_latent_topology}{RESET}",       flush=True   )

    self.encoder_layers       = []
    if len(self.pre_latent_topology)>1:                                                                    # if more than one pre-latent layer is defined, then establish those layers
      for i in range(len(self.pre_latent_topology)-1):
        layer = nn.Linear( self.pre_latent_topology[i], self.pre_latent_topology[i+1] )                    # add another linear later with dimensions derived from hidden_layer_encoder_topology vector
        torch.nn.init.xavier_uniform_(layer.weight)                                                        # specify Xavier initialization
        self.encoder_layers.append( nn.Sequential(layer, nn.ReLU(), nn.Dropout( nn_dense_dropout_1, inplace=False) ))
        
      self.encoder_layers.append( nn.Linear( hidden_layer_encoder_topology[-1], n_classes )  )

    self.encoder  = nn.Sequential( *self.encoder_layers ) if self.encoder_layers else nn.Dropout( p=0.0 )
    if DEBUG>2:
      print ( f"DEEPDENSE:      INFO:    encoder_layers = \n {CYAN}{self.encoder_layers}{RESET}", flush=True   )

    # ~ self.fc1  = nn.Linear( hidden_layer_encoder_topology[-1], n_classes            )
    # ~ self.dropout_1 = nn.Dropout( p=nn_dense_dropout_1 )  


  def encode(self, x, gpu, args ):


    z = self.encoder(x)

    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:         encode(): z.shape = {COQUELICOT}{z.shape}{RESET}",    flush=True   )
    if DEBUG>9:      
      print ( f"DEEPDENSE:      INFO:         encode(): z       = \n{CYAN}{x}{RESET}",        flush=True   ) 

    z = torch.squeeze(z)

    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:         encode(): z.shape = {BLEU}{z.shape}{RESET}",    flush=True  )

    return z



  def forward( self, x, gpu, args ):

    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:       forward() about to take a single step" )
    
    z = self.encode(x, gpu, args )
    # ~ if args.just_test != 'True':
      # ~ z = self.dropout_1(x)
    # ~ z = self.fc1(x)    
    
    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:         encode(): z.shape = {MAGENTA}{z.shape}{RESET}",    flush=True  )

    return z, 0

