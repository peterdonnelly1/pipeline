"""=============================================================================
Dense Autoencoder
============================================================================="""

import torch
from  torch import nn
from  torch import sigmoid
from  torch import relu
from  torch import tanh
import numpy as np

from constants  import *

DEBUG=1
# ------------------------------------------------------------------------------

class AEDENSEPOSITIVE(nn.Module):

    def __init__( self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, nn_dense_dropout_1, nn_dense_dropout_2  ):
      
        """Initialize simple linear model.
        """

        if DEBUG>0:        
          print ( "AEDENSEPOSITIVE: INFO:     at \033[35;1m __init__()\033[m" )
        
        super(AEDENSEPOSITIVE, self).__init__()
        
        self.input_dim       = cfg.N_GENES
        emb_dim              = cfg.GENE_EMBED_DIM
        hidden_layer_neurons = cfg.HIDDEN_LAYER_NEURONS
          
        
        self.fc1       = nn.Linear(self.input_dim, hidden_layer_neurons)               
        self.fc4       = nn.Linear(hidden_layer_neurons, emb_dim)


        self.rc1       = nn.Linear(emb_dim, hidden_layer_neurons)                             
        self.rc4       = nn.Linear(hidden_layer_neurons, self.input_dim)

        self.dropout_1 = nn.Dropout(p=nn_dense_dropout_1)     
        
        
        if DEBUG>99:
          print( f"AEDENSEPOSITIVE: INFO:       init(): layer self.fc1: (encode)    self.input_dim = cfg.N_GENES        = {CYAN}{self.input_dim}{RESET},   emb_dim        = cfg.GENE_EMBED_DIM = {CYAN}{emb_dim}{RESET}", flush=True   )
          print( f"AEDENSEPOSITIVE: INFO:       init(): layer self.fc2: (decode)           emb_dim = cfg.GENE_EMBED_DIM = {CYAN}{emb_dim}{RESET},  self.input_dim = cfg.N_GENES         = {CYAN}{self.input_dim}{RESET}", flush=True   )
          print (f"AEDENSEPOSITIVE: INFO:       init(): {ORANGE}caution: the input vectors must have the same dimensions as m1, viz: {CYAN}{self.input_dim}x{emb_dim}{RESET}",                                            flush=True )

# ------------------------------------------------------------------------------

    def encode(self, x, gpu, args ):
       
        if DEBUG>99:
          print ( f"AEDENSEPOSITIVE: INFO:       encode(): x.shape   = {CYAN}{x.shape}{RESET}", flush=True   ) 

        if args.encoder_activation=='none':
          z =  self.fc1(x)
        if args.encoder_activation=='sigmoid':
          z =  sigmoid(self.fc1(x))
        if args.encoder_activation=='tanh':
          z =  tanh(self.fc1(x))
        if args.encoder_activation=='relu':
          z =  relu(self.fc1(x))
          
        x =  self.dropout_1(x)  
        z =  self.fc4(z)

        if DEBUG>99:
          print ( f"AEDENSEPOSITIVE: INFO:       encode(): z.shape   = {CYAN}{z.shape}{RESET}", flush=True   ) 
          
        return z
        
# ------------------------------------------------------------------------------

    def decode(self, z):
      
        if DEBUG>99:
          print ( f"AEDENSEPOSITIVE: INFO:       decode(): z.shape   = {CYAN}{z.shape}{RESET}", flush=True         ) 
        
        x =  self.rc1(z)
        x =  self.rc4(x) 
        x[x<0] = 0            

        if DEBUG>99:
          print ( f"AEDENSEPOSITIVE: INFO:       decode(): x.shape   = {CYAN}{x.shape}{RESET}", flush=True   ) 
        
        return x

# ------------------------------------------------------------------------------

    def forward( self, x, gpu, args ):

        if DEBUG>9:
          print ( f"AEDENSEPOSITIVE: INFO:       forward(): x.shape           = {CYAN}{x.shape}{RESET}", flush=True             ) 
        
        z = self.encode(x.view(-1, self.input_dim), gpu, args )

        if DEBUG>9:
          print ( f"AEDENSEPOSITIVE: INFO:       forward(): z.shape           = {CYAN}{z.shape}{RESET}", flush=True             ) 
          
        return self.decode(z), 0, 0
