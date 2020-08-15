"""=============================================================================
Dense Autoencoder
============================================================================="""

import torch
from  torch import nn
from  torch import sigmoid
from  torch import relu
from  torch import tanh
import numpy as np

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

class AEDENSE(nn.Module):

    def __init__(self, cfg, args, encoder_activation, nn_dense_dropout_1, nn_dense_dropout_2 ):
      
        """Initialize simple linear model.
        """

        if DEBUG>9:        
          print ( "AEDENSE:        INFO:    at \033[35;1m __init__()\033[m" )
        
        super(AEDENSE, self).__init__()
        
        self.input_dim       = cfg.N_GENES
        emb_dim              = cfg.GENE_EMBED_DIM
        hidden_layer_neurons = cfg.HIDDEN_LAYER_NEURONS 
        
        self.fc1       = nn.Linear(self.input_dim, hidden_layer_neurons)               
        self.fc4       = nn.Linear(hidden_layer_neurons, emb_dim)


        self.rc1       = nn.Linear(emb_dim, hidden_layer_neurons)                             
        self.rc4       = nn.Linear(hidden_layer_neurons, self.input_dim)

        self.dropout_1 = nn.Dropout(p=nn_dense_dropout_1)     
        
        
        if DEBUG>2:
          print( f"AEDENSE:       INFO:       init(): layer self.fc1: (encode)    self.input_dim = cfg.N_GENES        = {CYAN}{self.input_dim}{RESET},   emb_dim        = cfg.GENE_EMBED_DIM = {CYAN}{emb_dim}{RESET}", flush=True   )
          print( f"AEDENSE:       INFO:       init(): layer self.fc2: (decode)           emb_dim = cfg.GENE_EMBED_DIM = {CYAN}{emb_dim}{RESET},  self.input_dim = cfg.N_GENES         = {CYAN}{self.input_dim}{RESET}", flush=True   )
          print (f"AEDENSE:       INFO:       init(): {ORANGE}caution: the input vectors must have the same dimensions as m1, viz: {CYAN}{self.input_dim}x{emb_dim}{RESET}",                                            flush=True )

# ------------------------------------------------------------------------------

    def encode(self, x, encoder_activation ):
       
        if DEBUG>2:
          print ( f"AEDENSE:       INFO:       encode(): x.shape   = {CYAN}{x.shape}{RESET}", flush=True   ) 

        if encoder_activation=='none':
          z =  self.fc1(x)
        if encoder_activation=='sigmoid':
          z =  sigmoid(self.fc1(x))
        if encoder_activation=='tanh':
          z =  tanh(self.fc1(x))
        if encoder_activation=='relu':
          z =  relu(self.fc1(x))
          
        x =  self.dropout_1(x)  
        z =  self.fc4(z)

        if DEBUG>2:
          print ( f"AEDENSE:       INFO:       encode(): z.shape   = {CYAN}{z.shape}{RESET}", flush=True   ) 
          
        return z
        
# ------------------------------------------------------------------------------

    def decode(self, z):
      
        if DEBUG>2:
          print ( f"AEDENSE:       INFO:       decode(): z.shape   = {CYAN}{z.shape}{RESET}", flush=True         ) 
        
        x =  self.rc1(z)
        x =  self.rc4(x)        

        if DEBUG>2:
          print ( f"AEDENSE:       INFO:       decode(): x.shape   = {CYAN}{x.shape}{RESET}", flush=True   ) 
        
        return x

# ------------------------------------------------------------------------------

    def forward(self, x, encoder_activation):  # NOT USED. RATHER, ENCODE AND DECODE ARE SEPARATELY CALLED 

        if DEBUG>9:
          print ( f"AEDENSE:       INFO:       forward(): x.shape           = {CYAN}{x.shape}{RESET}", flush=True             ) 
        
        z = self.encode( x.view(-1, self.input_dim), encoder_activation)

        if DEBUG>9:
          print ( f"AEDENSE:       INFO:       forward(): z.shape           = {CYAN}{z.shape}{RESET}", flush=True             ) 
        
        x2r = self.decode(z)
        
        return x2r, 0, 0
