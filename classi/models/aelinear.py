"""=============================================================================
Linear autoencoder.
============================================================================="""

import torch
from  torch import nn
import numpy as np

from constants  import *

DEBUG=0
# ------------------------------------------------------------------------------

class AELINEAR(nn.Module):

    def __init__( self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2  ):
      
        """Initialize simple linear model.
        """

        if DEBUG>0:        
          print ( "AELINEAR:       INFO:     at \033[35;1m __init__()\033[m" )
        
        super(AELINEAR, self).__init__()
        
        
        self.fc1       = nn.Linear( n_genes, embedding_dimensions)   # encode
        self.fc2       = nn.Linear(embedding_dimensions, n_genes)   # decode
   
        if DEBUG>0:
          print (f"AELINEAR:       INFO:       init(): {ORANGE}caution: the input vectors must have the same dimensions as m1, viz: {CYAN}{n_genes}x{embedding_dimensions}{RESET}",                                            flush=True )

# ------------------------------------------------------------------------------

    def encode( self, x, gpu, args ):
       
        if DEBUG>0:
          print ( f"AELINEAR:       INFO:       encode(): x.shape   = {CYAN}{x.shape}{RESET}", flush=True   ) 

        z =  self.fc1(x)

        if DEBUG>0:
          print ( f"AELINEAR:       INFO:       encode(): z.shape   = {CYAN}{z.shape}{RESET}", flush=True   ) 
          
        return z
        
# ------------------------------------------------------------------------------

    def decode(self, z):
      
        if DEBUG>0:
          print ( f"AELINEAR:       INFO:       decode(): z.shape   = {CYAN}{z.shape}{RESET}", flush=True         ) 
          
        x = self.fc2(z)

        if DEBUG>0:
          print ( f"AELINEAR:       INFO:       decode(): x.shape   = {CYAN}{x.shape}{RESET}", flush=True   ) 
        
        return x

# ------------------------------------------------------------------------------

    def forward(  self, args, x, gpu, encoder_activation ): 

        if DEBUG>0:
          print ( f"AELINEAR:       INFO:       forward(): x.shape           = {CYAN}{x.shape}{RESET}", flush=True             ) 
        
        z = self.encode( x.view(-1, self.input_dim), gpu, args )

        if DEBUG>0:
          print ( f"AELINEAR:       INFO:       forward(): z.shape           = {CYAN}{z.shape}{RESET}", flush=True             ) 
          
        return self.decode(z), 0, 0
