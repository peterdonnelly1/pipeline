"""=============================================================================
DENSE encoder 
============================================================================="""

import numpy as np

from   torch import nn
import torch.nn.functional as F


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
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
PALE_GREEN='\033[32m'

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1

np.set_printoptions(edgeitems=12)
np.set_printoptions(linewidth=50)

# ------------------------------------------------------------------------------

class DENSE(nn.Module):
  
    def __init__( self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  ):
        
        if DEBUG>0:
          print ( f"DENSE:          INFO:                 input  dimensions (n_genes)   = {MIKADO}{n_genes}{RESET}"            )  
          print ( f"DENSE:          INFO:                 hidden layer neurons          = {MIKADO}{hidden_layer_neurons}{RESET}"            )
          print ( f"DENSE:          INFO:                 output dimensions (n_classes) = {MIKADO}{n_classes}{RESET}"          )
          print ( f"DENSE:          INFO:                 dropout (proportion)          = {MIKADO}{nn_dense_dropout_1}{RESET}" )   
                              
        
        super(DENSE, self).__init__()
        
        self.input_dim        = n_genes
        hidden_layer_neurons  = hidden_layer_neurons

        self.fc1     = nn.Linear( self.input_dim,       hidden_layer_neurons)
        self.fc2     = nn.Linear( hidden_layer_neurons, n_classes)        
        self.dropout_1 = nn.Dropout(p=nn_dense_dropout_1)        

           
        if DEBUG>999:
          print( "DENSE:         INFO:   __init__() \033[1m values are: self.input_dim=\033[35;1m{:}\033[m, n_classes=\033[35;1m{:}\033[m, self.fc1=\033[35;1m{:}\033[m"\
.format( self.input_dim, n_classes, self.fc1 ) )
          print( "DENSE:         INFO:   __init__() MODEL dimensions: (input layer) m1 = \033[35;1m{:} x {:}\033[m; (output layer) m2 = \033[35;1m{:} x {:}\033[m"\
.format( self.input_dim, n_classes, n_classes, self.input_dim ) )
          print ("DENSE:         INFO:   __init__() \033[31;1mcaution: the gene input vectors must be the same dimensions as m1\033[m, i.e. \033[35;1m{:} x {:}\033[m".format( self.input_dim, n_classes, n_classes ) )
          print ("DENSE:         INFO:   __init__() \033[35;1mabout to return from DENSE()\033[m" )
        
# ------------------------------------------------------------------------------

    def encode( self, x, gpu, encoder_activation ):
    
      if DEBUG>999:
        print ( "DENSE:         INFO:     encode():   x.shape           = {:}".format( x.shape ) ) 
      if DEBUG>999:
        np.set_printoptions(formatter={'float': lambda x: "{:.2f}".format(x)})
        print ( f"DENSE:         INFO:     encode():   x                 = {x.cpu().numpy()[0]}" )          
    
      x = F.relu(self.fc1(x))
      x = self.dropout_1(x)
      x = self.fc2(x)
         
      return x

# ------------------------------------------------------------------------------

    def forward( self, x, gpu ):

        if DEBUG>9:
          print ( f"\033[2KDENSE:          INFO:     forward(): x.shape = {MIKADO}{x.shape}{RESET}" )
 
        x = F.relu(self.fc1(x.view(-1, self.input_dim)))
        x = self.dropout_1(x)
        embedding = x
        if DEBUG>8:
          print ( f"VGGNN:          INFO:     forward(): after FC2, x.size                          = {MIKADO}{x.size()}{RESET}" )
        if DEBUG>88:
          print ( f"VGGNN:          INFO:     forward(): x[:,0:20]                                  = {MIKADO}{x[:,0:20]}{RESET}" )        
        output = self.fc2(x) 
          
        #output = self.encode( x.view(-1, self.input_dim), gpu, encoder_activation )

        if DEBUG>9:
          print ( f"\033[2KDENSE:          INFO:     forward(): output.shape = {MIKADO}{output.shape}{RESET}" )
          
        return output, embedding
