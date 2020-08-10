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
ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
RESET='\033[m'


DEBUG=1

np.set_printoptions(edgeitems=12)
np.set_printoptions(linewidth=50)

# ------------------------------------------------------------------------------

class DENSE(nn.Module):
    
    def __init__( self, cfg, n_classes, n_genes, nn_dense_dropout_1, nn_dense_dropout_2 ):
        
        if DEBUG>0:
          print ( f"DENSE:          INFO:   __init__() n_classes           = {MIKADO}{n_classes}{RESET}" ) 
          print ( f"DENSE:          INFO:   __init__() n_genes             = {MIKADO}{n_genes}{RESET}" ) 
          print ( f"DENSE:          INFO:   __init__() nn_dense_dropout_1  = {MIKADO}{nn_dense_dropout_1}{RESET}" )                     
          print ( f"DENSE:          INFO:   __init__() nn_dense_dropout_2  = {MIKADO}{nn_dense_dropout_2}{RESET}" )                      
        
        super(DENSE, self).__init__()
        
        self.input_dim      = n_genes

        self.fc1     = nn.Linear(self.input_dim, 400)
        self.fc2     = nn.Linear(400, 200)
        self.fc3     = nn.Linear(200, 100)            
        self.fc4     = nn.Linear(100, n_classes)
        
        self.dropout_1 = nn.Dropout(p=nn_dense_dropout_1)        
        self.dropout_2 = nn.Dropout(p=nn_dense_dropout_2)
           
        if DEBUG>9:
          print( "DENSE:         INFO:   __init__() \033[1m values are: self.input_dim=\033[35;1m{:}\033[m, n_classes=\033[35;1m{:}\033[m, self.fc1=\033[35;1m{:}\033[m"\
.format( self.input_dim, n_classes, self.fc1 ) )
          print( "DENSE:         INFO:   __init__() MODEL dimensions: (input layer) m1 = \033[35;1m{:} x {:}\033[m; (output layer) m2 = \033[35;1m{:} x {:}\033[m"\
.format( self.input_dim, n_classes, n_classes, self.input_dim ) )
          print ("DENSE:         INFO:   __init__() \033[31;1mcaution: the gene input vectors must be the same dimensions as m1\033[m, i.e. \033[35;1m{:} x {:}\033[m".format( self.input_dim, n_classes, n_classes ) )
          print ("DENSE:         INFO:   __init__() \033[35;1mabout to return from DENSE()\033[m" )
        
# ------------------------------------------------------------------------------

    def encode(self, x):
    
      if DEBUG>999:
        print ( "DENSE:         INFO:     encode():   x.shape           = {:}".format( x.shape ) ) 
        np.set_printoptions(formatter={'float': lambda x: "{:.2f}".format(x)})
        print ( f"DENSE:         INFO:     encode():   x                 = {x.cpu().numpy()[0]}" )          
    
      x = F.relu(self.fc1(x))
      x = self.dropout_1(x)      
      x = F.relu(self.fc2(x))
      x = self.dropout_1(x)      
      x = F.relu(self.fc3(x))
      x = self.dropout_1(x)          
      x = self.fc4(x)
         
      return x

# ------------------------------------------------------------------------------

    def forward(self, x):

        if DEBUG>99:
          print ( "\033[2KLINEAR:         INFO:     forward(): x.shape = {:}".format( x.shape ) )
          
        output = self.encode(x.view(-1, self.input_dim))

        if DEBUG>99:
          print ( "\033[2KLINEAR:         INFO:     forward(): output.shape = {:}".format( output.shape ) )
          
        return output
