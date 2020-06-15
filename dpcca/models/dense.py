"""=============================================================================
DENSE encoder
============================================================================="""

from   torch import nn
import torch.nn.functional as F


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
DULL_ORANGE='\033[38;2;127;63;0m'
GREEN='\033[38;2;0;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'


DEBUG=0

# ------------------------------------------------------------------------------

class DENSE(nn.Module):

    def __init__(self, cfg):
        
        print ( f"DENSE:         INFO:    at {PURPLE} __init__(){RESET}" )
        
        super(DENSE, self).__init__()
        
        self.input_dim      = cfg.N_GENES
        number_of_classes   = 9

        if DEBUG>9:
          print ( f"DENSE:            INFO:       at {PURPLE} __init__(){RESET}: number of samples = {number_of_classes}" )
        
        self.fc1 = nn.Linear(self.input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 50)            
        self.fc8 = nn.Linear(50, number_of_classes)
   
        if DEBUG>0:
          print( "DENSE:         INFO:   __init__() \033[1m values are: self.input_dim=\033[35;1m{:}\033[m, number_of_classes=\033[35;1m{:}\033[m, self.fc1=\033[35;1m{:}\033[m"\
.format( self.input_dim, number_of_classes, self.fc1 ) )
          print( "DENSE:         INFO:   __init__() MODEL dimensions: (input layer) m1 = \033[35;1m{:} x {:}\033[m; (output layer) m2 = \033[35;1m{:} x {:}\033[m"\
.format( self.input_dim, number_of_classes, number_of_classes, self.input_dim ) )
          print ("DENSE:         INFO:   __init__() \033[31;1mcaution: the gene input vectors must be the same dimensions as m1\033[m, i.e. \033[35;1m{:} x {:}\033[m".format( self.input_dim, number_of_classes, number_of_classes ) )
          print ("DENSE:         INFO:   __init__() \033[35;1mabout to return from DENSE()\033[m" )
        
# ------------------------------------------------------------------------------

    def encode(self, x):
    
      if DEBUG>99:
        print ( "DENSE:         INFO:     encode():   x.shape           = {:}".format( x.shape ) ) 
        print ( "DENSE:         INFO:     encode():   self.fc1(x).shape = {:}".format( (self.fc1(x)).shape ) )        
    
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = F.relu(self.fc4(x))
      x = F.relu(self.fc5(x))
      x = F.relu(self.fc6(x))
      x = F.relu(self.fc7(x))
      x = self.fc8(x)
         
      return x

# ------------------------------------------------------------------------------

    def forward(self, x):

        if DEBUG>0:
          print ( "\033[2KLINEAR:         INFO:     forward(): x.shape = {:}".format( x.shape ) )
          
        output = self.encode(x.view(-1, self.input_dim))

        if DEBUG>0:
          print ( "\033[2KLINEAR:         INFO:     forward(): output.shape = {:}".format( output.shape ) )
          
        return output
