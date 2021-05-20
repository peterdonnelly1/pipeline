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
        
WHITE           ='\033[37;1m'
PURPLE          ='\033[35;1m'
DIM_WHITE       ='\033[37;2m'
CYAN            ='\033[36;1m'
PALE_RED        ='\033[31m'
PALE_GREEN      ='\033[32m'
AUREOLIN        ='\033[38;2;253;238;0m'
DULL_WHITE      ='\033[38;2;140;140;140m'
MIKADO          ='\033[38;2;255;196;12m'
AZURE           ='\033[38;2;0;127;255m'
AMETHYST        ='\033[38;2;153;102;204m'
ASPARAGUS       ='\033[38;2;135;169;107m'
CHARTREUSE      ='\033[38;2;223;255;0m'
COQUELICOT      ='\033[38;2;255;56;0m'
COTTON_CANDY    ='\033[38;2;255;188;217m'
HOT_PINK        ='\033[38;2;255;105;180m'
CAMEL           ='\033[38;2;193;154;107m'
MAGENTA         ='\033[38;2;255;0;255m'
YELLOW          ='\033[38;2;255;255;0m'
DULL_YELLOW     ='\033[38;2;179;179;0m'
ARYLIDE         ='\033[38;2;233;214;107m'
BLEU            ='\033[38;2;49;140;231m'
DULL_BLUE       ='\033[38;2;0;102;204m'
RED             ='\033[38;2;255;0;0m'
PINK            ='\033[38;2;255;192;203m'
BITTER_SWEET    ='\033[38;2;254;111;94m'
DARK_RED        ='\033[38;2;120;0;0m'
ORANGE          ='\033[38;2;255;103;0m'
PALE_ORANGE     ='\033[38;2;127;63;0m'
GOLD            ='\033[38;2;255;215;0m'
GREEN           ='\033[38;2;19;136;8m'
BRIGHT_GREEN    ='\033[38;2;102;255;0m'
CARRIBEAN_GREEN ='\033[38;2;0;204;153m'
GREY_BACKGROUND ='\033[48;2;60;60;60m'


BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'

FAIL    = 0
SUCCESS = 1

DEBUG   = 1

# ------------------------------------------------------------------------------

class DEEPDENSE( nn.Module) :
  
  
  """Pytorch DENSE encoder with fully connected layers and customizable topology.

  """

  def __init__( self, cfg, args, input_mode, nn_type, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  ):
    
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
        self.encoder_layers.append(nn.Sequential(layer, nn.ReLU()))

    self.encoder        = nn.Sequential( *self.encoder_layers ) if self.encoder_layers else nn.Dropout( p=0.0 )
    if DEBUG>1  :
      print ( f"DEEPDENSE:      INFO:    encoder_layers = \n {CYAN}{self.encoder_layers}{RESET}", flush=True   )

    self.fc1  = nn.Linear( hidden_layer_encoder_topology[-1], n_classes            )
    # ~ self.dropout_1 = nn.Dropout( p=nn_dense_dropout_1 )  


  def encode(self, x, gpu ):


    z = self.encoder(x)

    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:         encode(): z.shape = {COQUELICOT}{z.shape}{RESET}",    flush=True   )
    if DEBUG>9:      
      print ( f"DEEPDENSE:      INFO:         encode(): z       = \n{CYAN}{x}{RESET}",        flush=True   ) 

    z = torch.squeeze(z)

    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:         encode(): z.shape = {BLEU}{z.shape}{RESET}",    flush=True  )

    return z



  def forward( self, x, gpu ):

    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:       forward() about to take a single step" )
    
    x = self.encode(x, gpu )
    # ~ x = self.dropout_1(x)
    z = self.fc1(x)    
    
    if DEBUG>1:
      print ( f"DEEPDENSE:      INFO:         encode(): z.shape = {MAGENTA}{z.shape}{RESET}",    flush=True  )

    return z, 0

