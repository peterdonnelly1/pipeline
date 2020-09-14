"""============================================================================= 
Train LENET5
============================================================================="""

import sys
import math
import time
import datetime
import cuda
import pprint
import argparse
import numpy as np
import torch
from tiler_scheduler import *
from tiler_threader import *
from tiler_set_target import *
from tiler import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import pandas as pd
import seaborn as sns
from pandas.plotting import table
from tabulate import tabulate
from IPython.display import display 
#from matplotlib import figure
#from pytorch_memlab import profile

from   data                            import loader
from   data.dlbcl_image.config         import GTExV6Config
from   data.dlbcl_image.generate       import generate
from   models                          import LENETIMAGE
from   torch                           import optim
from   torch.nn.utils                  import clip_grad_norm_
from   torch.nn                        import functional
from   torch.nn                        import DataParallel
from   itertools                       import product, permutations
from   PIL                             import Image

import torchvision
import torch.utils.data
from   torch.utils.tensorboard import SummaryWriter
from   torchvision    import datasets, transforms

last_stain_norm='NULL'
last_gene_norm='NULL'

np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=300)

#torch.backends.cudnn.benchmark   = False                                                                  #for CUDA memory optimizations
torch.backends.cudnn.enabled     = True                                                                    #for CUDA memory optimizations

pd.set_option('display.max_rows',     99 )
pd.set_option('display.max_columns',  99 )
pd.set_option('display.width',       80 )
pd.set_option('display.max_colwidth', 8 ) 
pd.set_option('display.float_format', lambda x: '%6.2f' % x)

# ------------------------------------------------------------------------------

WHITE='\033[38;2;255;255;255m'
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
BITTER_SWEET='\033[38;2;254;111;94m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;255;103;0m'
DARK_ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
CARRIBEAN_GREEN='\033[38;2;0;204;153m'
PALE_GREEN='\033[32m'

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

FAIL    = 0
SUCCESS = 1

DEBUG   = 1

device = cuda.device()

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)

global global_batch_count

run_level_total_correct             = []

#run_level_classifications_matrix   = [ len(class_names), len(class_names) ]    
run_level_classifications_matrix    =  np.zeros( (6,6), dtype=int )
job_level_classifications_matrix    =  np.zeros( (6,6), dtype=int )

run_level_classifications_matrix_acc    =  np.zeros( ( 1000, 6,6 ), dtype=int )


global_batch_count    = 0
total_runs_in_job     = 0
final_test_batch_size = 0

# ------------------------------------------------------------------------------

#@profile
def main(args):

  """Main program: train -> test once per epoch
  """
  
  os.system("taskset -p 0xffffffff %d" % os.getpid())
  
  global last_stain_norm                                                                                   # Need to remember this across runs
  global last_gene_norm                                                                                    # Need to remember this across runs
  global run_level_classifications_matrix
  global run_level_classifications_matrix_acc

  global job_level_classifications_matrix   

  print ( f"TRAINLENEJ:     INFO:     mode                =    {MIKADO}{args.nn_mode}{RESET}" )
  print ( f"TRAINLENEJ:     INFO:     dataset             =    {BITTER_SWEET}{args.dataset}{RESET}" )
  
  now = time.localtime(time.time())
  print(time.strftime( f"TRAINLENEJ:     INFO:     start time          =    {MIKADO}%Y-%m-%d %H:%M:%S %Z{RESET}", now ))
  start_time = time.time() 

  print ( f"TRAINLENEJ:     INFO:     torch       version =    {MIKADO}{torch.__version__}{RESET}" )
  print ( f"TRAINLENEJ:     INFO:     torchvision version =    {MIKADO}{torchvision.__version__}{RESET}"  )
  print ( f"TRAINLENEJ:     INFO:     matplotlib  version =    {MIKADO}{matplotlib.__version__}{RESET}"   ) 
  print ( f"TRAINLENEJ:     INFO:     torchvision version =    {MIKADO}{torchvision.__version__}{RESET}"  )
  print ( f"TRAINLENEJ:     INFO:     seaborn     version =    {MIKADO}{sns.__version__}{RESET}"  )
  print ( f"TRAINLENEJ:     INFO:     pandas      version =    {MIKADO}{pd.__version__}{RESET}"  )  
  
  

  print( "TRAINLENEJ:     INFO:  common args:  \
dataset=\033[36;1m{:}\033[m,\
mode=\033[36;1m{:}\033[m,\
nn_optimizer=\033[36;1m{:}\033[m,\
batch_size=\033[36;1m{:}\033[m,\
learning_rate(s)=\033[36;1m{:}\033[m,\
epochs=\033[36;1m{:}\033[m,\
samples=\033[36;1m{:}\033[m,\
max_consec_losses=\033[36;1m{:}\033[m"\
  .format( args.dataset, args.input_mode, args.optimizer, args.batch_size, args.learning_rate, args.n_epochs, args.n_samples, args.max_consecutive_losses  ), flush=True )

  
  if ( args.input_mode=='image' ) | ( args.input_mode=='image_rna' ):
    print( "TRAINLENEJ:     INFO: image args: \
nn_type_img=\033[36;1m{:}\033[m,\
use_tiler=\033[36;1m{:}\033[m,\
n_tiles=\033[36;1m{:}\033[m,\
rand_tiles=\033[36;1m{:}\033[m,\
greyness<\033[36;1m{:}\033[m,\
sd<\033[36;1m{:}\033[m,\
min_uniques>\033[36;1m{:}\033[m,\
latent_dim=\033[36;1m{:}\033[m,\
label_swap=\033[36;1m{:}\033[m,\
make_grey=\033[36;1m{:}\033[m,\
stain_norm=\033[36;1m{:}\033[m,\
annotated_tiles=\033[36;1m{:}\033[m,\
probs_matrix_interpolation=\033[36;1m{:}\033[m"\
  .format( args.nn_type_img, args.use_tiler, args.n_tiles, args.rand_tiles, args.greyness, 
args.min_tile_sd, args.min_uniques, args.latent_dim, args.label_swap_perunit, args.make_grey_perunit, args.stain_norm, args.annotated_tiles, args.probs_matrix_interpolation  ), flush=True )

  elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
    print( f"TRAINLENEJ:     INFO:  rna-seq args: \
nn_type_rna={MIKADO}{args.nn_type_rna}{RESET},\
hidden_layer_neurons={MIKADO}{args.hidden_layer_neurons}{RESET}, \
gene_embed_dim={MIKADO}{args.gene_embed_dim}{RESET}, \
nn_dense_dropout_1={MIKADO}{args.nn_dense_dropout_1}{RESET}, \
nn_dense_dropout_2={MIKADO}{args.nn_dense_dropout_2}{RESET}, \
n_genes={MIKADO}{args.n_genes}{RESET}, \
gene_norm={YELLOW if not args.gene_data_norm[0]=='NONE' else YELLOW if len(args.gene_data_norm)>1 else MIKADO}{args.gene_data_norm}{RESET}, \
g_xform={YELLOW if not args.gene_data_transform[0]=='NONE' else YELLOW if len(args.gene_data_transform)>1 else MIKADO}{args.gene_data_transform}{RESET}" )

  skip_tiling                = args.skip_tiling
  skip_generation            = args.skip_generation
  dataset                    = args.dataset
  class_names                = args.class_names
  cancer_type                = args.cancer_type
  cancer_type_long           = args.cancer_type_long    
  long_class_names           = args.long_class_names  
  class_colours              = args.class_colours
  input_mode                 = args.input_mode
  use_tiler                  = args.use_tiler
  nn_mode                    = args.nn_mode
  nn_type_img                = args.nn_type_img
  nn_type_rna                = args.nn_type_rna
  use_same_seed              = args.use_same_seed
  hidden_layer_neurons       = args.hidden_layer_neurons
  gene_embed_dim             = args.gene_embed_dim
  nn_dense_dropout_1         = args.nn_dense_dropout_1
  nn_dense_dropout_2         = args.nn_dense_dropout_2
  label_swap_perunit         = args.label_swap_perunit
  nn_optimizer               = args.optimizer
  n_samples                  = args.n_samples
  n_tiles                    = args.n_tiles
  n_epochs                   = args.n_epochs
  batch_size                 = args.batch_size
  lr                         = args.learning_rate
  rand_tiles                 = args.rand_tiles
  n_genes                    = args.n_genes
  gene_data_norm             = args.gene_data_norm 
  gene_data_transform        = args.gene_data_transform    
  n_epochs                   = args.n_epochs
  greyness                   = args.greyness
  min_tile_sd                = args.min_tile_sd
  min_uniques                = args.min_uniques  
  make_grey_perunit          = args.make_grey_perunit
  stain_norm                 = args.stain_norm
  stain_norm_target          = args.stain_norm_target
  annotated_tiles            = args.annotated_tiles
  figure_width               = args.figure_width
  figure_height              = args.figure_height  
  probs_matrix_interpolation = args.probs_matrix_interpolation
  max_consecutive_losses     = args.max_consecutive_losses
  target_tile_coords         = args.target_tile_coords
  
  base_dir                   = args.base_dir
  data_dir                   = args.data_dir
  log_dir                    = args.log_dir
  tile_size                  = args.tile_size
  rna_file_name              = args.rna_file_name
  class_numpy_file_name      = args.class_numpy_file_name
  regenerate                 = args.regenerate
  just_profile               = args.just_profile
  just_test                  = args.just_test
  save_model_name            =  args.save_model_name
  save_model_every           = args.save_model_every
  supergrid_size             = args.supergrid_size

  remove_unexpressed_genes    = args.remove_unexpressed_genes
  remove_low_expression_genes = args.remove_low_expression_genes
  low_expression_threshold    = args.low_expression_threshold
  encoder_activation          = args.encoder_activation
  hidden_layer_neurons        = args.hidden_layer_neurons
  gene_embed_dim              = args.gene_embed_dim
  
  use_autoencoder_output      = args.use_autoencoder_output  
  
#  pprint.set_logfiles( log_dir )

  if ( input_mode=='image' ) | ( input_mode=='image_rna' ): 
    if 1 in batch_size:
      print ( f"{RED}TRAINLENEJ:     INFO: analyse_probs(): Sorry - parameter '{CYAN}BATCH_SIZE{RESET}{RED}' (currently '{MIKADO}{batch_size}{RESET}{RED}' cannot include a value <2 for images ... halting now{RESET}" )
      sys.exit(0) 
 
  if  ( just_test=='True' ) & ( use_autoencoder_output=='True' ):
    print( f"{ORANGE}TRAINLENEJ:     INFO:  flag USE_AUTOENCODER_OUTPUT' isn't compatible with flag 'JUST_TEST' ... will disable test mode and continues{RESET}" )
    args.just_test=False
  
  if  ( ( nn_mode == 'dlbcl_image' ) & ( 'AE' in nn_type_img[0] ) ):
    print( f"{RED}TRAINLENEJ:     FATAL: the network model must not be an autoencoder if nn_mode='{MIKADO}{nn_mode}{RESET}{RED}' (you have NN_TYPE_IMG='{MIKADO}{nn_type_img[0]}{RESET}{RED}', which is an autoencoder) ... halting now{RESET}" )
    sys.exit(0)
  
  if supergrid_size<1:
    print( f"{RED}TRAINLENEJ:     FATAL:  parameter 'supergrid_size' (current value {supergrid_size}) must be an integer greater than zero ... halting now{RESET}" )
    sys.exit(0)
  
  n_samples_max  = np.max(n_samples)
  tile_size_max  = np.max(tile_size)  
  n_tiles_max    = np.max(n_tiles)
  n_tiles_last   = 0                                                                                       # used to trigger regeneration of tiles if a run requires more tiles that the preceeding run 
  n_samples_last = 0
  tile_size_last = 0                                                                                       # also used to trigger regeneration of tiles if a run requires a different file size than the preceeding run 
  n_classes      = len(class_names)
  
  
  if just_test=='True':
    print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'  flag is set. No training will be performed{RESET}" )
    if len(args.hidden_layer_neurons)>1:
      print( f"{RED}TRAINLENEJ:     INFO:  in test mode, ({CYAN}JUST_TEST=\"True\"{RESET}{RED}), only one value is allowed for the parameter '{CYAN}HIDDEN_LAYER_NEURONS{RESET}{RED}'. At the moment it has {MIKADO}{len(args.hidden_layer_neurons)}{RESET}{RED} values ... halting{RESET}" )
      sys.exit(0)        
    if not input_mode=='rna': 
      if not tile_size_max**0.5 == int(tile_size_max**0.5):
        print( f"{RED}TRAINLENEJ:     INFO:  in test_mode, 'tile_size' ({MIKADO}{tile_size}{RESET}{RED}) must be a perfect square (eg. 49, 64, 144, 256 ) ... halting {RESET}" )
        sys.exit(0)      
    if n_epochs>1:
      print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'  flag is set, so n_epochs (currently {MIKADO}{n_epochs}{RESET}{ORANGE}) has been set to 1 for this job{RESET}" ) 
      n_epochs=1
    if len(batch_size)>1:
      print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'  flag is set but but 'batch_size' has {MIKADO}{len(batch_size)}{RESET}{ORANGE} values ({MIKADO}{batch_size}{RESET}{ORANGE}). Only the first value ({MIKADO}{batch_size[0]}{ORANGE}) will be used{RESET}" )
      del batch_size[1:]       
    if len(n_tiles)>1:
      print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'  flag is set but but 'n_tiles'    has {MIKADO}{len(n_tiles)}{RESET}{ORANGE} values ({MIKADO}{n_tiles}{RESET}{ORANGE}). Only the first value ({MIKADO}{n_tiles[0]}{RESET}{ORANGE}) will be used{RESET}" )
      del n_tiles[1:] 
    n_tiles[0] = supergrid_size**2 * batch_size[0]
    print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'  flag is set, therefore 'n_tiles' has been set to 'supergrid_size^2 * batch_size' ({MIKADO}{supergrid_size} * {supergrid_size} * {batch_size} =  {n_tiles}{RESET} {ORANGE}) for this job{RESET}" )          
  else:
    if supergrid_size>1:
      print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'      flag is NOT set, so supergrid_size (currently {MIKADO}{supergrid_size}{RESET}{ORANGE}) will be ignored{RESET}" )
      args.supergrid_size=1

           
  if rand_tiles=='False':
    print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'rand_tiles' flag is not set. Tiles will be selected sequentially rather than at random{RESET}" )     

  if (DEBUG>99):
    print ( f"TRAINLENEJ:     INFO:  n_classes   = {MIKADO}{n_classes}{RESET}",                 flush=True)
    print ( f"TRAINLENEJ:     INFO:  class_names = {MIKADO}{class_names}{RESET}",               flush=True)

  if use_same_seed=='True':
    print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'use_same_seed'  flag is set. The same seed will be used for all runs in this job{RESET}" )
    torch.manual_seed(0.223124)                                                                                     # for reproducability across runs (i.e. so that results can be validly compared)


  # (A)  SET UP JOB LOOP

  already_tiled=False
  already_generated=False
                          
  parameters = dict( 
                                 lr  =   lr,
                          n_samples  =   n_samples,
                         batch_size  =   batch_size,
                            n_tiles  =   n_tiles,
                          tile_size  =   tile_size,
                         rand_tiles  =  [ rand_tiles ],
                        nn_type_img  =   nn_type_img,
                        nn_type_rna  =   nn_type_rna,
               hidden_layer_neurons  =   hidden_layer_neurons,
                     gene_embed_dim  =   gene_embed_dim,
                 nn_dense_dropout_1  =   nn_dense_dropout_1,
                 nn_dense_dropout_2  =   nn_dense_dropout_2,
                        nn_optimizer =  nn_optimizer,
                          stain_norm =  stain_norm,
                      gene_data_norm =  gene_data_norm, 
                 gene_data_transform =  gene_data_transform,                                                
                  label_swap_perunit = [   0.0   ],
                   make_grey_perunit = [   0.0   ],
                              jitter = [  [ 0.0, 0.0, 0.0, 0.0 ] ]  )

  param_values = [v for v in parameters.values()]

  start_column  = 0
  offset        = 12
  second_offset = 12

  total_runs_in_job = len(list(product(*param_values)))
  if DEBUG>0:
    print ( f"TRAINLENEJ:     INFO:  total_runs_in_job      =  {CARRIBEAN_GREEN}{total_runs_in_job}{RESET}"  )
    
  if DEBUG>0:
    print(f"\n{UNDER}JOB:{RESET}")
    print(f"\033[2C\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Csamples\
\r\033[{start_column+2*offset}Cbatch_size\
\r\033[{start_column+3*offset}Ctiles\
\r\033[{start_column+4*offset}Ctile_size\
\r\033[{start_column+5*offset}Crand_tiles\
\r\033[{start_column+6*offset}Cnet_img\
\r\033[{start_column+7*offset}Cnet_rna\
\r\033[{start_column+8*offset}Chidden\
\r\033[{start_column+9*offset}Cembeded\
\r\033[{start_column+10*offset}Cnn_drop_1\
\r\033[{start_column+11*offset}Cnn_drop_1\
\r\033[{start_column+12*offset}Coptimizer\
\r\033[{start_column+13*offset}Cstain_norm\
\r\033[{start_column+14*offset}Cg_norm\
\r\033[{start_column+15*offset}Cg_xform\
\r\033[{start_column+16*offset}Clabel_swap\
\r\033[{start_column+17*offset}Cgreyscale\
\r\033[{start_column+18*offset}Cjitter vector\033[m")
    for lr, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values):
      print( f"\
\r\033[{start_column+0*offset}C{CARRIBEAN_GREEN}{lr:<9.6f}\
\r\033[{start_column+1*offset}C{n_samples:<5d}\
\r\033[{start_column+2*offset}C{batch_size:<5d}\
\r\033[{start_column+3*offset}C{n_tiles:<5d}\
\r\033[{start_column+4*offset}C{tile_size:<3d}\
\r\033[{start_column+5*offset}C{rand_tiles:<5s}\
\r\033[{start_column+6*offset}C{nn_type_img:<10s}\
\r\033[{start_column+7*offset}C{nn_type_rna:<10s}\
\r\033[{start_column+8*offset}C{hidden_layer_neurons:<5d}\
\r\033[{start_column+9*offset}C{gene_embed_dim:<5d}\
\r\033[{start_column+10*offset}C{nn_dense_dropout_1:<5.2f}\
\r\033[{start_column+11*offset}C{nn_dense_dropout_2:<5.2f}\
\r\033[{start_column+12*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+13*offset}C{stain_norm:<10s}\
\r\033[{start_column+14*offset}C{gene_data_norm:<10s}\
\r\033[{start_column+15*offset}C{gene_data_transform:<10s}\
\r\033[{start_column+16*offset}C{label_swap_perunit:<6.1f}\
\r\033[{start_column+17*offset}C{make_grey_perunit:<5.1f}\
\r\033[{start_column+18*offset}C{jitter:}{RESET}" )      
  # ~ for lr, batch_size  in product(*param_values): 
      # ~ comment = f' batch_size={batch_size} lr={lr}'

  if just_test=='True':
    if not input_mode=='rna':     
      if not ( batch_size == int( math.sqrt(batch_size) + 0.5) ** 2 ):
        print( f"\033[31;1mTRAINLENEJ:     FATAL:  in test mode 'batch_size' (currently {batch_size}) must be a perfect square (4, 19, 16, 25 ...) to permit selection of a a 2D contiguous patch. Halting.\033[m" )
        sys.exit(0)      

  
  # (B) RUN JOB LOOP

  run=0
  
  for lr, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values): 
    
    run+=1

    if DEBUG>0:
      print(f"\n\n{UNDER}RUN: { run} of {total_runs_in_job}{RESET}")
      print(f"\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Csamples\
\r\033[{start_column+2*offset}Cbatch_size\
\r\033[{start_column+3*offset}Ctiles\
\r\033[{start_column+4*offset}Ctile_size\
\r\033[{start_column+5*offset}Crand_tiles\
\r\033[{start_column+6*offset}Cnet_img\
\r\033[{start_column+7*offset}Cney_rna\
\r\033[{start_column+8*offset}Chidden\
\r\033[{start_column+9*offset}Cembeded\
\r\033[{start_column+10*offset}Cnn_drop_1\
\r\033[{start_column+11*offset}Cnn_drop_2\
\r\033[{start_column+12*offset}Coptimizer\
\r\033[{start_column+13*offset}Cstain_norm\
\r\033[{start_column+14*offset}Cg_norm\
\r\033[{start_column+15*offset}Cg_xform\
\r\033[{start_column+16*offset}Clabel_swap\
\r\033[{start_column+17*offset}Cgreyscale\
\r\033[{start_column+18*offset}Cjitter vector\033[m")
      print( f"\
\r\033[{start_column+0*offset}C{BITTER_SWEET}{lr:<9.6f}\
\r\033[{start_column+1*offset}C{n_samples:<5d}\
\r\033[{start_column+2*offset}C{batch_size:<5d}\
\r\033[{start_column+3*offset}C{n_tiles:<5d}\
\r\033[{start_column+4*offset}C{tile_size:<3d}\
\r\033[{start_column+5*offset}C{rand_tiles:<5s}\
\r\033[{start_column+6*offset}C{nn_type_img:<10s}\
\r\033[{start_column+7*offset}C{nn_type_rna:<10s}\
\r\033[{start_column+8*offset}C{hidden_layer_neurons:<5d}\
\r\033[{start_column+9*offset}C{gene_embed_dim:<5d}\
\r\033[{start_column+10*offset}C{nn_dense_dropout_1:<5.2f}\
\r\033[{start_column+11*offset}C{nn_dense_dropout_2:<5.2f}\
\r\033[{start_column+12*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+13*offset}C{stain_norm:<10s}\
\r\033[{start_column+14*offset}C{gene_data_norm:<10s}\
\r\033[{start_column+15*offset}C{gene_data_transform:<10s}\
\r\033[{start_column+16*offset}C{label_swap_perunit:<6.1f}\
\r\033[{start_column+17*offset}C{make_grey_perunit:<5.1f}\
\r\033[{start_column+18*offset}C{jitter:}{RESET}" )    

      print ("")
    
    # (1) Potentially schedule and run tiler threads
    
    if (input_mode=='image') | (input_mode=='image_rna'):
      if skip_tiling=='False':
        if use_tiler=='internal':
          # need to re-tile if certain parameters have eiher INCREASED ('n_tiles' or 'n_samples') or simply CHANGED ( 'stain_norm' or 'tile_size') since the last run
          if ( ( already_tiled==True ) & ( ( stain_norm==last_stain_norm ) | (last_stain_norm=="NULL") ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) ):
            pass          # no need to re-tile                                                              
          else:           # must re-tile
            if DEBUG>0:
              print( f"TRAINLENEJ:     INFO: {BOLD}2 about to launch tiling processes{RESET}" )
              print( f"TRAINLENEJ:     INFO:   about to delete all existing tiles from {MIKADO}{data_dir}{RESET}")
              print( f"TRAINLENEJ:     INFO:   stain normalization method = {MIKADO}{stain_norm}{RESET}" )
            delete_selected( data_dir, "png" )
            last_stain_norm=stain_norm
            already_tiled=True
  
            if DEBUG>999:
              print( f"TRAINLENEJ:       INFO:   n_samples_max                   = {MIKADO}{n_samples_max}{RESET}")
              print( f"TRAINLENEJ:       INFO:   n_tiles_max                     = {MIKADO}{n_tiles_max}{RESET}")
    
            if stain_norm=="NONE":                                                                         # we are NOT going to stain normalize ...
              norm_method='NONE'
            else:                                                                                          # we are going to stain normalize ...
              if DEBUG>0:
                print( f"TRAINLENEJ:       INFO: {BOLD}2 about to set up stain normalization target{RESET}" )
              if stain_norm_target.endswith(".svs"):                                                       # ... then grab the user provided target
                norm_method = tiler_set_target( args, stain_norm, stain_norm_target, writer )
              else:                                                                                        # ... and there MUST be a target
                print( f"TRAINLENEJ:     FATAL:    for {MIKADO}{stain_norm}{RESET} an SVS file must be provided from which the stain normalization target will be extracted" )
                sys.exit(0)
    
             
            if just_test=='True':
              if DEBUG>0:
                print( f"TRAINLENEJ:     INFO: about to call tile threader with n_samples_max={MIKADO}{n_samples_max}{RESET}; n_tiles={MIKADO}{n_tiles}{RESET}  " )
              result = tiler_threader( args, n_samples_max, n_tiles, tile_size, batch_size, stain_norm, norm_method )                   # we tile the precise number of tiles required for the grid, as calc ulated above
            else:
              if DEBUG>99:
                print( f"TRAINLENEJ:     INFO: about to call tile threader with n_samples_max={MIKADO}{n_samples_max}{RESET}; n_tiles_max={MIKADO}{n_tiles_max}{RESET}  " )
              result = tiler_threader( args, n_samples_max, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job
              
            
            if just_profile=='True':                                                                       # then we are all done
              sys.exit(0)


    # (2) Regenerate Torch '.pt' file, if required. The logic for 'image_rna' is just the concatenation of the logic for 'image' and the logic for 'r na'

    if skip_generation=='False':
      
      if (input_mode=='image') | (input_mode=='image_rna'):
        
        if ( ( already_tiled==True ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) & ( stain_norm==last_stain_norm ) ):    # all three have to be true, or else we must regenerate the .pt file
          pass
        else:
          if global_batch_count==0:
            print( f"TRAINLENEJ:     INFO: \033[1m3  now generating torch '.pt' file from contents of dataset directories{RESET}" )
          else:
            print( f"TRAINLENEJ:     INFO: \033[1m3  will regenerate torch '.pt' file from files, for the following reason(s):{RESET}" )            
            if n_tiles>n_tiles_last:
              print( f"                                    -- value of n_tiles   {MIKADO}({n_tiles})        \r\033[60Chas increased since last run{RESET}" )
            if n_samples>n_samples_last:
              print( f"                                    -- value of n_samples {MIKADO}({n_samples_last}) \r\033[60Chas increased since last run{RESET}")
            if not tile_size_last==tile_size:
              print( f"                                    -- value of tile_size {MIKADO}({tile_size})      \r\033[60Chas changed   since last run{RESET}")
                        
        n_genes = generate( args, n_samples, n_tiles, tile_size, gene_data_norm, gene_data_transform  )
          
        n_tiles_last   = n_tiles                                                                           # for the next run
        n_samples_last = n_samples                                                                         # for the next run
        tile_size_last = tile_size                                                                         # for the next run

      
      elif input_mode=='rna':
        
        must_generate=False
        if ( already_generated==False ):                                                                 # if we've never generated
          must_generate=True
        
        if not ( ( gene_data_norm==last_gene_norm ) & (last_gene_norm=="NULL") ):                        # if the type of normalization has changed since the last run, we have to regenerate
          must_generate=True
          
        if must_generate==True:
         
          if DEBUG>2:
            print( f"TRAINLENEJ:     INFO: args                    = {CYAN}{args}{RESET}"           )
            print( f"TRAINLENEJ:     INFO: n_samples               = {MIKADO}{n_samples}{RESET}"      )
            print( f"TRAINLENEJ:     INFO: n_tiles                 = {MIKADO}{n_tiles}{RESET}"        )
            print( f"TRAINLENEJ:     INFO: n_genes (from args)     = {MIKADO}{n_genes}{RESET}"        )
            print( f"TRAINLENEJ:     INFO: gene_data_norm          = {CYAN}{gene_data_norm}{RESET}" )            

          n_genes = generate( args, n_samples, n_tiles, tile_size, gene_data_norm, gene_data_transform  )
          last_gene_norm=gene_data_norm
          already_generated=True
        else:
          if DEBUG>0:      
            print( f"\nTRAINLENEJ:     INFO: \033[1m3 gene_data_norm = {MIKADO}{gene_data_norm}{RESET} and last_gene_norm = {MIKADO}{last_gene_norm}{RESET} so no need to regenerate torch '.pt' file" )

      else:
        print( f"{RED}TRAINLENEJ:   FATAL:    input mode of type '{MIKADO}{input_mode}{RESET}{RED}' is not supported [200]{RESET}" )
        sys.exit(0)




    #(3) set up Tensorboard
    
    print( "TRAINLENEJ:     INFO: \033[1m1 about to set up Tensorboard\033[m" )
    
    if input_mode=='image':
      writer = SummaryWriter(comment=f' {dataset} {input_mode} {nn_type_img} {nn_optimizer} n={n_samples} test={args.pct_test}% batch={batch_size} lr={lr} t/samp={n_tiles} t_sz={tile_size} t_tot={n_tiles*n_samples} swaps={args.label_swap_perunit}' )
    elif input_mode=='rna':
      writer = SummaryWriter(comment=f' {dataset} {input_mode} {nn_type_rna} {nn_optimizer} n={n_samples} test={args.pct_test}% batch={batch_size} lr={lr} d1={nn_dense_dropout_1} d2={nn_dense_dropout_2} hid={hidden_layer_neurons} emb={gene_embed_dim}  genes={n_genes} gene_norm={gene_data_norm} g_xform={gene_data_transform} swaps={args.label_swap_perunit}')
    elif input_mode=='image_rna':
      writer = SummaryWriter(comment=f' {dataset} {input_mode} {nn_type_img} {nn_type_rna} {nn_optimizer} n={n_samples} test={args.pct_test}% batch={batch_size} lr={lr} n_t={n_tiles} tile={tile_size} t_tot={n_tiles*n_samples} genes={n_genes} gene_norm={gene_data_norm} g_xform={gene_data_transform}')
    else:
      print( f"{RED}TRAINLENEJ:   FATAL:    input mode of type '{MIKADO}{input_mode}{RESET}{RED}' is not supported [314]{RESET}" )
      sys.exit(0)

    print( "TRAINLENEJ:     INFO:   \033[3mTensorboard has been set up\033[m" )
   



    # (4) Load experiment config.  Most configurable parameters are now provided via user arguments

    print( f"TRAINLENEJ:     INFO: {BOLD}2 about to load experiment config{RESET}" )
#    pprint.log_section('Loading config.')
    cfg = loader.get_config( nn_mode, lr, batch_size )                                                #################################################################### change to just args at some point
#    GTExV6Config.INPUT_MODE         = input_mode                                                           # now using args
    GTExV6Config.MAKE_GREY          = make_grey_perunit                                                    # modify config class variable to take into account user preference
    GTExV6Config.JITTER             = jitter                                                               # modify config class variable to take into account user preference
#          if args.input_mode=='rna':  pprint.log_config(cfg) 
#    pprint.log_section('Loading script arguments.')
#    pprint.log_args(args)
  
    print( f"TRAINLENEJ:     INFO:   {ITALICS}experiment config has been loaded{RESET}" )
   



    #(5) Load model
                                                                                                      
    print( f"TRAINLENEJ:     INFO: {BOLD}3 about to load models {MIKADO}{nn_type_img}{RESET}{BOLD} and {MIKADO}{nn_type_rna}{RESET}" )                                  
    model = LENETIMAGE( args, cfg, input_mode, nn_type_img, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, args.latent_dim, args.em_iters  )

    print( f"TRAINLENEJ:     INFO:    {ITALICS}model loaded{RESET}" )

    if just_test=='True':                                                                                  # then load already trained model from HDD
      if DEBUG>0:
        print( f"TRAINLENEJ:     INFO:   'just_test'  flag is set: about to load model state dictionary from {MIKADO}{save_model_name}{RESET} in directory {MIKADO}{log_dir}{RESET}" )
      fpath = '%s/model.pt' % log_dir
      try:
        model.load_state_dict(torch.load(fpath))       
      except Exception as e:
        print ( f"{RED}GENERATE:             FATAL: error when trying to load model {MAGENTA}'{save_model_name}'{RESET}", flush=True)    
        print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)      
        time.sleep(2)
        pass
                                            






    #(6) Send model to GPU(s)
    
    print( f"TRAINLENEJ:     INFO: {BOLD}4 about to send model to device{RESET}" )   
    model = model.to(device)
    print( f"TRAINLENEJ:     INFO:     {ITALICS}model sent to device{RESET}" ) 
  
    #pprint.log_section('Model specs.')
    #pprint.log_model(model)
     
    
    if DEBUG>9:
      print( f"TRAINLENEJ:     INFO:   pytorch Model = {MIKADO}{model}{RESET}" )
    
    #(7) Load dataset
    
    gpu        = 0
    world_size = 0
    rank       = 0
    
    do_all_test_examples=False
    print( "TRAINLENEJ:     INFO: \033[1m5 about to call dataset loader" )
    train_loader, test_loader, final_test_batch_size, final_test_loader = loader.get_data_loaders( args,
                                                         gpu,
                                                         cfg,
                                                         world_size,
                                                         rank,
                                                         batch_size,
                                                         do_all_test_examples,
                                                         args.n_workers,
                                                         args.pin_memory,                                                       
                                                         args.pct_test
                                                        )
    if DEBUG>4:
      print( "TRAINLENEJ:     INFO:   \033[3mdataset loaded\033[m" )
  
    #if just_test=='False':                                                                                # c.f. loader() Sequential'SequentialSampler' doesn't return indices
    #  pprint.save_test_indices(test_loader.sampler.indices)
  
  
  
  



    #(8) Select and configure optimizer
      
    print( f"TRAINLENEJ:     INFO: {BOLD}6 about to select and configure optimizer\033[m with learning rate = {MIKADO}{lr}{RESET}" )
    if nn_optimizer=='ADAM':
      optimizer = optim.Adam       ( model.parameters(),  lr=lr,  weight_decay=0,  betas=(0.9, 0.999),  eps=1e-08,               amsgrad=False                                    )
      print( "TRAINLENEJ:     INFO:   \033[3mAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAMAX':
      optimizer = optim.Adamax     ( model.parameters(),  lr=lr,  weight_decay=0,  betas=(0.9, 0.999),  eps=1e-08                                                                 )
      print( "TRAINLENEJ:     INFO:   \033[3mAdamax optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAGRAD':
      optimizer = optim.Adagrad    ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-10,               lr_decay=0, initial_accumulator_value=0          )
      print( "TRAINLENEJ:     INFO:   \033[3mAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='SPARSEADAM':
      optimizer = optim.SparseAdam ( model.parameters(),  lr=lr,                   betas=(0.9, 0.999),  eps=1e-08                                                                 )
      print( "TRAINLENEJ:     INFO:   \033[3mSparseAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADADELTA':
      optimizer = optim.Adadelta   ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-06, rho=0.9                                                        )
      print( "TRAINLENEJ:     INFO:   \033[3mAdagrad optimizer selected and configured\033[m" )
    elif nn_optimizer=='ASGD':
      optimizer = optim.ASGD       ( model.parameters(),  lr=lr,  weight_decay=0,                                               alpha=0.75, lambd=0.0001, t0=1000000.0            )
      print( "TRAINLENEJ:     INFO:   \033[3mAveraged Stochastic Gradient Descent optimizer selected and configured\033[m" )
    elif   nn_optimizer=='RMSPROP':
      optimizer = optim.RMSprop    ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-08,  momentum=0,  alpha=0.99, centered=False                       )
      print( "TRAINLENEJ:     INFO:   \033[3mRMSProp optimizer selected and configured\033[m" )
    elif   nn_optimizer=='RPROP':
      optimizer = optim.Rprop      ( model.parameters(),  lr=lr,                                                                etas=(0.5, 1.2), step_sizes=(1e-06, 50)           )
      print( "TRAINLENEJ:     INFO:   \033[3mResilient backpropagation algorithm optimizer selected and configured\033[m" )
    elif nn_optimizer=='SGD':
      optimizer = optim.SGD        ( model.parameters(),  lr=lr,  weight_decay=0,                                   momentum=0.9, dampening=0, nesterov=True                       )
      print( "TRAINLENEJ:     INFO:   \033[3mStochastic Gradient Descent optimizer selected and configured\033[m" )
    elif nn_optimizer=='LBFGS':
      optimizer = optim.LBFGS      ( model.parameters(),  lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None  )
      print( "TRAINLENEJ:     INFO:   \033[3mL-BFGS optimizer selected and configured\033[m" )
    else:
      print( "TRAINLENEJ:     FATAL:    Optimizer '{:}' not supported".format( nn_optimizer ) )
      sys.exit(0)
 
 
 
 
         
    #(9) Select Loss function
    
    print( "TRAINLENEJ:     INFO: \033[1m7 about to select CrossEntropyLoss function\033[m" )  
    loss_function = torch.nn.CrossEntropyLoss()
      
    print( "TRAINLENEJ:     INFO:   \033[3mCross Entropy loss function selected\033[m" )  
    
    
#    show,  via Tensorboard, what the samples look like
#    images, labels = next(iter(train_loader))                                                              # PGD 200129 -
#    images = images.to(device)
#    labels = labels.to (device)
  
#    show,  via Tensorboard, what the samples look like
#    grid = torchvision.utils.make_grid( images, nrow=16 )                                                  # PGD 200129 - 
#    writer.add_image('images', grid, 0)                                                                    # PGD 200129 - 
#    writer.add_graph(model, images)                                                                        # PGD 200129 -  
  
    
    #pprint.log_section('Training model.\n\n'\
    #                   'Epoch\t\tTrain x1 err\tTrain x2 err\tTrain l1\t'\
    #                   '\tTest x1 err\tTest x2 err\tTest l1')
   
   
   
   
   
   
    #(10) Train/Test
                     
    print( "TRAINLENEJ:     INFO: \033[1m8 about to commence main loop, one iteration per epoch\033[m" )

    global_correct_prediction_count = 0
    global_number_tested            = 0
    max_correct_predictions         = 0
    max_percent_correct             = 0
    
    test_loss_min           = 999999
    train_loss_min          = 999999  
   
    consecutive_training_loss_increases    = 0
    consecutive_test_loss_increases        = 0
    

    last_epoch_loss_increased              = True

    train_total_loss_sum_ave_last          = 99999                       # used to determine whether total loss is increasing or decreasing
    train_lowest_total_loss_observed       = 99999                       # used to track lowest total loss
    train_lowest_total_loss_observed_epoch = 0                           # used to track lowest total loss

    train_images_loss_sum_ave_last         = 99999
    train_lowest_image_loss_observed       = 99999
    train_lowest_image_loss_observed_epoch = 0

    test_total_loss_sum_ave_last           = 99999                       # used to determine whether total loss is increasing or decreasing
    test_lowest_total_loss_observed        = 99999
    test_lowest_total_loss_observed_epoch  = 0

    test_image_loss_sum_ave_last           = 99999
    test_lowest_image_loss_observed        = 99999    
    test_lowest_image_loss_observed_epoch  = 0     

    test_genes_loss_sum_ave_last           = 99999 
    test_lowest_genes_loss_observed        = 99999      
    test_lowest_genes_loss_observed_epoch  = 0 
  
    
    for epoch in range(1, n_epochs + 1):
  
        if   args.input_mode=='image':
          print( f'\nTRAINLENEJ:     INFO:   epoch: {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}, mode: {MIKADO}{input_mode}{RESET}, lr: {MIKADO}{lr}{RESET}, samples: {MIKADO}{n_samples}{RESET}, batch size: {MIKADO}{batch_size}{RESET}, tile: {MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide: {MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )
        elif args.input_mode=='rna':
          print( f'\nTRAINLENEJ:     INFO:   epoch: {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}, mode: {MIKADO}{input_mode}{RESET}, lr: {MIKADO}{lr}{RESET}, samples: {MIKADO}{n_samples}{RESET}, batch size: {MIKADO}{batch_size}{RESET}, hidden_layer_neurons: {MIKADO}{hidden_layer_neurons}{RESET}, gene_embed_dim: {MIKADO}{batch_size if args.use_autoencoder_output==True  else "N/A" }{RESET}.  {DULL_WHITE}will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )
        else:
          print( f'\nTRAINLENEJ:     INFO:   epoch: {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}, mode: {MIKADO}{input_mode}{RESET}, lr: {MIKADO}{lr}{RESET}, samples: {MIKADO}{n_samples}{RESET}, batch size: {MIKADO}{batch_size}{RESET}, tile: {MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide: {MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )

    
        if just_test=='True':                                                                              # bypass training altogether in test mode
          pass     
        
        else:
          
          if DEBUG>1:
            print('TRAINLENEJ:     INFO:   6.1 running training step ')
    
          train_loss_images_sum_ave, train_loss_genes_sum_ave, train_l1_loss_sum_ave, train_total_loss_sum_ave =\
                                                                                                       train ( args, epoch, train_loader, model, optimizer, loss_function, writer, train_loss_min, batch_size )
    
          if train_total_loss_sum_ave < train_lowest_total_loss_observed:
            train_lowest_total_loss_observed       = train_total_loss_sum_ave
            train_lowest_total_loss_observed_epoch = epoch
    
          if train_loss_images_sum_ave < train_lowest_image_loss_observed:
            train_lowest_image_loss_observed       = train_loss_images_sum_ave
            train_lowest_image_loss_observed_epoch = epoch

          if DEBUG>0:
            if ( (train_total_loss_sum_ave < train_total_loss_sum_ave_last) | (epoch==1) ):
              consecutive_training_loss_increases = 0
              last_epoch_loss_increased = False
            else:
              last_epoch_loss_increased = True

            if ( input_mode=='image' ):
              print ( f"\
  \r\033[1C{CLEAR_LINE}{DULL_WHITE}\
  \r\033[27Ctrain():\
  \r\033[49Closs_images={train_loss_images_sum_ave:5.2f}\
  \r\033[96Cl1_loss={train_l1_loss_sum_ave:5.2f}\
  \r\033[120CBATCH AVE OVER EPOCH={PALE_GREEN if last_epoch_loss_increased==False else PALE_RED}{train_total_loss_sum_ave:9.4f}{DULL_WHITE}\
  \r\033[166Cmins: total: {train_lowest_total_loss_observed:>6.2f}@e={train_lowest_total_loss_observed_epoch:<2d}"
  , end=''  )
            elif ( input_mode=='rna' ):
              print ( f"\
  \r\033[1C{CLEAR_LINE}{DULL_WHITE}\
  \r\033[27Ctrain():\
  \r\033[73Closs_rna={train_loss_genes_sum_ave:5.2f}\
  \r\033[96Cl1_loss={train_l1_loss_sum_ave:5.2f}\
  \r\033[120CBATCH AVE OVER EPOCH={PALE_GREEN if last_epoch_loss_increased==False else PALE_RED}{train_total_loss_sum_ave:9.4f}{DULL_WHITE}\
  \r\033[166Cmins: total: {train_lowest_total_loss_observed:>6.2f}@e={train_lowest_total_loss_observed_epoch:<2d} |"
  , end=''  )
            elif ( input_mode=='image_rna' ):
              print ( f"\
  \r\033[1C{CLEAR_LINE}{DULL_WHITE}\
  \r\033[27Ctrain():\
  \r\033[49Closs_images={train_loss_images_sum_ave:5.2f}\
  \r\033[73Closs_rna={train_loss_genes_sum_ave:5.2f}\
  \r\033[96Cl1_loss={train_l1_loss_sum_ave:5.2f}\
  \r\033[120CBATCH AVE OVER EPOCH={PALE_GREEN if last_epoch_loss_increased==False else PALE_RED}{train_total_loss_sum_ave:9.4f}{DULL_WHITE}\
  \r\033[166Cmins: total: {train_lowest_total_loss_observed:>6.2f}@e={train_lowest_total_loss_observed_epoch:<2d} |\
  \r\033[194Cimage:{train_lowest_image_loss_observed:6.2f}@e={train_lowest_image_loss_observed_epoch:<2d}{RESET}"
  , end=''  )
  
  
  
            if last_epoch_loss_increased == True:
              consecutive_training_loss_increases +=1
              if consecutive_training_loss_increases == 1:
                print ( f"\r\033[232C{DARK_RED} < training loss increased{RESET}", end='' )
              else:
                print ( f"\r\033[232C{DARK_RED} < {consecutive_training_loss_increases} {DARK_RED}consec increases !{RESET}", end='' )
              print ( "" )
    
            if (last_epoch_loss_increased == False):
              print ('')
    
          train_total_loss_sum_ave_last = train_total_loss_sum_ave
  
  
        if DEBUG>1:
          print('TRAINLENEJ:     INFO:   6.2 running test step ')
  
        do_all_test_examples=False
        test_loss_images_sum_ave, test_loss_genes_sum_ave, test_l1_loss_sum_ave, test_total_loss_sum_ave, correct_predictions, number_tested, max_correct_predictions, max_percent_correct, test_loss_min     =\
                      test ( cfg, args, epoch, test_loader,  model,  tile_size, loss_function, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                                           test_loss_min, do_all_test_examples, batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours)

        global_correct_prediction_count += correct_predictions
        global_number_tested            += number_tested
        
        if DEBUG>99:
          print( f"TRAINLENEJ:       INFO:   global_correct_prediction_count   = {MIKADO}{global_correct_prediction_count:>}{RESET}")        
          print( f"TRAINLENEJ:       INFO:   global_number_tested              = {MIKADO}{global_number_tested}{RESET:>}")
          print( f"TRAINLENEJ:       INFO:   global_percent_correct            = {MIKADO}{global_correct_prediction_count/global_number_tested*100:<3.0f}%{RESET}")                    
        
        if ( (test_total_loss_sum_ave < ( test_total_loss_sum_ave_last )) | (epoch==1) ):
          consecutive_test_loss_increases = 0
          last_epoch_loss_increased = False
        else:
          last_epoch_loss_increased = True
          
        if ( input_mode=='image' ):
          print ( f"\
\033[5A\
\r\033[1C\033[2K{DULL_WHITE}\
\r\033[27Ctest():\
\r\033[49Closs_images={CARRIBEAN_GREEN}{test_loss_images_sum_ave:5.2f}{DULL_WHITE}\
\r\033[96Cl1_loss={test_l1_loss_sum_ave:5.2f}{DULL_WHITE}\
\r\033[120CBATCH AVE OVER EPOCH={GREEN if last_epoch_loss_increased==False else RED}{test_total_loss_sum_ave:9.4f}{DULL_WHITE}\
\r\033[166Cmins: total: {test_lowest_total_loss_observed:6.2f}@{WHITE}e={test_lowest_total_loss_observed_epoch:<2d}{DULL_WHITE} |\
\r\033[194Cimage:{test_lowest_image_loss_observed:>6.2f}@{CARRIBEAN_GREEN}e={test_lowest_image_loss_observed_epoch:<2d}{DULL_WHITE} |\
\033[5B\
", end=''  )
        elif ( input_mode=='rna' ):
          print ( f"\
\033[5A\
\r\033[1C\033[2K{DULL_WHITE}\
\r\033[27Ctest():\
\r\033[73Closs_rna={BITTER_SWEET}{test_loss_genes_sum_ave:5.2f}{DULL_WHITE}\
\r\033[96Cl1_loss={test_l1_loss_sum_ave:5.2f}{DULL_WHITE}\
\r\033[120CBATCH AVE OVER EPOCH={GREEN if last_epoch_loss_increased==False else RED}{test_total_loss_sum_ave:9.4f}{DULL_WHITE}\
\r\033[166Cmins: total: {test_lowest_total_loss_observed:6.2f}@{WHITE}e={test_lowest_total_loss_observed_epoch:<2d}{DULL_WHITE} |\
\r\033[214Cgenes:{BITTER_SWEET}{test_lowest_genes_loss_observed:>6.2f}@e={test_lowest_genes_loss_observed_epoch:<2d}{RESET}\
\033[5B\
", end=''  )
        elif ( input_mode=='image_rna' ):
          print ( f"\
\033[5A\
\r\033[1C\033[2K{DULL_WHITE}\
\r\033[27Ctest():\
\r\033[49Closs_images={CARRIBEAN_GREEN}{test_loss_images_sum_ave:5.2f}{DULL_WHITE}\
\r\033[73Closs_rna={BITTER_SWEET}{test_loss_genes_sum_ave:5.2f}{DULL_WHITE}\
\r\033[96Cl1_loss={test_l1_loss_sum_ave:5.2f}{DULL_WHITE}\
\r\033[120CBATCH AVE OVER EPOCH={GREEN if last_epoch_loss_increased==False else RED}{test_total_loss_sum_ave:9.4f}{DULL_WHITE}\
\r\033[166Cmins: total: {test_lowest_total_loss_observed:6.2f}@{WHITE}e={test_lowest_total_loss_observed_epoch:<2d}{DULL_WHITE} |\
\r\033[194Cimage:{test_lowest_image_loss_observed:>6.2f}@{CARRIBEAN_GREEN}e={test_lowest_image_loss_observed_epoch:<2d}{DULL_WHITE} |\
\r\033[214Cgenes:{test_lowest_genes_loss_observed:>6.2f}@{BITTER_SWEET}e={test_lowest_genes_loss_observed_epoch:<2d}{RESET}\
\033[5B\
", end=''  )



        if last_epoch_loss_increased == True:
          consecutive_test_loss_increases +=1
          if consecutive_test_loss_increases == 1:
            print ( "\033[5A", end='' )
            print ( f"\r\033[232C{PALE_RED} < test loss increased{RESET}", end='' )
            print ( "\033[5B", end=''  )
          else:
            print ( "\033[5A", end='' )
            print ( f"\r\033[232C{RED} < {consecutive_test_loss_increases} consec increases !{RESET}", end='' )
            print ( "\033[5B", end=''  )
            
          if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before, so that the most recent model for which the loss improved will be saved
              now = time.localtime(time.time())
              print(time.strftime("TRAINLENEJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
              sys.exit(0)
        else:
          print ( "\033[5A", end='' )
          print ( f"\r\033[232C{PALE_GREEN} < test loss decreased{RESET}", end='' )
          print ( "\033[5B", end=''  )
        
      

        test_total_loss_sum_ave_last = test_total_loss_sum_ave
        
        if test_total_loss_sum_ave < test_lowest_total_loss_observed:
          test_lowest_total_loss_observed       = test_total_loss_sum_ave
          test_lowest_total_loss_observed_epoch = epoch
          if DEBUG>0:
            print ( "\033[5A", end='' )
            print ( f"\r\033[232C\033[0K{BRIGHT_GREEN} < new low/saving{RESET}", end='' )
            print ( "\033[5B", end='' )
          save_model(args.log_dir, model) 
  
        if test_loss_genes_sum_ave < test_lowest_genes_loss_observed:
          test_lowest_genes_loss_observed       = test_loss_genes_sum_ave
          test_lowest_genes_loss_observed_epoch = epoch 
          if DEBUG>0:
            print ( "\033[5A", end='' )
            print ( f"\r\033[250C{BITTER_SWEET} < rna low {RESET}", end='' )
            print ( "\033[5B", end='' )
            
        if test_loss_images_sum_ave < test_lowest_image_loss_observed:
          test_lowest_image_loss_observed       = test_loss_images_sum_ave
          test_lowest_image_loss_observed_epoch = epoch
          if DEBUG>0:
            print ( "\033[5A", end='' )
            print ( f"\r\033[260C{CARRIBEAN_GREEN} < image low{RESET}", end='' )
            print ( "\033[5B", end='' )
  

  #            if DEBUG>0:
  #              print( "TRAINLENEJ:     INFO:   saving samples to \033[35;1m{:}\033[m".format( args.log_dir ) )
  #            save_samples(args.log_dir, model, test_loader, cfg, epoch)
   
  #      if epoch%save_model_every == 0:
  #          if DEBUG>0:
  #            print( f"TRAINLENEJ:     INFO:   about to save model to \033[35;1m{log_dir}\033[m" )
  #          save_model(args.log_dir, model)
  #          if DEBUG>0:
  #            print( f"TRAINLENEJ:     INFO:   \033[3mmodel saved \033[m" )


        if args.input_mode=='rna':
          print ( "\033[8A", end='' )
        else:
          print ( "\033[8A", end='' )           

    #  ^^^  RUN FINISHES HERE ^^^
    
  
    # (C)  MAYBE CLASSIFY ALL TEST SAMPLES USING THE BEST MODEL PRODUCED DURING TRAINING 
  
    if DEBUG>0:
      print ( "\033[8B" )        
      print ( f"TRAINLENEJ:     INFO:  about to classify all test samples through the best model this run produced"        )

      if DEBUG>0:
        print( f"TRAINLENEJ:     INFO:  about to load model state dictionary for best model (from {MIKADO}{save_model_name}{RESET} in directory {MIKADO}{log_dir}{RESET})" )
      fpath = '%s/model.pt' % log_dir
      try:
        model.load_state_dict(torch.load(fpath))
        model = model.to(device)
      except Exception as e:
        print ( f"{RED}GENERATE:             FATAL: error when trying to load model {MAGENTA}'{save_model_name}'{RESET}", flush=True)    
        print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)      
        time.sleep(2)
        pass

    do_all_test_examples=True
    if DEBUG>0:
      print ( f"TRAINLENEJ:     INFO:      test():             final_test_batch_size               = {MIKADO}{final_test_batch_size}{RESET}" )    
    test_loss_images_sum_ave, test_loss_genes_sum_ave, test_l1_loss_sum_ave, test_total_loss_sum_ave, correct_predictions, number_tested, max_correct_predictions, max_percent_correct, test_loss_min     =\
                      test ( cfg, args, epoch, final_test_loader,  model,  tile_size, loss_function, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                                       test_loss_min, do_all_test_examples, final_test_batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours )    

    job_level_classifications_matrix               += run_level_classifications_matrix                     # accumulate for the job level stats. Has to be just after this call to 'test()'    


    # (D)  PROCESS AND DISPLAY RUN LEVEL STATISTICS

    if DEBUG>4:
      print ( f"\n{run_level_classifications_matrix}" )
         
    run_level_classifications_matrix_acc[run-1,:,:] = run_level_classifications_matrix[:,:]                # accumulate run_level_classifications_matrices
 
    if DEBUG>9:
      print ( f"\n{run_level_classifications_matrix_acc[run-1,:,:]}" )    

    if args.input_mode=='rna':    
      print ( "\033[8B", end='' )
    else:
      print ( "\033[8B", end='' )

    #np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE if x==0 else WHITE if x<=5 else CARRIBEAN_GREEN} {x:>15d}"})  
    #print ( f"TRAINLENEJ:     INFO:  {ORANGE}run_level{RESET}_classifications_matrix (all test samples, using the best model that was saved during this run =\n" )
    #print ( f"         ", end='' ) 
    #print ( [ f"{name:.50s}" for name in class_names ] )    
    #print ( f"\n{run_level_classifications_matrix}{RESET}" )
  
    print( f'\n')
    print( f'TRAINLENEJ:       INFO:    {BITTER_SWEET}run level stats{RESET}'  )
    print( f"TRAINLENEJ:       INFO:    {BITTER_SWEET}==============={RESET}"  )  
  
    total_correct, total_examples  = show_classifications_matrix( writer, total_runs_in_job, epoch, run_level_classifications_matrix, level='run' )


    print( f"TRAINLENEJ:       INFO:    correct / examples  =  {BITTER_SWEET}{np.sum(total_correct, axis=0)} / {np.sum(run_level_classifications_matrix, axis=None)}{WHITE}  ({BITTER_SWEET}{100 * np.sum(total_correct, axis=0) / np.sum(run_level_classifications_matrix):3.1f}%){RESET}")

    for i in range( 0, len( run_level_classifications_matrix) ):                                           # reset for the next run   
      run_level_classifications_matrix[i] = 0  
  

    hours   = round( (time.time() - start_time) / 3600,  1   )
    minutes = round( (time.time() - start_time) /   60,  1   )
    seconds = round( (time.time() - start_time),     0       )
    #pprint.log_section('run complete in {:} mins'.format( minutes ) )

    print( f'TRAINLENEJ:       INFO:    elapsed time since job started: {MIKADO}{minutes}{RESET} mins ({MIKADO}{seconds:.1f}{RESET} secs)')


  #  ^^^  JOB FINISHES HERE ^^^


  # (E)  PROCESS AND DISPLAY JOB LEVEL STATISTICS
  print( f'\n\n\n\n')
  print( f'TRAINLENEJ:       INFO:    {CARRIBEAN_GREEN}job level stats{RESET}'  )
  print( f"TRAINLENEJ:       INFO:    {CARRIBEAN_GREEN}==============={RESET}"  )  

  total_correct, total_examples  = show_classifications_matrix( writer, total_runs_in_job, epoch, job_level_classifications_matrix, level='job' )

  np.seterr( invalid='ignore', divide='ignore' ) 
  print( f"\n" )
  print( f'TRAINLENEJ:       INFO:    number of runs in this job                 = {MIKADO}{total_runs_in_job}{RESET}')
  print( f"TRAINLENEJ:       INFO:    total for ALL test examples over ALL runs  =  {CARRIBEAN_GREEN}{np.sum(total_correct, axis=0)} / {np.sum(job_level_classifications_matrix, axis=None)}  ({CARRIBEAN_GREEN}{100 * np.sum(total_correct, axis=0) / np.sum(job_level_classifications_matrix):3.1f}%){RESET}")

  np.set_printoptions(formatter={'int': lambda x: f"{CARRIBEAN_GREEN}{x:>6d}    "})
  print( f'TRAINLENEJ:       INFO:    total correct per subtype over all runs:                          {total_correct}{RESET}')
  np.set_printoptions(formatter={'float': lambda x: f"{CARRIBEAN_GREEN}{x:>6.2f}    "})
  print( f'TRAINLENEJ:       INFO:     %    correct per subtype over all runs:                          { 100 * np.divide( total_correct, total_examples) }{RESET}')
  np.seterr(divide='warn', invalid='warn')  
  
  if DEBUG>9:
    np.set_printoptions(formatter={'int': lambda x: f"{CARRIBEAN_GREEN}{x:>6d}    "})    
    print ( f"TRAINLENEJ:       INFO:    run_level_classifications_matrix_acc[0:total_runs_in_job,:,:]            = \n{run_level_classifications_matrix_acc[0:total_runs_in_job,:,:] }{RESET}" )
  if DEBUG>9:
    print ( f"TRAINLENEJ:       INFO:  run_level_classifications_matrix_acc                 = {MIKADO}{run_level_classifications_matrix_acc}{RESET}"     )


  box_plot_by_subtype( writer, total_runs_in_job, run_level_classifications_matrix_acc )



  # (F)  CLOSE UP AND END
  writer.close()        
  
  print( f"\n\n\nTRAINLENEJ:       INFO: {WHITE}job complete{RESET}" )

  hours   = round( (time.time() - start_time) / 3600,  1   )
  minutes = round( (time.time() - start_time) /   60,  1   )
  seconds = round( (time.time() - start_time),     0       )
  #pprint.log_section('Job complete in {:} mins'.format( minutes ) )

  print( f'TRAINLENEJ:       INFO: the whole job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
            
  #pprint.log_section('Model saved.')
  
  
# --------------------------------------------------------------------------------------------  
def box_plot_by_subtype( writer, total_runs_in_job, pandas_matrix ):
  
  # (1) Just some stats
  flattened              =  np.sum  ( pandas_matrix, axis=0 )                                                                          # sum across all examples to produce a 2D matrix
  
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    flattened.shape     = {CARRIBEAN_GREEN}{flattened.shape}{RESET}')
  total_examples_by_subtype     =  np.expand_dims(np.sum  (  flattened, axis=0 ), axis=0 )                                         # sum down the columns to produces a row vector
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    total_examples_by_subtype.shape  = {CARRIBEAN_GREEN}{total_examples_by_subtype.shape}{RESET}')
  if DEBUG>9:    
    print( f'TRAINLENEJ:       INFO:    total_examples_by_subtype        = {CARRIBEAN_GREEN}{total_examples_by_subtype}{RESET}') 
    
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    flattened.shape     = {CARRIBEAN_GREEN}{flattened.shape}{RESET}')
  total_correct_by_subtype      =  np.array( [ flattened[i,i] for i in  range( 0 , len( flattened ))  ] )                              # pick out diagonal elements (= number correct) to produce a row vector
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    total_correct_by_subtype.shape   = {CARRIBEAN_GREEN}{total_correct_by_subtype.shape}{RESET}')
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    total_correct_by_subtype         = {CARRIBEAN_GREEN}{total_correct_by_subtype}{RESET}')                                
  
  
  # (2) process and present box plot

  total_values_plane            =   np.sum(  pandas_matrix, axis=1 )[ 0:total_runs_in_job, : ]                                         # sum elements (= numbers correct) from 3D volume down columns (axis 1)  to produce a matrix
  if DEBUG>0:
    print( f'TRAINLENEJ:       INFO:    total_values_plane.shape         = {CARRIBEAN_GREEN}{total_values_plane.shape}{RESET}')
  if DEBUG>0:
    np.set_printoptions(formatter={ 'int' : lambda x: f"   {CARRIBEAN_GREEN}{x:>6d}   "} )    
    print( f'TRAINLENEJ:       INFO:    total_values_plane               = \n{CARRIBEAN_GREEN}{total_values_plane}{RESET}')

  correct_values_plane          =   np.transpose( np.array( [ pandas_matrix[:,i,i] for i in  range( 0 , pandas_matrix.shape[1] ) ]  )  ) [ 0:total_runs_in_job, : ]      # pick out diagonal elements (= numbers correct) from 3D volume  to produce a matrix
  if DEBUG>0:
    print( f'TRAINLENEJ:       INFO:    correct_values_plane.shape       = {CARRIBEAN_GREEN}{correct_values_plane.shape}{RESET}')
  if DEBUG>0:
    np.set_printoptions(formatter={ 'int' : lambda x: f"   {CARRIBEAN_GREEN}{x:>6d}   "} )          
    print( f'TRAINLENEJ:       INFO:    correct_values_plane             = \n{CARRIBEAN_GREEN}{correct_values_plane}{RESET}')

  
  np.seterr( invalid='ignore', divide='ignore' )          
  percentage_correct_plane      =   100 * np.divide( correct_values_plane, total_values_plane )
  if DEBUG>0:
    print( f'TRAINLENEJ:       INFO:    percentage_correct_plane.shape   = {CARRIBEAN_GREEN}{percentage_correct_plane.shape}{RESET}')
  if DEBUG>0:
    np.set_printoptions(formatter={'float': lambda x: f"   {CARRIBEAN_GREEN}{x:>6.2f}   "} )    
    print( f'TRAINLENEJ:       INFO:    percentage_correct_plane         = \n{CARRIBEAN_GREEN}{percentage_correct_plane}{RESET}')
  np.seterr(divide='warn', invalid='warn') 
  
  npy_class_names = np.transpose(np.expand_dims( np.array(args.class_names), axis=0 ) )
  if DEBUG>0:
    print( f'TRAINLENEJ:       INFO:    npy_class_names.shape   = {CARRIBEAN_GREEN}{npy_class_names.shape}{RESET}')
    print( f'TRAINLENEJ:       INFO:    npy_class_names         = \n{CARRIBEAN_GREEN}{npy_class_names}{RESET}')
        
  pd_percentage_correct_plane =   pd.DataFrame( (100-correct_values_plane), columns=npy_class_names )                 

  
  figure_width  = 8
  figure_height = 12 
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  plt.xticks(rotation=90)
  #sns.set_theme(style="whitegrid")   
  ax = sns.boxplot( data=pd_percentage_correct_plane, orient='v' )
  ax.set(ylim=(0, 100))
  #plt.show()
  writer.add_figure('Box Plot V', fig, 1)
  
  # save portrait version of box plot to logs directory
  now              = datetime.datetime.now()
  file_name_prefix = f"_r{total_runs_in_job}_e{args.n_epochs}_n{args.n_samples[0]}_b{args.batch_size[0]}_t{int(100*args.pct_test)}_lr{args.learning_rate[0]}_h{args.hidden_layer_neurons[0]}_d{int(100*args.nn_dense_dropout_1[0])}"
  
  fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__box_plot_portrait.png"
  fig.savefig(fqn)
  
  figure_width  = 12
  figure_height = 8 
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  plt.xticks(rotation=0)
  #sns.set_theme(style="whitegrid")   
  ax = sns.boxplot( data=pd_percentage_correct_plane, orient='h' )
  ax.set(xlim=(0, 100))
  #plt.show()
  writer.add_figure('Box Plot H', fig, 1)
  
  # save landscape version of box plot to logs directory
  fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__box_plot_landscape.png"
  fig.savefig(fqn)  
  
  
  plt.close('all')
  
  return

# --------------------------------------------------------------------------------------------  
def show_classifications_matrix( writer, total_runs_in_job, epoch, pandas_matrix, level ):

  global final_test_batch_size

  # (1) Process and Present the Table
  
  total_examples_by_subtype         =  np.sum  (   pandas_matrix, axis=0  )                                                          # sum down the columns. produces a row vector
  total_correct_by_subtype          =  np.array( [ pandas_matrix[i,i] for i in  range( 0 , len( total_examples_by_subtype ))  ] )    # produces a row vector                                      
  np.seterr( invalid='ignore', divide='ignore' )                                                                                     # produces a row vector
  percent_correct_by_subtype        =  100*np.divide (       total_correct_by_subtype, total_examples_by_subtype )                   # produces a row vector
  percent_wrong_by_subtype          =  100*np.divide (   1-percent_correct_by_subtype, total_examples_by_subtype )                   # produces a row vector
  np.seterr(divide='warn', invalid='warn') 
                 

  exp_total_examples_by_subtype     =  np.expand_dims( total_examples_by_subtype,                          axis=0 )
  ext1_pandas_matrix                =  np.append     ( pandas_matrix, exp_total_examples_by_subtype,       axis=0 )  
  
  exp_total_correct_by_subtype      =  np.expand_dims( total_correct_by_subtype, axis=0 )      
  ext2_pandas_matrix                =  np.append     ( ext1_pandas_matrix, exp_total_correct_by_subtype,   axis=0 )      
  
  exp_percent_correct_by_subtype    =  np.expand_dims( percent_correct_by_subtype,                         axis=0 )
  ext3_pandas_matrix                =  np.append     ( ext2_pandas_matrix, exp_percent_correct_by_subtype, axis=0 )            
           

  index_names = args.class_names.copy()   
    
  #print ( "" )                                                                                            # peel off an all numbers pandas version to use for graphing etc
  pandas_version = pd.DataFrame( pandas_matrix, columns=args.class_names, index=index_names )
  #print(tabulate(pandas_version, headers='keys', tablefmt = 'psql'))     

  index_names.append( "subtype totals"  )
  index_names.append( "subtype correct" ) 
  index_names.append( "percent correct" )

  
  print ( "" )                                                                                             # this version has subtotals etc at the bottom so it's just for display
  pandas_version_ext = pd.DataFrame( ext3_pandas_matrix, columns=args.class_names, index=index_names )  
  print(tabulate( pandas_version_ext, headers='keys', tablefmt = 'fancy_grid' ) )   
  
  #display(pandas_version_ext)
 
 
  # (1) Save job level classification matrix as a csv file in logs directory

  if level=='job':

    now              = datetime.datetime.now()
    file_name_prefix = f"_r{total_runs_in_job}_e{args.n_epochs}_n{args.n_samples[0]}_b{args.batch_size[0]}_t{int(100*args.pct_test)}_lr{args.learning_rate[0]}_h{args.hidden_layer_neurons[0]}_d{int(100*args.nn_dense_dropout_1[0])}"
    fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__job_level_classifications_matrix.csv"

    try:
      pandas_version.to_csv( fqn, sep='\t' )
      if DEBUG>0:
        print ( f"TRAINLENEJ:     INFO:     saving          job level classification file to {MAGENTA}{fqn}{RESET}"  )
    except Exception as e:
      print ( f"{RED}TRAINLENEJ:     FATAL:     could not save file {MAGENTA}{fqn}{RESET}"  )
      print ( f"{RED}TRAINLENEJ:     FATAL:     error was: {e}{RESET}" )
      sys.exit(0)    
    
    fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__job_level_classifications_matrix_with_totals.csv"
    try:
      pandas_version_ext.to_csv( fqn, sep='\t' )
      if DEBUG>0:
        print ( f"TRAINLENEJ:     INFO:     saving extended job level classification file to {MAGENTA}{fqn}{RESET}"  )
    except Exception as e:
      print ( f"{RED}TRAINLENEJ:     FATAL:     could not save file         = {MAGENTA}{fqn}{RESET}"  )
      print ( f"{RED}TRAINLENEJ:     FATAL:     error was: {e}{RESET}" )      
      sys.exit(0)
  
  return ( total_correct_by_subtype, total_examples_by_subtype )

  
  
# --------------------------------------------------------------------------------------------
def triang( df ):

  print( f"{BRIGHT_GREEN}TRAINLENEJ:     INFO: at top of triang(){RESET} ")  
  temp=df.copy()
  ut=np.triu(np.ones(df.shape),1).astype(np.bool)
  lt=np.tril(np.ones(df.shape),-1).astype(np.bool)

  temp=temp.where(ut==False, 'up')
  temp=temp.where(lt==False, 'lt')
  np.fill_diagonal(temp.values,'dg')
  return(temp)
    
# --------------------------------------------------------------------------------------------
def color_vals(val):

  # pandas_version_ext.style.apply( color_vals )

  print( f"{MIKADO}TRAINLENEJ:     INFO: at top of color_vals(){RESET} ")   
  """
  Color dataframe using values
  """
  d = {'up' : 'orange',
       'dg' : 'black',
       'lt' : 'blue'}
  return [f'color : {i}' for i in triang(df_vals).loc[val.name, val.index].map(d).tolist()] 

# --------------------------------------------------------------------------------------------
def color_negative_red(val):

    #pd_percentage_correct_plane.style.applymap(color_negative_red) 

    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 1 else 'white'
    return 'color: %s' % color
# --------------------------------------------------------------------------------------------



def train(args, epoch, train_loader, model, optimizer, loss_function, writer, train_loss_min, batch_size  ):
    """
    Train model and update parameters in batches of the whole training set
    """
    
    if DEBUG>1:
      print( "TRAINLENEJ:     INFO:     at top of train() and parameter train_loader() = \033[35;1m{:}\033[m".format( train_loader ) )
    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:     at top of train() with parameters \033[35;1margs: \033[m{:}, \033[35;1mtrain_loader: \033[m{:}, \033[35;1mmodel: \033[m{:}, \033[35;1moptimizer: \033[m{:}".format(args, train_loader, model, optimizer ) )

    if DEBUG>1:
      print( "TRAINLENEJ:     INFO:     train(): about to call \033[33;1mmodel.train()\033[m" )

    model.train()                                                                                          # set model to training mode

    loss_images_sum  = 0
    loss_genes_sum   = 0
    l1_loss_sum      = 0
    total_loss_sum   = 0


    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:     train(): about to enumerate over dataset" )
    
    for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in enumerate( train_loader ):
        
        if DEBUG>88:
          print( f"TRAINLENEJ:     INFO:     train(): len(batch_images) = \033[33;1m{len(batch_images)}\033[m" )
          print( f"TRAINLENEJ:     INFO:     train(): len(image_labels) = \033[33;1m{len(image_labels)}\033[m" )
          print( f"TRAINLENEJ:     INFO:     train(): len(rna_labels)   = \033[33;1m{len(rna_labels)}\033[m" )
        if DEBUG>888:
          print ( "\033[6B" )
          print( f"{ image_labels.cpu().detach().numpy()},  ", flush=True, end="" )
          print( f"{   rna_labels.cpu().detach().numpy()},  ", flush=True, end="" )    
          print ( "\033[6A" )
                            
        if DEBUG>888:
          print( f"TRAINLENEJ:     INFO:     train(): about to call {CYAN}optimizer.zero_grad(){RESET}" )

        optimizer.zero_grad()

        batch_images = batch_images.to ( device )                                                          # send to GPU
        batch_genes  = batch_genes.to  ( device )                                                          # send to GPU
        image_labels = image_labels.to ( device )                                                          # send to GPU
        rna_labels   = rna_labels.to   ( device )                                                          # send to GPU

        if DEBUG>9:
          print ( "TRAINLENEJ:     INFO:     train():       type(batch_images)                 = {:}".format( type(batch_images)       ) )
          print ( "TRAINLENEJ:     INFO:     train():       batch_images.size()                = {:}".format( batch_images.size()       ) )

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:      train(): about to call \033[33;1mmodel.forward()\033[m" )

        gpu                = 0                                                                             # to maintain compatability with NN_MODE=pre_compress
        encoder_activation = 0                                                                             # to maintain compatability with NN_MODE=pre_compress
        if args.input_mode=='image':
          y1_hat, y2_hat = model.forward( [batch_images, 0          ], gpu, encoder_activation  )          # perform a step. y1_hat = image outputs; y2_hat = rna outputs
        elif args.input_mode=='rna':
          y1_hat, y2_hat = model.forward( [0,            batch_genes], gpu, encoder_activation )           # perform a step
        elif args.input_mode=='image_rna':
          y1_hat, y2_hat = model.forward( [batch_images, batch_genes], gpu, encoder_activation  )          # perform a step


        if (args.input_mode=='image') | (args.input_mode=='image_rna'):
          if DEBUG>99:
            np.set_printoptions(formatter={'int': lambda x:   "{:>4d}".format(x)})
            image_labels_numpy = (image_labels.cpu().data).numpy()
            print ( "TRAINLENEJ:     INFO:      train():       image_labels_numpy                = \n{:}".format( image_labels_numpy  ) )
          if DEBUG>99:
            np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
            y1_hat_numpy = (y1_hat.cpu().data).numpy()
            print ( "TRAINLENEJ:     INFO:      train():       y1_hat_numpy                      = \n{:}".format( y1_hat_numpy) )
          loss_images       = loss_function( y1_hat, image_labels )
          loss_images_value = loss_images.item()                                                           # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories
          
          if DEBUG>888:
              print ( f"TRAINLENEJ:     INFO:      train():             loss_images_value           = {PURPLE}{loss_images_value:6.3f}{RESET}" )
        
        if (args.input_mode=='rna')   | (args.input_mode=='image_rna'):
          if DEBUG>9:
            np.set_printoptions(formatter={'int': lambda x:   "{:>4d}".format(x)})
            rna_labels_numpy = (rna_labels.cpu().data).numpy()
            print ( "TRAINLENEJ:     INFO:      train():       rna_labels_numpy                = \n{:}".format( image_labels_numpy  ) )
          if DEBUG>9:
            np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
            y2_hat_numpy = (y2_hat.cpu().data).numpy()
            print ( "TRAINLENEJ:     INFO:      train():       y2_hat_numpy                      = \n{:}".format( y2_hat_numpy) )
          loss_genes        = loss_function( y2_hat, rna_labels )
          loss_genes_value  = loss_genes.item()                                                            # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories

        #l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0

        if (args.input_mode=='image'):
          total_loss        = loss_images_value + l1_loss
        elif (args.input_mode=='rna'):
          total_loss        = loss_genes_value + l1_loss
        elif (args.input_mode=='image_rna'):
          total_loss        = loss_genes_value + l1_loss


        if DEBUG>0:
          if ( args.input_mode=='image' ):
            print ( f"\
\033[2K\r\033[27C{DULL_WHITE}train():\
\r\033[40Cn={i+1:>3d}{CLEAR_LINE}\
\r\033[49Closs_images={ loss_images_value:5.2f}\
\r\033[96Cl1_loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{156+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
            print ( "\033[2A" )
          elif ( args.input_mode=='rna' ):
            print ( f"\
\033[2K\r\033[27C{DULL_WHITE}train():\
\r\033[40Cn={i+1:>3d}{CLEAR_LINE}\
\r\033[73Closs_rna={loss_genes_value:5.2f}\
\r\033[96Cl1_loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{156+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
            print ( "\033[2A" )          
          if ( args.input_mode=='image_rna' ):
            print ( f"\
\033[2K\r\033[27C{DULL_WHITE}train():\
\r\033[40Cn={i+1:>3d}{CLEAR_LINE}\
\r\033[49Closs_images={ loss_images_value:5.2f}\
\r\033[73Closs_rna={loss_genes_value:5.2f}\
\r\033[96Cl1_loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{156+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
            print ( "\033[2A" )


        if (args.input_mode=='image') | (args.input_mode=='image_rna'):
          loss_images.backward()
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          loss_genes.backward()

        # Perform gradient clipping *before* calling `optimizer.step()`.
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        
        if (args.input_mode=='image') | (args.input_mode=='image_rna'):
          loss_images_sum      += loss_images_value
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          loss_genes_sum       += loss_genes_value
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss

        if (args.input_mode=='image') | (args.input_mode=='image_rna'):
          del y1_hat                 
          del loss_images
          del image_labels          
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          del y2_hat   
          del loss_genes
          del rna_labels

        torch.cuda.empty_cache()

        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      train():       type(loss_images_sum)                      = {:}".format( type(loss_images_sum)       ) )
          
    loss_images_sum_ave = loss_images_sum / (i+1)                                                          # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    loss_genes_sum_ave  = loss_genes_sum  / (i+1)                                                          # average genes loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    l1_loss_sum_ave     = l1_loss_sum     / (i+1)                                                          # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    total_loss_ave      = total_loss_sum  / (i+1)                                                          # average total loss for the entire epoch (divide cumulative loss by number of batches in the epoch)

    if total_loss_sum < train_loss_min:
      train_loss_min = total_loss_sum

    if args.just_test=='False':                                                                            # don't record stats in test mode because it's only one epoch and is of no interest
      writer.add_scalar( 'loss_train',      total_loss_sum, epoch )
      writer.add_scalar( 'loss_train_min',  train_loss_min, epoch )

    return loss_images_sum_ave, loss_genes_sum_ave, l1_loss_sum_ave, total_loss_ave

# ------------------------------------------------------------------------------




def test( cfg, args, epoch, test_loader,  model,  tile_size, loss_function, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                                        test_loss_min, do_all_test_examples, batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours ):

    """Test model by pusing a held out batch through the network
    """

    global global_batch_count
    
    global run_level_total_correct    
    global run_level_classifications_matrix
    global job_level_classifications_matrix    

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to test model by computing the average loss on a held-out dataset. No parameter updates" )

    model.eval()                                                                                           # set model to evaluation mod

    loss_images_sum     = 0
    loss_genes_sum      = 0
    l1_loss_sum         = 0
    total_loss_sum      = 0

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to enumerate  " )
      
    for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in  enumerate( test_loader ):         # PGD 200613 - added 'batch_genes'
        
        batch_images = batch_images.to(device)
        batch_genes  = batch_genes .to(device)                                                              # PGD 200613 - added 
        image_labels = image_labels.to(device)
        rna_labels   = rna_labels  .to(device)        
        
        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     test(): about to call \033[33;1mmodel.forward()\033[m" )

        gpu                = 0                                                                             # to maintain compatability with NN_MODE=pre_compress
        encoder_activation = 0                                                                             # to maintain compatability with NN_MODE=pre_compress
        if args.input_mode=='image':
          with torch.no_grad():                                                                            # PGD 200129 - Don't need gradients for testing, so this should save some GPU memory (tested: it does)
            y1_hat, y2_hat = model.forward( [batch_images, 0          ], gpu, encoder_activation  )        # perform a step
        elif args.input_mode=='rna':
          with torch.no_grad():                                                                            # PGD 200129 - Don't need gradients for testing, so this should save some GPU memory (tested: it does)
            y1_hat, y2_hat = model.forward( [0,            batch_genes], gpu, encoder_activation )         # perform a step
        elif args.input_mode=='image_rna':
          with torch.no_grad():                                                                            # PGD 200129 - Don't need gradients for testing, so this should save some GPU memory (tested: it does)
            y1_hat, y2_hat = model.forward( [batch_images, batch_genes], gpu, encoder_activation  )        # perform a step
          
        image_labels_values   = image_labels.cpu().detach().numpy()
        rna_labels_values     =   rna_labels.cpu().detach().numpy()

        if not args.input_mode=='rna':  
          preds, p_full_softmax_matrix, p_highest, p_2nd_highest, p_true_class = analyse_probs( y1_hat, image_labels_values )
        
    
        if ( args.just_test=='True' ) & ( (args.input_mode=='image') | (args.input_mode=='image_rna') ):

          if DEBUG>0:
              print ( f"TRAINLENEJ:     INFO:      test():       global_batch_count {DIM_WHITE}(super-patch number){RESET} = {global_batch_count+1:5d}  {DIM_WHITE}({((global_batch_count+1)/(args.supergrid_size**2)):04.2f}){RESET}" )
                      
          if global_batch_count%(args.supergrid_size**2)==0:
            grid_images                = batch_images.cpu().numpy()
            grid_labels                = image_labels.cpu().numpy()
            grid_preds                 = preds
            grid_p_highest             = p_highest
            grid_p_2nd_highest         = p_2nd_highest
            grid_p_true_class          = p_true_class
            grid_p_full_softmax_matrix = p_full_softmax_matrix 

            if DEBUG>99:
              print ( f"TRAINLENEJ:     INFO:      test():             batch_images.shape                      = {batch_images.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_images.shape                       = {grid_images.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             image_labels.shape                      = {image_labels.shape}" )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_labels.shape                       = {grid_labels.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             preds.shape                             = {preds.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_preds.shape                        = {grid_preds.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             p_highest.shape                         = {p_highest.shape}" )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_highest.shape                    = {grid_p_highest.shape}" )            
              print ( f"TRAINLENEJ:     INFO:      test():             p_2nd_highest.shape                     = {p_2nd_highest.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_2nd_highest.shape                = {grid_p_2nd_highest.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             p_full_softmax_matrix.shape             = {p_full_softmax_matrix.shape}" )                                    
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_full_softmax_matrix.shape        = {grid_p_full_softmax_matrix.shape}" )
                      
          else:
            grid_images                = np.append( grid_images,                batch_images.cpu().numpy(), axis=0 )
            grid_labels                = np.append( grid_labels,                image_labels.cpu().numpy(), axis=0 )
            grid_preds                 = np.append( grid_preds,                 preds,                      axis=0 )
            grid_p_highest             = np.append( grid_p_highest,             p_highest,                  axis=0 )
            grid_p_2nd_highest         = np.append( grid_p_2nd_highest,         p_2nd_highest,              axis=0 )
            grid_p_true_class          = np.append( grid_p_true_class,          p_true_class,               axis=0 )
            grid_p_full_softmax_matrix = np.append( grid_p_full_softmax_matrix, p_full_softmax_matrix,      axis=0 )
  
            if DEBUG>99:
              print ( f"TRAINLENEJ:     INFO:      test():             grid_images.shape                       = {grid_images.shape}"        )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_labels.shape                       = {grid_labels.shape}"        )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_preds.shape                        = {grid_preds.shape}"         )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_highest.shape                    = {grid_p_highest.shape}"     )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_2nd_highest.shape                = {grid_p_2nd_highest.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_true_class.shape                 = {grid_p_true_class.shape}"  )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_full_softmax_matrix.shape        = {grid_p_full_softmax_matrix.shape}"            )  

          global_batch_count+=1
        
          if DEBUG>999:
              print ( f"TRAINLENEJ:     INFO:      test():             global_batch_count%(args.supergrid_size**2)                       = {global_batch_count%(args.supergrid_size**2)}"  )
          
          if global_batch_count%(args.supergrid_size**2)==0:
            if args.input_mode=='image':
              print("")
              
              if args.annotated_tiles=='True':
                
                fig=plot_classes_preds(args, model, tile_size, grid_images, grid_labels, grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_full_softmax_matrix, class_names, class_colours )
                writer.add_figure('1 annotated tiles', fig, epoch)
                plt.close(fig)


              batch_fnames_npy = batch_fnames.numpy()                                                # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details
              
              if DEBUG>99:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():       batch_fnames_npy.shape      = {batch_fnames_npy.shape:}" )        
                print ( f"TRAINLENEJ:     INFO:      test():       batch_fnames_npy            = {batch_fnames_npy:}"       )
    
              fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"
              
              if DEBUG>0:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():       fq_link                     = {PINK}{fq_link:}{RESET}"                )
                print ( f"TRAINLENEJ:     INFO:      test():       file fq_link points to      = {PINK}{os.readlink(fq_link)}{RESET}"    )
              
              background_image = np.load(f"{fq_link}")
              
              if DEBUG>999:
                print ( f"TRAINLENEJ:     INFO:      test():        background_image.shape = {background_image.shape}" )
                
              if args.scattergram=='True':
                
                plot_scatter(args, writer, epoch, background_image, tile_size, grid_labels, class_names, class_colours, grid_preds, show_patch_images='True')
                plot_scatter(args, writer, epoch, background_image, tile_size, grid_labels, class_names, class_colours, grid_preds, show_patch_images='False')

              if args.probs_matrix=='True':
                
                # without interpolation
                matrix_types = [ 'margin_1st_2nd', 'confidence_RIGHTS', 'p_std_dev' ]
                for i, matrix_type in enumerate(matrix_types):
                  plot_matrix (matrix_type, args, writer, epoch, background_image, tile_size, grid_labels, class_names, class_colours, grid_p_full_softmax_matrix, grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_true_class, 'none' )    # always display without probs_matrix_interpolation 
                # with  interpolation
                matrix_types = [ 'probs_true' ]
                for i, matrix_type in enumerate(matrix_types): 
                  plot_matrix (matrix_type, args, writer, epoch, background_image, tile_size, grid_labels, class_names, class_colours, grid_p_full_softmax_matrix, grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_true_class, args.probs_matrix_interpolation )
                

        if DEBUG>9:
          y1_hat_numpy = (y1_hat.cpu().data).numpy()
          print ( "TRAINLENEJ:     INFO:      test():        type(y1_hat)                      = {:}".format( type(y1_hat_numpy)       ) )
          print ( "TRAINLENEJ:     INFO:      test():        y1_hat.shape                      = {:}".format( y1_hat_numpy.shape       ) )
          print ( "TRAINLENEJ:     INFO:      test():        image_labels.shape                = {:}".format( image_labels.shape  ) )
        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      test():        y1_hat                            = \n{:}".format( y1_hat_numpy) )
          print ( "TRAINLENEJ:     INFO:      test():        image_labels                      = \n{:}".format( image_labels  ) )


        if (args.input_mode=='image') | (args.input_mode=='image_rna'):
          loss_images       = loss_function(y1_hat, image_labels)
          loss_images_value = loss_images.item()                                                             # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories
        if (args.input_mode=='rna')   | (args.input_mode=='image_rna'):
          loss_genes        = loss_function(y2_hat, rna_labels)
          loss_genes_value  = loss_genes.item()                                                              # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories

        if DEBUG>9:
          print ( "\033[2K                           test():      loss_images, loss_images_values ={:}, {:}".format( loss_images_value,  loss_images_value))


        #l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0
        
        if (args.input_mode=='image'):
          total_loss        = loss_images_value + l1_loss
        elif (args.input_mode=='rna'):
          total_loss        = loss_genes_value + l1_loss
        elif (args.input_mode=='image_rna'):
          total_loss        = loss_images_value + loss_genes_value + l1_loss        
        

        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      test():       type(loss)                      = {:}".format( type(loss)       ) )

        if DEBUG>0:
          if (not args.just_test=='True'):

            if ( args.input_mode=='image' ):
              print ( f"\
\033[2K\r\033[27Ctest():\
\r\033[40C{DULL_WHITE}n={i+1:>3d}{CLEAR_LINE}\
\r\033[49Closs_images={loss_images_value:5.2f}\
\r\033[96Cl1_ loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{150+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
              print ( "\033[2A" )
            elif ( args.input_mode=='rna' ):
              print ( f"\
\033[2K\r\033[27Ctest():\
\r\033[40C{DULL_WHITE}n={i+1:>3d}{CLEAR_LINE}\
\r\033[73Closs_rna={loss_genes_value:5.2f}\
\r\033[96Cl1_loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{150+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
              print ( "\033[2A" )
            elif ( args.input_mode=='image_rna' ):
              print ( f"\
\033[2K\r\033[27Ctest():\
\r\033[40C{DULL_WHITE}n={i+1:>3d}{CLEAR_LINE}\
\r\033[49Closs_images={loss_images_value:5.2f}\
\r\033[73Closs_rna={loss_genes_value:5.2f}\
\r\033[96Cl1_loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{150+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
              print ( "\033[2A" )


        if not args.input_mode=='rna':
          loss_images_sum      += loss_images_value
        if not args.input_mode=='image':
          loss_genes_sum       += loss_genes_value
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss    

        if ( args.input_mode=='image') | (args.input_mode=='image_rna'):   
          del loss_images
        if ( args.input_mode=='rna'  ) | (args.input_mode=='image_rna'):       
          del loss_genes
        torch.cuda.empty_cache()


    ### ENDOF for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in  enumerate( test_loader ):

    if epoch % 1 == 0:
      if args.input_mode=='image':      
        y1_hat_values             = y1_hat.cpu().detach().numpy()
        y1_hat_values_max_indices = np.argmax( np.transpose(y1_hat_values), axis=0 )                       # indices of the highest values of y1_hat = highest probability class
      if args.input_mode=='rna':      
        y2_hat_values             = y2_hat.cpu().detach().numpy()
        y2_hat_values_max_indices = np.argmax( np.transpose(y2_hat_values), axis=0 )                       # indices of the highest values of y2_hat = highest probability class
      elif args.input_mode=='image_rna':
        y1_hat_values             = y1_hat.cpu().detach().numpy()
        y1_hat_values_max_indices = np.argmax( np.transpose(y1_hat_values), axis=0 )                       # indices of the highest values of y1_hat = highest probability class
        y2_hat_values             = y2_hat.cpu().detach().numpy()
        y2_hat_values_max_indices = np.argmax( np.transpose(y2_hat_values), axis=0 )                       # indices of the highest values of y2_hat = highest probability class     
      
      image_labels_values       = image_labels.cpu().detach().numpy()
      rna_labels_values         =   rna_labels.cpu().detach().numpy()

      torch.cuda.empty_cache()
      
      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      test():        y1_hat.shape                      = {:}".format( y1_hat.shape                     ) )
        print ( "TRAINLENEJ:     INFO:      test():        y1_hat_values_max_indices.shape   = {:}".format( y1_hat_values_max_indices.shape  ) )
        print ( "TRAINLENEJ:     INFO:      test():        image_labels_values.shape         = {:}".format( image_labels_values.shape        ) )
        print ( "TRAINLENEJ:     INFO:      test():        rna_labels_values.shape           = {:}".format(   rna_labels_values.shape        ) )
      
      number_to_display= 9 if args.dataset=='tcl' else batch_size
      np.set_printoptions(linewidth=10000)   
      np.set_printoptions(edgeitems=10000)
      np.set_printoptions(threshold=10000)      
      print ( "" )
      
      if args.input_mode=='image':          
        correct=np.sum( np.equal(y1_hat_values_max_indices, image_labels_values))
      elif args.input_mode=='rna':          
        correct=np.sum( np.equal(y2_hat_values_max_indices, rna_labels_values))
      elif args.input_mode=='image_rna':
        correct=np.sum( np.equal(y2_hat_values_max_indices, rna_labels_values))                          # Use number of rna-seq preds correct until multimode is fully working
            
      pct=100*correct/batch_size if batch_size>0 else 0
      global_pct = 100*(global_correct_prediction_count+correct) / (global_number_tested+batch_size) 
      if do_all_test_examples==False:
        print ( f"{CLEAR_LINE}                           test(): truth/prediction for first {MIKADO}{number_to_display}{RESET} examples from the last test batch \
  ( number correct this batch: {correct}/{batch_size} \
  = {BRIGHT_GREEN if pct>=90 else PALE_GREEN if pct>=80 else ORANGE if pct>=70 else GOLD if pct>=60 else WHITE if pct>=50 else DIM_WHITE}{pct:>5.2f}%{RESET} )  \
  ( number correct overall: {global_correct_prediction_count+correct}/{global_number_tested+batch_size} (number tested per run = n_epochs x batch_size)  \
  = {BRIGHT_GREEN if pct>=90 else PALE_GREEN if pct>=80 else ORANGE if pct>=70 else GOLD if pct>=60 else WHITE if pct>=50 else DIM_WHITE}{pct:>5.2f}%{RESET} )" )
      else:
        run_level_total_correct.append( correct )
        print ( f"{CLEAR_LINE}                           test(): truth/prediction for all {MIKADO}{number_to_display}{RESET} test examples \
  ( number correct  - all test examples - this run: {correct}/{batch_size} \
  = {BRIGHT_GREEN if pct>=90 else PALE_GREEN if pct>=80 else ORANGE if pct>=70 else GOLD if pct>=60 else WHITE if pct>=50 else DIM_WHITE}{pct:>5.2f}%{RESET} )  \
  ( number correct  - all test examples - cumulative over all runs: {global_correct_prediction_count+correct}/{global_number_tested+batch_size}  \
  = {BRIGHT_GREEN if pct>=90 else PALE_GREEN if pct>=80 else ORANGE if pct>=70 else GOLD if pct>=60 else WHITE if pct>=50 else DIM_WHITE}{pct:>5.2f}%{RESET} )" )


      if args.input_mode=='image':   
        labs  = image_labels_values       [0:number_to_display]
        preds = y1_hat_values_max_indices [0:number_to_display]
        delta  = np.abs(preds - labs)
        np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>1d}{RESET}"})
        print (  f"truth = {labs}", flush=True   )
        print (  f"preds = {preds}", flush=True  )
        np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else DIM_WHITE}{x:>1d}{RESET}"})     
        print (  f"delta = {delta}", flush=True  )
      elif args.input_mode=='rna':   
        labs  = rna_labels_values         [0:number_to_display]
        preds = y2_hat_values_max_indices [0:number_to_display]
        delta = np.abs(preds - labs)
        np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>1d}{RESET}"})
        print (  f"truth = {labs}", flush=True   )
        print (  f"preds = {preds}", flush=True  )
        np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else DIM_WHITE}{x:>1d}{RESET}"})     
        print (  f"delta = {delta}", flush=True  )
        
        
        if do_all_test_examples==True:
          for i in range(0, len(preds) ):
            run_level_classifications_matrix[ labs[i], preds[i] ] +=1
                
      elif args.input_mode=='image_rna':   
        labs  = rna_labels_values         [0:number_to_display]
        preds = y2_hat_values_max_indices [0:number_to_display]
        delta = np.abs(preds - labs)
        np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>1d}{RESET}"})
        print (  f"truth = {labs}", flush=True   )
        print (  f"preds = {preds}", flush=True  )
        np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else DIM_WHITE}{x:>1d}{RESET}"})     
        print (  f"delta = {delta}", flush=True  )


      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      test():       y1_hat.shape                     = {:}".format( y1_hat_values.shape          ) )
        np.set_printoptions(formatter={'float': lambda x: "{0:10.2e}".format(x)})
        print (  "{:}".format( (np.transpose(y1_hat_values))[:,:number_to_display] )  )
        np.set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})

      if DEBUG>2:
        number_to_display=16  
        print ( "TRAINLENEJ:     INFO:      test():       FIRST  GROUP BELOW: y1_hat"                                                                      ) 
        print ( "TRAINLENEJ:     INFO:      test():       SECOND GROUP BELOW: y1_hat_values_max_indices (prediction)"                                      )
        print ( "TRAINLENEJ:     INFO:      test():       THIRD  GROUP BELOW: image_labels_values (truth)"                                                 )
        np.set_printoptions(formatter={'float': '{: >6.2f}'.format}        )
        print ( f"{(np.transpose(y1_hat_values)) [:,:number_to_display] }" )
        np.set_printoptions(formatter={'int': '{: >6d}'.format}            )
        print ( " {:}".format( y1_hat_values_max_indices    [:number_to_display]        ) )
        print ( " {:}".format( image_labels_values          [:number_to_display]        ) )
 
 
    if args.input_mode=='image':   
      y1_hat_values               = y1_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y1_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y1_hat_values_max_indices   = np.argmax( np.transpose(y1_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
    elif args.input_mode=='rna':   
      y2_hat_values               = y2_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y2_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y2_hat_values_max_indices   = np.argmax( np.transpose(y2_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
    elif args.input_mode=='image_rna':   
      y1_hat_values               = y1_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y1_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y1_hat_values_max_indices   = np.argmax( np.transpose(y1_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
      y2_hat_values               = y2_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y2_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y2_hat_values_max_indices   = np.argmax( np.transpose(y2_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
      
    
    image_labels_values         = image_labels.cpu().detach().numpy()                                      # these are the true      classes corresponding to batch_images


    if args.input_mode=='image':   
      correct_predictions              = np.sum( y1_hat_values_max_indices == image_labels_values )
    if args.input_mode=='rna':   
      correct_predictions              = np.sum( y2_hat_values_max_indices == rna_labels_values )
    if args.input_mode=='image_rna':   
      correct_predictions              = np.sum( y1_hat_values_max_indices == image_labels_values )             # PGD 200630 Use number of images correct until multimode is working


    pct_correct                 = correct_predictions / batch_size * 100

    loss_images_sum_ave = loss_images_sum / (i+1)                                                          # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    loss_genes_sum_ave  = loss_genes_sum  / (i+1)                                                          # average genes loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    l1_loss_sum_ave     = l1_loss_sum     / (i+1)                                                          # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    total_loss_ave      = total_loss_sum  / (i+1)                                                          # average total loss for the entire epoch (divide cumulative loss by number of batches in the epoch)

    if total_loss_sum    <  test_loss_min:
       test_loss_min     =  total_loss_sum

    if correct_predictions    >  max_correct_predictions:
      max_correct_predictions =  correct_predictions

    if pct_correct       >  max_percent_correct:
      max_percent_correct    =  pct_correct
    
    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      test():             total_loss_sum                           = {:}".format( total_loss_sum           ) )
      print ( "TRAINLENEJ:     INFO:      test():             test_loss_min                            = {:}".format( test_loss_min            ) )
      print ( "TRAINLENEJ:     INFO:      test():             correct_predictions                           = {:}".format( correct_predictions           ) )
      print ( "TRAINLENEJ:     INFO:      test():             max_correct_predictions                       = {:}".format( max_correct_predictions       ) )
      print ( "TRAINLENEJ:     INFO:      test():             pct_correct                              = {:}".format( pct_correct              ) )
      print ( "TRAINLENEJ:     INFO:      test():             max_percent_correct                          = {:}".format( max_percent_correct          ) )


    if args.just_test=='False':                                                                            # don't record training stats in test mode because it's only one epoch and is of no interest 
      writer.add_scalar( '1a_test_loss',     total_loss_sum,     epoch )
      writer.add_scalar( '1c_test_loss_min', test_loss_min,      epoch )    
      writer.add_scalar( 'num_correct',      correct_predictions,     epoch )
      writer.add_scalar( 'num_correct_max',  max_correct_predictions, epoch )
      writer.add_scalar( 'pct_correct',      pct_correct,        epoch ) 
      writer.add_scalar( 'max_percent_correct',  max_percent_correct,    epoch ) 
  
    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      test():             batch_images.shape                       = {:}".format( batch_images.shape ) )
      print ( "TRAINLENEJ:     INFO:      test():             image_labels.shape                       = {:}".format( image_labels.shape ) )
      
#    if not args.just_test=='True':
#      if args.input_mode=='image':
#        writer.add_figure('Predictions v Truth', plot_classes_preds(args, model, tile_size, batch_images, image_labels, preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours), epoch)
        
    if args.just_test=='False':                                                                            # This call to plot_classes_preds() is for use by test() during training, and not for use in "just_test" mode (the latter needs support for supergrids)
      if args.annotated_tiles=='True':
        fig=plot_classes_preds(args, model, tile_size, batch_images.cpu().numpy(), image_labels.cpu().numpy(), preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours)
        writer.add_figure('Predictions v Truth', fig, epoch)
        plt.close(fig)

    if (args.input_mode=='image') | (args.input_mode=='image_rna'):
      del batch_images
      del image_labels

    
#    if args.just_test=='True':
#      if args.input_mode=='image':
#        it=list(permutations( range(0, batch_size)  ) )
#        writer.add_figure('Predictions v Truth', plot_classes_preds(args, model, tile_size, batch_images, image_labels, preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours), epoch)

    if DEBUG>99:
      print ( "TRAINLENEJ:     INFO:      test():       type(loss_images_sum_ave)                      = {:}".format( type(loss_images_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(loss_genes_sum_ave)                      = {:}".format( type(loss_genes_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(l1_loss_sum_ave)                    = {:}".format( type(l1_loss_sum_ave)   ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(total_loss_ave)                     = {:}".format( type(total_loss_ave)    ) )

    return loss_images_sum_ave, loss_genes_sum_ave, l1_loss_sum_ave, total_loss_ave, correct_predictions, batch_size, max_correct_predictions, max_percent_correct, test_loss_min



# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def newline(ax, p1, p2):
    
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()
       
    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    if DEBUG>0:
      print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             xmin                                    = {xmin}"                            )
      print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             xmax                                    = {xmax}"                            )
      print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             ymin                                    = {ymin}"                            )
      print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             ymax                                    = {ymax}"                            )

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

# ------------------------------------------------------------------------------
def analyse_probs( y1_hat, image_labels_values ):

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(y1_hat, axis=1)

    if DEBUG>99:
      y1_hat_numpy = (y1_hat.cpu().data).numpy()
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds_tensor.shape           = {:}".format( preds_tensor.shape    ) ) 
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds_tensor                 = \n{:}".format( preds_tensor      ) ) 
    
    preds = np.squeeze( preds_tensor.cpu().numpy() )

    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               type(preds)                  = {:}".format( type(preds)           ) )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds.shape                  = {:}".format( preds.shape           ) ) 
      print ( "TRAINLENEJ:     INFO:      analyse_probs():         FIRST  GROUP BELOW: preds"            ) 
      print ( "TRAINLENEJ:     INFO:      analyse_probs():         SECOND GROUP BELOW: y1_hat_numpy.T"   )
      np.set_printoptions(formatter={'int':   lambda x: "\033[1m{:^10d}\033[m".format(x)    }    )
      print ( preds[0:22] )
      #np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      #print (  np.transpose(y1_hat_numpy[0:22,:])  )

    p_full_softmax_matrix = functional.softmax( y1_hat, dim=1).cpu().numpy()

    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():              type(p_full_softmax_matrix)     = {:}".format( type(p_full_softmax_matrix) )  )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               p_full_softmax_matrix          = \n{:}".format( np.transpose(p_full_softmax_matrix[0:22,:])   )  )

    # make a vector of the HIGHEST probability (for each example in the batch) 
    p_highest  = np.array(  [ functional.softmax( el, dim=0)[i].item() for i, el in zip(preds, y1_hat) ]   )

    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               p_highest.shape                = {:}".format( (np.array(p_highest)).shape )  )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               p_highest                      = \n{:}".format( np.array(p_highest) )  )
      
    # make a vector of the SECOND HIGHEST probability (for each example in the batch) (which is a bit trickier)
    p_2nd_highest = np.zeros((len(preds)))
    for i in range (0, len(p_2nd_highest)):
      p_2nd_highest[i] = max( [ el for el in p_full_softmax_matrix[i,:] if el != max(p_full_softmax_matrix[i,:]) ] )

    if DEBUG>99:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               p_2nd_highest              = \n{:}".format( p_2nd_highest   )  )  

    # make a vector of the probability the network gave for the true class (for each example in the batch)
    for i in range (0, len(image_labels_values)):
      p_true_class = np.choose( image_labels_values, p_full_softmax_matrix.T)
    
    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( f"TRAINLENEJ:     INFO:      analyse_probs():               p_true_class              = \n{p_true_class}"  )  
      
   
    return preds, p_full_softmax_matrix, p_highest, p_2nd_highest, p_true_class



# ------------------------------------------------------------------------------
def plot_scatter( args, writer, epoch, background_image, tile_size, image_labels, class_names, class_colours, preds, show_patch_images ):

  number_to_plot = len(image_labels)  
  classes        = len(class_names)
  total_tiles    = len(image_labels)
  nrows          = int(number_to_plot**.5)
  ncols          = nrows
  
  figure_width   = args.figure_width
  figure_height  = args.figure_height

#  (0) define two functions which will be used to draw the secondary 'tile' axes (top and right)
  def forward(x):
      return x/tile_size

  def inverse(x):
      return x*tile_size
  
  # (1) capture scattergram data
  
  scatter_data = [[] for n in range(0, classes)]
    
  correct_predictions = 0
  for r in range(nrows):
  
    for c in range(ncols):

      idx = (r*nrows)+c

      if (preds[idx]==image_labels[idx]):
        correct_predictions+=1
      
      scatter_data[preds[idx]].append( [c*tile_size+int(tile_size/2), r*tile_size+int(tile_size/2)] )
  
  if DEBUG>9:
    for n in range(0, classes):
      if image_labels[idx]==n:                                                                         # Truth class for this slide
        print ( f"{GREEN}", end="")
      else:
        print ( f"{RED}", end="")
      print ( f" scatter_data[{n}] = {class_names[n]:20s} coordinates set = {scatter_data[n]}", flush=True  )                                                                     # Truth class for this slide
      print ( f"{RESET}", end="")
 
  
  marker_wrong='x'                                                                                         # marker used for tiles where the NNprediction was incorrect
  
  #plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
  #plt.rcParams['xtick.top']    = plt.rcParams['xtick.labeltop']    = True
  #plt.rcParams['ytick.left']   = plt.rcParams['ytick.labelleft']   = True
  #plt.rcParams['ytick.right']  = plt.rcParams['ytick.labelright']  = True   
  
  # (2) create the figure and axis
  
  #fig=plt.figure( figsize=( figure_width, figure_height ) )
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

  # (3) imshow the background image first, so that it will be behind the set of axes we will do shortly
  
  if show_patch_images=='True':
    
    img=background_image
    plt.imshow(img, aspect='auto')
  
  # (4) add the legend

  l=[]
  for n in range (0, len(class_colours)):
    l.append(mpatches.Patch(color=class_colours[n], linewidth=0))
    fig.legend(l, args.long_class_names, loc='upper right', fontsize=10, facecolor='lightgrey')  
  
  
  # (5) plot the points, organised so as to be at the centre of where the tiles would be on the background image, if it were tiled (the grid lines are on the tile borders)
  
  for n in range(0, classes ):

    threshold_0=0
    threshold_1=10
    threshold_2=40
    threshold_3=80
    threshold_4=120
    threshold_5=200
    threshold_6=300

    pixel_width  = nrows*tile_size
    pixel_height = pixel_width
    major_ticks  = np.arange(0, pixel_width+1, tile_size)
    second_ticks = np.arange(2, nrows, 1)

    if DEBUG>999:
      print ( f"TRAINLENEJ:     INFO:      major_ticks = {major_ticks}" )    
    
    if not image_labels[idx]==n:                                                                           # if the prediction was WRONG
      
      try:
        x,y = zip(*scatter_data[n])
        x_npy=np.array(x)
        y_npy=np.array(y)
        
        if   threshold_0<=nrows<threshold_1:
          marker_size =int(800000/pixel_width) # seems ok
        elif threshold_1<=nrows<threshold_2:
          marker_size =int(400000/pixel_width) # seems ok
        elif threshold_2<=nrows<threshold_3:
          marker_size =int(200000/pixel_width) # seems ok
        elif threshold_3<=nrows<threshold_4:
          marker_size =int(80000/pixel_width)
        elif threshold_4<=nrows<threshold_5:   # seems ok
          marker_size =int(60000/pixel_width)
        else:
          marker_size = 1
        
        if DEBUG>99:
          print ( f"TRAINLENEJ:     INFO:      nrows       = {nrows}" )
          print ( f"TRAINLENEJ:     INFO:      marker_size = {marker_size}" )
          
        plt.scatter( x_npy, y_npy, c=class_colours[n], marker='x', s=marker_size, zorder=100 )  # 80000 is a good value for sqrt(14*14*64)=112x112
        
      except Exception as e:
        pass

      plt.grid(True, which='major', alpha=1.0, color='dimgrey', linestyle='-', linewidth=1 )
      #plt.tick_params(axis='y', left='on',    which='major', labelsize=12)
      #plt.tick_params(axis='y', right='off',  which='both', labelsize=12)      
      #plt.tick_params(axis='x', bottom='on',  which='major', labelsize=12)
      #plt.tick_params(axis='x', top='off',    which='both', labelsize=12)      

      ax.set_xlim(0,nrows*tile_size)
      ax.set_xlabel('pixels', color='lightgrey', fontsize=14)                                                                                           # definitely working
      ax.set_xticks(major_ticks)                                                                                                                        # definitely working
      ax.tick_params(axis='x', bottom='on', which='major',  color='lightgrey', labelsize=9,  labelcolor='lightgrey', width=1, length=6, direction = 'out')   # definitely working
      #ax.tick_params(axis='x', top='on',    which='major', color='teal',   width=4, length=12, direction = 'in')                                       # definitely working - if un-commented

      ax.set_ylim(ncols*tile_size,0)
      ax.set_ylabel('pixels', color='lightgrey', fontsize=14)                                                                                           # definitely working
      ax.set_yticks(major_ticks)                                                                                                                        # definitely working
      ax.tick_params(axis='y', left='on',  which='major', color='lightgrey', labelsize=9,  labelcolor='lightgrey', width=1, length=6, direction = 'out')     # definitely working
      #ax.tick_params(axis='y', right='on',  which='major', color='red', width=4, length=12, direction = 'out')                                         # definitely working - if un-commented
      

      secax = ax.secondary_xaxis( 'top', functions=( forward, inverse )   )                                                                             # definitely working      
      secax.set_xlabel('tile', color="lightsteelblue", fontsize=14)                                                                                     # definitely working
      secax.xaxis.set_minor_locator(AutoMinorLocator(n=2))                                                                                              # not doing anything
      
      secax = ax.secondary_yaxis( 'right', functions=( forward, inverse )   )                                                                           # definitely working                                                                               
      secax.set_ylabel('tile', color='lightsteelblue', fontsize=14)                                                                                     # definitely working
      secax.yaxis.set_minor_locator(AutoMinorLocator(n=2))                                                                                              # not doing anything

  
  pct_correct = correct_predictions/total_tiles
#  stats=f"Statistics: tile count: {total_tiles}; background tiles: {non_specimen_tiles}; specimen tiles: {specimen_tiles}; correctly predicted: {correct_predictions}/{specimen_tiles} ({pct_correct*100}%)"
  stats=f"Statistics: tile count: {total_tiles}; correctly predicted: {correct_predictions}/{total_tiles} ({pct_correct*100}%)"
  plt.figtext( 0.15, 0.055, stats, size=14, color="black", style="normal" )
  
  scattergram_name = [ "2 scattergram on tiles" if show_patch_images=='True' else "9 scattergram " ][0]
  plt.show
  writer.add_figure( scattergram_name, fig, epoch )
  plt.close(fig)  
    
  return
      

# ------------------------------------------------------------------------------
def plot_matrix( matrix_type, args, writer, epoch, background_image, tile_size, image_labels, class_names, class_colours, grid_p_full_softmax_matrix, preds, p_highest, p_2nd_highest, p_true_class, probs_matrix_interpolation ):

  number_to_plot = len(image_labels)  
  nrows          = int(number_to_plot**.5)
  ncols          = nrows
  
  figure_width   = args.figure_width
  figure_height  = args.figure_height
  fig = plt.figure( figsize=( figure_width, figure_height ) )
      
  if matrix_type=='probs_true':
    
    p_true_class     = p_true_class[np.newaxis,:] 
    p_true_class     = p_true_class.T
    reshaped_to_2D   = np.reshape(p_true_class, (nrows,ncols))
    
    cmap=cm.RdYlGn
    tensorboard_label = "3 probs assigned to true class"

  elif matrix_type=='confidence_RIGHTS':                                                                     # probability of the prediction, where the prectiction was correct only
     
    if DEBUG>2:
      print ( f"TRAINLENEJ:     INFO:        p_true_class.tolist() = {p_true_class.tolist()}" )
      print ( f"TRAINLENEJ:     INFO:        preds.tolist()        = {preds.tolist()}"        )
      print ( f"TRAINLENEJ:     INFO:        image_labels.tolist() = {image_labels.tolist()}"        )     
     
    only_corrects  = np.array ( [ p_true_class.tolist()[i] if preds.tolist()[i]==image_labels.tolist()[i] else 0 for i in range(len(p_true_class.tolist()) ) ] )
    only_corrects  = only_corrects[np.newaxis,:] 
    only_corrects  = only_corrects.T
    reshaped_to_2D = np.reshape(only_corrects, (nrows,ncols))
    
    cmap=cm.Greens
    tensorboard_label = "4 probs for tiles where prediction is correct"

  elif matrix_type=='confidence':                                                                          # probability of the prediction, (whether it was correct or incorrect)
      
    p_highest        = p_highest[np.newaxis,:] 
    p_highest        = p_highest.T
    reshaped_to_2D   = np.reshape(p_highest, (nrows,ncols))
    
    cmap=cm.Greens
    tensorboard_label = "5 highest probs whether correct or incorrect"

  elif matrix_type=='margin_1st_2nd':                                                                      # probability of the prediction, (whether it was correct or incorrect)
    
    delta_1st_2nd    = p_highest - p_2nd_highest
    delta_1st_2nd    = delta_1st_2nd[np.newaxis,:] 
    delta_1st_2nd    = delta_1st_2nd.T
    reshaped_to_2D   = np.reshape(delta_1st_2nd, (nrows,ncols))
    
    cmap=cm.Greens
    tensorboard_label = "6 prob margins 1st:2nd"

  elif matrix_type=='p_std_dev':                                                                            # standard deviation of probailities of each class

    if DEBUG>0:
      print ( f"TRAINLENEJ:     INFO:        plot_matrix():  (type: {MIKADO}{matrix_type}{RESET}) grid_p_full_softmax_matrix.shape  = {grid_p_full_softmax_matrix.shape}" ) 
      
    sd             = np.std( grid_p_full_softmax_matrix, axis=1 )    
    sd             = sd[np.newaxis,:]
    sd             = sd.T
    reshaped_to_2D = np.reshape(sd, (nrows,ncols))
    
    if DEBUG>9:
      print ( f"TRAINLENEJ:     INFO:        plot_matrix():  (type: {MIKADO}{matrix_type}{RESET}) reshaped_to_2D.shape  = {reshaped_to_2D.shape}" ) 
      print ( f"TRAINLENEJ:     INFO:        plot_matrix():  (type: {MIKADO}{matrix_type}{RESET}) reshaped_to_2D values = \n{reshaped_to_2D.T}" ) 
          
    cmap=cm.Greens
    tensorboard_label = "7 sd of class probs"

  else:
    print( f"\n{ORANGE}TRAINLENEJ:     WARNING: no such matrix_type {RESET}{MIKADO}{matrix_type}{RESET}{ORANGE}. Skipping.{RESET}", flush=True)

  #gwr = ListedColormap(['r', 'w', 'g'])  
  #plt.matshow( reshaped_to_2D, fignum=1, interpolation='spline16', cmap=cm.binary, vmin=0, vmax=1 )
  #plt.matshow( reshaped_to_2D, fignum=1, cmap=gwr, vmin=0, vmax=1 )
  
  plt.matshow( reshaped_to_2D, fignum=1, interpolation=probs_matrix_interpolation, cmap=cmap, vmin=0, vmax=1 )
  plt.show
#  writer.add_figure( f"{tensorboard_label} ({probs_matrix_interpolation})", fig, epoch)
  writer.add_figure( f"{tensorboard_label}", fig, epoch)
  plt.close(fig)
    
  return
      

# ------------------------------------------------------------------------------
def plot_classes_preds(args, model, tile_size, batch_images, image_labels, preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours):
    '''
    Generates matplotlib Figure using a trained network, along with a batch of images and labels, that shows the network's top prediction along with its probability, alongside the actual label, colouring this
    information based on whether the prediction was correct or not. Uses the "images_to_probs" function. 
    
    '''
    

    ##################################################################################################################################
    #
    #  (1) Training mode: the simple case because we are just displaying a set of random tiles which have been passed through training
    #
    if args.just_test=='False':
  
      number_to_plot = len(image_labels)    
      figure_width   = 15
      figure_height  = int(number_to_plot * .4)
          
      # plot the images in the batch, along with predicted and true labels
      fig = plt.figure( figsize=( figure_width, figure_height ) )                                         # overall size ( width, height ) in inches
  
      if DEBUG>99:
        print ( "\nTRAINLENEJ:     INFO:      plot_classes_preds():             number_to_plot                          = {:}".format( number_to_plot    ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             figure width  (inches)                  = {:}".format( figure_width    ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             figure height (inches)                  = {:}".format( figure_height   ) )
  
      #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
      #plt.grid( False )
  
      ncols = int((   number_to_plot**.5 )           // 1 )
      nrows = int(( ( number_to_plot // ncols ) + 1 ) // 1 )
  
      if DEBUG>99:
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             number_to_plot                          = {:}".format( number_to_plot  ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             nrows                                   = {:}".format( nrows           ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             ncols                                   = {:}".format( ncols           ) ) 
  
      for idx in np.arange( number_to_plot-1 ):
  
          ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[])            # nrows, ncols, "index starts at 1 in the upper left corner and increases to the right", List of x-axis tick locations, List of y-axis tick locations
          ax.set_frame_on( False )
  
          img     = batch_images[idx]
          npimg_t = np.transpose(img, (1, 2, 0))
          plt.imshow(npimg_t)
  
          if DEBUG>99:
            print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:}".format( idx ) )
            print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:} probs[idx] = {:4.2e}, classes[preds[idx]] = {:<20s}, classes[labels[idx]] = {:<20s}".format( idx, probs[idx], classes[preds[idx]], classes[labels[idx]]  ) )
  
          ax.set_title( "p_1={:<.4f}\n p_2nd_highest={:<.4f}\n pred: {:}\ntruth: {:}".format( p_highest[idx], p_2nd_highest[idx], class_names[preds[idx]], class_names[image_labels[idx]] ),
                      loc        = 'center',
                      pad        = None,
                      size       = 8,
                      color      = ( "green" if preds[idx]==image_labels[idx] else "red") )
  
      fig.tight_layout( rect=[0, 0.03, 1, 0.95] )

      
      return fig


    ##################################################################################################################################
    #
    # (2) Test mode is much more complex, because we need to present an annotated 2D contiguous grid of tiles
    #
    if args.just_test=='True':
 
      non_specimen_tiles=0
      correct_predictions=0  
  
      number_to_plot = image_labels.shape[0]  
      ncols = int(number_to_plot**.5)
      nrows = ncols
      figure_width   = args.figure_width
      figure_height  = args.figure_height
      
      break_1=6    # rows
      break_2=18   # rows
      break_3=25   # rows
      break_4=40   # rows    
  
  
      # (2a) set up all axes
         
      if DEBUG>0:
        print ( f"TRAINLENEJ:     INFO:        plot_classes_preds():  {ORANGE if args.just_test=='True' else MIKADO} about to set up {MIKADO}{figure_width}x{figure_height} inch{RESET} figure and axes for {MIKADO}{nrows}x{ncols}={number_to_plot}{RESET} subplots. (Note: This takes a long time for larger values of nrows/ncols)", end="", flush=True )
            
      fig, axes = plt.subplots( nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=True, figsize=( figure_width, figure_height ) )        # This takes a long time to execute for larger values of nrows and ncols
    
      if DEBUG>0:
        print ( f"  ... done", flush=True )
      

      # (2b) add the legend 
      
      l=[]
      for n in range (0, len(class_colours)):
        l.append(mpatches.Patch(color=class_colours[n], linewidth=0))
        fig.legend(l, args.long_class_names, loc='upper right', fontsize=14, facecolor='lightgrey')      
      #fig.tight_layout( pad=0 )     
      
      # (2c) remove axes from the region we want to reserve for the bar chart 
    
      gs = axes[1, -1].get_gridspec()
      if nrows<=break_1:                                            
          axes[nrows-1, ncols-1].remove()                                                                                 # delete this cell (the one in the bottom right hand corner)
      elif break_1<=nrows<break_2:
        for i, j in product(range(nrows-2, nrows), range(ncols-2, ncols )):                                               # delete all these cells (cartesian product)
          axes[i,j].remove()
      elif break_2<=nrows<break_3:
        for i, j in product(range(nrows-3, nrows), range(ncols-3, ncols )):                                               # delete all these cells (cartesian product)
          axes[i,j].remove()
      elif break_3<=nrows<break_4:
        for i, j in product(range(nrows-4, nrows), range(ncols-4, ncols )):                                               # delete all these cells (cartesian product)
          axes[i,j].remove()
      elif nrows>=break_4:
        for i, j in product(range(nrows-5, nrows), range(ncols-5, ncols )):                                               # delete all these cells (cartesian product)
          axes[i,j].remove()
    
      # ax0 will be used for the bar chart
      if nrows<=break_1:      
           ax0 = fig.add_subplot( gs[nrows-1:, ncols-1:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart
      elif break_1<=nrows<break_2:
           ax0 = fig.add_subplot( gs[nrows-2:, ncols-2:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart
      elif break_2<=nrows<break_3:
           ax0 = fig.add_subplot( gs[nrows-3:, ncols-3:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart
      elif break_3<=nrows<break_4:
           ax0 = fig.add_subplot( gs[nrows-4:, ncols-4:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart           
      elif nrows>=break_4:
           ax0 = fig.add_subplot( gs[nrows-5:, ncols-5:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart

      ax0.grid( color='silver', linestyle='--', linewidth=1, axis='y', alpha=0 )
      ax0.set_xlabel("sum of tile probs Vs. class", size=11)
      ax0.yaxis.set_ticks_position("right")
      ax0.tick_params(labelsize=10) 
      ax0.set_ylim(0,number_to_plot) 
      ax0.set_facecolor("xkcd:mint" if image_labels[0]==np.argmax(np.sum(p_full_softmax_matrix,axis=0)) else "xkcd:faded pink" )      
      ax0.bar( x=['1', '2', '3', '4', '5', '6', '7'], height=np.sum(p_full_softmax_matrix,axis=0),  width=int(number_to_plot/len(image_labels)), color=class_colours )
      # [c[0] for c in class_names]


      # (2d) process each tile; which entails allocating the tile to the correct spot in the subplot grid together plus annotated class information encoded as border color and centred 'x' of prediction was incorrect
      
      flag=0
      
      for r in range(nrows):
      
        for c in range(ncols):

          idx = (r*nrows)+c
          
          if args.just_test=='True':
            
            if DEBUG>0:
              if flag==0:
                  print ( f"TRAINLENEJ:     INFO:        plot_classes_preds():  {ORANGE if args.just_test=='True' else MIKADO} now processing sub-plot {RESET}", end="", flush=True )
                  flag=1
              if ( idx==0 ):
                  print ( f"..1", end="", flush=True )                  
              elif ( idx%10==0 ):
                  print ( f"{DIM_WHITE}..{idx}", end="", flush=True )
              elif ( idx==(nrows*ncols)-1 ):
                  print ( f"{RESET}..{idx+1}", end="", flush=True )              
            
            if nrows<break_1:
              if ( r==nrows-1) & (c==ncols-1):
                pass
              else:
                axes[r,c]=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
            elif break_1<=nrows<break_2:
              if ( r>=nrows-2) & (c>=ncols-2):
                pass
              else:
                axes[r,c]=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
            elif break_2<=nrows<break_3:
              if ( r>=nrows-3) & (c>=ncols-3):
                pass
              else:
                axes[r,c]=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
            elif break_3<=nrows<break_4:
              if ( r>=nrows-4) & (c>=ncols-4):
                pass
              else:
                axes[r,c]=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
            else:
              if ( r>=nrows-5) & (c>=ncols-5):
                pass
              else:
                axes[r,c]=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
           
            threshold_0=36
            threshold_1=100
            threshold_2=400
            threshold_3=900
                   
            if idx==0:
              t1=f"{int(number_to_plot**.5)//1}x{int(number_to_plot**.5)//1}"
              axes[r,c].text( -120,  20, t1, size=12, ha="left", color="goldenrod", style="normal" )
              t2=f"Cancer type: {args.cancer_type_long}"
              t3=f"Truth label for this WSI:"
              t4=f"{args.long_class_names[image_labels[idx]]}"
              t5=f"NN prediction from patch:"
              t6=f"{args.long_class_names[np.argmax(np.sum(p_full_softmax_matrix,axis=0))]}"
              if len(image_labels)>=threshold_3:
                axes[r,c].text( -550, -400, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
                axes[r,c].text( -550, -300, t3, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  550, -300, t4, size=14, ha="left",   color="black", style="italic" )
                axes[r,c].text( -550, -200, t5, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  550, -200, t6, size=14, ha="left",   color="black", style="italic" )
              elif threshold_3>len(image_labels)>=threshold_2: #OK
                axes[r,c].text( -380, -300, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
                axes[r,c].text( -380, -200, t3, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  400, -200, t4, size=14, ha="left",   color="black", style="italic" )
                axes[r,c].text( -380, -120, t5, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  400, -120, t6, size=14, ha="left",   color="black", style="italic" )
              elif threshold_2>len(image_labels)>=threshold_1: #OK
                axes[r,c].text( -200, -180, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
                axes[r,c].text( -200, -120, t3, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  375, -120, t4, size=14, ha="left",   color="black", style="italic" )
                axes[r,c].text( -200, -80, t5, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  375, -80, t6, size=14, ha="left",   color="black", style="italic" )
              elif threshold_1>len(image_labels)>=threshold_0: #OK
                axes[r,c].text( -100, -75, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
                axes[r,c].text( -100, -50, t3, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  230, -50, t4, size=14, ha="left",   color="black", style="italic" )
                axes[r,c].text( -100, -30, t5, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  230, -30, t6, size=14, ha="left",   color="black", style="italic" )               
              else: # (< threshold0) #OK
                axes[r,c].text( -60,  -60, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
                axes[r,c].text( -60,  -35, t3, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  95, -35, t4, size=14, ha="left",   color="black", style="italic" )
                axes[r,c].text( -60,  -20, t5, size=14, ha="left",   color="black", style="normal" )
                axes[r,c].text(  95, -20, t6, size=14, ha="left",   color="black", style="italic" )                           
              
              if DEBUG>99:
                predicted_class=np.argmax(np.sum(p_full_softmax_matrix, axis=0))
                print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             predicted_class                                   = {predicted_class}" )
            
            #  check 'badness' status. such tiles were never looked at during training, so we don't want to mark them up             
            tile_rgb_npy=batch_images[idx]
            tile_rgb_npy_T = np.transpose(tile_rgb_npy, (1, 2, 0))         
            tile_255 = tile_rgb_npy_T * 255
            tile_uint8 = np.uint8( tile_255 )
            tile_norm_PIL = Image.fromarray( tile_uint8 )
            tile = tile_norm_PIL.convert("RGB")
      
            IsBadTile = check_badness( args, tile )
            
            if IsBadTile:
              non_specimen_tiles+=1
              pass
              
            else:
              if len(image_labels)>=threshold_3:
                font_size=8
                left_offset=int(0.6*tile_size)
                top_offset =int(0.9*tile_size)            
                p=int(10*(p_highest[idx]-.01)//1)
                p_txt=p
              elif len(image_labels)>=threshold_2:
                font_size=10
                left_offset=int(0.45*tile_size)
                top_offset =int(0.90*tile_size)            
                p=np.around(p_highest[idx]-.01,decimals=1)
                p_txt=p
              elif len(image_labels)>=threshold_1:
                font_size=14
                left_offset=int(0.6*tile_size)
                top_offset =int(0.92*tile_size)            
                p=np.around(p_highest[idx]-.01,decimals=1)
                p_txt=p
              else: 
                p=np.around(p_highest[idx],2)
                p_txt = f"p={p}"   
                font_size=16
                left_offset=4
                top_offset =int(0.95*tile_size)
                
              if p_highest[idx]>=0.75:
                col="orange"
              elif p_highest[idx]>0.50:
                col="orange"
              else:
                col="orange"
      
              if len(image_labels)>=threshold_3:
                col="red"
                                                 
              axes[r,c].text( left_offset, top_offset, p_txt, size=font_size, color=col, style="normal", weight="bold" )
      
              if (preds[idx]==image_labels[idx]):
                correct_predictions+=1
              else:
                col=class_colours[preds[idx]]
                if len(image_labels)>=threshold_3:
                  font_size=13
                  left_offset=int(0.3*tile_size)
                  top_offset =int(0.6*tile_size)  
                elif len(image_labels)>=threshold_2:
                  left_offset=int(0.4*tile_size)
                  top_offset =int(0.6*tile_size)  
                  font_size=16
                elif len(image_labels)>=threshold_1:
                  left_offset=int(0.4*tile_size)
                  top_offset =int(0.55*tile_size)  
                  font_size=25
                else:
                  left_offset=int(0.45*tile_size)
                  top_offset =int(0.52  *tile_size)                
                  font_size=50
                  
                if p>0.7:
                  text="x"
                elif p>0.5:
                  text="x"
                else:
                  text="x"
                  
                axes[r,c].text( left_offset, top_offset, text, size=font_size, color=col, style="normal", weight="bold" )
                

          total_tiles     =  len(image_labels)
          specimen_tiles  =  total_tiles - non_specimen_tiles
          if specimen_tiles>0:
            pct_correct     =   (correct_predictions/specimen_tiles)
          else:
            pct_correct     =   0
    
          if idx==total_tiles-2:
            stats=f"Statistics: tile count: {total_tiles}; background tiles: {non_specimen_tiles}; specimen tiles: {specimen_tiles}; correctly predicted: {correct_predictions}/{specimen_tiles} ({pct_correct*100}%)"
            plt.figtext( 0.15, 0.055, stats, size=14, color="black", style="normal" )
            
          img=batch_images[idx]
          npimg_t = np.transpose(img, (1, 2, 0))
          plt.imshow(npimg_t, aspect='auto')
          plt.subplots_adjust(wspace=0, hspace=0)    
  
          if not IsBadTile:
            if preds[idx]==image_labels[idx]:
              axes[r,c].patch.set_edgecolor(class_colours[preds[idx]])
              if len(image_labels)>threshold_3:
                axes[r,c].patch.set_linewidth('1')
              if len(image_labels)>threshold_2:
                axes[r,c].patch.set_linewidth('2')
              elif len(image_labels)>threshold_1:
                axes[r,c].patch.set_linewidth('3')
              else:
                axes[r,c].patch.set_linewidth('4')
            else:
              axes[r,c].patch.set_edgecolor('magenta')
              axes[r,c].patch.set_linestyle(':')
              if len(image_labels)>threshold_3:
                axes[r,c].patch.set_linewidth('1')              
              if len(image_labels)>threshold_2:
                axes[r,c].patch.set_linewidth('2')
              elif len(image_labels)>threshold_1:
                axes[r,c].patch.set_linewidth('3')
              else:
                axes[r,c].patch.set_linewidth('6')

      print ( f"{RESET}")
          
      
      if DEBUG>99:
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:}".format( idx ) )
      if DEBUG>99:
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:} p_highest[idx] = {:4.2f}, class_names[preds[idx]] = {:<20s}, class_names[image_labels[idx]] = {:<20s}".format( idx, p_highest[idx], class_names[preds[idx]], class_names[image_labels[idx]]  ) )
  
      if DEBUG>99:
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             idx                                     = {idx}"                            )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             p_highest[idx]                          = {p_highest[idx]:4.2f}"            )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             p_2nd_highest[idx]]                     = {p_2nd_highest[idx]:4.2f}"        )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             preds[idx]                              = {preds[idx]}"                     )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names                             = {class_names}"                    )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names                             = {class_names[1]}"                 )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names                             = {class_names[2]}"                 )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names[preds[idx]]                 = {class_names[preds[idx]]}"        )
        print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names[image_labels[idx]]          = {class_names[image_labels[idx]]}" )
      
      return fig

# ------------------------------------------------------------------------------

def l1_penalty(model, l1_coef):
    """Compute L1 penalty. For implementation details, see:

    See: https://discuss.pytorch.org/t/simple-l2-regularization/139
    """
    reg_loss = 0
    for param in model.lnetimg.parameters_('y2_hat'):
        reg_loss += torch.norm(param, 1)
    return l1_coef * reg_loss

# ------------------------------------------------------------------------------
# NOT USED
def save_samples(log_dir, model, test_loader, cfg, epoch):
    """Save samples from test set.
    """
    
    with torch.no_grad():
        n  = len(test_loader.sampler.indices)
        x1_batch = torch.Tensor(n, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)
        x2_batch = torch.Tensor(n, cfg.N_GENES)
        labels   = []

        for i in range(n):

            j = test_loader.sampler.indices[i]

            x1, x2 = test_loader.dataset[j]
            lab    = test_loader.dataset.labels[j]
            x1_batch[i] = x1
            x2_batch[i] = x2
            labels.append(lab)

        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)

        cfg.save_samples(log_dir, model, epoch, x1_batch, x2_batch, labels)

# ------------------------------------------------------------------------------

def save_model(log_dir, model):
    """Save PyTorch model state dictionary
    """
    
    fpath = '%s/model.pt' % log_dir
    if DEBUG>2:
      print( f"TRAINLENEJ:     INFO:   save_model(): new lowest loss on this epoch... saving model state dictionary to {MAGENTA}{fpath}{RESET}" )      
    model_state = model.state_dict()
    torch.save(model_state, fpath)

# ------------------------------------------------------------------------------
    
def delete_selected( root, extension ):

  walker = os.walk( root, topdown=True )

  for root, dirs, files in walker:

    for f in files:
      fqf = root + '/' + f
      if DEBUG>99:
        print( f"TRAINLENEJ:     INFO:   examining file:   '\r\033[43C\033[36;1m{fqf}\033[m' \r\033[180C with extension '\033[36;1m{extension}\033[m'" )
      if ( f.endswith( extension ) ): 
        try:
          if DEBUG>99:
            print( f"TRAINLENEJ:     INFO:   will delete file  '\r\033[43C{MIKADO}{fqf}{RESET}'" )
          os.remove( fqf )
        except:
          pass

# ------------------------------------------------------------------------------
def excludes( number_to_plot, plot_box_side_length ):

  if DEBUG>99:
    print ( f"number to plot =       {    number_to_plot    }" )
    print ( f"plot_box_side_length = { plot_box_side_length }" )   
  
  patch_side_length=int(number_to_plot**0.5)
  
  concat_excludes=[]
  for row in range( patch_side_length-plot_box_side_length+1 , patch_side_length+1 ):
    
    start_cell = row*patch_side_length - plot_box_side_length
    end_cell   = row*patch_side_length-1
  
    exclude_cells=[*range(start_cell, end_cell+1)]
  
    concat_excludes=concat_excludes+exclude_cells
  
  if DEBUG>99:
    print ( concat_excludes )

  return concat_excludes

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--skip_tiling',             type=str,   default='False')                                # USED BY main() to enable user to skip tile generation
    p.add_argument('--skip_generation',                type=str,   default='False')                                # USED BY main() to enable user to skip torch database generation
    p.add_argument('--log_dir',                        type=str,   default='data/dlbcl_image/logs')                # used to store logs and to periodically save the model
    p.add_argument('--base_dir',                       type=str,   default='/home/peter/git/pipeline')             # NOT CURRENTLY USED
    p.add_argument('--data_dir',                       type=str,   default='/home/peter/git/pipeline/dataset')     # USED BY generate()
    p.add_argument('--save_model_name',                type=str,   default='model.pt')                             # USED BY main()
    p.add_argument('--save_model_every',               type=int,   default=10)                                     # USED BY main()    
    p.add_argument('--rna_file_name',                  type=str,   default='rna.npy')                              # USED BY generate()
    p.add_argument('--rna_file_suffix',                type=str,   default='*FPKM-UQ.txt' )                        # USED BY generate()
    p.add_argument('--rna_file_reduced_suffix',        type=str,   default='_reduced')                             # USED BY generate()
    p.add_argument('--use_unfiltered_data',            type=str,   default='True' )                                # USED BY generate()     
    p.add_argument('--class_numpy_file_name',          type=str,   default='class.npy')                            # USED BY generate()
    p.add_argument('--wall_time',                                                     type=int,    default=24)
    p.add_argument('--seed',                                                          type=int,    default=0)
    p.add_argument('--nn_mode',                                                       type=str,    default='pre_compress')
    p.add_argument('--use_same_seed',                                                 type=str,    default='False')
    p.add_argument('--nn_type_img',                                       nargs="+",  type=str,    default='VGG11')
    p.add_argument('--nn_type_rna',                                       nargs="+",  type=str,    default='DENSE')
    p.add_argument('--hidden_layer_encoder_topology', '--nargs-int-type', nargs='*',  type=int,                      )                             # USED BY AEDEEPDENSE(), TTVAE()
    p.add_argument('--encoder_activation',                                nargs="+",  type=str,    default='sigmoid')                              # USED BY AEDENSE(), AEDENSEPOSITIVE()
    p.add_argument('--nn_dense_dropout_1',                                nargs="+",  type=float,  default=0.0)                                    # USED BY DENSE()    
    p.add_argument('--nn_dense_dropout_2',                                nargs="+",  type=float,  default=0.0)                                    # USED BY DENSE()
    p.add_argument('--dataset',                                                       type=str,    default='STAD')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--input_mode',                                                    type=str,    default='NONE')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--n_samples',                                         nargs="+",  type=int,    default=101)                                    # USED BY generate()      
    p.add_argument('--n_tiles',                                           nargs="+",  type=int,    default=100)                                    # USED BY generate() and all ...tiler() functions 
    p.add_argument('--supergrid_size',                                                type=int,    default=1)                                      # USED BY main()
    p.add_argument('--patch_points_to_sample',                                        type=int,    default=1000)                                   # USED BY tiler()    
    p.add_argument('--tile_size',                                          nargs="+", type=int,    default=128)                                    # USED BY many
    p.add_argument('--gene_data_norm',                                     nargs="+", type=str,    default='NONE')                                 # USED BY generate()
    p.add_argument('--gene_data_transform',                                nargs="+", type=str,    default='NONE' )
    p.add_argument('--n_genes',                                                       type=int,    default=506)                                   # USED BY main() and generate()
    p.add_argument('--remove_unexpressed_genes',                                      type=str,    default='True' )                               # USED generate()
    p.add_argument('--remove_low_expression_genes',                                   type=str,   default='True' )                               # USED generate()
    p.add_argument('--low_expression_threshold',                                      type=float, default=0      )                               # USED generate()
    p.add_argument('--batch_size',                                         nargs="+", type=int,   default=256)                                   # USED BY tiler() 
    p.add_argument('--learning_rate',                                      nargs="+", type=float, default=.00082)                                # USED BY main()                               
    p.add_argument('--n_epochs',                                                      type=int,   default=10)
    p.add_argument('--pct_test',                                                      type=float, default=0.2)
    p.add_argument('--lr',                                                nargs="+",  type=float, default=0.0001)
    p.add_argument('--latent_dim',                                                    type=int,   default=7)
    p.add_argument('--l1_coef',                                                       type=float, default=0.1)
    p.add_argument('--em_iters',                                                      type=int,   default=1)
    p.add_argument('--clip',                                                          type=float, default=1)
    p.add_argument('--max_consecutive_losses',                                        type=int,   default=7771)
    p.add_argument('--optimizer',                                         nargs="+",  type=str,   default='ADAM')
    p.add_argument('--label_swap_perunit',                                            type=float, default=0.0)                                    
    p.add_argument('--make_grey_perunit',                                             type=float, default=0.0) 
    p.add_argument('--figure_width',                                                  type=float, default=16)                                  
    p.add_argument('--figure_height',                                                 type=float, default=16)
    p.add_argument('--annotated_tiles',                                               type=str,   default='True')
    p.add_argument('--scattergram',                                                   type=str,   default='True')
    p.add_argument('--probs_matrix',                                                  type=str,   default='True')
    p.add_argument('--probs_matrix_interpolation',                                    type=str,   default='none')
    p.add_argument('--show_patch_images',                                             type=str,   default='True')
    p.add_argument('--regenerate',                                                    type=str,   default='True')
    p.add_argument('--just_profile',                                                  type=str,   default='False')                        # USED BY tiler()    
    p.add_argument('--just_test',                                                     type=str,   default='False')                        # USED BY tiler()    
    p.add_argument('--rand_tiles',                                                    type=str,   default='True')                         # USED BY tiler()      
    p.add_argument('--points_to_sample',                                              type=int,   default=100)                            # USED BY tiler()
    p.add_argument('--min_uniques',                                                   type=int,   default=0)                              # USED BY tiler()
    p.add_argument('--min_tile_sd',                                                   type=float, default=3)                              # USED BY tiler()
    p.add_argument('--greyness',                                                      type=int,   default=0)                              # USED BY tiler()
    p.add_argument('--stain_norm',                                        nargs="+",  type=str,   default='NONE')                         # USED BY tiler()
    p.add_argument('--stain_norm_target',                                             type=str,   default='NONE')                         # USED BY tiler_set_target()
    p.add_argument('--use_tiler',                                                     type=str,   default='external'  )                   # USED BY main()
    p.add_argument('--cancer_type',                                                   type=str,   default='NONE'      )                   # USED BY main()
    p.add_argument('--cancer_type_long',                                              type=str,   default='NONE'      )                   # USED BY main()
    p.add_argument('--class_names',                                       nargs="+"                                   )                    # USED BY main()
    p.add_argument('--long_class_names',                                  nargs="+"                                   )                    # USED BY main()
    p.add_argument('--class_colours',                                     nargs="*"                                   )    
    p.add_argument('--target_tile_coords',                                nargs=2,    type=int,    default=[2000,2000]       )               # USED BY tiler_set_target()

    p.add_argument('--a_d_use_cupy',                                                  type=str,   default='True'     )                    # USED BY main()
    p.add_argument('--cov_threshold',                                                 type=float, default=8.0        )                    # USED BY main()   
    p.add_argument('--cov_uq_threshold',                                              type=float, default=0.0        )                    # USED BY main() 
    p.add_argument('--cutoff_percentile',                                             type=float, default=0.05       )                    # USED BY main() 
    
    p.add_argument('--show_rows',                                                     type=int,   default=500)                            # USED BY main()
    p.add_argument('--show_cols',                                                     type=int,   default=100)                            # USED BY main() 
    
    p.add_argument('-ddp', '--ddp',                                                   type=str,   default='False'                                                  )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    p.add_argument('-n', '--nodes',                                                   type=int,   default=1,  metavar='N'                                          )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    p.add_argument('-g', '--gpus',                                                    type=int,   default=1,  help='number of gpus per node'                       )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    p.add_argument('-nr', '--nr',                                                     type=int,   default=0,  help='ranking within node'                           )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    
    p.add_argument('--hidden_layer_neurons',                              nargs="+",  type=int,    default=2000)     
    p.add_argument('--gene_embed_dim',                                    nargs="+",  type=int,    default=1000)    
    
    p.add_argument('--use_autoencoder_output',                                        type=str,   default='True')                         # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values)
    
        
    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 12
    args.pin_memory = torch.cuda.is_available()

    main(args)
