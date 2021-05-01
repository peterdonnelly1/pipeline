
"""=============================================================================
Code to support Dimensionality Reduction Mode
============================================================================="""

import argparse
import time
import numpy as np
import os

import torch
from   torch                       import optim
import torch.utils.data
from   torch.nn                    import functional as F
from   torch.nn                    import MSELoss, BCELoss
from   torch.nn.utils              import clip_grad_norm_
from   torch.nn.parallel           import DistributedDataParallel as DDP
from   torch.utils.tensorboard     import SummaryWriter
import torchvision
from   torchvision                 import datasets, transforms
import torch.multiprocessing   as mp
import torch.distributed       as dist

import matplotlib
import matplotlib.pyplot   as plt
import matplotlib.lines    as mlines
import matplotlib.patches  as mpatches
import matplotlib.gridspec as gridspec
from   matplotlib.colors           import ListedColormap
from   matplotlib                  import cm
from   matplotlib.ticker           import (AutoMinorLocator, MultipleLocator)

from tiler_scheduler               import *
from tiler_threader                import *
from tiler_set_target              import *
from tiler                         import *

from   schedulers                  import *
from   data                        import loader
from   data.pre_compress.generate  import generate

from   itertools                   import product, permutations
from   PIL                         import Image

import cuda
from   models                      import PRECOMPRESS
from   models.ttvae                import vae_loss
import pprint

torch.set_printoptions(edgeitems=6)
torch.set_printoptions(linewidth=250)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
  
  
np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=200)

torch.backends.cudnn.enabled     = True                                                                     # for CUDA memory optimizations
# ------------------------------------------------------------------------------

LOG_EVERY        = 2
SAVE_MODEL_EVERY = 100

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
ASPARAGUS='\033[38;2;135;169;107m'
CHARTREUSE='\033[38;2;223;255;0m'
COQUELICOT='\033[38;2;255;56;0m'
COTTON_CANDY='\033[38;2;255;188;217m'
HOT_PINK='\033[38;2;255;105;180m'
CAMEL='\033[38;2;193;154;107m'
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
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
CARRIBEAN_GREEN='\033[38;2;0;204;153m'
PALE_GREEN='\033[32m'
GREY_BACKGROUND='\033[48;2;60;60;60m'


BRIGHT='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'


DEBUG=1



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main( args ):
  
  print ( f"MAIN:           INFO:     mode = {CYAN}{args.nn_mode}{RESET}" )
  
#  if  not args.input_mode=='rna':
#    print( f"{RED}MAIN:           FATAL:  currently only rna input is supported by pre_compress {RED}' (you have INPUT_MODE='{MIKADO}{args.input_mode}{RESET}{RED}') ... halting now{RESET}" )
#    sys.exit(0)

  print ( f"MAIN:           INFO:     torch       version =    {MIKADO}{torch.__version__}{RESET}" )
  print ( f"MAIN:           INFO:     torchvision version =    {MIKADO}{torchvision.__version__}{RESET}"  )
  print ( f"MAIN:           INFO:     matplotlib version  =    {MIKADO}{matplotlib.__version__}{RESET}"   ) 



# THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 

# COMMENTED OUT BECAUSE WE NOW WANT TO USE THE use_autoencoder_output FLAG TO INSTRUCT THE LOADER TO RANDOM SHUFFLE SELECTED TILES, WHICH IS SOMETHING WE DON'T DO FOR THE EXISTING USE OF JUST_TEST (WHERE WE CONSTRUCT A PATCH FROM A SEQUENTIAL SELECTION OF TILES)
# AT SOME POINT, CHANGE THE NAME OF THIS FLAG TO "USING_AUTOECODER"

  # ~ if  args.use_autoencoder_output=='True':
    # ~ print( f"{ORANGE}PRE_COMPRESS:     WARNING:  main():  Flag {CYAN}'USE_AUTOENCODER_OUTPUT'{RESET}{ORANGE} isn't compatible with {CYAN}'pre_compress'{RESET}{ORANGE} mode ... it will be changed to {CYAN}'False'{RESET}" )
    # ~ args.use_autoencoder_output=False

  if args.use_autoencoder_output=='True':
    if args.just_test!='True':
      print( f"MAIN:           INFO: {RED}{BRIGHT}AUTOENCODER WORKING HAS BEEN ENABLED FOR THIS TRAINING RUN{RESET} (flag {CYAN}'USE_AUTOENCODER_OUTPUT'{RESET}={CYAN}{args.use_autoencoder_output}{RESET}). LOWEST LOSS MODEL WILL BE SAVED AS {CYAN}logs/lowest_loss_ae_model.pt{RESET}" )
    else:
      print( f"MAIN:           INFO: {RED}{BRIGHT}AUTOENCODER WORKING HAS BEEN ENABLED FOR THIS TEST RUN{RESET}     (flag {CYAN}'USE_AUTOENCODER_OUTPUT'{RESET}). EMBEDDINGS FROM {CYAN}ae_output_features.pt{RESET} WILL BE USED AS INPUT RATHER THAN IMAGE TILES OR RNA_SEQ VECTORS{RESET}" )

    if args.ae_add_noise=='True':
      print( f"MAIN:           INFO: {RED}{BRIGHT}NOISE ADDITION HAS BEEN ENABLED FOR THIS TRAINING RUN{RESET}      (flag {CYAN}'AE_USE_NOISE'{RESET}={CYAN}{args.ae_add_noise}{RESET})" )

  if args.peer_noise_perunit>0.0:
    print( f"P_C_DATASET:    INFO: CAUTION! {RED}{BRIGHT}ADD PEER NOISE{RESET} IS ACTIVE!; DURING AUTOENCODING, IN TRAINING MODE (ONLY), EACH TILE WILL RECEIVE {MIKADO}{args.peer_noise_perunit * 100:3.0f}%{RESET} NOISE FROM ANOTHER RANDONLY SELECTED TILE{RESET}" )  
        

# THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 THIS DIFFERS FROM TRAINLENT5 




  if ( args.ddp=='True' ) & ( args.just_test=='True' ):
    print( f"{RED}PRE_COMPRESS:     WARNING: 'JUST_TEST' flag and 'DDP' flag are both set. However, in test mode, DDP must be disabled ('DDP=False') ... DDP will now be disabled {RESET}" ) 
    args.ddp="False"
  
  if args.ddp=='True':

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'

    if DEBUG>0:
      print ( f"MAIN:           INFO:      main(): number of GPUs available   = {MIKADO}{torch.cuda.device_count()}{RESET}" )      
      print ( f"MAIN:           INFO:      main(): args.gpus                  = {MIKADO}{args.gpus}{RESET}" )
      print ( f"MAIN:           INFO:      main(): args.nprocs                = {MIKADO}{args.gpus}{RESET}" )

    args.gpus = torch.cuda.device_count()
    mp.spawn( run_job,                                                                                     # One copy of run_job for each of two processors and the two GPUs
              nprocs = args.gpus,                                                                          # number of processes
              args   = (args,)  )                                                                          # total number of GPUs
  
  else:
    run_job( 0, args )

  print ( "\033[12B", end='', flush=True )  



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_job(gpu, args ):

  
  multimode_case_count = unimode_case_count = not_a_multimode_case_count = not_a_multimode_case____image_count = not_a_multimode_case____image_test_count = 0
  
  if args.ddp=='True':
    if gpu>0:
      MIKADO='\033[38;2;0;168;107m'
      GREEN ='\033[38;2;164;198;57m'
      RED   ='\033[38;2;201;0;22m'
    else:
      MIKADO='\033[38;2;233;115;255m' 
      GREEN ='\033[38;2;141;182;0m'
      RED   ='\033[38;2;204;51;51m'
    
    world_size = args.gpus * args.nodes
    rank       = args.nr * args.gpus + gpu
    torch.cuda.set_device(rank)
    
    if DEBUG>0:
      print ( f"{BRIGHT_GREEN}PRE_COMPRESS:   INFO:   DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! processor = {MIKADO}{gpu}{RESET}" )
      print ( f"{BRIGHT_GREEN}PRE_COMPRESS:   INFO:   DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! rank      = {MIKADO}{gpu}{RESET}" )
  
  else:
    MIKADO = '\033[38;2;255;196;12m'
    GREEN  = '\033[38;2;19;136;8m'
    RED    = '\033[38;2;255;0;0m'
      
      
  """Main program: train -> test once per epoch while saving samples as needed.
  """
  
  mode = 'TRAIN' if args.just_test!='True' else 'TEST'

  print( f"{GREY_BACKGROUND}PRE_COMPRESS:   INFO: common args: \
{WHITE}mode={CHARTREUSE}{mode}{WHITE}, \
input={CHARTREUSE}{args.input_mode}{WHITE}, \
network={CHARTREUSE}{args.nn_mode}{WHITE}, \
multimode={CHARTREUSE}{args.multimode}{WHITE}, \
cases={CHARTREUSE}{args.cases}{WHITE}, \
dataset={CHARTREUSE}{args.dataset}{WHITE}, \
n_samples={CHARTREUSE}{args.n_samples}{WHITE}, \
pct_test={CHARTREUSE}{args.pct_test}{WHITE}, \
epochs={CHARTREUSE}{args.n_epochs}{WHITE}, \
nn_optimizer={CHARTREUSE}{args.optimizer}{WHITE}, \
batch_size={CHARTREUSE}{args.batch_size}{WHITE}, \
learning_rate(s)={CHARTREUSE}{args.learning_rate}{WHITE}, \
max_consec_losses={CHARTREUSE}{args.max_consecutive_losses}{WHITE} \
{RESET}"\
, flush=True )

  
  if args.input_mode=='image':
    print( f"{GREY_BACKGROUND}PRE_COMPRESS:   INFO: image args: \
nn_type_img={CHARTREUSE}{args.nn_type_img}{WHITE}, \
n_tiles={CHARTREUSE}{args.n_tiles}{WHITE}, \
h_class={CHARTREUSE}{args.highest_class_number}{WHITE}, \
tile_size={CHARTREUSE}{args.tile_size}{WHITE}, \
rand_tiles={CHARTREUSE}{args.rand_tiles}{WHITE}, \
greyness<{CHARTREUSE}{args.greyness}{WHITE}, \
sd<{CHARTREUSE}{args.min_tile_sd}{WHITE}, \
min_uniques>{CHARTREUSE}{args.min_uniques}{WHITE}, \
latent_dim={CHARTREUSE}{args.latent_dim}{WHITE}, \
label_swap={CHARTREUSE}{args.label_swap_perunit}{WHITE}, \
make_grey={CHARTREUSE}{args.make_grey_perunit}{WHITE}, \
stain_norm={CHARTREUSE}{args.stain_norm}{WHITE}, \
annotated_tiles={CHARTREUSE}{args.annotated_tiles}{WHITE}, \
probs_matrix_interpolation={CHARTREUSE}{args.probs_matrix_interpolation}{WHITE} \
                {RESET}"
, flush=True )

  elif args.input_mode=="rna":
    print( f"{GREY_BACKGROUND}PRE_COMPRESS:   INFO: rna-seq args: \
nn_dense_dropout_1={CHARTREUSE}{args.nn_dense_dropout_1 if args.nn_type_rna=='DENSE' else 'n/a'}{WHITE}, \
nn_dense_dropout_2={CHARTREUSE}{args.nn_dense_dropout_2 if args.nn_type_rna=='DENSE' else 'n/a'}{WHITE}, \
hidden_layer_neurons={CHARTREUSE}{args.hidden_layer_neurons}{WHITE}, \
gene_embed_dim={CHARTREUSE}{args.gene_embed_dim}{WHITE}, \
n_genes={CHARTREUSE}{args.n_genes}{WHITE}, \
gene_data_norm={WHITE}{ORANGE if not args.gene_data_norm[0]     =='NONE' else MAGENTA if len(args.gene_data_norm)>1      else CHARTREUSE}{args.gene_data_norm}{WHITE}, \
g_xform={WHITE}{ORANGE        if not args.gene_data_transform[0]=='NONE' else MAGENTA if len(args.gene_data_transform)>1 else CHARTREUSE}{args.gene_data_transform}                                                                                                                 {RESET}" )

  pretrain                      = args.pretrain
  skip_tiling                   = args.skip_tiling
  skip_generation               = args.skip_generation
  dataset                       = args.dataset
  cases                         = args.cases
  divide_cases                  = args.divide_cases
  cases_reserved_for_image_rna  = args.cases_reserved_for_image_rna
  data_source                   = args.data_source
  global_data                   = args.global_data
  mapping_file_name             = args.mapping_file_name
  target_genes_reference_file   = args.target_genes_reference_file
  class_names                   = args.class_names
  cancer_type                   = args.cancer_type
  cancer_type_long              = args.cancer_type_long    
  long_class_names              = args.long_class_names  
  class_colours                 = args.class_colours
  colour_map                    = args.colour_map
  input_mode                    = args.input_mode
  multimode                     = args.multimode
  nn_mode                       = args.nn_mode
  nn_type_img                   = args.nn_type_img
  nn_type_rna                   = args.nn_type_rna
  use_same_seed                 = args.use_same_seed
  hidden_layer_neurons          = args.hidden_layer_neurons
  gene_embed_dim                = args.gene_embed_dim
  nn_dense_dropout_1            = args.nn_dense_dropout_1
  nn_dense_dropout_2            = args.nn_dense_dropout_2
  label_swap_perunit            = args.label_swap_perunit
  nn_optimizer                  = args.optimizer
  n_samples                     = args.n_samples
  n_tests                       = args.n_tests
  pct_test                      = args.pct_test
  n_tiles                       = args.n_tiles
  n_epochs                      = args.n_epochs
  n_iterations                  = args.n_iterations
  batch_size                    = args.batch_size
  lr                            = args.learning_rate
  rand_tiles                    = args.rand_tiles
  n_genes                       = args.n_genes
  gene_data_norm                = args.gene_data_norm 
  gene_data_transform           = args.gene_data_transform    
  n_epochs                      = args.n_epochs
  greyness                      = args.greyness
  min_tile_sd                   = args.min_tile_sd
  min_uniques                   = args.min_uniques  
  make_grey_perunit             = args.make_grey_perunit
  peer_noise_perunit            = args.peer_noise_perunit
  stain_norm                    = args.stain_norm
  stain_norm_target             = args.stain_norm_target
  annotated_tiles               = args.annotated_tiles
  figure_width                  = args.figure_width
  figure_height                 = args.figure_height  
  probs_matrix_interpolation    = args.probs_matrix_interpolation
  max_consecutive_losses        = args.max_consecutive_losses
  target_tile_coords            = args.target_tile_coords
  
  base_dir                   = args.base_dir
  data_dir                   = args.data_dir
  log_dir                    = args.log_dir
  tile_size                  = args.tile_size
  rna_file_name              = args.rna_file_name
  class_numpy_file_name      = args.class_numpy_file_name
  highest_class_number       = args.highest_class_number
  regenerate                 = args.regenerate
  just_profile               = args.just_profile
  just_test                  = args.just_test
  save_model_name            = args.save_model_name
  save_model_every           = args.save_model_every
  supergrid_size             = args.supergrid_size
  
  
  use_autoencoder_output     = args.use_autoencoder_output
  ae_add_noise               = args.ae_add_noise
  
  ddp                        = args.ddp
  gpus                       = args.gpus
  nodes                      = args.nodes
  
  remove_unexpressed_genes    = args.remove_unexpressed_genes
  remove_low_expression_genes = args.remove_low_expression_genes
  low_expression_threshold    = args.low_expression_threshold
  encoder_activation          = args.encoder_activation
  hidden_layer_neurons        = args.hidden_layer_neurons
  gene_embed_dim              = args.gene_embed_dim


  last_stain_norm='NULL'
  last_gene_norm='NULL'
  n_samples_max  = np.max(n_samples)
  tile_size_max  = np.max(tile_size)  
  n_tiles_max    = np.max(n_tiles)
  n_tiles_last   = 0                                                                                       # used to trigger regeneration of tiles if a run requires more tiles that the preceeding run 
  n_samples_last = 0
  tile_size_last = 0                                                                                       #   
  global_batch_count    = 0
  total_runs_in_job     = 0
  final_test_batch_size = 0

  n_classes=len(class_names)

  if ddp=='True':
    
    if DEBUG>0:
      print ( f"{BRIGHT_GREEN}PRE_COMPRESS:   INFO:   DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! pre-processing and generation steps will be bypassed (do pre-processing and generation with {YELLOW}DDP='False'{RESET}{BRIGHT_GREEN} if necessary){RESET}" )

    comms_package = 'nccl'
    if DEBUG>0:
      print ( f"PRE_COMPRESS:   INFO:      about to initialize process group:" )
      print ( f"PRE_COMPRESS:   INFO:        NVDIA comms package = {MIKADO}{comms_package}{RESET}" )
      print ( f"PRE_COMPRESS:   INFO:        rank                = {MIKADO}{rank}{RESET}" )
      print ( f"PRE_COMPRESS:   INFO:        world_size          = {MIKADO}{world_size}{RESET}" )      
      
    dist.init_process_group( backend      = 'nccl',
                             init_method  ='env://',
                             rank         = rank, 
                             world_size   = world_size )
  
  if  ( ( input_mode == 'image' ) & ( nn_mode == 'pre_compress' ) &  ( not ( 'AE' in nn_type_img[0] ) )):
    print( f"{RED}PRE_COMPRESS:   FATAL:  the network model must be an autoencoder if nn_mode='{MIKADO}{nn_mode}{RESET}{RED}' (you have nn_type_img='{MIKADO}{nn_type_img[0]}{RESET}{RED}', which is not an autoencoder) ... halting now{RESET}" )
    sys.exit(0)
    
  if  ( ( input_mode == 'rna'   ) & ( nn_mode == 'pre_compress' ) &  ( not ( 'AE' in nn_type_rna[0] ) )):
    print( f"{RED}PRE_COMPRESS:   FATAL:  when {CYAN}NN_TYPE_RNA{RESET}{RED}='{MIKADO}{nn_mode}{RESET}{RED}', the network model must be an autoencoder (you have {CYAN}NN_TYPE_RNA{RESET}{RED}={MIKADO}{nn_type_rna[0]}{RESET}{RED}, which is not an autoencoder){RESET}" )
    print( f"{RED}PRE_COMPRESS:   FATAL:  possible solution: try one of the DENSE autoencoders; for example {CYAN}NN_TYPE_RNA{RESET}{RED}={MIKADO}AEDENSE{RESET}" )
    print( f"{RED}PRE_COMPRESS:   FATAL:  ... halting now{RESET}" )
    sys.exit(0)


  if just_test=='True':
    print( f"{ORANGE}PRE_COMPRESS:   INFO:  CAUTION! 'just_test'  flag is set. No training will be performed{RESET}" )

# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE WANT TO BE ABLE TO HAVE MULTUIPLE RUNS OF TEST AND ACCCUMULATE THE RESULTING EMBEDDINGS

    # ~ if args.use_autoencoder_output!=True:
      # ~ if n_epochs>1:
        # ~ print( f"{BRIGHT}{ORANGE}PRE_COMPRESS:   INFO:  CAUTION! 'just_test'  flag is set, so n_epochs (currently {MIKADO}{n_epochs}{RESET}{ORANGE}) has been set to {MIKADO}1{RESET}{ORANGE} for this job{RESET}" ) 
        # ~ n_epochs=1
    # ~ else:
      # ~ pass

# THIS IS DIFFERENT TO THE DLBCL VERSION, BECAUSE WE WANT TO BE ABLE TO HAVE MULTUIPLE RUNS OF TEST AND ACCCUMULATE THE RESULTING EMBEDDINGS


    if len(batch_size)>1:
      print( f"{ORANGE}PRE_COMPRESS:   INFO:  CAUTION! 'just_test'  flag is set but but 'batch_size' has {MIKADO}{len(batch_size)}{RESET}{ORANGE} values ({MIKADO}{batch_size}{RESET}{ORANGE}). Only the first value ({MIKADO}{batch_size[0]}{ORANGE}) will be used{RESET}" )
      del batch_size[1:]



  # (A)  SET UP JOB LOOP

  already_tiled=False
  already_generated=False
                          
  parameters = dict( 
                                 lr  =   lr,
                           pct_test  =   pct_test,
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
  second_offset = 10
  
  total_runs_in_job = len(list(product(*param_values)))
    
  
  if DEBUG>0:
    print ( f"PRE_COMPRESS:   INFO: total_runs_in_job       =  {CARRIBEAN_GREEN}{total_runs_in_job}{RESET}"  )

  image_headings =\
f"\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Cpct_test\
\r\033[{start_column+2*offset}Csamples\
\r\033[{start_column+3*offset}Cbatch_size\
\r\033[{start_column+4*offset}Ctiles/image\
\r\033[{start_column+5*offset}Ctile_size\
\r\033[{start_column+6*offset}Crand_tiles\
\r\033[{start_column+7*offset}Cnet_img\
\r\033[{start_column+8*offset}Coptimizer\
\r\033[{start_column+9*offset}Cstain_norm\
\r\033[{start_column+10*offset}Clabel_swap\
\r\033[{start_column+11*offset}Cgreyscale\
\r\033[{start_column+12*offset}Cjitter vector\
"

  rna_headings =\
f"\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Cpct_test\
\r\033[{start_column+2*offset}Csamples\
\r\033[{start_column+3*offset}Cbatch_size\
\r\033[{start_column+4*offset}Cnet_rna\
\r\033[{start_column+5*offset}Chidden\
\r\033[{start_column+6*offset}Cembeded\
\r\033[{start_column+7*offset}Cnn_drop_1\
\r\033[{start_column+8*offset}Cnn_drop_1\
\r\033[{start_column+9*offset}Coptimizer\
\r\033[{start_column+10*offset}Cg_norm\
\r\033[{start_column+11*offset}Cg_xform\
\r\033[{start_column+12*offset}Clabel_swap\
\r\033[{start_column+13*offset}Cjitter vector\
"
  
  if DEBUG>0:
    if input_mode=='image':
      print(f"\n{UNDER}JOB:{RESET}")
      print(f"\033[2C{image_headings}{RESET}")      
      for lr, pct_test, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values):    
        print( f"{CARRIBEAN_GREEN}\
\r\033[2C\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{pct_test:<9.6f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{n_tiles:<5d}\
\r\033[{start_column+5*offset}C{tile_size:<3d}\
\r\033[{start_column+6*offset}C{rand_tiles:<5s}\
\r\033[{start_column+7*offset}C{nn_type_img:<10s}\
\r\033[{start_column+8*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+9*offset}C{stain_norm:<10s}\
\r\033[{start_column+10*offset}C{label_swap_perunit:<6.1f}\
\r\033[{start_column+11*offset}C{make_grey_perunit:<5.1f}\
\r\033[{start_column+12*offset}C{jitter:}\
{RESET}" )  

    elif input_mode=='rna':
      print(f"\n{UNDER}JOB:{RESET}")
      print(f"\033[2C\{rna_headings}{RESET}")
      
      for lr, pct_test, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values):
        print( f"{CARRIBEAN_GREEN}\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{pct_test:<9.2f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{nn_type_rna:<10s}\
\r\033[{start_column+5*offset}C{hidden_layer_neurons:<5d}\
\r\033[{start_column+6*offset}C{gene_embed_dim:<5d}\
\r\033[{start_column+7*offset}C{nn_dense_dropout_1:<5.2f}\
\r\033[{start_column+8*offset}C{nn_dense_dropout_2:<5.2f}\
\r\033[{start_column+9*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+10*offset}C{gene_data_norm:<10s}\
\r\033[{start_column+11*offset}C{gene_data_transform:<10s}\
\r\033[{start_column+12*offset}C{label_swap_perunit:<6.1f}\
\r\033[{start_column+13*offset}C{jitter:}\
{RESET}" )

  if (just_test=='True') & (input_mode=='image') & (multimode!= 'image_rna') & (use_autoencoder_output!="True"):   
    if not ( batch_size == int( math.sqrt(batch_size) + 0.5) ** 2 ):
      print( f"\033[31;1mTRAINLENEJ:     FATAL:  in test mode 'batch_size' (currently {batch_size}) must be a perfect square (4, 9, 16, 25 ...) to permit selection of a a 2D contiguous patch. Halting [2989].\033[m" )
      sys.exit(0)      

  
  # (B) RUN JOB LOOP

  run=0
  
  for lr, pct_test, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values): 
 
 
    if ( divide_cases == 'True' ):
      
      if just_test!='True':                                                                      
        multimode_case_count, unimode_case_count, not_a_multimode_case_count, not_a_multimode_case____image_count, not_a_multimode_case____image_test_count =     segment_cases( pct_test )  # boils down to setting flags in the directories of certain cases, esp. 'MULTIMODE_CASE_FLAG'
      else:
        print( f"{RED}PRE_COMPRESS:     FATAL: user option  {CYAN}-v ('args.cases'){RESET}{RED} is not allowed in test mode ({CYAN}JUST_TEST=True{RESET}, {CYAN}--just_test 'True'{RESET}){RED}{RESET}" )
        print( f"{RED}PRE_COMPRESS:     FATAL: explanation:  it will resegment the cases, meaning there is every chance cases you've trained on will end up in the test set{RESET}" )
        print( f"{RED}PRE_COMPRESS:     FATAL: ... halting now{RESET}" )
        sys.exit(0)        
 
 
 
    use_unfiltered_data=""
    if use_unfiltered_data=='True':
      rna_genes_tranche="all_ENSG_genes_including_non_coding_genes"
    else:
      rna_genes_tranche=os.path.basename(target_genes_reference_file)    
    
    if input_mode=='image':
      file_name_prefix = f"_{args.cases[0:18]}_{args.dataset}_r{total_runs_in_job}_e{args.n_epochs:03d}_n{args.n_samples[0]:03d}_b{args.batch_size[0]:02d}_t{int(100*pct_test):03d}_lr{args.learning_rate[0]:01.5f}"
    elif input_mode=='rna':
      file_name_prefix = f"_{args.cases[0:18]}_{args.dataset}_r{total_runs_in_job}_e{args.n_epochs:03d}_n{args.n_samples[0]:03d}_b{args.batch_size[0]:02d}_t{int(100*pct_test):03d}_lr{args.learning_rate[0]:01.5f}_h{args.hidden_layer_neurons[0]:04d}_d{int(100*args.nn_dense_dropout_1[0]):04d}_{rna_genes_tranche}"
    else:
      file_name_prefix = f"_{args.cases[0:18]}_{args.dataset}_r{total_runs_in_job}_e{args.n_epochs:03d}_n{args.n_samples[0]:03d}_b{args.batch_size[0]:02d}_t{int(100*pct_test):03d}_lr{args.learning_rate[0]:01.5f}_h{args.hidden_layer_neurons[0]:04d}_d{int(100*args.nn_dense_dropout_1[0]):04d}"          
    
    run+=1


    # ~ if DEBUG>0:
      # ~ print(f"\n\n{UNDER}RUN:{run}{RESET}")
      # ~ print(f"\
# ~ \r\033[{start_column+0*offset}Clr\
# ~ \r\033[{start_column+1*offset}Cn_samples\
# ~ \r\033[{start_column+2*offset}Cbatch_size\
# ~ \r\033[{start_column+3*offset}Cn_tiles\
# ~ \r\033[{start_column+4*offset}Ctile_size\
# ~ \r\033[{start_column+5*offset}Crand_tiles\
# ~ \r\033[{start_column+6*offset}Cnn_type_img\
# ~ \r\033[{start_column+7*offset}Cnn_type_rna\
# ~ \r\033[{start_column+8*offset}Cactivation\
# ~ \r\033[{start_column+9*offset}Chidden_layer_neurons\
# ~ \r\033[{start_column+10*offset+second_offset}Cembed_dims\
# ~ \r\033[{start_column+11*offset+second_offset}Cnn_drop_1\
# ~ \r\033[{start_column+12*offset+second_offset}Cnn_drop_2\
# ~ \r\033[{start_column+13*offset+second_offset}Coptimizer\
# ~ \r\033[{start_column+14*offset+second_offset}Cstain_norm\
# ~ \r\033[{start_column+15*offset+second_offset}Cg_norm\
# ~ \r\033[{start_column+16*offset+second_offset}Cg_xform\
# ~ \r\033[{start_column+17*offset+second_offset}Clabel_swap\
# ~ \r\033[{start_column+18*offset+second_offset}Cgreyscale\
# ~ \r\033[{start_column+19*offset+second_offset}Cjitter vector\033[m")
      # ~ print( f"\
# ~ \r\033[{start_column+0*offset}C{MIKADO}{lr:<9.6f}\
# ~ \r\033[{start_column+1*offset}C{n_samples:<5d}\
# ~ \r\033[{start_column+2*offset}C{batch_size:<5d}\
# ~ \r\033[{start_column+3*offset}C{n_tiles:<5d}\
# ~ \r\033[{start_column+4*offset}C{tile_size:<3d}\
# ~ \r\033[{start_column+5*offset}C{rand_tiles:<5s}\
# ~ \r\033[{start_column+6*offset}C{nn_type_img:<10s}\
# ~ \r\033[{start_column+7*offset}C{nn_type_rna:<10s}\
# ~ \r\033[{start_column+8*offset}C{encoder_activation:<12s}\
# ~ \r\033[{start_column+9*offset}C{hidden_layer_neurons:<5d}\
# ~ \r\033[{start_column+10*offset+second_offset}C{gene_embed_dim:<5d}\
# ~ \r\033[{start_column+11*offset+second_offset}C{nn_dense_dropout_1:<5.2f}\
# ~ \r\033[{start_column+12*offset+second_offset}C{nn_dense_dropout_2:<5.2f}\
# ~ \r\033[{start_column+13*offset+second_offset}C{nn_optimizer:<8s}\
# ~ \r\033[{start_column+14*offset+second_offset}C{stain_norm:<10s}\
# ~ \r\033[{start_column+15*offset+second_offset}C{gene_data_norm:<10s}\
# ~ \r\033[{start_column+16*offset+second_offset}C{gene_data_transform:<10s}\
# ~ \r\033[{start_column+17*offset+second_offset}C{label_swap_perunit:<6.1f}\
# ~ \r\033[{start_column+18*offset+second_offset}C{make_grey_perunit:<5.1f}\
# ~ \r\033[{start_column+1*offset+second_offset}C{jitter:}{RESET}" )    

      # ~ print ( "\n" )
      
      
    #(N-2) set up Tensorboard
    
    print( "PRE_COMPRESS:   INFO: \033[1m1 about to set up Tensorboard\033[m" )
    
    if input_mode=='image':
      writer = SummaryWriter(comment=f' {dataset}; {input_mode} {nn_type_img}; n_smp={n_samples}; sg_sz={supergrid_size}; n_t={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; n_e={n_epochs}; b_sz={batch_size}' )
    elif input_mode=='rna':
      writer = SummaryWriter(comment=f' {dataset}; {input_mode}; {nn_type_rna}; ACT={encoder_activation}; HID={hidden_layer_neurons}; EMB={gene_embed_dim}; d1={nn_dense_dropout_1}; d2={nn_dense_dropout_2}; {nn_optimizer}; samples={n_samples}; genes={n_genes}; g_norm={gene_data_norm}; g_xform={gene_data_transform}; EPOCHS={n_epochs}; BATCH={batch_size}; lr={lr}')
    elif input_mode=='image_rna':
      writer = SummaryWriter(comment=f' {dataset}; {input_mode}; {nn_type_img}; {nn_type_rna}; ACT={encoder_activation}; HID={hidden_layer_neurons}; {nn_optimizer}; samples={n_samples}; tiles={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; genes={n_genes}; g_norm={gene_data_norm}; g_xform={gene_data_transform}; epochs={n_epochs}; batch={batch_size}; lr={lr}')
    else:
      print( f"{RED}PRE_COMPRESS: FATAL:    input mode of type '{MIKADO}{input_mode}{RESET}{RED}' is not supported [314]{RESET}" )
      sys.exit(0)

    print( "PRE_COMPRESS:   INFO:   \033[3mTensorboard has been set up\033[m", flush=True ) 
    


    # (1) Potentially schedule and run tiler threads
    
    if (input_mode=='image') & (multimode!='image_rna'):
      
      if skip_tiling=='False':
        
        # need to re-tile if certain parameters have eiher INCREASED ('n_tiles' or 'n_samples') or simply CHANGED ( 'stain_norm' or 'tile_size') since the last run
        if ( ( already_tiled==True ) & ( ( stain_norm==last_stain_norm ) | (last_stain_norm=="NULL") ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) ):
          pass          # no need to re-tile                                                              
        else:           # must re-tile
          if DEBUG>0:
            print( f"PRE_COMPRESS:     INFO:{BRIGHT}1 about to launch tiling processes{RESET}" )
          if DEBUG>1:
            print( f"PRE_COMPRESS:     INFO:    stain normalization method = {CYAN}{stain_norm}{RESET}" )
          delete_selected( data_dir, "png" )
          last_stain_norm=stain_norm
          already_tiled=True

          if DEBUG>999:
            print( f"PRE_COMPRESS:       INFO:   n_samples_max                   = {MIKADO}{n_samples_max}{RESET}")
            print( f"PRE_COMPRESS:       INFO:   n_tiles_max                     = {MIKADO}{n_tiles_max}{RESET}")
  
          if stain_norm=="NONE":                                                                         # we are NOT going to stain normalize ...
            norm_method='NONE'
          else:                                                                                          # we are going to stain normalize ...
            if DEBUG>0:
              print( f"PRE_COMPRESS:       INFO: {BRIGHT}about to set up stain normalization target{RESET}" )
            if stain_norm_target.endswith(".svs"):                                                       # ... then grab the user provided target
              norm_method = tiler_set_target( args, n_tiles, tile_size, stain_norm, stain_norm_target, writer )
            else:                                                                                        # ... and there MUST be a target
              print( f"PRE_COMPRESS:     FATAL:    for {MIKADO}{stain_norm}{RESET} an SVS file must be provided from which the stain normalization target will be extracted" )
              sys.exit(0)


          print ( f"{SAVE_CURSOR}" )
            
          if just_test=='True':

              try:
                fqn = f"{args.data_dir}/SUFFICIENT_SLIDES_TILED"
                os.remove( fqn )
              except:
                pass

              if (  args.cases == 'NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG' ):
                
                flag  = 'NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG'
                count = n_samples
                if DEBUG>0:
                  print( f"{SAVE_CURSOR}\r\033[{num_cpus}B{WHITE}PRE_COMPRESS:     INFO:about to call tiler_threader with flag = {CYAN}{flag}{RESET}; count = {MIKADO}{count:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles = {MIKADO}{n_tiles}{RESET}{RESTORE_CURSOR}", flush=True )
                slides_tiled_count = tiler_threader( args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method )

              elif (  args.cases == 'DESIGNATED_MULTIMODE_CASE_FLAG' ):
                
                flag  = 'DESIGNATED_MULTIMODE_CASE_FLAG'
                count = cases_reserved_for_image_rna
                if DEBUG>0:
                  print( f"{SAVE_CURSOR}\r\033[{num_cpus}B{WHITE}PRE_COMPRESS:     INFO:about to call tiler_threader with flag = {CYAN}{flag}{RESET}; count = {MIKADO}{count:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles = {MIKADO}{n_tiles}{RESET}{RESTORE_CURSOR}", flush=True )
                slides_tiled_count = tiler_threader( args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method )

# ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS 

              elif (  args.cases == 'NOT_A_MULTIMODE_CASE_FLAG' ):
                
                flag  = 'NOT_A_MULTIMODE_CASE_FLAG'
                count =  n_samples
                if DEBUG>0:
                  print( f"{SAVE_CURSOR}\r\033[{num_cpus}B{WHITE}PRE_COMPRESS:     INFO:about to call tiler_threader with flag = {CYAN}{flag}{RESET}; count = {MIKADO}{count:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles = {MIKADO}{n_tiles}{RESET}{RESTORE_CURSOR}", flush=True )
                slides_tiled_count = tiler_threader( args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method )

# ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS ONLY PRE_COMPRESS MODE HAS THIS


          else:

            if (  args.cases == 'ALL_ELIGIBLE_CASES' ):
              
              slides_to_be_tiled = n_samples

              try:
                fqn = f"{args.data_dir}/SUFFICIENT_SLIDES_TILED"
                os.remove( fqn )
              except:
                pass

              flag  = 'HAS_IMAGE_FLAG'
            
              if DEBUG>0:
                print( f"{SAVE_CURSOR}\r\033[{num_cpus+1}B{WHITE}PRE_COMPRESS:     INFO:about to call tiler_threader with flag = {CYAN}{flag}{RESET}; slides_to_be_tiled = {MIKADO}{slides_to_be_tiled:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles_max = {MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )
              slides_tiled_count = tiler_threader( args, flag, slides_to_be_tiled, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job

              
            if (  args.cases == 'NOT_A_MULTIMODE_CASE_FLAG' ):

              test_count  =  int(pct_test * n_samples)
              train_count =  n_samples - test_count
              
              slides_to_be_tiled = train_count + test_count

              try:
                fqn = f"{args.data_dir}/SUFFICIENT_SLIDES_TILED"
                os.remove( fqn )
              except:
                pass

              flag  = 'NOT_A_MULTIMODE_CASE____IMAGE_FLAG'
              if DEBUG>0:
                print( f"{SAVE_CURSOR}\r{WHITE}PRE_COMPRESS:     INFO:about to call {MAGENTA}tiler_threader{RESET}: flag={CYAN}{flag}{RESET}; train_count={MIKADO}{train_count:3d}{RESET}; %_test={MIKADO}{pct_test:2.2f}{RESET}; n_samples={MIKADO}{n_samples_max:3d}{RESET}; n_tiles={MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )
              slides_tiled_count = tiler_threader( args, flag, train_count, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job


              flag  = 'NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG'
              if DEBUG>0:
                print( f"{SAVE_CURSOR}\r{WHITE}PRE_COMPRESS:     INFO:about to call {MAGENTA}tiler_threader{RESET}: flag={CYAN}{flag}{RESET}; test_count={MIKADO}{test_count:3d}{RESET}; %_test={MIKADO}{pct_test:2.2f}{RESET}; n_samples={MIKADO}{n_samples_max:3d}{RESET}; n_tiles={MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )
              slides_tiled_count = tiler_threader( args, flag, test_count, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job

              

          print ( f"{RESTORE_CURSOR}" )



        
          if just_profile=='True':                                                                       # then we are all done
            sys.exit(0)


    # (2) Regenerate Torch '.pt' file, if required. The logic for 'image_rna' is just the concatenation of the logic for 'image' and the logic for 'rna'

    if skip_generation=='False':
      
      if DEBUG>5:
        print( f"PRE_COMPRESS:     INFO:n_samples               = {MAGENTA}{n_samples}{RESET}"       )
        print( f"PRE_COMPRESS:     INFO:args.n_samples          = {MAGENTA}{args.n_samples}{RESET}"  )
        print( f"PRE_COMPRESS:     INFO:n_tiles                 = {MAGENTA}{n_tiles}{RESET}"         )
        print( f"PRE_COMPRESS:     INFO:args.n_tiles            = {MAGENTA}{args.n_tiles}{RESET}"    )
        print( f"PRE_COMPRESS:     INFO:batch_size              = {MAGENTA}{batch_size}{RESET}"      )
        print( f"PRE_COMPRESS:     INFO:args.batch_size         = {MAGENTA}{args.batch_size}{RESET}" )
        print( f"PRE_COMPRESS:     INFO:n_genes (from args)     = {MAGENTA}{n_genes}{RESET}"         )
        print( f"PRE_COMPRESS:     INFO:gene_data_norm          = {MAGENTA}{gene_data_norm}{RESET}"  )            
                      
      n_genes = generate( args, n_samples, highest_class_number, multimode_case_count, unimode_case_count, not_a_multimode_case_count, not_a_multimode_case____image_count, not_a_multimode_case____image_test_count, pct_test, n_tiles, tile_size, gene_data_norm, gene_data_transform  ) 

      if DEBUG>5:
        print( f"PRE_COMPRESS:     INFO:n_samples               = {BLEU}{n_samples}{RESET}"       )
        print( f"PRE_COMPRESS:     INFO:args.n_samples          = {BLEU}{args.n_samples}{RESET}"  )
        print( f"PRE_COMPRESS:     INFO:n_tiles                 = {BLEU}{n_tiles}{RESET}"         )
        print( f"PRE_COMPRESS:     INFO:args.n_tiles            = {BLEU}{args.n_tiles}{RESET}"    )
        print( f"PRE_COMPRESS:     INFO:batch_size              = {BLEU}{batch_size}{RESET}"      )
        print( f"PRE_COMPRESS:     INFO:args.batch_size         = {BLEU}{args.batch_size}{RESET}" )
        print( f"PRE_COMPRESS:     INFO:n_genes (from args)     = {BLEU}{n_genes}{RESET}"         )
        print( f"PRE_COMPRESS:     INFO:gene_data_norm          = {BLEU}{gene_data_norm}{RESET}"  )  
        
      n_tiles_last   = n_tiles                                                                           # for the next run
      n_samples_last = n_samples                                                                         # for the next run
      tile_size_last = tile_size                                                                         # for the next run




    # (N) Load experiment config.  Most configurable parameters are now provided via user arguments
    
    cfg = loader.get_config( args.nn_mode, lr, batch_size )

    if ddp=='True':
      world_size = world_size
      rank       = rank      
      gpu        = gpu
      
    else:
      world_size = 0
      rank       = 0
      gpu        = 0


    #(N+1) Load dataset      

    do_all_test_examples=False    
    train_loader, test_loader, _,  _  = loader.get_data_loaders( args,
                                                         gpu,
                                                         cfg,
                                                         world_size,
                                                         rank,
                                                         batch_size,
                                                         args.n_workers,
                                                         args.pin_memory,                                                      
                                                         pct_test,
                                                         writer
                                                        )                                                 


    #(N+2) Load model
    
    if DEBUG>1:                                                                                                       
      print( f"PRE_COMPRESS:     INFO:{BRIGHT}5 about to load networks {MIKADO}{nn_type_img}{RESET}{BRIGHT} and {MIKADO}{nn_type_rna}{RESET}" )                                  
    model = PRECOMPRESS( args, gpu, rank, cfg, input_mode, nn_type_img, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, args.latent_dim, args.em_iters  )   


    #(N+3) Send model to GPU(s)
        
    model = model.to(rank)

    if just_test=='True':                                                                                  # then load already trained model from HDD
      if DEBUG>0:
        print( f"{ORANGE}PRE_COMPRESS:   INFO:  'just_test'  flag is set: about to load model state dictionary from {MAGENTA}{save_model_name}{log_dir}{ORANGE} in directory {MAGENTA}{log_dir}{RESET}" )
      fpath = '%s/lowest_loss_ae_model.pt' % log_dir
      try:
        model.load_state_dict(torch.load(fpath))       
      except Exception as e:
        print( f"{RED}PRE_COMPRESS:   INFO:  CAUTION! There is no trained model. Predictions will be meaningless ... continuing{RESET}" )        
        time.sleep(2)
        pass
                      



    #(N+4) Select and configure optimizer

    optimizer = optim.Adam( model.parameters(), lr )

    if nn_type_rna=='TTVAE':
      scheduler_opts = dict( scheduler           = 'warm_restarts',
                             lr_scheduler_decay  = 0.5, 
                             T_max               = 10, 
                             eta_min             = 5e-8, 
                             T_mult              = 2                  )

      scheduler      = Scheduler( optimizer = optimizer,  opts=scheduler_opts )   
    else:
      scheduler = 0

    number_correct_max   = 0
    pct_correct_max      = 0
    test_loss_min        = 9999999999999
    train_loss_min       = 9999999999999



    #(N_5) Select Loss function
      ############################################# empty



    #(N+6) Train/Test
    
    
    consecutive_training_loss_increases    = 0
    consecutive_test_loss_increases        = 0
    

    last_epoch_loss_increased              = True

    train_total_loss_sum_ave_last          = 9999999999999                                                 # used to determine whether total loss is increasing or decreasing
    train_lowest_total_loss_observed       = 9999999999999                                                 # used to track lowest total loss
    train_lowest_total_loss_observed_epoch = 0                                                             # used to track lowest total loss

    train_images_loss_sum_ave_last         = 9999999999999
    train_lowest_image_loss_observed       = 9999999999999
    train_lowest_image_loss_observed_epoch = 0

    test_batch_loss_epoch_ave_last         = 9999999999999                                                 # used to determine whether total loss is increasing or decreasing
    test_lowest_total_loss_observed        = 9999999999999
    test_lowest_total_loss_observed_epoch  = 0
    
    test_genes_loss_sum_ave_last           = 9999999999999 
    test_lowest_genes_loss_observed        = 9999999999999      
    test_lowest_genes_loss_observed_epoch  = 0 
        


# PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION 
 
    if just_test=='True':
      
      if DEBUG>12:
        print( f"PRE_COMPRESS:   INFO:        batch_size     =  {MAGENTA}{batch_size}{RESET}",     flush=True )
        print( f"PRE_COMPRESS:   INFO:        gene_embed_dim =  {MAGENTA}{gene_embed_dim}{RESET}", flush=True )
                                                                                            
      embeddings_accum = torch.zeros( [ 0, gene_embed_dim ],  requires_grad=False,  dtype=torch.float )
      
      if DEBUG>12:
        print( f"PRE_COMPRESS:   INFO:        embeddings_accum.dtype =  {MAGENTA}{embeddings_accum.dtype }{RESET}", flush=True  )
        print( f"PRE_COMPRESS:   INFO:        embeddings_accum.size  =  {MAGENTA}{embeddings_accum.size() }{RESET}", flush=True )
                  
      labels_accum     = torch.zeros( [ 0                 ],  requires_grad=False,  dtype=torch.int64 ) 

    else:
      embeddings_accum = 0
      labels_accum     = 0
        
# PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION PREP FOR EMBEDDINGS ACCUMULATION 





    print( "PRE_COMPRESS:   INFO: \033[12m1 about to commence main loop, one iteration per epoch\033[m" )

    for epoch in range(1, n_epochs + 1):   

      if input_mode=='rna':
        print( f'\n{DIM_WHITE}PRE_COMPRESS:   INFO:      {RESET}run {MIKADO}{run}:{RESET} epoch: {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}, {PINK}({nn_type_rna}){RESET} mode: {MIKADO}{input_mode}{RESET}, samples: {MIKADO}{n_samples}{RESET}, batch size: {MIKADO}{batch_size}{RESET}.  {DULL_WHITE}Will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )          
      else:
        print( f'\n{DIM_WHITE}PRE_COMPRESS:   INFO:      {RESET}run {MIKADO}{run}:{RESET} epoch: {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}, {PINK}({nn_type_img}){RESET} mode: {MIKADO}{input_mode}{RESET}, samples: {MIKADO}{n_samples}{RESET}, batch size: {MIKADO}{batch_size}{RESET}, tile: {MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide: {MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}Will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )

 
      
 
    
      if just_test=='True':                                                                              # bypass training altogether in test mode
        pass         
      else:
        train_batch_loss_epoch_ave, train_loss_genes_sum_ave, train_l1_loss_sum_ave, train_total_loss_sum_ave =\
                                           train (      args, gpu, epoch, encoder_activation, train_loader, model, nn_type_img, lr, scheduler, optimizer, writer, train_loss_min, batch_size )
  

      embeddings_accum, labels_accum, test_batch_loss_epoch_ave, test_l1_loss_sum_ave, test_loss_min                =\
                                          test ( cfg, args, gpu, epoch, encoder_activation, test_loader,  model, nn_type_img, embeddings_accum, labels_accum,  tile_size, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, annotated_tiles, class_names, class_colours)

      
      
      
      torch.cuda.empty_cache()
      
      if DEBUG>0:
        if ( (test_batch_loss_epoch_ave < (test_batch_loss_epoch_ave_last)) | (epoch==1) ):
          consecutive_test_loss_increases = 0
          last_epoch_loss_increased = False
        else:
          last_epoch_loss_increased = True
          
        print ( f"\
\033[2K\
{DIM_WHITE}PRE_COMPRESS:   INFO:   {RESET}\
\r\033[27Cepoch summary:\
\r\033[92Cae={GREEN}{test_batch_loss_epoch_ave:<11.3f}{DULL_WHITE}\
\r\033[109Cl1={test_l1_loss_sum_ave:<11.3f}{DULL_WHITE}\
\r\033[124CBATCH AVE OVER EPOCH={GREEN if last_epoch_loss_increased==False else RED}{test_batch_loss_epoch_ave:<11.3f}\r\033[155C{UP_ARROW if last_epoch_loss_increased==True else DOWN_ARROW}{DULL_WHITE}\
\r\033[167Clowest BAOE: {test_lowest_total_loss_observed:<11.3f}@{ORANGE}\r\033[202Ce={test_lowest_total_loss_observed_epoch:<2d}{RESET}\
\033[3B\
", end='', flush=True )

        if last_epoch_loss_increased == True:
          consecutive_test_loss_increases +=1

          if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before, so that the most recent model for which the loss improved will be saved
              now = time.localtime(time.time())
              print(time.strftime("PRE_COMPRESS:   INFO: %Y-%m-%d %H:%M:%S %Z", now))
              sys.exit(0)

      test_batch_loss_epoch_ave_last = test_batch_loss_epoch_ave
      
      if DEBUG>9:
        print( f"{DIM_WHITE}PRE_COMPRESS:   INFO:   test_lowest_total_loss_observed = {MIKADO}{test_lowest_total_loss_observed}{RESET}" )
        print( f"{DIM_WHITE}PRE_COMPRESS:   INFO:   test_batch_loss_epoch_ave         = {MIKADO}{test_batch_loss_epoch_ave}{RESET}"         )
      
      if test_batch_loss_epoch_ave < test_lowest_total_loss_observed:
        test_lowest_total_loss_observed       = test_batch_loss_epoch_ave
        test_lowest_total_loss_observed_epoch = epoch
        if DEBUG>0:
          print ( f"\r\033[200C{DIM_WHITE}{GREEN}{ITALICS} \r\033[210C<< new minimum test loss{RESET}\033[1A", flush=True )

      #if ( (epoch+1)%LOG_EVERY==0 ):
      #    save_samples( args.log_dir, model, test_loader, cfg, epoch )




    # end of run
    
    # THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE

    if args.just_test=='True':                                                                       # In test mode (only), the embeddings are the reduced dimensionality features that we want to save for use with NN models
      
      fqn = f"{args.log_dir}/ae_output_features.pt"

      if DEBUG>0:
        print( f"\nPRE_COMPRESS:   INFO:        about to save embeddings and labels as torch dictionary to {MAGENTA}{fqn}{RESET}" )
       
      if args.input_mode=='image':
          
        if DEBUG>0:
          print( f"PRE_COMPRESS:   INFO:        embeddings_accum.size  =      {CARRIBEAN_GREEN}{embeddings_accum.size() }{RESET}", flush=True )        
          print( f"PRE_COMPRESS:   INFO:        labels_accum    .size  =      {CARRIBEAN_GREEN}{labels_accum.size() }{RESET}",     flush=True )        
        
        try:          
          torch.save({
              'embeddings': embeddings_accum,
              'labels':     labels_accum,
          }, fqn )
        except Exception as e:
          print( f"{RED}PRE_COMPRESS:   ERROR:  couldn't save{RESET}" )  
          time.sleep(5)          
      
      if args.input_mode=='rna':            
        torch.save({
            'embeddings': embeddings_accum,
            'labels':     labels_accum,
        }, fqn )
        
      if DEBUG>0:
        print( f"PRE_COMPRESS:   INFO:        embeddings have been saved" )          

    # THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE THIS IS WHERE THE EMBEDDINGS ARE SAVED TO FILE 
      

  if DEBUG>0:
    print( f"{DIM_WHITE}PRE_COMPRESS:   INFO:    pytorch Model = {MIKADO}{model}{RESET}" )





# ------------------------------------------------------------------------------

def train(  args, gpu, epoch, encoder_activation, train_loader, model, nn_type_rna, lr, scheduler, optimizer, writer, train_loss_min, batch_size  ):  
    """Train PCCA model and update parameters in batches of the whole train set.
    """

    model.train()

    ae_loss2_sum     = 0
    l1_loss_sum      = 0

    loss             = 0.
    total_recon_loss = 0.
    total_kl_loss    = 0.

    if DEBUG>2:
      print ( f"PRE_COMPRESS:   INFO:      train(): about to enumerate over train_loader" )
    
    for i, ( x1, x2, _, _, _ ) in enumerate( train_loader ):                                               # = enumerate over Torch DataLoader object, which is established in loader 

        if args.input_mode=='image': ### HACK
          x2=x1

        if DEBUG>2:
          print ( f"PRE_COMPRESS:   INFO:      train(): i  = {MIKADO}{i}{RESET}" )
        
        optimizer.zero_grad()

        x2 = x2.to( gpu )

        if DEBUG>2:
          # ~ print ( f"PRE_COMPRESS:   INFO:      train(): x2.type             = {MIKADO}{x2.type}{RESET}" )
          print ( f"PRE_COMPRESS:   INFO:      train(): encoder_activation  = {MIKADO}{encoder_activation}{RESET}" )

        x2r, mean, logvar = model.forward( args, x2, args.input_mode, gpu, encoder_activation )

        if DEBUG>99:
          print ( f"PRE_COMPRESS:   INFO:      train(): nn_type_rna        = {MIKADO}{nn_type_rna}{RESET}" )
          
        if nn_type_rna=='TTVAE':                                                                               # Fancy loss function for TTVAE
          
          if DEBUG>0:
            print ( f"PRE_COMPRESS:   INFO:      train(): x2[0:12,0:12]  = {MIKADO}{x2[0:12,0:12]}{RESET}" ) 
            print ( f"PRE_COMPRESS:   INFO:      train(): x2r[0:12,0:12] = {MIKADO}{x2r[0:12,0:12]}{RESET}" )


                                
          bce_loss       = False
          loss_reduction = 'sum'
          loss_fn        = BCELoss( reduction=loss_reduction ).cuda(gpu) if bce_loss else MSELoss( reduction=loss_reduction ).cuda(gpu)                                                 # Have to use Binary cross entropy loss for TTVAE (and VAEs generally)
          ae_loss2, reconstruction_loss, kl_loss = vae_loss( x2r, x2, mean, logvar, loss_fn, epoch, kl_warm_up=0, beta=1. )
          del mean
          del logvar
          
        else:                                                                                              # Used for AELINEAR, AEDENSE, AEDENSEPOSITIVE, AE3LAYERCONV3D, DCGANAE128
          ae_loss2 = F.mse_loss( x2r, x2.squeeze())                                                        # mean squared error loss function
        
        del x2
        del x2r
        
        l1_loss                = l1_penalty(model, args.l1_coef)                                            # NOT CURRENTLY USING l1_loss
        #loss                  = ae_loss1 + ae_loss2 + l1_loss                                              # NOT CURRENTLY USING l1_loss
        ae_loss2_sum          += ae_loss2.item()
        if nn_type_rna=='TTVAE':        
          reconstruction_loss += reconstruction_loss.item()
          kl_loss             += kl_loss.item()
        loss                   = ae_loss2

        loss.backward()
        
        del loss
        
        if not nn_type_rna=='TTVAE':
          # Perform gradient clipping *before* calling `optimizer.step()
          clip_grad_norm_( model.parameters(), args.clip )

        optimizer.step()

        if nn_type_rna=='TTVAE':
          scheduler.step()                                                                                 # has to be after optimizer.step()
          current_lr = scheduler.get_lr()
          if DEBUG>99:         
            print ( f"PRE_COMPRESS:   INFO:      train(): lr        = {MIKADO}{scheduler.get_lr():<2.2e}{RESET}" )
        else:
          current_lr = lr
          
        ae_loss2_sum += ae_loss2.item()
        l1_loss_sum  += l1_loss.item()                                                                     # NOT CURRENTLY USING l1_loss
        #total_loss   = ae_loss1_sum + ae_loss2_sum + l1_loss_sum                                          # NOT CURRENTLY USING l1_loss
        total_loss    = ae_loss2_sum                                                                       # NOT CURRENTLY USING l1_loss
        
        if DEBUG>0:
          print ( f"\
\033[2K\
{DIM_WHITE}PRE_COMPRESS:   INFO:{RESET}\
\r\033[29C{DULL_WHITE}train:\
\r\033[40Cn={i+1:>3d}\
\r\033[47Clr={current_lr:<2.2e}\
\r\033[92Cae={ ae_loss2:<11.3f}\
\r\033[109Cl1={l1_loss:<11.3f}\
\r\033[135CBATCH AVE={ae_loss2:11.3f}{RESET}" )
#\r\033[124C    BATCH LOSS=\r\033[{139+4*int((ae_loss2*10)//1) if ae_loss2<1 else 150+4*int((ae_loss2*2)//1) if ae_loss2<12 else 160}C{PALE_GREEN if ae_loss2<1 else GOLD if 1<=ae_loss2<2 else PALE_RED}{ae_loss2:11.3f}{RESET}" )

          if nn_type_rna=='TTVAE':
            print ( "\033[2A" )
            print ( f"{DULL_WHITE}\r\033[60Crecon={reconstruction_loss:<11.1f} \
            \r\033[78Ckl={ kl_loss:<6.3f}{RESET}" )

          print ( "\033[2A" )


    ae_loss2_sum  /= (i+1)                                                                                 # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    l1_loss_sum   /= (i+1)                                                                                 # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    train_msgs     = [ae_loss2_sum, l1_loss_sum]

    if ae_loss2_sum    <  train_loss_min:
      train_loss_min   =  ae_loss2_sum
       
    writer.add_scalar( '2_loss_train',      ae_loss2_sum,    epoch  )
    writer.add_scalar( 'loss_train_min',  train_loss_min,  epoch  ) 
    if nn_type_rna=='TTVAE':
      writer.add_scalar( 'loss_recon_VAE', reconstruction_loss,      epoch  )
      writer.add_scalar( 'loss_kl_vae',    kl_loss,                  epoch  )
      writer.add_scalar( 'lr',             scheduler.get_lr(),       epoch  )

      del kl_loss
      del reconstruction_loss

    del ae_loss2
    del l1_loss

    torch.cuda.empty_cache()

    return ae_loss2_sum, l1_loss_sum, total_loss, train_loss_min

# ------------------------------------------------------------------------------

def test( cfg, args, gpu, epoch, encoder_activation, test_loader, model,  nn_type_rna, embeddings_accum, labels_accum, tile_size, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, annotated_tiles, class_names, class_colours ):
 
            
    """Test model by computing the average loss on a held-out dataset. No parameter updates.
    """
    model.eval()

    ae_loss2_sum      = 0
    l1_loss_sum       = 0
    
    loss             = 0.
    total_recon_loss = 0.
    total_kl_loss    = 0.    
  
    for i, ( x1, x2, image_labels, rna_labels, batch_fnames ) in enumerate( test_loader ):


# MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS 

        if args.input_mode=='image':                                                                       # HACK
          x2=x1                                                                                            # HACK
          
        x2=x2.to(gpu)                                                                                      # HACK

        if args.input_mode=='image': 
          image_labels = image_labels.to(gpu)
          
        if args.input_mode=='rna':
          rna_labels   = rna_labels  .to(gpu)
        
# MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS MOD TO CATER FOR LABELS 

        with torch.no_grad():                                                                              # Don't need gradients for testing, so this will save some GPU memory
          x2r, mean, logvar = model.forward( args, x2, args.input_mode, gpu, encoder_activation )


# BELOW IS WHERE THE EMBEDDINGS ARE ACCUMULATED THIS IS WHERE THE EMBEDDINGS ARE ACCUMULATED THIS IS WHERE THE EMBEDDINGS ARE ACCUMULATED THIS IS WHERE THE EMBEDDINGS ARE ACCUMULATED

          if args.just_test=='True':                                                                       # In test mode (only), the embeddings are the reduced dimensionality features that we want to save for use with NN models
            
            if DEBUG>12:   
              print( f"PRE_COMPRESS:   INFO:        about to push x2 through the autoencoder to obtain the reduced dimensionality features using the best model generated by the last training run{RESET}" )

            embeddings  = model.encode  ( x2, args.input_mode, gpu, encoder_activation )             

            if DEBUG>12:
              print( f"PRE_COMPRESS:   INFO:        sanity check: embeddings      .size     = {MIKADO}{embeddings.size()}{RESET}" )
              print( f"PRE_COMPRESS:   INFO:        sanity check: embeddings      .dtype    = {MIKADO}{embeddings.dtype}{RESET}" )
              print( f"PRE_COMPRESS:   INFO:        sanity check: image_labels    .size     = {MIKADO}{image_labels.size()}{RESET}" )
              print( f"PRE_COMPRESS:   INFO:        sanity check: image_labels    .dtype    = {MIKADO}{image_labels.dtype}{RESET}" )
              
            embeddings_accum = torch.cat( (embeddings_accum, embeddings.cpu()   ), dim=0, out=None )
            
            if args.input_mode=="image":
              labels_accum     = torch.cat( (labels_accum,     image_labels.cpu() ), dim=0, out=None )
            if args.input_mode=="rna":
              labels_accum     = torch.cat( (labels_accum,     rna_labels  .cpu() ), dim=0, out=None ) 
            
            if DEBUG>12:
              print( f"PRE_COMPRESS:   INFO:        sanity check: embeddings_accum.size     = {AMETHYST}{embeddings_accum.size()}{RESET}" )
              print( f"PRE_COMPRESS:   INFO:        sanity check: labels_accum    .size     = {AMETHYST}{labels_accum.size()}{RESET}" )


# ABOVE IS WHERE THE EMBEDDINGS ARE ACCUMULATED THIS IS WHERE THE EMBEDDINGS ARE ACCUMULATED THIS IS WHERE THE EMBEDDINGS ARE ACCUMULATED THIS IS WHERE THE EMBEDDINGS ARE ACCUMULATED 


        if DEBUG>99:
          print ( f"PRE_COMPRESS:   INFO:      test(): nn_type_rna        = {MIKADO}{nn_type_rna}{RESET}" )
          
        if nn_type_rna=='TTVAE':                                                                               # Fancy loss function for TTVAE. ------------------> Disabling for the moment because it's not working
          bce_loss       = False
          loss_reduction ='sum'
          loss_fn        = BCELoss( reduction=loss_reduction ).cuda(gpu) if bce_loss else MSELoss( reduction=loss_reduction ).cuda(gpu)      
          ae_loss2, reconstruction_loss, kl_loss = vae_loss( x2r, x2, mean, logvar, loss_fn, epoch, kl_warm_up=0, beta=1.0 )
        else:                                                                                              # Used for AELINEAR, AEDENSE, AEDENSEPOSITIVE, AE3LAYERCONV3D, DCGANAE128
          ae_loss2 = F.mse_loss(x2r, x2.squeeze())


        l1_loss                = l1_penalty( model, args.l1_coef)                                           # NOT CURRENTLY USING l1_loss
        ae_loss2_sum          += ae_loss2.item()
        if nn_type_rna=='TTVAE':  
          reconstruction_loss += reconstruction_loss.item()
          kl_loss             += kl_loss.item()
        l1_loss_sum           += l1_loss.item()                                                             # NOT CURRENTLY USING l1_loss                                                    
        total_loss             =  ae_loss2_sum                                                              # NOT CURRENTLY USING l1_loss
        
        if DEBUG>0:
          if i==0:
            print ("")
          print ( f"\
\033[2K\
{DIM_WHITE}PRE_COMPRESS:   INFO:{RESET}\
\r\033[29Ctest:\
\r\033[40C{DULL_WHITE}n={i+1:>3d}\
\r\033[92Cae={ ae_loss2:<11.3f}\
\r\033[109Cl1={l1_loss:<11.3f}\
\r\033[135CBATCH AVE={ae_loss2:11.3f}{RESET}" )
#\r\033[124C    BATCH AVE=\r\033[{139+4*int((ae_loss2*10)//1) if ae_loss2<1 else 150+4*int((ae_loss2*2)//1) if ae_loss2<12 else 160}C{GREEN if ae_loss2<1 else ORANGE if 1<=ae_loss2<2 else RED}{ae_loss2:<11.3f}{RESET}" )

        print ( "\033[2A" )
    
    if DEBUG>1:
      print ("")
  
      ae_loss2_sum  /= (i+1)                                                                                 # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
      l1_loss_sum   /= (i+1)                                                                                 # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
  
      if DEBUG>2:
        print ( f"PRE_COMPRESS:   INFO:      test(): x2.size  = {MIKADO}{x2.size()}{RESET}" )
        print ( f"PRE_COMPRESS:   INFO:      test(): x2r.size = {MIKADO}{x2r.size()}{RESET}" )
  
      x2_nums    = x2 .cpu().detach().numpy().squeeze()
      x2r_nums   = x2r.cpu().detach().numpy().squeeze()
  
      if DEBUG>2:
        print ( f"PRE_COMPRESS:   INFO:      test(): x2_nums.shape  = {MIKADO}{x2_nums.shape}{RESET}" )
        print ( f"PRE_COMPRESS:   INFO:      test(): x2r_nums.shape = {MIKADO}{x2r_nums.shape}{RESET}" )
  
      x2r_nums[x2r_nums<0] = 0                                                                             # change negative values (which are impossible) to zero
      ratios    = np.around(np.absolute( (       (x2r_nums+.02) /  (x2_nums+.02)  ) ), decimals=2 )        # to avoid divide by zero error
      delta     = np.around(np.absolute( ( 0.98 + (x2r_nums   ) -  (x2_nums    )  ) ), decimals=2 )
      closeness = np.minimum( 1+np.abs(1-ratios), 1+np.abs(1-delta) )                                      # Figure of Merit. ratio isn't a good measure when X2 is close to zero
      closeness_ave = np.average( closeness )                                                              # average closeness for the current last batch in the epoch (different to the formall loss calculations above coz it's just over a representative batch)
  
    
      if ( (args.input_mode=='image') | ( args.input_mode=='rna') &  ( (epoch+1)%3==0  )  | (ae_loss2_sum<test_loss_min )  ):     # every nth batch, or if a new minimum was reached on this batch
  
        np.set_printoptions(linewidth=300)   
        np.set_printoptions(edgeitems=300)
         
        if args.just_test=='False':
          rows_to_display     = 3
          columns_to_display  = 49
          sample = np.random.randint( x2.shape[0] )
          if args.input_mode=='image':
            channel = np.random.randint( x2.shape[1] )
            print ( f"{DIM_WHITE}PRE_COMPRESS:   INFO:        test: original/reconstructed values for a randomly selected sample ({MIKADO}{sample}{RESET}{DIM_WHITE}) and channel ({MIKADO}{channel}{RESET}{DIM_WHITE}) for first {MIKADO}{rows_to_display}{RESET}{DIM_WHITE} rows x  {MIKADO}{columns_to_display:<}{RESET}{DIM_WHITE} columns (out of {MIKADO}{x2_nums.shape[2]}{RESET}{DIM_WHITE} x  {MIKADO}{x2_nums.shape[3]:<}{RESET})" )
            np.set_printoptions(formatter={'float': lambda x: "{:>5.2f}".format(x)})
            print (  f"x2        = \n{x2_nums  [ sample, channel, 0:rows_to_display, 0:columns_to_display ]  }",  flush=True    )
            print (  f"x2r       = \n{x2r_nums [ sample, channel, 0:rows_to_display, 0:columns_to_display ]  }",  flush=True    )
            np.set_printoptions(formatter={ 'float': lambda x: f"{BRIGHT_GREEN if abs(x-1)<0.01 else PALE_GREEN if abs(x-1)<0.05 else ORANGE if abs(x-1)<0.25 else GOLD if abs(x-1)<0.5 else BLEU if abs(x-1)<1.0 else DIM_WHITE}{x:>5.2f}{RESET}" } )
            print (  f"{CARRIBEAN_GREEN}closeness = \n{closeness[ sample, channel, 0:rows_to_display, 0:columns_to_display ]  }{RESET}", flush=True      )
          elif args.input_mode=='rna':
            genes_to_display  = 49
            print ( f"{DIM_WHITE}PRE_COMPRESS:   INFO:        test: original/reconstructed values for and first {MIKADO}{genes_to_display}{RESET}{DIM_WHITE} genes of a rrandomly selected sample ({MIKADO}{sample}{RESET})" )
            np.set_printoptions(formatter={'float': lambda x: "{:>5.2f}".format(x)})
            if DEBUG>55:
              print (  f"x2_nums  .shape      = {x2_nums.shape}" )
              print (  f"x2r_nums .shape      = {x2r_nums.shape}" )
              print (  f"closeness.shape      = {closeness.shape}" )
            print (  f"x2        = \r\033[12C{x2_nums  [ sample, 0:genes_to_display  ]}",  flush=True    )
            print (  f"x2r       = \r\033[12C{x2r_nums [ sample, 0:genes_to_display  ]}",  flush=True    )
            np.set_printoptions(formatter={'float': lambda w: f"{BRIGHT_GREEN if abs(w-1)<0.01 else PALE_GREEN if abs(w-1)<0.05 else ORANGE if abs(w-1)<0.25 else GOLD if abs(w-1)<0.5 else BLEU if abs(w-1)<1.0 else DIM_WHITE}{w:>5.2f}{RESET}"})
            print (  f"{BITTER_SWEET}closeness = \r\033[12C{closeness[ sample, 0:genes_to_display] }{RESET}", flush=True      )
        else:                                                                                             # in test mode, display every sample and every gene; in three different views
          genes_to_display=35
          print ( f"{DIM_WHITE}PRE_COMPRESS:   INFO:        test: original/reconstructed values for batch and first {MIKADO}{genes_to_display}{RESET} genes" )
          np.set_printoptions(formatter={'float': lambda x: "{:>7.2f}".format(x)})
          for sample in range( 0, batch_size ):
            np.set_printoptions(formatter={'float': lambda w: "{:>7.3f}".format(w)})
            print (  f"x2        = \r\033[12C{ x2_nums[ sample,  0:genes_to_display] }", flush=True    )
            print (  f"x2r       = \r\033[12C{x2r_nums[ sample,  0:genes_to_display] }", flush=True    )
            np.set_printoptions(formatter={'float': lambda w: f"{BRIGHT_GREEN if abs(w-1)<0.01 else PALE_GREEN if abs(w-1)<0.05 else ORANGE if abs(w-1)<0.25 else GOLD if abs(w-1)<0.5 else BLEU if abs(w-1)<1.0 else DIM_WHITE}{w:>7.3f}{RESET}"})     
            print (  f"ratios    = \r\033[12C{ratios   [ sample,  0:genes_to_display] }{RESET}",    flush=True    )
            print (  f"delta     = \r\033[12C{delta    [ sample,  0:genes_to_display] }{RESET}",     flush=True   )
            print (  f"{BITTER_SWEET}closeness = \r\033[12C{closeness[ sample,  0:genes_to_display] }{RESET}", flush=True       )
          np.set_printoptions(formatter={'float': lambda w: "{:>8.3f}".format(w)})
          print ( "\n\n" )
          for sample in range( 0, batch_size ):
            np.set_printoptions(formatter={'float': lambda w: "{:>7.3f}".format(w)})
            print (  f"x2        = \r\033[12C{ x2_nums[ sample,  0:genes_to_display] }", flush=True    )
            print (  f"x2r       = \r\033[12C{x2r_nums[ sample,  0:genes_to_display] }", flush=True    )
            np.set_printoptions(formatter={'float': lambda w: f"{BRIGHT_GREEN if abs(w-1)<0.01 else PALE_GREEN if abs(w-1)<0.05 else ORANGE if abs(w-1)<0.25 else GOLD if abs(w-1)<0.5 else BLEU if abs(w-1)<1.0 else DIM_WHITE}{w:>7.3f}{RESET}"})     
            print (  f"{BITTER_SWEET}closeness = \r\033[12C{closeness[ sample, 0:genes_to_display] }{RESET}", flush=True      )
          np.set_printoptions(formatter={'float': lambda w: "{:>8.3f}".format(w)})          
          print ( "\n\n" )
          for sample in range( 0, batch_size ):
            np.set_printoptions(formatter={'float': lambda w: "{:>7.3f}".format(w)})
            np.set_printoptions(formatter={'float': lambda w: f"{BRIGHT_GREEN if abs(w-1)<0.01 else PALE_GREEN if abs(w-1)<0.05 else ORANGE if abs(w-1)<0.25 else GOLD if abs(w-1)<0.5 else BLEU if abs(w-1)<1.0 else DIM_WHITE}{w:>7.3f}{RESET}"})     
            print (  f"{BITTER_SWEET}closeness = \r\033[12C{closeness[ sample,  0:genes_to_display] }{RESET}", flush=True      )
          np.set_printoptions(formatter={'float': lambda w: "{:>8.3f}".format(w)})
            
      del ratios
      del delta
      del closeness          

    if ( epoch%LOG_EVERY==1 ):
      if DEBUG>2:
          print ( f"PRE_COMPRESS:   INFO:      test(): x2.size  = {ARYLIDE}{x2.size()}{RESET}" )
          print ( f"PRE_COMPRESS:   INFO:      test(): x2r.size = {BITTER_SWEET}{x2r.size()}{RESET}" )
      if args.input_mode=='image':
          cfg.save_comparison  ( args.log_dir, x2, x2r, epoch,  is_image=True )
      else:
          cfg.save_comparison  ( args.log_dir, x2, x2r, epoch,  is_image=False )

    del x2

    if args.just_test=='True':
      # save the values
      pass
    
    del x2r
    
    if DEBUG>1:
      if gpu==0:                                                                                             # record output for one gpu only, or else tensorboard will get very confused
        writer.add_scalar( '1a_test_loss',      ae_loss2_sum,   epoch )
        writer.add_scalar( '1b_test_closeness', closeness_ave,  epoch )
        writer.add_scalar( '1c_test_loss_min',  test_loss_min,  epoch )
        if nn_type_rna=='TTVAE':
          writer.add_scalar( '1d_test_loss_recon_VAE', reconstruction_loss,  epoch  )
          writer.add_scalar( '1e_test_loss_kl_vae',    kl_loss,              epoch  )
  
    del ae_loss2
    del l1_loss
    
    
    if DEBUG>9:
      print ( f"{DIM_WHITE}PRE_COMPRESS:   INFO:      test(): test_loss_min  = {MIKADO}{test_loss_min:5.2f}{RESET}" )
      print ( f"{DIM_WHITE}PRE_COMPRESS:   INFO:      test(): ae_loss2_sum   = {MIKADO}{ae_loss2_sum:5.2f}{RESET}" )

    if  args.just_test!='True':                                                                            # only save models in training mode (the validation testing step of training mode, not to be confused with 'just_test')
      if ae_loss2_sum < test_loss_min:
        test_loss_min = ae_loss2_sum
        if  (epoch>1) | (epoch==args.n_epochs-1):                                                          # wait till a reasonable number of epochs have completed before saving mode, else it will be saving all the time early on
          if gpu==0:
            save_model( args.log_dir, model)                                                               # save model with the lowest cost to date. Over-write earlier least cost model, if one exists.
    else:
      # in test mode we need to save the z values during the one and only run
      pass
    
    torch.cuda.empty_cache()
    
    return embeddings_accum, labels_accum, ae_loss2_sum, l1_loss_sum, test_loss_min
    
    
    

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------

def segment_cases( pct_test ):

  # (1A) analyse dataset directory

  if args.use_unfiltered_data=='True':
    rna_suffix = args.rna_file_suffix[1:]
  else:
    rna_suffix = args.rna_file_reduced_suffix
    
  cumulative_image_file_count = 0
  cumulative_png_file_count   = 0
  cumulative_rna_file_count   = 0
  cumulative_other_file_count = 0
  dir_count                   = 0
  
  for dir_path, dirs, files in os.walk( args.data_dir ):                                                        # each iteration takes us to a new directory under data_dir

    if not (dir_path==args.data_dir):                                                                           # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data      
      
      dir_count += 1
      image_file_count   = 0
      rna_file_count     = 0
      png_file_count     = 0
      other_file_count   = 0

      for f in sorted( files ):
       
        if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
          image_file_count            +=1
          cumulative_image_file_count +=1
        elif  ( f.endswith( 'png' ) ):
          png_file_count              +=1
          cumulative_png_file_count   +=1
        elif  ( f.endswith( rna_suffix ) ):
          rna_file_count              +=1
          cumulative_rna_file_count   +=1
        else:
          other_file_count            +=1
          cumulative_other_file_count +=1
        
      if DEBUG>77:
        if ( ( rna_file_count>1 ) | ( image_file_count>1 ) ): 
          print( f"PRE_COMPRESS:      INFO:    \033[58Cdirectory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq file(s) and {MIKADO}{image_file_count:<2d}{RESET} image files(s) and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )
          time.sleep(0.5)       
        else:
          print( f"PRE_COMPRESS:      INFO:    directory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq files, {MIKADO}{image_file_count:<2d}{RESET} image files and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )

  if DEBUG>0:
    print( f"PRE_COMPRESS:     INFO:   directories count  =  {MIKADO}{dir_count:<6d}{RESET}",                   flush=True  )
    print( f"PRE_COMPRESS:     INFO:   image file  count  =  {MIKADO}{cumulative_image_file_count:<6d}{RESET}", flush=True  )
    print( f"PRE_COMPRESS:     INFO:   tile  file  count  =  {MIKADO}{cumulative_png_file_count:<6d}{RESET}",   flush=True  )
    print( f"PRE_COMPRESS:     INFO:   rna   file  count  =  {MIKADO}{cumulative_rna_file_count:<6d}{RESET}",   flush=True  )
    print( f"PRE_COMPRESS:     INFO:   other file  count  =  {MIKADO}{cumulative_other_file_count:<6d}{RESET}", flush=True  )


  # (1B) Locate and flag directories that contain BOTH an image and and rna-seq files

  if args.divide_cases=='True':

    if DEBUG>0:
      print ( f"{ORANGE}PRE_COMPRESS:     INFO: divide_cases  ( {CYAN}-v{RESET}{ORANGE} option ) = {MIKADO}{args.divide_cases}{RESET}{ORANGE}, so will divide cases and set applicable flag files{RESET}",    flush=True )

    dirs_which_have_matched_image_rna_files    = 0
  
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>888:  
        print( f"{DIM_WHITE}PRE_COMPRESS:     INFO:  now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) is skipped because it only contains sub-directories, not data
                
        dir_has_rna_data    = False
        dir_also_has_image  = False
  
        for f in sorted( files ):
          if  ( f.endswith( args.rna_file_suffix[1:]) ):
            dir_has_rna_data=True
            rna_file  = f
          if ( ( f.endswith( 'svs' ))  |  ( f.endswith( 'tif' ) )  |  ( f.endswith( 'tiff' ) )  ):
            dir_also_has_image=True
            fqn = f"{dir_path}/HAS_IMAGE_FLAG"
            with open(fqn, 'w') as f:
              f.write( f"this directory contains image data" )
            f.close                           
        
        if dir_has_rna_data & dir_also_has_image:
          
          if DEBUG>555:
            print ( f"{WHITE}PRE_COMPRESS:     INFO:  case {PINK}{args.data_dir}/{os.path.basename(dir_path)}{RESET} \r\033[100C has both matched and rna files (listed above) (count= {MIKADO}{dirs_which_have_matched_image_rna_files+1}{RESET})",  flush=True )
          fqn = f"{dir_path}/HAS_MATCHED_IMAGE_RNA_FLAG"
          with open(fqn, 'w') as f:
            f.write( f"this directory contains matched image and rna-seq data" )
          f.close  
          dirs_which_have_matched_image_rna_files+=1
  
    if DEBUG>0:
      print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  number of cases (directories) which contain BOTH matched and rna files = {MIKADO}{dirs_which_have_matched_image_rna_files}{RESET}",  flush=True )


  
  
  
  
    # (1C) Segment the cases as follows:
    #      (1Ca)  DESIGNATED_MULTIMODE_CASE_FLAG ............... all MATCHED cases, used for multimode testing only. The amount of cases to be so flagged is given by config parameter CASES_RESERVED_FOR_IMAGE_RNA
    #      (1Cb)  DESIGNATED_UNIMODE_CASE_FLAG ................. all MATCHED cases minus designated multimode cases; used for unimode training (generated embeddings are used for multimode training)
    #      (1Cc)  NOT_A_MULTIMODE_CASE_FLAG .................... ALL cases minus multimode cases, and don't have to be matched. Constitute the largest possible set of cases for use in unimode image or rna training and testing (including as a prelude to multimode testing with the designated multimode test set where comparing unimode to multimode performance (which requires the use of the same cases for unimode and multimode) is not of interest
    #      (1Cd ) NOT_A_MULTIMODE_CASE____IMAGE_FLAG ........... ALL cases minus multimode cases which contain an image -     used for unimode training ) constitute the largest possible (but umatched) set of cases for use in unimode image training (including as a prelude to multimode testing with the designated multimode test set, where comparing unimode to multimode performance (the latter requires the use of the SAME cases for unimode and multimode) is not of interest
    #      (1Ce ) NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG ...... ALL cases minus multimode cases which contain an image - reserved for inimode testing  ) same criteria as NOT_A_MULTIMODE_CASE____IMAGE_FLAG, but reserved for testing


    #        - yes it's confusing. sorry!

    if DEBUG>0:
      print ( f"{WHITE}PRE_COMPRESS:     INFO:     segment_cases():  about to segment cases by placing flags according to the following logic:         {CAMEL}DESIGNATED_UNIMODE_CASE_FLAG{RESET}{DULL_WHITE}   XOR {RESET}{ASPARAGUS} DESIGNATED_MULTIMODE_CASE_FLAG{RESET}",  flush=True )
      print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  config parameter '{CYAN}CASES_RESERVED_FOR_IMAGE_RNA{RESET}{DULL_WHITE}' = {MIKADO}{args.cases_reserved_for_image_rna}{RESET}{DULL_WHITE}, therefore {MIKADO}{args.cases_reserved_for_image_rna}{RESET}{DULL_WHITE} cases selected at random will be flagged with the    {ASPARAGUS}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}{DULL_WHITE} thereby exclusively setting them aside for multimode testing",  flush=True )


    # (1Ce) designate MULTIMODE cases.  Infinite loop with a break condition (necessary to give every case an equal chance of being randonly selected for inclusion in the MULTIMODE case set)
    
    directories_considered_count     = 0
    designated_multimode_case_count  = 0
    
    while True:
      
      for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):                                                      # select the multimode cases ...
    
        if DEBUG>55:  
          print( f"{DIM_WHITE}PRE_COMPRESS:     INFO:  now considering case {ARYLIDE}{os.path.basename(dir_path)}{RESET}{DIM_WHITE} as a multimode case  " ) 
    
        
        if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
  
          if DEBUG>55:
            print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  case   \r\033[60C{RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[120C has both image and rna files\r\033[140C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
            
          try:
            fqn = f"{dir_path}/HAS_MATCHED_IMAGE_RNA_FLAG"        
            f = open( fqn, 'r' )
            if DEBUG>55:
              print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files (listed above)  \r\033[160C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
              print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  designated_multimode_case_count          = {AMETHYST}{designated_multimode_case_count}{RESET}",          flush=True )
              print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  dirs_which_have_matched_image_rna_files  = {AMETHYST}{dirs_which_have_matched_image_rna_files}{RESET}",  flush=True )
              print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  cases_reserved_for_image_rna             = {AMETHYST}{args.cases_reserved_for_image_rna}{RESET}",        flush=True )
            selector = random.randint(0,500)                                                               # the high number has to be larger than the total number of matched cases to give every case a chance of being included 
            if ( selector==22 ) & ( designated_multimode_case_count<args.cases_reserved_for_image_rna ):   # used 22 but it could be any number

              fqn = f"{dir_path}/DESIGNATED_MULTIMODE_CASE_FLAG"         
              try:
                with open(fqn, 'r') as f:                                                                  # have to check that the case (directory) was not already flagged as a multimode cases, else it will do it again and think it was an additional case, therebody creating one (or more) fewer cases
                  pass
              except Exception:
                fqn = f"{dir_path}/DESIGNATED_MULTIMODE_CASE_FLAG"         
                try:
                  with open(fqn, 'w') as f:
                    f.write( f"this case is designated as a multimode case" )
                    designated_multimode_case_count+=1
                    f.close
                  if DEBUG>2:
                    print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:     segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[122C has been randomly flagged as '{ASPARAGUS}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}{PALE_GREEN}'  \r\033[204C (count= {MIKADO}{designated_multimode_case_count}{RESET}{PALE_GREEN})",  flush=True )
                except Exception:
                  print( f"{RED}PRE_COMPRESS:   FATAL:  could not create '{CYAN}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}' file" )
                  time.sleep(10)
                  sys.exit(0)
  
          except Exception:
            if DEBUG>55:
              print ( f"{RED}PRE_COMPRESS:   not a matched case" )
    
      directories_considered_count+=1
      if DEBUG>555:
        print ( f"c={c}" )      

      if designated_multimode_case_count== args.cases_reserved_for_image_rna:
        if DEBUG>55:
          print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  designated_multimode_case_count          = {AMETHYST}{designated_multimode_case_count}{RESET}",          flush=True )
          print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  cases_reserved_for_image_rna             = {AMETHYST}{args.cases_reserved_for_image_rna}{RESET}",             flush=True )
        break


    # (1Cb) designate UNIMODE cases. Go through all MATCHED directories one time. Flag any MATCHED case other than those flagged as DESIGNATED_MULTIMODE_CASE_FLAG case at 1Ci above with the DESIGNATED_UNIMODE_CASE_FLAG
    
    designated_unimode_case_count    = 0

    for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):                                   # ... designate every matched case (HAS_MATCHED_IMAGE_RNA_FLAG) other than those flagged as DESIGNATED_MULTIMODE_CASE_FLAG above to be a unimode case
  
      if DEBUG>1:  
        print( f"{DIM_WHITE}PRE_COMPRESS:     INFO:  now considering case (directory) as a unimode case {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                    # the top level directory (dataset) is skipped because it only contains sub-directories, not data

        if DEBUG>55:
          print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files (listed above)  \r\033[160C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
  
          
        try:
          fqn = f"{dir_path}/HAS_MATCHED_IMAGE_RNA_FLAG"
          f = open( fqn, 'r' )

          try:
            fqn = f"{dir_path}/DESIGNATED_MULTIMODE_CASE_FLAG"                                             # then we designated it to be a MULTIMODE case above, so ignore 
            f = open( fqn, 'r' )
          except Exception:                                                                               # these are the ones we want
            if DEBUG>555:
              print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files and has not already been designated as a mutimode case  \r\033[200C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
              print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  designated_unimode_case_count            = {AMETHYST}{designated_unimode_case_count}{RESET}",            flush=True )
            if ( ( designated_unimode_case_count + designated_multimode_case_count ) <= dirs_which_have_matched_image_rna_files ):                 # if we don't yet have enough designated multimode cases (and hence designations in total)
              fqn = f"{dir_path}/DESIGNATED_UNIMODE_CASE_FLAG"            
              with open(fqn, 'w') as f:
                f.write( f"this case is designated as a unimode case" )
              f.close
              designated_unimode_case_count+=1
              if DEBUG>44:
                print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{DULL_YELLOW} \r\033[122C has been randomly designated as a   unimode case  \r\033[204C (count= {MIKADO}{designated_unimode_case_count}{RESET}{DULL_WHITE})",  flush=True )


        except Exception:
          if DEBUG>555:
            print ( "not a multimode case" )
      
      
    # (1Cc) designate the 'NOT MULTIMODE' cases. Go through all directories one time. Flag ANY case (whether matched or not) other than those flagged as DESIGNATED_MULTIMODE_CASE_FLAG case at 1Ci above with the NOT_A_MULTIMODE_CASE_FLAG
    
    if DEBUG>0:
      print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  about to further segment cases by placing flags according to the following logic: {RESET}{ASPARAGUS}DESIGNATED_MULTIMODE_CASE_FLAG {RESET}{DULL_WHITE}XOR{RESET}{PALE_GREEN}  NOT_A_MULTIMODE_CASE_FLAG{RESET}",  flush=True )
    
    not_a_multimode_case_count=0
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>55:  
        print( f"{DIM_WHITE}PRE_COMPRESS:      INFO:   now processing case (directory) {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data  

        for f in sorted( files ):          
                    
          try:
            fqn = f"{dir_path}/DESIGNATED_MULTIMODE_CASE_FLAG"        
            f = open( fqn, 'r' )
            if DEBUG>555:
              print ( f"{RED}PRE_COMPRESS:      INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{RED} \r\033[100C is a multimode case. Skipping",  flush=True )
            break
          except Exception:
            try:
              fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE_FLAG"        
              f = open( fqn, 'r' )
              if DEBUG>555:
                print ( f"{RED}PRE_COMPRESS:      INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{RED} \r\033[100C is in a directory containing the NOT_A_MULTIMODE_CASE_FLAG. Skipping",  flush=True )
              break
            except Exception:
              if DEBUG>44:
                print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[122C has been flagged with the  {ASPARAGUS}NOT_A_MULTIMODE_CASE_FLAG{RESET}  \r\033[204C (count= {MIKADO}{not_a_multimode_case_count+1}{RESET})",  flush=True )
              fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE_FLAG"            
              with open(fqn, 'w') as f:
                f.write( f"this case is not a designated multimode case" )
              f.close
              not_a_multimode_case_count+=1                                                                # only segment_cases knows the value of not_a_multimode_case_count, and we need in generate(), so we return it
                                                                  

    # (1Cd) Designate those IMAGE cases which are not also MULTIMODE cases. Go through directories one time. Flag NOT_A_MULTIMODE_CASE_FLAG which are also image cases as NOT_A_MULTIMODE_CASE____IMAGE_FLAG
    
    if DEBUG>3:
      print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  about to designate '{ARYLIDE}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{DULL_WHITE}' cases{RESET}",  flush=True )  
    
    directories_considered_count                    = 0
    designated_not_a_multimode_case____image_count  = 0
    
    for dir_path, dirs, files in os.walk( args.data_dir ):
  
      if DEBUG>55:  
        print( f"{DIM_WHITE}PRE_COMPRESS:      INFO:   now processing case (directory) {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir): 
                    
        try:
          fqn = f"{dir_path}/HAS_IMAGE_FLAG"        
          f = open( fqn, 'r' )
          if DEBUG>44:
            print ( f"{GREEN}PRE_COMPRESS:      INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{GREEN}' \r\033[122C is an image case",  flush=True )
          try:
            fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE_FLAG"        
            f = open( fqn, 'r' )
            if DEBUG>2:
              print ( f"{GREEN}PRE_COMPRESS:      INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{GREEN} \r\033[122C is in a directory containing the NOT_A_MULTIMODE_CASE_FLAG",  flush=True )
            fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE____IMAGE_FLAG"            
            with open(fqn, 'w') as f:
              f.write( f"this case is a NOT_A_MULTIMODE_CASE____IMAGE_FLAG case" )
            f.close
            if DEBUG>22:
              print ( f"{PALE_GREEN}PRE_COMPRESS:      INFO:       segment_cases():  case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_GREEN}' \r\033[122C has been flagged with the NOT_A_MULTIMODE_CASE____IMAGE_FLAG  \r\033[204C (count= {MIKADO}{designated_not_a_multimode_case____image_count+1}{RESET})",  flush=True )
            designated_not_a_multimode_case____image_count+=1                                                                # only segment_cases knows the value of not_a_multimode_case_count, and we need in generate(), so we return it
          except Exception:
            if DEBUG>44:
              print ( f"{RED}PRE_COMPRESS:      INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{RED}' \r\033[122C  is not a NOT_A_MULTIMODE_CASE_FLAG case - - skipping{RESET}",  flush=True )
        except Exception:
          if DEBUG>44:
            print ( f"{PALE_RED}PRE_COMPRESS:      INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[122C is not an image case - - skipping{RESET}",  flush=True )                                                                    
        

    # (1Ce) Designate 'NOT MULTIMODE IMAGE TEST' cases. Go through directories one time. Flag PCT_TEST % of the NOT_A_MULTIMODE_CASE_FLAG cases as NOT_A_MULTIMODE_CASE_IMAGE_TEST_FLAG
    #        These cases are used for unimode image testing. Necessary to strictly separated cases in this manner for image mode so that tiles from a single image do not end up in both the training and test sets   
    #        In image mode, tiles allocated to the training set cann't come from an image which is also contributing tiles to the test set. Ditto the reverse.
    #        This issue does not affect rna mode, where there is only one artefact per case. I.e. when input mode is rna, any rna sample can be allocated to either the training set or test set
    #
    #        Strategy: res-designate an appropriate number of the 'NOT_A_MULTIMODE_CASE____IMAGE_FLAG' to be 'NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG' (delete the first flag)
  

    cases_to_designate = int(pct_test * designated_not_a_multimode_case____image_count)
        
    if DEBUG>0:
      print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  about to randomly designate {ARYLIDE}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{DULL_WHITE} cases as reserved image test cases by placing the flag {ARYLIDE}NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG{RESET}{DULL_WHITE} in their case directories",  flush=True )

      print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  cases_to_designate = int({CYAN}PCT_TEST{RESET}{DULL_WHITE} {MIKADO}{pct_test*100:4.2f}%{RESET}{DULL_WHITE} * {CYAN}designated_not_a_multimode_case____image_count{RESET}{DULL_WHITE} {MIKADO}{designated_not_a_multimode_case____image_count}{RESET}{DULL_WHITE}) = {MIKADO}{cases_to_designate}",  flush=True )
    
    directories_considered_count                         = 0
    designated_not_a_multimode_case____image_test_count  = 0
    
    while True:
      
      for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):
    
        if DEBUG>55:  
          print( f"{DIM_WHITE}PRE_COMPRESS:     INFO:  now considering case {ARYLIDE}{os.path.basename(dir_path)}{RESET}{DIM_WHITE} \r\033[130C as a candidate NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG case  " ) 
            
        if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
          try:
            fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE____IMAGE_FLAG"    
            f = open( fqn, 'r' )                
            if DEBUG>66:
              print ( f"{PALE_GREEN}PRE_COMPRESS:      INFO:   case   \r\033[55C'{RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[130C is a     {CYAN}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{PALE_GREEN} case{RESET}",  flush=True )
            selector = random.randint(0,500)                                                                    # the high number has to be larger than the total number of not a multimode cases to give every case a chance of being included 
            if ( selector==22 ) & ( designated_not_a_multimode_case____image_test_count<cases_to_designate ):   # used 22 but it could be any number
              fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG"         
              try:
                with open(fqn, 'w') as f:
                  f.write( f"this case is designated as a NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG case" )
                  designated_not_a_multimode_case____image_test_count+=1
                  f.close
                  os.remove ( f"{dir_path}/NOT_A_MULTIMODE_CASE____IMAGE_FLAG" )
                if DEBUG>66:
                  print ( f"{BLEU}PRE_COMPRESS:      INFO:      segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{BLEU} \r\033[130C has been randomly designated as a NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG case  \r\033[204C (count= {MIKADO}{designated_not_a_multimode_case____image_test_count}{BLEU}{RESET})",  flush=True )
              except Exception:
                print( f"{RED}PRE_COMPRESS:  FATAL:  either could not create '{CYAN}NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG{RESET}' file or delete the '{CYAN}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}' " )  
                time.sleep(10)
                sys.exit(0)              
          except Exception:
            if DEBUG>66:
              print ( f"{RED}PRE_COMPRESS:      INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[130C is not a {CYAN}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{RED} case - - skipping{RESET}",  flush=True )
    
      directories_considered_count+=1
     
      if designated_not_a_multimode_case____image_test_count == cases_to_designate:
        if DEBUG>55:
          print ( f"{PALE_GREEN}PRE_COMPRESS:     INFO:  designated_not_a_multimode_case____image_test_count  = {AMETHYST}{designated_not_a_multimode_case____image_test_count}{RESET}",          flush=True )
        break


    designated_not_a_multimode_case____image_count = designated_not_a_multimode_case____image_count - designated_not_a_multimode_case____image_test_count

    if DEBUG>0:
        print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  HAS_MATCHED_IMAGE_RNA_FLAG ................ flags placed = {MIKADO}{dirs_which_have_matched_image_rna_files}{RESET}",              flush=True )
        print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  DESIGNATED_MULTIMODE_CASE_FLAG ............ flags placed = {MIKADO}{designated_multimode_case_count}{RESET}",                      flush=True )
        print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  DESIGNATED_UNIMODE_CASE_FLAG .............. flags placed = {MIKADO}{designated_unimode_case_count}{RESET}",                        flush=True )
        print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  NOT_A_MULTIMODE_CASE_FLAG ................. flags placed = {MIKADO}{not_a_multimode_case_count}{RESET}",                           flush=True )
        print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  NOT_A_MULTIMODE_CASE____IMAGE_FLAG ........ flags placed = {MIKADO}{designated_not_a_multimode_case____image_count}{RESET}",       flush=True )
        print ( f"{DULL_WHITE}PRE_COMPRESS:     INFO:     segment_cases():  NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG ... flags placed = {MIKADO}{designated_not_a_multimode_case____image_test_count}{RESET}",  flush=True )


    
    return designated_multimode_case_count, designated_unimode_case_count, not_a_multimode_case_count, designated_not_a_multimode_case____image_count, designated_not_a_multimode_case____image_test_count


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
      print ( f"PRE_COMPRESS:     INFO:     newline():             xmin                                    = {xmin}"                            )
      print ( f"PRE_COMPRESS:     INFO:     newline():             xmax                                    = {xmax}"                            )
      print ( f"PRE_COMPRESS:     INFO:     newline():             ymin                                    = {ymin}"                            )
      print ( f"PRE_COMPRESS:     INFO:     newline():             ymax                                    = {ymax}"                            )

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l


# ------------------------------------------------------------------------------


def l1_penalty(model, l1_coef):
    """Compute L1 penalty. For implementation details, see:

    https://discuss.pytorch.org/t/simple-l2-regularization/139
    """
    reg_loss = 0
    for param in model.pcca.parameters_('y2'):
        reg_loss += torch.norm(param, 1)
    return l1_coef * reg_loss

# ------------------------------------------------------------------------------

def save_samples(directory, model, test_loader, cfg, epoch):
    """Save samples from test set.
    """
    with torch.no_grad():
        n  = len(test_loader.sampler.indices)
        x2_batch = torch.Tensor(n, cfg.N_GENES)
        labels   = []

        for i in range(n):

            j = test_loader.sampler.indices[i]

            x2     = test_loader.dataset[j]
            lab    = test_loader.dataset.img_labels[j]
            x2_batch[i] = x2
            labels.append(lab)

        x2_batch = x2_batch.to(gpu)

        cfg.save_samples(directory, model, epoch, x2_batch, labels)

# ------------------------------------------------------------------------------

def save_model(log_dir, model):
  
    """Save PyTorch model state dictionary
    """
     
    fpath = '%s/lowest_loss_ae_model.pt' % log_dir
    if DEBUG>0:   
      print( f"PRE_COMPRESS:   INFO:      save_model(){DULL_YELLOW}{ITALICS}: new lowest test loss on this epoch... saving model state dictionary to {fpath}{RESET}" )       
    model_state = model.state_dict()
    torch.save(model_state, fpath)

# ------------------------------------------------------------------------------
    
def delete_selected( root, extension ):

  walker = os.walk( root, topdown=True )

  for root, dirs, files in walker:

    for f in files:
      fqf = root + '/' + f
      if DEBUG>99:
        print( f"PRE_COMPRESS:   INFO:   examining file:   '\r\033[43C\033[36;1m{fqf}\033[m' \r\033[180C with extension '\033[36;1m{extension}\033[m'" )
      if ( f.endswith( extension ) ): 
        try:
          if DEBUG>99:
            print( f"PRE_COMPRESS:   INFO:   will delete file  '\r\033[43C{MIKADO}{fqf}{RESET}'" )
          os.remove( fqf )
        except:
          pass

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--pretrain',                                                      type=str,   default='False'                            )                                
    p.add_argument('--skip_tiling',                                                   type=str,   default='False'                            )                                
    p.add_argument('--skip_generation',                                               type=str,   default='False'                            )                                
    p.add_argument('--log_dir',                                                       type=str,   default='data/dlbcl_image/logs'            )                
    p.add_argument('--base_dir',                                                      type=str,   default='/home/peter/git/pipeline'         )             # NOT CURRENTLY USED
    p.add_argument('--data_dir',                                                      type=str,   default='/home/peter/git/pipeline/dataset' )     
    p.add_argument('--save_model_name',                                               type=str,   default='model.pt'                         )                             
    p.add_argument('--save_model_every',                                              type=int,   default=10                                 )                                     
    p.add_argument('--rna_file_name',                                                 type=str,   default='rna.npy'                          )                              
    p.add_argument('--rna_file_suffix',                                               type=str,   default='*FPKM-UQ.txt'                     )                        
    p.add_argument('--embedding_file_suffix_rna',                                     type=str                                               )                        
    p.add_argument('--embedding_file_suffix_image',                                   type=str                                               )                        
    p.add_argument('--embedding_file_suffix_image_rna',                               type=str                                               )                        
    p.add_argument('--rna_file_reduced_suffix',                                       type=str,   default='_reduced'                         )                             
    p.add_argument('--use_unfiltered_data',                                           type=str,   default='True'                             )                                
    p.add_argument('--class_numpy_file_name',                                         type=str,   default='class.npy'                        )                            
    p.add_argument('--wall_time',                                                     type=int,    default=24                                )
    p.add_argument('--seed',                                                          type=int,    default=0                                 )
    p.add_argument('--nn_mode',                                                       type=str,    default='pre_compress'                    )
    p.add_argument('--use_same_seed',                                                 type=str,    default='False'                           )
    p.add_argument('--nn_type_img',                                       nargs="+",  type=str,    default='VGG11'                           )
    p.add_argument('--nn_type_rna',                                       nargs="+",  type=str,    default='DENSE'                           )
    p.add_argument('--hidden_layer_encoder_topology', '--nargs-int-type', nargs='*',  type=int,                                              )                             
    p.add_argument('--encoder_activation',                                nargs="+",  type=str,    default='sigmoid'                         )                              
    p.add_argument('--nn_dense_dropout_1',                                nargs="+",  type=float,  default=0.0                               )                                    
    p.add_argument('--nn_dense_dropout_2',                                nargs="+",  type=float,  default=0.0                               )                                    
    p.add_argument('--dataset',                                                       type=str                                               )
    p.add_argument('--cases',                                                         type=str,    default='ALL_ELIGIBLE_CASES'              )
    p.add_argument('--divide_cases',                                                  type=str,    default='False'                           )
    p.add_argument('--cases_reserved_for_image_rna',                                  type=int                                               )
    p.add_argument('--data_source',                                                   type=str                                               )
    p.add_argument('--global_data',                                                   type=str                                               )
    p.add_argument('--mapping_file_name',                                             type=str,    default='mapping_file'                    )
    p.add_argument('--target_genes_reference_file',                                   type=str                                               )
    p.add_argument('--input_mode',                                                    type=str,    default='NONE'                            )
    p.add_argument('--multimode',                                                     type=str,    default='NONE'                            )
    p.add_argument('--n_samples',                                         nargs="+",  type=int,    default=101                               )                                    
    p.add_argument('--n_tiles',                                           nargs="+",  type=int,    default=50                                )                                    
    p.add_argument('--n_tests',                                                       type=int,    default="16"                              )       
    p.add_argument('--highest_class_number',                                          type=int,    default=999                               )                                                             
    p.add_argument('--supergrid_size',                                                type=int,    default=1                                 )                                      
    p.add_argument('--patch_points_to_sample',                                        type=int,    default=1000                              )                                   
    p.add_argument('--tile_size',                                         nargs="+",  type=int,    default=128                               )                                    
    p.add_argument('--gene_data_norm',                                    nargs="+",  type=str,    default='NONE'                            )                                 
    p.add_argument('--gene_data_transform',                               nargs="+",  type=str,    default='NONE'                            )
    p.add_argument('--n_genes',                                                       type=int,    default=60483                             )                                   
    p.add_argument('--remove_unexpressed_genes',                                      type=str,    default='True'                            )                               
    p.add_argument('--remove_low_expression_genes',                                   type=str,   default='True'                             )                                
    p.add_argument('--low_expression_threshold',                                      type=float, default=0                                  )                                
    p.add_argument('--batch_size',                                        nargs="+",  type=int,   default=64                                 )                                     
    p.add_argument('--learning_rate',                                     nargs="+",  type=float, default=.00082                             )                                 
    p.add_argument('--n_epochs',                                                      type=int,   default=10                                 )
    p.add_argument('--n_iterations',                                                  type=int,   default=251                                )
    p.add_argument('--pct_test',                                          nargs="+",  type=float, default=0.2                                )
    p.add_argument('--final_test_batch_size',                                         type=int,   default=1000                               )                                   
    p.add_argument('--lr',                                                nargs="+",  type=float, default=0.0001                             )
    p.add_argument('--latent_dim',                                                    type=int,   default=7                                  )
    p.add_argument('--l1_coef',                                                       type=float, default=0.1                                )
    p.add_argument('--em_iters',                                                      type=int,   default=1                                  )
    p.add_argument('--clip',                                                          type=float, default=1                                  )
    p.add_argument('--max_consecutive_losses',                                        type=int,   default=7771                               )
    p.add_argument('--optimizer',                                         nargs="+",  type=str,   default='ADAM'                             )
    p.add_argument('--label_swap_perunit',                                            type=float, default=0.0                                )                                    
    p.add_argument('--make_grey_perunit',                                             type=float, default=0.0                                ) 
    p.add_argument('--peer_noise_perunit',                                            type=float, default=0.0                                ) 
    p.add_argument('--regenerate',                                                    type=str,   default='True'                             )
    p.add_argument('--just_profile',                                                  type=str,   default='False'                            )                        
    p.add_argument('--just_test',                                                     type=str,   default='False'                            )                        
    p.add_argument('--rand_tiles',                                                    type=str,   default='True'                             )                         
    p.add_argument('--points_to_sample',                                              type=int,   default=100                                )                            
    p.add_argument('--min_uniques',                                                   type=int,   default=0                                  )                              
    p.add_argument('--min_tile_sd',                                                   type=float, default=3                                  )                              
    p.add_argument('--greyness',                                                      type=int,   default=0                                  )                              
    p.add_argument('--stain_norm',                                        nargs="+",  type=str,   default='NONE'                             )                         
    p.add_argument('--stain_norm_target',                                             type=str,   default='NONE'                             )                         
    p.add_argument('--cancer_type',                                                   type=str,   default='NONE'                             )                 
    p.add_argument('--cancer_type_long',                                              type=str,   default='NONE'                             )                 
    p.add_argument('--class_names',                                       nargs="*",  type=str,   default='NONE'                             )                 
    p.add_argument('--long_class_names',                                  nargs="+",  type=str,   default='NONE'                             ) 
    p.add_argument('--class_colours',                                     nargs="*"                                                          )                 
    p.add_argument('--colour_map',                                                    type=str,   default='tab10'                            )    
    p.add_argument('--target_tile_coords',                                nargs=2,    type=int,    default=[2000,2000]                       )                 
    p.add_argument('--zoom_out_prob',                                     nargs="*",  type=float,                                            )                 
    p.add_argument('--zoom_out_mags',                                     nargs="*",  type=int,                                              )                 

    p.add_argument('--a_d_use_cupy',                                                  type=str,   default='True'                             )                    
    p.add_argument('--cov_threshold',                                                 type=float, default=8.0                                )                    
    p.add_argument('--cov_uq_threshold',                                              type=float, default=0.0                                )                    
    p.add_argument('--cutoff_percentile',                                             type=float, default=0.05                               )                    
 
    p.add_argument('--figure_width',                                                  type=float, default=16                                 )                                  
    p.add_argument('--figure_height',                                                 type=float, default=16                                 )
    p.add_argument('--annotated_tiles',                                               type=str,   default='True'                             )
    p.add_argument('--scattergram',                                                   type=str,   default='True'                             )
    p.add_argument('--box_plot',                                                      type=str,   default='True'                             )
    p.add_argument('--minimum_job_size',                                              type=float, default=5                                  )
    p.add_argument('--probs_matrix',                                                  type=str,   default='True'                             )
    p.add_argument('--probs_matrix_interpolation',                                    type=str,   default='none'                             )
    p.add_argument('--show_patch_images',                                             type=str,   default='True'                             )    
    p.add_argument('--show_rows',                                                     type=int,   default=500                                )                            
    p.add_argument('--show_cols',                                                     type=int,   default=100                                ) 
    p.add_argument('--bar_chart_x_labels',                                            type=str,   default='rna_case_id'                      )
    p.add_argument('--bar_chart_show_all',                                            type=str,   default='True'                             )
    p.add_argument('--bar_chart_sort_hi_lo',                                          type=str,   default='True'                             )
    p.add_argument('-ddp', '--ddp',                                                   type=str,   default='False'                            )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    p.add_argument('-n', '--nodes',                                                   type=int,   default=1,  metavar='N'                    )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    p.add_argument('-g', '--gpus',                                                    type=int,   default=1,  help='number of gpus per node' )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    p.add_argument('-nr', '--nr',                                                     type=int,   default=0,  help='ranking within node'     )  # only supported for 'NN_MODE=pre_compress' ATM (auto-encoder front-end)
    
    p.add_argument('--hidden_layer_neurons',                              nargs="+",  type=int,    default=2000                              )     
    p.add_argument('--gene_embed_dim',                                    nargs="+",  type=int,    default=1000                              )    
    
    p.add_argument('--use_autoencoder_output',                                        type=str,   default='False'                            ) # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values)
    p.add_argument('--ae_add_noise',                                                  type=str,   default='False'                             )

    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 12
    args.pin_memory = torch.cuda.is_available()

    main(args)
