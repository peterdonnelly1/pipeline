
"""=============================================================================
Code to support Dimensionality Reduction Mode
============================================================================="""

import argparse
import time
import numpy as np
import os

import torch
from   torch                   import optim
import torch.utils.data
import torch.multiprocessing   as mp
import torch.distributed       as dist
from   torch.nn                import functional as F
from   torch.nn                import MSELoss, BCELoss
from   torch.nn.utils          import clip_grad_norm_
from   torch.nn.parallel       import DistributedDataParallel as DDP
from   torch.utils.tensorboard import SummaryWriter

import torchvision
from   torchvision             import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from   matplotlib.colors import ListedColormap
from   matplotlib import cm
from   matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

from tiler_scheduler import *
from tiler_threader import *
from tiler_set_target import *
from tiler import *

from   schedulers import *
from   data import loader
from   data.pre_compress.generate       import generate

from   itertools                       import product, permutations
from   PIL                             import Image

import cuda
from   models import PRECOMPRESS
from   models.ttvae import vae_loss
import pprint

torch.set_printoptions(edgeitems=6)
torch.set_printoptions(linewidth=250)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
  
  
np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=200)

torch.backends.cudnn.enabled     = True                                                                     # for CUDA memory optimizations
# ------------------------------------------------------------------------------

LOG_EVERY        = 10
SAVE_MODEL_EVERY = 100

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
BRIGHT_GREEN='\033[38;2;102;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
RESET='\033[m'

UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1

device = cuda.device()




# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main(args):

  os.system("taskset -p 0xffffffff %d" % os.getpid())
    
  print ( "PRECOMPRESS:    INFO:   torch       version =    {:}".format (  torch.__version__       )  )
  print ( "PRECOMPRESS:    INFO:   torchvision version =    {:}".format (  torchvision.__version__ )  )
  print ( "PRECOMPRESS:    INFO:   matplotlib version  =    {:}".format (  matplotlib.__version__ )   )
  
  
  now = time.localtime(time.time())
  print(time.strftime("\nPRECOMPRESS:    INFO: %Y-%m-%d %H:%M:%S %Z", now))
  start_time = time.time() 

  if args.ddp=='True':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12312'     
    world_size = 1
    
    print( "PRECOMPRESS:    INFO: about to set run mp.spawn" )    
    mp.spawn( run_job( args ),
              args   = ( world_size,),
              #nprocs = world_size,
              nprocs = 1,
              #daemon = True,
              join   = True )
  else:
    gpu=0
    run_job( gpu, args )

  print( "TRAINLENEJ:     INFO: \033[33;1mtraining complete\033[m" )

  hours   = round((time.time() - start_time) / 3600, 1  )
  minutes = round((time.time() - start_time) / 60,   1  )
  seconds = round((time.time() - start_time), 0  )
  #pprint.log_section('Job complete in {:} mins'.format( minutes ) )

  print(f'TRAINLENEJ:     INFO: run completed in {minutes} mins ({seconds:.1f} secs)')  



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_job( args ):

  """Main program: train -> test once per epoch while saving samples as needed.
  """ 

  print( "TRAINLENEJ:     INFO: yep the spawn" )
  
    
  if args.ddp=='True':
    print( f"TRAINLENEJ:     INFO: args.ddp=='True'" )

    print( f"TRAINLENEJ:     INFO: torch.distributed.is_available() = {torch.distributed.is_available()}" )
    
    torch.distributed.is_available()
    #world_size = gpus * nodes
    #rank       = args.nr * args.gpus + gpu

    dist.init_process_group (
      backend       = 'nccl',
      init_method   = 'env://',
      world_size    = 1,
      rank          = 1    )
    
    #dist.init_process_group( "nccl", rank=1, world_size=1 )
    print( "TRAINLENEJ:     INFO: yep dist.init_process_group" )
    
    
    
    
  print( "PRECOMPRESS:    INFO:   common args:   \
dataset=\033[36;1m{:}\033[m,\
mode=\033[36;1m{:}\033[m,\
nn=\033[38;2;255;192;203m{:}\033[m,\
nn_optimizer=\033[36;1m{:}\033[m,\
batch_size=\033[36;1m{:}\033[m,\
learning_rate(s)=\033[36;1m{:}\033[m,\
epochs=\033[36;1m{:}\033[m,\
samples=\033[36;1m{:}\033[m,\
max_consec_losses=\033[36;1m{:}\033[m"\
  .format( args.dataset, args.input_mode, args.nn_type, args.optimizer, args.batch_size, args.learning_rate, args.n_epochs, args.n_samples, args.max_consecutive_losses  ), flush=True )

  
  if args.input_mode=="image":
    print( "PRECOMPRESS:    INFO: image args: \
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
  .format( args.use_tiler, args.n_tiles, args.rand_tiles, args.greyness, 
args.min_tile_sd, args.min_uniques, args.latent_dim, args.label_swap_perunit, args.make_grey_perunit, args.stain_norm, args.annotated_tiles, args.probs_matrix_interpolation  ), flush=True )

  elif args.input_mode=="rna":
    print( f"PRECOMPRESS:    INFO:   {UNDER}rna-seq args:{RESET}  \
nn_dense_dropout_1={CYAN}{args.nn_dense_dropout_1 if args.nn_type=='DENSE' else 'n/a'}{RESET}, \
nn_dense_dropout_2={CYAN}{args.nn_dense_dropout_2 if args.nn_type=='DENSE' else 'n/a'}{RESET}, \
n_genes={CYAN}{args.n_genes}{RESET}, \
gene_data_norm={YELLOW if not args.gene_data_norm[0]=='NONE' else YELLOW if len(args.gene_data_norm)>1 else CYAN}{args.gene_data_norm}{RESET}, \
g_xform={YELLOW if not args.gene_data_transform[0]=='NONE' else YELLOW if len(args.gene_data_transform)>1 else CYAN}{args.gene_data_transform}{RESET}" )

  skip_preprocessing         = args.skip_preprocessing
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
  nn_type                    = args.nn_type
  use_same_seed              = args.use_same_seed
  nn_dense_dropout_1         = args.nn_dense_dropout_1
  nn_dense_dropout_2         = args.nn_dense_dropout_2
  nn_optimizer               = args.optimizer
  n_samples                  = args.n_samples
  n_tiles                    = args.n_tiles
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
  label_swap_perunit         = args.label_swap_perunit
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
  save_model_name            = args.save_model_name
  save_model_every           = args.save_model_every
  supergrid_size             = args.supergrid_size
  
  ddp                        = args.ddp
  gpus                       = args.gpus
  nodes                      = args.nodes
  
  remove_unexpressed_genes    = args.remove_unexpressed_genes
  remove_low_expression_genes = args.remove_low_expression_genes
  low_expression_threshold    = args.low_expression_threshold
  encoder_activation          = args.encoder_activation

  n_classes=len(class_names)

  
  if  ( ( nn_mode == 'pre_compress' ) &  ( not ( 'AE' in nn_type[0] ) )):
    print( f"{RED}PRECOMPRESS:    FATAL:  the network model must be an autoencoder if nn_mode='{CYAN}{nn_mode}{RESET}{RED}' (you have NN_TYPE='{CYAN}{nn_type[0]}{RESET}{RED}', which is not an autoencoder) ... halting now{RESET}" )
    sys.exit(0)

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
                            nn_type  =   nn_type,
                encoder_activation  =   encoder_activation,
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
  offset        = 14
  second_offset = 10
  
  if DEBUG>0:
    print(f"\033[2C\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Cn_samples\
\r\033[{start_column+2*offset}Cbatch_size\
\r\033[{start_column+3*offset}Cn_tiles\
\r\033[{start_column+4*offset}Ctile_size\
\r\033[{start_column+5*offset}Crand_tiles\
\r\033[{start_column+6*offset}Cnn_type\
\r\033[{start_column+7*offset+second_offset}Cactivation\
\r\033[{start_column+8*offset+second_offset}Cnn_drop_1\
\r\033[{start_column+9*offset+second_offset}Cnn_drop_2\
\r\033[{start_column+10*offset+second_offset}Coptimizer\
\r\033[{start_column+11*offset+second_offset}Cstain_norm\
\r\033[{start_column+12*offset+second_offset}Cg_norm\
\r\033[{start_column+13*offset+second_offset}Cg_xform\
\r\033[{start_column+14*offset+second_offset}Clabel_swap\
\r\033[{start_column+15*offset+second_offset}Cgreyscale\
\r\033[{start_column+16*offset+second_offset}Cjitter vector\033[m")
    for lr, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type, encoder_activation, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values):
      print( f"\
\r\033[{start_column+0*offset}C{CYAN}{lr:9.6f}\
\r\033[{start_column+1*offset}C{n_samples:<5d}\
\r\033[{start_column+2*offset}C{batch_size:<5d}\
\r\033[{start_column+3*offset}C{n_tiles:<5d}\
\r\033[{start_column+4*offset}C{tile_size:<3d}\
\r\033[{start_column+5*offset}C{rand_tiles:<5s}\
\r\033[{start_column+6*offset}C{nn_type:<10s}\
\r\033[{start_column+7*offset+second_offset}C{encoder_activation:<12s}\
\r\033[{start_column+8*offset+second_offset}C{nn_dense_dropout_1:<5.2f}\
\r\033[{start_column+9*offset+second_offset}C{nn_dense_dropout_2:<5.2f}\
\r\033[{start_column+10*offset+second_offset}C{nn_optimizer:<8s}\
\r\033[{start_column+11*offset+second_offset}C{stain_norm:<10s}\
\r\033[{start_column+12*offset+second_offset}C{gene_data_norm:<10s}\
\r\033[{start_column+13*offset+second_offset}C{gene_data_transform:<10s}\
\r\033[{start_column+14*offset+second_offset}C{label_swap_perunit:<6.1f}\
\r\033[{start_column+15*offset+second_offset}C{make_grey_perunit:<5.1f}\
\r\033[{start_column+16*offset+second_offset}C{jitter:}{RESET}" )      

  # ~ for lr, batch_size  in product(*param_values): 
      # ~ comment = f' batch_size={batch_size} lr={lr}'

  if just_test=='True':
    if not ( batch_size == int( math.sqrt(batch_size) + 0.5) ** 2 ):
      print( f"\033[31;1mPRECOMPRESS:    FATAL:test_batch_loss_epoch_ave  in test mode 'batch_size' (currently {batch_size}) must be a perfect square (4, 19, 16, 25 ...) to permit selection of a a 2D contiguous patch. Halting.\033[m" )
      sys.exit(0)      

  if input_mode=='image_rna':                                                                             # PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING - ########################################################################################################################################################
    n_samples=args.n_samples[0]*args.n_tiles[0]                                                           # PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING - ########################################################################################################################################################
    print( f"{WHITE} PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING  n_samples= {CYAN}{n_samples}{RESET}" )   # PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING - ########################################################################################################################################################


  
  # (B) RUN JOB LOOP

  run=0
  
  for lr, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type, encoder_activation, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values): 

    if DEBUG>0:
      print(f"PRECOMPRESS:    INFO: job level parameters: {CYAN} \n\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Cn_samples\
\r\033[{start_column+2*offset}Cbatch_size\
\r\033[{start_column+3*offset}Cn_tiles\
\r\033[{start_column+4*offset}Ctile_size\
\r\033[{start_column+5*offset}Crand_tiles\
\r\033[{start_column+6*offset}Cnn_type\
\r\033[{start_column+7*offset}Cencoder_activation\
\r\033[{start_column+8*offset}Cnn_drop_1\
\r\033[{start_column+9*offset}Cnn_drop_2\
\r\033[{start_column+10*offset}Coptimizer\
\r\033[{start_column+11*offset}Cstain_norm\
\r\033[{start_column+12*offset}Cgene_norm\
\r\033[{start_column+13*offset}Cgene_data_transform\
\r\033[{start_column+14*offset}Clabel_swap\
\r\033[{start_column+15*offset}Cgreyscale\
\r\033[{start_column+16*offset}Cjitter vector\
\r{RESET}\n{param_values}" )
    
    run+=1


    #(1) set up Tensorboard
    
    print( "PRECOMPRESS:    INFO: \033[1m1 about to set up Tensorboard\033[m" )
    
    if input_mode=='image':
#      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type}; opt={nn_optimizer}; n_samps={n_samples}; n_t={n_tiles}; t_sz={tile_size}; rnd={rand_tiles}; tot_tiles={n_tiles * n_samples}; n_epochs={n_epochs}; bat={batch_size}; stain={stain_norm};  uniques>{min_uniques}; grey>{greyness}; sd<{min_tile_sd}; lr={lr}; lbl_swp={label_swap_perunit*100}%; greyscale={make_grey_perunit*100}% jit={jitter}%' )
      writer = SummaryWriter(comment=f' NN={nn_type}; n_smp={n_samples}; sg_sz={supergrid_size}; n_t={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; n_e={n_epochs}; b_sz={batch_size}' )
    elif input_mode=='rna':
      writer = SummaryWriter(comment=f' {dataset}; {input_mode}; {nn_type}; act={encoder_activation}; d1={nn_dense_dropout_1}; d2={nn_dense_dropout_2}; opt={nn_optimizer}; samples={n_samples}; genes={n_genes}; g_norm={gene_data_norm}; g_xform={gene_data_transform}; epochs={n_epochs}; batch={batch_size}; lr={lr}')
    elif input_mode=='image_rna':
      writer = SummaryWriter(comment=f' {dataset}; {input_mode}; {nn_type}; act={encoder_activation}; {nn_optimizer}; samples={n_samples}; tiles={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; genes={n_genes}; g_norm={gene_data_norm}; g_xform={gene_data_transform}; epochs={n_epochs}; batch={batch_size}; lr={lr}')
    else:
      print( f"{RED}PRECOMPRESS:  FATAL:    input mode of type '{CYAN}{input_mode}{RESET}{RED}' is not supported [314]{RESET}" )
      sys.exit(0)

    print( "PRECOMPRESS:    INFO:   \033[3mTensorboard has been set up\033[m" ) 
    

    # (2) potentially schedule and run tiler threads
    
    if skip_preprocessing=='False':
      
      if not input_mode=='rna':

        n_samples_max = np.max(n_samples)
        tile_size_max = np.max(tile_size)
        n_tiles_max   = np.max(n_tiles)    
      
        if stain_norm=="NONE":                                                                         # we are NOT going to stain normalize ...
          norm_method='NONE'
        else:                                                                                          # we are going to stain normalize ...
          if DEBUG>0:
            print( f"PRECOMPRESS:      INFO: {BOLD}2 about to set up stain normalization target{RESET}" )
          if stain_norm_target.endswith(".svs"):                                                       # ... then grab the user provided target
            norm_method = tiler_set_target( args, stain_norm, stain_norm_target, writer )
          else:                                                                                        # ... and there MUST be a target
            print( f"PRECOMPRESS:    FATAL:    for {CYAN}{stain_norm}{RESET} an SVS file must be provided from which the stain normalization target will be extracted" )
            sys.exit(0)
    
        print( f"PRECOMPRESS:    INFO: about to call tile threader with n_samples_max={CYAN}{n_samples_max}{RESET}; n_tiles_max={CYAN}{n_tiles_max}{RESET}  " )
        result = tiler_threader( args, n_samples_max, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job
  
    n_genes = generate( args, n_samples, n_tiles, tile_size, gene_data_norm, gene_data_transform  )

    cfg = loader.get_config( args.nn_mode, lr, batch_size )

    if ddp=='True':
      rank       = rank      
      world_size = world_size
    else:
      rank       = 0
      world_size = 0
    
    train_loader, test_loader = loader.get_data_loaders( args,
                                                         cfg,
                                                         world_size,
                                                         rank,
                                                         batch_size,
                                                         args.n_workers,
                                                         args.pin_memory,                                                       
                                                         args.pct_test
                                                        )
    
    model = PRECOMPRESS( args, cfg, input_mode, nn_type, encoder_activation, n_classes, n_genes, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, args.latent_dim, args.em_iters)   

    if ddp=='TTrue':
      torch.cuda.set_device(gpu)
      model.cuda(gpu)
      model = DDP( model, device_ids=[gpu] )
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)

    if nn_type=='TTVAE':
      scheduler_opts = dict( scheduler          = 'warm_restarts', 
                             lr_scheduler_decay = 0.5, 
                             T_max              = 10, 
                             eta_min            = 5e-8, 
                             T_mult             = 2                )

      scheduler      = Scheduler( optimizer = optimizer,  opts=scheduler_opts )   
    else:
      scheduler = 0
  
    pprint.log_section('Training model.\n\n'\
                       'Epoch\t\tTrain x1 err\tTrain x2 err\tTrain l1\t'\
                       '\tTest x1 err\tTest x2 err\tTest l1')

    number_correct_max   = 0
    pct_correct_max      = 0
    test_loss_min        = 9999999999999
    train_loss_min       = 9999999999999



    #(10) Train/Test
    
    
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
        
                     
    print( "PRECOMPRESS:    INFO: \033[12m1 about to commence training loop, one iteration per epoch\033[m" )

    for epoch in range(1, args.n_epochs + 1):   
    
        if input_mode=='rna':
          print( f'\n{DIM_WHITE}PRECOMPRESS:    INFO:      {RESET}epoch: {MIKADO}{epoch}{RESET} of {CYAN}{n_epochs}{RESET}, {PINK}({nn_type}){RESET} mode: {CYAN}{input_mode}{RESET}, samples: {CYAN}{n_samples}{RESET}, batch size: {CYAN}{batch_size}{RESET}.  {DULL_WHITE}will halt if test loss increases for {CYAN}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )          
        else:
          print( f'\n{DIM_WHITE}PRECOMPRESS:    INFO:      {RESET}epoch: {MIKADO}{epoch}{RESET} of {CYAN}{n_epochs}{RESET}, {PINK}({nn_type}){RESET} mode: {CYAN}{input_mode}{RESET}, samples: {CYAN}{n_samples}{RESET}, batch size: {CYAN}{batch_size}{RESET}, tile: {CYAN}{tile_size}x{tile_size}{RESET} tiles per slide: {CYAN}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {CYAN}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )


        train_batch_loss_epoch_ave, train_loss_genes_sum_ave, train_l1_loss_sum_ave, train_total_loss_sum_ave =\
                                           train (      args, epoch, encoder_activation, train_loader, model, nn_type, lr, scheduler, optimizer, writer, train_loss_min, batch_size )

  
        test_batch_loss_epoch_ave, test_l1_loss_sum_ave, test_loss_min                =\
                                            test ( cfg, args, epoch, encoder_activation, test_loader,  model, nn_type, tile_size, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, annotated_tiles, class_names, class_colours)

        torch.cuda.empty_cache()
        
        if DEBUG>0:
          if ( (test_batch_loss_epoch_ave < (test_batch_loss_epoch_ave_last)) | (epoch==1) ):
            consecutive_test_loss_increases = 0
            last_epoch_loss_increased = False
          else:
            last_epoch_loss_increased = True
            
          print ( f"\
\033[2K\
{DIM_WHITE}PRECOMPRESS:    INFO:   {RESET}\
\r\033[27Cepoch summary:\
\r\033[92Cae={GREEN}{test_batch_loss_epoch_ave:<11.3f}{DULL_WHITE}\
\r\033[109Cl1={test_l1_loss_sum_ave:<11.3f}{DULL_WHITE}\
\r\033[124CBATCH AVE OVER EPOCH={GREEN if last_epoch_loss_increased==False else RED}{test_batch_loss_epoch_ave:<11.3f}\r\033[155C{UP_ARROW if last_epoch_loss_increased==True else DOWN_ARROW}{DULL_WHITE}\
\r\033[167Cmins: total: {test_lowest_total_loss_observed:<11.3f}@{ORANGE}\r\033[202Ce={test_lowest_total_loss_observed_epoch:<2d}{RESET}\
\033[3B\
", end='', flush=True )

          if last_epoch_loss_increased == True:
            consecutive_test_loss_increases +=1

            if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before, so that the most recent model for which the loss improved will be saved
                now = time.localtime(time.time())
                print(time.strftime("PRECOMPRESS:    INFO: %Y-%m-%d %H:%M:%S %Z", now))
                sys.exit(0)
  
        test_batch_loss_epoch_ave_last = test_batch_loss_epoch_ave
        
        if DEBUG>9:
          print( f"{DIM_WHITE}PRECOMPRESS:    INFO:   test_lowest_total_loss_observed = {CYAN}{test_lowest_total_loss_observed}{RESET}" )
          print( f"{DIM_WHITE}PRECOMPRESS:    INFO:   test_batch_loss_epoch_ave         = {CYAN}{test_batch_loss_epoch_ave}{RESET}"         )
        
        if test_batch_loss_epoch_ave < test_lowest_total_loss_observed:
          test_lowest_total_loss_observed       = test_batch_loss_epoch_ave
          test_lowest_total_loss_observed_epoch = epoch
          if DEBUG>0:
            print ( f"\r\033[200C{DIM_WHITE}{GREEN}{ITALICS} \r\033[210C<< new minimum test loss{RESET}\033[1A", flush=True )

    if DEBUG>9:
      print( f"{DIM_WHITE}PRECOMPRESS:    INFO:    pytorch Model = {CYAN}{model}{RESET}" )

# ------------------------------------------------------------------------------

def train(  args, epoch, encoder_activation, train_loader, model, nn_type, lr, scheduler, optimizer, writer, train_loss_min, batch_size  ):  
    """Train PCCA model and update parameters in batches of the whole train set.
    """
    
    model.train()

    ae_loss2_sum     = 0
    l1_loss_sum      = 0

    loss             = 0.
    total_recon_loss = 0.
    total_kl_loss    = 0.
    
    for i, (x2) in enumerate( train_loader ):

        optimizer.zero_grad()

        x2 = x2.to(device)

        if DEBUG>99:
          print ( f"PRECOMPRESS:    INFO:      train(): x2.type             = {CYAN}{x2.type}{RESET}" )
          print ( f"PRECOMPRESS:    INFO:      train(): encoder_activation  = {CYAN}{encoder_activation}{RESET}" )

        x2r, mean, logvar = model.forward( x2, encoder_activation )

        if DEBUG>99:
          print ( f"PRECOMPRESS:    INFO:      train(): nn_type        = {CYAN}{nn_type}{RESET}" )
          
        if nn_type=='TTVAE':                                                                               # Fancy loss function for TTVAE
          
          if DEBUG>99:
            print ( f"PRECOMPRESS:    INFO:      train(): x2[0:12,0:12]  = {MIKADO}{x2[0:12,0:12]}{RESET}" ) 
            print ( f"PRECOMPRESS:    INFO:      train(): x2r[0:12,0:12] = {MIKADO}{x2r[0:12,0:12]}{RESET}" )


                                
          bce_loss       = False
          loss_reduction = 'sum'
          loss_fn        = BCELoss( reduction=loss_reduction ) if bce_loss else MSELoss( reduction=loss_reduction )                                                 # Have to use Binary cross entropy loss for TTVAE (and VAEs generally)
          ae_loss2, reconstruction_loss, kl_loss = vae_loss( x2r, x2, mean, logvar, loss_fn, epoch, kl_warm_up=0, beta=1. )
          del mean
          del logvar
          
        else:                                                                                              # Used for AELINEAR, AEDENSE, AEDENSEPOSITIVE, DCGANAE128
          ae_loss2 = F.mse_loss( x2r, x2)                                                                  # mean squared error loss function
        
        del x2
        del x2r
        
        l1_loss                = l1_penalty(model, args.l1_coef)                                            # NOT CURRENTLY USING l1_loss
        #loss                  = ae_loss1 + ae_loss2 + l1_loss                                              # NOT CURRENTLY USING l1_loss
        ae_loss2_sum          += ae_loss2.item()
        if nn_type=='TTVAE':        
          reconstruction_loss += reconstruction_loss.item()
          kl_loss             += kl_loss.item()
        loss                   = ae_loss2

        loss.backward()
        
        del loss
        
        if not nn_type=='TTVAE':
          # Perform gradient clipping *before* calling `optimizer.step()
          clip_grad_norm_( model.parameters(), args.clip )

        optimizer.step()

        if nn_type=='TTVAE':
          scheduler.step()                                                                                 # has to be after optimizer.step()
          current_lr = scheduler.get_lr()
          if DEBUG>99:         
            print ( f"PRECOMPRESS:    INFO:      train(): lr        = {CYAN}{scheduler.get_lr():<2.2e}{RESET}" )
        else:
          current_lr = lr
          
        ae_loss2_sum += ae_loss2.item()
        l1_loss_sum  += l1_loss.item()                                                                     # NOT CURRENTLY USING l1_loss
        #total_loss   = ae_loss1_sum + ae_loss2_sum + l1_loss_sum                                          # NOT CURRENTLY USING l1_loss
        total_loss    = ae_loss2_sum                                                                       # NOT CURRENTLY USING l1_loss
        
        if DEBUG>0:
          print ( f"\
\033[2K\
{DIM_WHITE}PRECOMPRESS:    INFO:{RESET}\
\r\033[29C{DULL_WHITE}train:\
\r\033[40Cn={i+1:>3d}\
\r\033[47Clr={current_lr:<2.2e}\
\r\033[92Cae={ ae_loss2:<11.3f}\
\r\033[109Cl1={l1_loss:<11.3f}\
\r\033[135CBATCH AVE={ae_loss2:11.3f}{RESET}" )
#\r\033[124C    BATCH LOSS=\r\033[{139+4*int((ae_loss2*10)//1) if ae_loss2<1 else 150+4*int((ae_loss2*2)//1) if ae_loss2<12 else 160}C{PALE_GREEN if ae_loss2<1 else GOLD if 1<=ae_loss2<2 else PALE_RED}{ae_loss2:11.3f}{RESET}" )

          if nn_type=='TTVAE':
            print ( "\033[2A" )
            print ( f"{DULL_WHITE}\r\033[60Crecon={reconstruction_loss:<11.1f} \
            \r\033[78Ckl={ kl_loss:<6.3f}{RESET}" )

          print ( "\033[2A" )


    ae_loss2_sum  /= (i+1)                                                                                 # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    l1_loss_sum   /= (i+1)                                                                                 # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    train_msgs     = [ae_loss2_sum, l1_loss_sum]

    if ae_loss2_sum    <  train_loss_min:
      train_loss_min   =  ae_loss2_sum
       
    writer.add_scalar( '1_loss_train',      ae_loss2_sum,    epoch  )
    writer.add_scalar( 'loss_train_min',  train_loss_min,  epoch  ) 
    if nn_type=='TTVAE':
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

def test( cfg, args, epoch, encoder_activation, test_loader, model,  nn_type, tile_size, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, annotated_tiles, class_names, class_colours ):
  
    """Test model by computing the average loss on a held-out dataset. No
    parameter updates.
    """
    model.eval()

    ae_loss2_sum      = 0
    l1_loss_sum       = 0
    
    loss             = 0.
    total_recon_loss = 0.
    total_kl_loss    = 0.    

    for i, (x2) in enumerate(test_loader):

        x2 = x2.to(device)

        with torch.no_grad():                                                                              # Don't need gradients for testing, so this will save some GPU memory
#       x2r =               model.forward( x2, encoder_activation )
          x2r, mean, logvar = model.forward( x2, encoder_activation )

        if DEBUG>99:
          print ( f"PRECOMPRESS:    INFO:      test(): nn_type        = {CYAN}{nn_type}{RESET}" )
          
        if nn_type=='TTVAE':                                                                               # Fancy loss function for TTVAE. ------------------> Disabling for the moment because it's not working
          bce_loss       = False
          loss_reduction ='sum'
          loss_fn        = BCELoss( reduction=loss_reduction ) if bce_loss else MSELoss( reduction=loss_reduction )          
          ae_loss2, reconstruction_loss, kl_loss = vae_loss( x2r, x2, mean, logvar, loss_fn, epoch, kl_warm_up=0, beta=1.0 )
        else:                                                                                              # Used for AELINEAR, AEDENSE, AEDENSEPOSITIVE, DCGANAE128
          ae_loss2 = F.mse_loss(x2r, x2)


        l1_loss                = l1_penalty( model, args.l1_coef)                                           # NOT CURRENTLY USING l1_loss
        ae_loss2_sum          += ae_loss2.item()
        if nn_type=='TTVAE':  
          reconstruction_loss += reconstruction_loss.item()
          kl_loss             += kl_loss.item()
        l1_loss_sum           += l1_loss.item()                                                             # NOT CURRENTLY USING l1_loss                                                    
        total_loss             =  ae_loss2_sum                                                              # NOT CURRENTLY USING l1_loss
        
        
        if i == 0 and epoch % LOG_EVERY == 0:
            cfg.save_comparison(args.log_dir, x2, x2r, epoch, is_x1=False)
        
        if DEBUG>0:
          if i==0:
            print ("")
          print ( f"\
\033[2K\
{DIM_WHITE}PRECOMPRESS:    INFO:{RESET}\
\r\033[29Ctest:\
\r\033[40C{DULL_WHITE}n={i+1:>3d}\
\r\033[92Cae={ ae_loss2:<11.3f}\
\r\033[109Cl1={l1_loss:<11.3f}\
\r\033[135CBATCH AVE={ae_loss2:11.3f}{RESET}" )
#\r\033[124C    BATCH AVE=\r\033[{139+4*int((ae_loss2*10)//1) if ae_loss2<1 else 150+4*int((ae_loss2*2)//1) if ae_loss2<12 else 160}C{GREEN if ae_loss2<1 else ORANGE if 1<=ae_loss2<2 else RED}{ae_loss2:<11.3f}{RESET}" )

        print ( "\033[2A" )
    
    print ("")

    ae_loss2_sum  /= (i+1)                                                                                 # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    l1_loss_sum   /= (i+1)                                                                                 # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)

    if DEBUG>9:
      print ( f"PRECOMPRESS:    INFO:      test(): x2.shape  = {CYAN}{x2.shape}{RESET}" )
      print ( f"PRECOMPRESS:    INFO:      test(): x2r.shape = {CYAN}{x2r.shape}{RESET}" )
    
    if ( (epoch+1)%10==0 ) | ( ae_loss2_sum<test_loss_min ):
      
      if DEBUG>0:
        np.set_printoptions(linewidth=600)   
        np.set_printoptions(edgeitems=600)
        number_to_display=24
        sample = np.random.randint( x2.shape[0] )
        print ( f"{DIM_WHITE}PRECOMPRESS:    INFO:        test: original/reconstructed values for a randomly selected sample ({CYAN}{sample}{RESET}) and first {CYAN}{number_to_display}{RESET} genes" )
        np.set_printoptions(formatter={'float': lambda x: "{:>7.2f}".format(x)})
        x2_nums  = x2.cpu().detach().numpy()  [12,0:number_to_display]                                     
        x2r_nums = x2r.cpu().detach().numpy() [12,0:number_to_display]
        x2r_nums[x2r_nums<0] = 0                                                                             # change negative values (which are impossible) to zero        
        print (  f"x2     = \r\033[29C{x2_nums}",  flush='True'     )
        print (  f"x2r    = \r\033[29C{x2r_nums}", flush='True'     )
        ratios= np.around(np.absolute( ( (x2r_nums+.02) / (x2_nums+.02)  ) ), decimals=2 )           # to avoid divide by zero error
        np.set_printoptions(formatter={'float': lambda w: f"{BRIGHT_GREEN if abs(w-1)<0.01 else PALE_GREEN if abs(w-1)<0.05 else ORANGE if abs(w-1)<0.25 else GOLD if abs(w-1)<0.5 else BLEU if abs(w-1)<1.0 else DIM_WHITE}{w:>7.2f}{RESET}"})     
        print (  f"ratios = \r\033[29C{ratios}{RESET}", flush='True'     )
        np.set_printoptions(formatter={'float': lambda w: "{:>8.2f}".format(w)})
    
    del x2
    del x2r
    
    writer.add_scalar( '2_loss_test',      ae_loss2_sum,   epoch )
    writer.add_scalar( 'loss_test_min',  test_loss_min,  epoch )
    if nn_type=='TTVAE':
      writer.add_scalar( 'loss_recon_VAE', reconstruction_loss,  epoch  )
      writer.add_scalar( 'loss_kl_vae',    kl_loss,              epoch  )
      del kl_loss
      del reconstruction_loss

    del ae_loss2
    del l1_loss
    
    
    if DEBUG>9:
      print ( f"{DIM_WHITE}PRECOMPRESS:    INFO:      test(): test_loss_min  = {CYAN}{test_loss_min:5.2f}{RESET}" )
      print ( f"{DIM_WHITE}PRECOMPRESS:    INFO:      test(): ae_loss2_sum   = {CYAN}{ae_loss2_sum:5.2f}{RESET}" )
                
    if ae_loss2_sum < test_loss_min:
      test_loss_min = ae_loss2_sum
      if epoch>9:                                                                                          # wait till a reasonable number of epochs have completed befor saving mode, else it will be saving all the time early on
        save_model( args.log_dir, model)                                                                   # save model with the lowest cost to date. Over-write earlier least cost model, if one exists.
    
    torch.cuda.empty_cache()
    
    return ae_loss2_sum, l1_loss_sum, test_loss_min
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
            lab    = test_loader.dataset.labels[j]
            x2_batch[i] = x2
            labels.append(lab)

        x2_batch = x2_batch.to(device)

        cfg.save_samples(directory, model, epoch, x2_batch, labels)

# ------------------------------------------------------------------------------

def save_model(log_dir, model):
    """Save PyTorch model state dictionary
    """
    
    fpath = '%s/model_pre_compressed_version.pt' % log_dir
    if DEBUG>0:
#      print( f"TRAINLENEJ:     INFO:   save_model(){DULL_YELLOW}{ITALICS}: new lowest loss on this epoch... saving model to {fpath}{RESET}\033[1A" )       
      print( f"TRAINLENEJ:     INFO:   save_model(){DULL_YELLOW}{ITALICS}: new lowest loss on this epoch... saving model to {fpath}{RESET}" )       
    model_state = model.state_dict()
    torch.save(model_state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--skip_preprocessing',             type=str,   default='False')                                # USED BY main() to enable user to skip tile generation
    p.add_argument('--skip_generation',                type=str,   default='False')                                # USED BY main() to enable user to skip torch database generation
    p.add_argument('--log_dir',                        type=str,   default='data/pre_compress/logs')                # used to store logs and to periodically save the model
    p.add_argument('--base_dir',                       type=str,   default='/home/peter/git/pipeline')             # NOT CURRENTLY USED
    p.add_argument('--data_dir',                       type=str,   default='/home/peter/git/pipeline/dataset')     # USED BY generate()
    p.add_argument('--save_model_name',                type=str,   default='model.pt')                             # USED BY main()
    p.add_argument('--save_model_every',               type=int,   default=10)                                     # USED BY main()    
    p.add_argument('--rna_file_name',                  type=str,   default='rna.npy')                              # USED BY generate()
    p.add_argument('--rna_file_suffix',                type=str,   default='*FPKM-UQ.txt' )                        # USED BY generate()
    p.add_argument('--use_unfiltered_data',            type=str,   default='True' )                                # USED BY generate() 
    p.add_argument('--rna_file_reduced_suffix',        type=str,   default='_reduced')                             # USED BY generate()
    p.add_argument('--class_numpy_file_name',          type=str,   default='class.npy')                            # USED BY generate()
    p.add_argument('--wall_time',                                                     type=int,    default=24)
    p.add_argument('--seed',                                                          type=int,    default=0)
    p.add_argument('--nn_mode',                                                       type=str,    default='pre_compress')
    p.add_argument('--use_same_seed',                                                 type=str,    default='False')
    p.add_argument('--nn_type',                                           nargs="+",  type=str,    default='VGG11')
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
    p.add_argument('--batch_size',         nargs="+",  type=int,   default=256)                                   # USED BY tiler() 
    p.add_argument('--learning_rate',      nargs="+",  type=float, default=.00082)                                # USED BY main()                               
    p.add_argument('--n_epochs',                       type=int,   default=10)
    p.add_argument('--pct_test',                       type=float, default=0.2)
    p.add_argument('--latent_dim',                     type=int,   default=100)
    p.add_argument('--l1_coef',                        type=float, default=0.1)
    p.add_argument('--em_iters',                       type=int,   default=1)
    p.add_argument('--clip',                           type=float, default=1)
    p.add_argument('--max_consecutive_losses',         type=int,   default=7771)
    p.add_argument('--optimizer',          nargs="+",  type=str,   default='ADAM')
    p.add_argument('--label_swap_perunit',             type=int,   default=0)                                    
    p.add_argument('--make_grey_perunit',              type=float, default=0.0) 
    p.add_argument('--figure_width',                   type=float, default=16)                                  
    p.add_argument('--figure_height',                  type=float, default=16)
    p.add_argument('--annotated_tiles',                type=str,   default='True')
    p.add_argument('--scattergram',                    type=str,   default='True')
    p.add_argument('--probs_matrix',                   type=str,   default='True')
    p.add_argument('--probs_matrix_interpolation',     type=str,   default='none')
    p.add_argument('--show_patch_images',              type=str,   default='True')
    p.add_argument('--regenerate',                     type=str,   default='True')
    p.add_argument('--just_profile',                   type=str,   default='False')                                # USED BY tiler()    
    p.add_argument('--just_test',                      type=str,   default='False')                                # USED BY tiler()    
    p.add_argument('--rand_tiles',                     type=str,   default='True')                                 # USED BY tiler()      
    p.add_argument('--points_to_sample',               type=int,   default=100)                                    # USED BY tiler()
    p.add_argument('--min_uniques',                    type=int,   default=0)                                      # USED BY tiler()
    p.add_argument('--min_tile_sd',                    type=float, default=3)                                      # USED BY tiler()
    p.add_argument('--greyness',                       type=int,   default=0)                                      # USED BY tiler()
    p.add_argument('--stain_norm',         nargs="+",  type=str,   default='NONE')                                 # USED BY tiler()
    p.add_argument('--stain_norm_target',              type=str,   default='NONE')                                 # USED BY tiler_set_target()
    p.add_argument('--use_tiler',                      type=str,   default='external'  )                           # USED BY main()
    p.add_argument('--cancer_type',                    type=str,   default='NONE'      )                           # USED BY main()
    p.add_argument('--cancer_type_long',               type=str,   default='NONE'      )                           # USED BY main()
    p.add_argument('--class_names',        nargs="+"                                  )                            # USED BY main()
    p.add_argument('--long_class_names',   nargs="+"                                  )                            # USED BY main()
    p.add_argument('--class_colours',      nargs="*"                                  )    
    p.add_argument('--target_tile_coords', nargs=2,    type=int, default=[2000,2000]       )                       # USED BY tiler_set_target()

    p.add_argument('--a_d_use_cupy',                   type=str,   default='True'     )                            # USED BY main()
    p.add_argument('--cov_threshold',                  type=float, default=8.0        )                            # USED BY main()   
    p.add_argument('--cov_uq_threshold',               type=float, default=0.0        )                            # USED BY main() 
    p.add_argument('--cutoff_percentile',              type=float, default=0.05       )                            # USED BY main() 
    
    p.add_argument('--show_rows',                      type=int,   default=500)                                    # USED BY main()
    p.add_argument('--show_cols',                      type=int,   default=100)                                    # USED BY main()
    
    p.add_argument('-ddp', '--ddp',                    type=str,   default='True'                                                   )
    p.add_argument('-n', '--nodes',                    type=int,   default=1,  metavar='N'                                          )
    p.add_argument('-g', '--gpus',                     type=int,   default=1,  help='number of gpus per node'                       )
    p.add_argument('-nr', '--nr',                      type=int,   default=0,  help='ranking within the nodes'                      )


    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 12
    args.pin_memory = torch.cuda.is_available()

    main(args)
