"""=============================================================================
Code to support Data Analysis Mode
============================================================================="""

import os
import sys
import math
import time
import cuda
import pprint
import argparse
import numpy as np
import cupy
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
#from matplotlib import figure
#from pytorch_memlab import profile


from   data.pre_compress.config   import pre_compressConfig

from   data                            import loader
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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE 
from sklearn.datasets import load_digits                                                                   # For the UCI ML handwritten digits dataset

from   data import loader
from   data.pre_compress.generate       import generate

from   itertools                       import product, permutations
from   PIL                             import Image

import cuda
from   models import PRECOMPRESS
import pprint

#===========================================
import pandas as pd
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import missingno as msno
import matplotlib.colors as mcolors

from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

inline_rc = dict(mpl.rcParams)
pd.set_option('max_colwidth', 50)
#===========================================

np.set_printoptions(edgeitems=500)
np.set_printoptions(linewidth=200)

pd.set_option('display.max_rows',     50 )
pd.set_option('display.max_columns',  13 )
pd.set_option('display.width',       300 )
pd.set_option('display.max_colwidth', 99 )  

torch.backends.cudnn.enabled     = True                                                                     #for CUDA memory optimizations
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

UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1

device = cuda.device()

pool = cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)
cupy.cuda.set_allocator(pool.malloc)

# ------------------------------------------------------------------------------

def main(args):

  """Main program: train -> test once per epoch while saving samples as
  needed.
  """
  
  
  os.system("taskset -p 0xffffffff %d" % os.getpid())

  now = time.localtime(time.time())
  print(time.strftime("\nANALYSEDATA:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
  start_time = time.time()
    
  print ( "ANALYSEDATA:     INFO:   torch       version =    {:}".format (  torch.__version__       )  )
  print ( "ANALYSEDATA:     INFO:   torchvision version =    {:}".format (  torchvision.__version__ )  )
  print ( "ANALYSEDATA:     INFO:   matplotlib version  =    {:}".format (  matplotlib.__version__ )   )   


  print( f"ANALYSEDATA:     INFO:  common args: \
dataset={MIKADO}{args.dataset}{RESET},\
mode={MIKADO}{args.input_mode}{RESET},\
nn={MIKADO}{args.nn_type}{RESET},\
nn_optimizer={MIKADO}{args.optimizer}{RESET},\
batch_size={MIKADO}{args.batch_size}{RESET},\
learning_rate(s)={MIKADO}{args.learning_rate}{RESET},\
epochs={MIKADO}{args.n_epochs}{RESET},\
samples={MIKADO}{args.n_samples}{RESET},\
max_consec_losses={MIKADO}{args.max_consecutive_losses}{RESET}",  
flush=True )

  
  if args.input_mode=="image":
    print( "ANALYSEDATA:     INFO: image args: \
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
    print( f"ANALYSEDATA:     INFO: rna-seq args: \
nn_dense_dropout_1={MIKADO}{args.nn_dense_dropout_1}{RESET}, \
nn_dense_dropout_2={MIKADO}{args.nn_dense_dropout_2}{RESET}, \
n_genes={MIKADO}{args.n_genes}{RESET}, \
gene_norm={YELLOW if not args.gene_data_norm[0]=='NONE' else YELLOW if len(args.gene_data_norm)>1 else MIKADO}{args.gene_data_norm}{RESET}, \
g_xform={YELLOW if not args.gene_data_transform[0]=='NONE' else YELLOW if len(args.gene_data_transform)>1 else MIKADO}{args.gene_data_transform}{RESET}" )

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
  
  cov_threshold              = args.cov_threshold
  cov_uq_threshold           = args.cov_uq_threshold  

  n_classes=len(class_names)
  
            
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

  if DEBUG>0:
    print("\033[2Clr\r\033[14Cn_samples\r\033[26Cbatch_size\r\033[38Cn_tiles\r\033[48Ctile_size\r\033[59Crand_tiles\r\033[71Cnn_type\r\033[90Cnn_drop_1\r\033[100Cnn_drop_2\r\033[110Coptimizer\r\033[120Cstain_norm\
\r\033[130Cg_norm\r\033[140Cg_xform\r\033[155Clabel_swap\r\033[170Cgreyscale\r\033[182Cjitter vector\033[m")
    for       lr,      n_samples,        batch_size,                 n_tiles,         tile_size,        rand_tiles,         nn_type,          nn_dense_dropout_1, nn_dense_dropout_2,       nn_optimizer,          stain_norm, \
    gene_data_norm,    gene_data_transform,   label_swap_perunit, make_grey_perunit,   jitter in product(*param_values):
      print( f"\033[0C{MIKADO}{lr:9.6f} \r\033[14C{n_samples:<5d} \r\033[26C{batch_size:<5d} \r\033[38C{n_tiles:<5d} \r\033[48C{tile_size:<3d} \r\033[59C{rand_tiles:<5s} \r\033[71C{nn_type:<8s} \r\033[90C{nn_dense_dropout_1:<5.2f}\
\r\033[100C{nn_dense_dropout_2:<5.2f} \r\033[110C{nn_optimizer:<8s} \r\033[120C{stain_norm:<10s} \r\033[130C{gene_data_norm:<10s} \r\033[140C{gene_data_transform:<10s} \r\033[155C{label_swap_perunit:<6.1f}\
\r\033[170C{make_grey_perunit:<5.1f}\r\033[182C{jitter:}{RESET}" )      

  # ~ for lr, batch_size  in product(*param_values): 
      # ~ comment = f' batch_size={batch_size} lr={lr}'

  if just_test=='True':
    if not ( batch_size == int( math.sqrt(batch_size) + 0.5) ** 2 ):
      print( f"\033[31;1mANALYSEDATA:     FATAL:test_total_loss_sum_ave  in test mode 'batch_size' (currently {batch_size}) must be a perfect square (4, 19, 16, 25 ...) to permit selection of a a 2D contiguous patch. Halting.\033[m" )
      sys.exit(0)      

  if input_mode=='image_rna':                                                                             # PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING - ########################################################################################################################################################
    n_samples=args.n_samples[0]*args.n_tiles[0]                                                           # PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING - ########################################################################################################################################################
    print( f"{WHITE} PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING  n_samples= {MIKADO}{n_samples}{RESET}" )   # PGD 200531 - TEMP TILL MULTIMODE IS UP AND RUNNING - ########################################################################################################################################################


  
  # (B) RUN JOB LOOP

  run=0
  
  for lr, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values): 

    if DEBUG>0:
      print(f"ANALYSEDATA:     INFO: job level parameters:  \nlr\r\033[10Cn_samples\r\033[26Cbatch_size\r\033[38Cn_tiles\r\033[51Ctile_size\r\033[61Crand_tiles\r\033[71Cnn_type\r\033[81Cnn_drop_1\r\033[91Cnn_drop_2\r\033[101Coptimizer\r\033[111Cstain_norm\
\r\033[123Cgene_norm\r\033[133Cgene_data_transform\r\033[144Clabel_swap\r\033[154Cgreyscale\r\033[164Cjitter vecto{MIKADO}\n{param_values}{RESET}", flush=True  )
    
    run+=1


    #(1) set up Tensorboard
    
    print( "ANALYSEDATA:     INFO: \033[1m1 about to set up Tensorboard\033[m" )
    
    if input_mode=='image':
#      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type}; opt={nn_optimizer}; n_samps={n_samples}; n_t={n_tiles}; t_sz={tile_size}; rnd={rand_tiles}; tot_tiles={n_tiles * n_samples}; n_epochs={n_epochs}; bat={batch_size}; stain={stain_norm};  uniques>{min_uniques}; grey>{greyness}; sd<{min_tile_sd}; lr={lr}; lbl_swp={label_swap_perunit*100}%; greyscale={make_grey_perunit*100}% jit={jitter}%' )
      writer = SummaryWriter(comment=f' NN={nn_type}; n_smp={n_samples}; sg_sz={supergrid_size}; n_t={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; n_e={n_epochs}; b_sz={batch_size}' )
    elif input_mode=='rna':
      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type}; d1={nn_dense_dropout_1}; d2={nn_dense_dropout_2}; opt={nn_optimizer}; n_smp={n_samples}; n_g={n_genes}; gene_norm={gene_data_norm}; g_xform={gene_data_transform}; n_e={n_epochs}; b_sz={batch_size}; lr={lr}')
    elif input_mode=='image_rna':
      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type}; opt={nn_optimizer}; n_smp={n_samples}; n_t={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; n_g={n_genes}; gene_norm={gene_data_norm}; g_xform={gene_data_transform}; n_e={n_epochs}; b_sz={batch_size}; lr={lr}')
    else:
      print( f"{RED}ANALYSEDATA:   FATAL:    input mode of type '{MIKADO}{input_mode}{RESET}{RED}' is not supported [314]{RESET}" )
      sys.exit(0)

    print( "ANALYSEDATA:     INFO:   \033[3mTensorboard has been set up\033[m" ) 
    

    # (2) potentially schedule and run tiler threads
    
    if input_mode=='image':
      
      if skip_preprocessing=='False':
  
        n_samples_max = np.max(n_samples)
        tile_size_max = np.max(tile_size)
        n_tiles_max   = np.max(n_tiles)    
      
        if stain_norm=="NONE":                                                                         # we are NOT going to stain normalize ...
          norm_method='NONE'
        else:                                                                                          # we are going to stain normalize ...
          if DEBUG>0:
            print( f"ANALYSEDATA:       INFO: {BOLD}2 about to set up stain normalization target{RESET}" )
          if stain_norm_target.endswith(".svs"):                                                       # ... then grab the user provided target
            norm_method = tiler_set_target( args, stain_norm, stain_norm_target, writer )
          else:                                                                                        # ... and there MUST be a target
            print( f"ANALYSEDATA:     FATAL:    for {MIKADO}{stain_norm}{RESET} an SVS file must be provided from which the stain normalization target will be extracted" )
            sys.exit(0)
    
        print( f"ANALYSEDATA:     INFO: about to call tile threader with n_samples_max={MIKADO}{n_samples_max}{RESET}; n_tiles_max={MIKADO}{n_tiles_max}{RESET}  " )
        result = tiler_threader( args, n_samples_max, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job



    generate( args, n_samples, n_tiles, tile_size, n_genes, gene_data_norm, gene_data_transform  )
      

    #(1) set up Tensorboard
    
    print( f"ANALYSEDATA:     INFO: {BOLD}2 about to start selected data analyses{RESET}" )
      
   #pd.set_option( 'display.max_columns',    25 )
   #pd.set_option( 'display.max_categories', 24 )
   #pd.set_option( 'precision',               1 )
    pd.set_option( 'display.min_rows',    8     )
    pd.set_option( 'display.float_format', lambda x: '%6.1f' % x)    
    np.set_printoptions(formatter={'float': lambda x: "{:>6.1f}".format(x)})


    save_file_name  = f'{base_dir}/dpcca/data/{nn_mode}/genes_df_lo.pickle'                                # if it exists, just use it
    
    if os.path.isfile(save_file_name):    
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:        checking to see if saved file '{MAGENTA}{save_file_name}{RESET}' exists" )           
      df_sml = pd.read_pickle(save_file_name)
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:        saved dataframe               '{MAGENTA}{save_file_name}{RESET}' exists ... will load and use the previously saved file" )      
    else:
      print( f"ANALYSEDATA:        INFO:          file                        '{RED}{save_file_name}{RESET}' does not exist ... will create" )  
      
      generate_file_name  = f'{base_dir}/dpcca/data/{nn_mode}/genes.pickle'
      print( f"ANALYSEDATA:        INFO:          about to load pickle file   '{MIKADO}{generate_file_name}{RESET}'" ) 
      df  = pd.read_pickle(generate_file_name)
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:          data.shape =  {MIKADO}{df.shape}{RESET}"   )
        print( f"ANALYSEDATA:        INFO:          loading complete"                          )     
      
      #print (  df.head(samples_to_print)  ) 
      #print ( df.max( axis=0 )            )
  
      print ( f"ANALYSEDATA:        INFO:          summary description of data =  \n{MIKADO}{df.describe()}{RESET}", flush='True' )
    
      #print( f"\nANALYSEDATA:        INFO:        data frame with {ORANGE}ALL{RESET} columns" )  
      #df_lo = df.loc[:, :]
      #print (  df_lo, flush='True'  )
  
      #threshold=1.0
      #print( f"\nANALYSEDATA:        INFO:        data frame with {ORANGE} median value <{threshold}{RESET} columns removed" )     
      #df_lo = df.loc[:, (df.median(axis=0)<threshold) ]
      #print (  df_lo  )

      threshold=cov_threshold
      if DEBUG>0:
        print( f"\nANALYSEDATA:        NOTE:        {RED}only genes (columns) having {UNDER}all{RESET}{RED} examples (rows) with an rna-exp value {UNDER}greater than{RESET}{RED} cov_threshold={MIKADO}{threshold}{RESET} {RED}will be retained{RESET}" )      
      df_sml = df.loc[:, (df>threshold).all(axis=0)]
      if DEBUG>9:
        print( f"ANALYSEDATA:        INFO:        {YELLOW}df_sml = df.loc[:, (df>threshold).any(axis=0)].shape = \n{MIKADO}{df_sml.shape}{RESET}" )
      if DEBUG>0:                  
        print( f"ANALYSEDATA:        INFO:        about to save pandas file as {MIKADO}{save_file_name}{RESET}"   )
      df_sml.to_pickle(save_file_name)


    if DEBUG>0:     
      print( f"ANALYSEDATA:        INFO:        {PINK}df_sml.shape             = {MIKADO}{df_sml.shape}{RESET}" )    
    if DEBUG>99:     
      print( f"ANALYSEDATA:        INFO:        {PINK}df_sml                    = \n{MIKADO}{df_sml}{RESET}" ) 
    if DEBUG>90:     
      print( f"ANALYSEDATA:        INFO:        df_sm l.columns.tolist()         = \n{MIKADO}{df_sml.columns.tolist()}{RESET}" )        
    
    # Normalize -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
    df_sml = pd.DataFrame( StandardScaler().fit_transform(df_sml), index=df_sml.index, columns=df_sml.columns )    
    if DEBUG>0:    
      print( f"ANALYSEDATA:        INFO:        {PINK}normalized df_sml.shape  = {MIKADO}{df_sml.shape}{RESET}" ) 
    if DEBUG>99:        
      print( f"ANALYSEDATA:        INFO:        {PINK}normalized df_sml        = \n{{YELLOW}}{df_sml}{RESET}" )       

    # Plot settings --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

    figure_dim=16
    title_size=14
    text_size=12
    sns.set(font_scale = 1.0)
    np.set_printoptions(formatter={'float': lambda x: "{:>7.3f}".format(x)})    
    do_annotate=False
    
    
    do_cpu_covariance='False'
    # CPU version of coveriance ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if do_cpu_covariance=='True':
      fig_1 = plt.figure(figsize=(figure_dim, figure_dim))
      cov=df_sml.cov()
      if DEBUG>0:
        print( f"\n{YELLOW}ANALYSEDATA:        INFO:        cov                 = {MIKADO}{cov.shape}{RESET}" )       
        print( f"{YELLOW}ANALYSEDATA:        INFO:        cov                 = \n{MIKADO}{cov}{RESET}" )         
      if cov.shape[1]==0:
        print( f"{RED}ANALYSEDATA:   FATAL:    covariance matrix is empty ... exiting now [980]{RESET}" )
        sys.exit(0)

      if cov.shape[1]<20:
        label_size=9  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.3f'
      elif cov.shape[1]<30:
        label_size=8  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.2f'
      elif cov.shape[1]<50:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.6)                
        fmt='.1f'
      elif cov.shape[1]<100:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.4)                
        fmt='.1f'
      else:
        label_size=4.5        
        do_annotate=False
        sns.set( font_scale = 0.2 )
        fmt='.1f'  
          
      sns.heatmap(cov, cmap='coolwarm', annot=do_annotate, fmt='.1f')
      plt.xticks(range(cov.shape[1]), cov.columns, fontsize=text_size, rotation=90)
      plt.yticks(range(cov.shape[1]), cov.columns, fontsize=text_size)
      plt.title('Covariance Heatmap', fontsize=title_size) 
      writer.add_figure('Covariance Matrix', fig_1, 0)
      #plt.show()


    do_gpu_covariance='False'
    # GPU version of coveriance ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if do_gpu_covariance=='True':
      fig_11 = plt.figure(figsize=(figure_dim, figure_dim))       
      df_sml_npy = df_sml.to_numpy()
      df_cpy = cupy.asarray( df_sml_npy )                                                                                   # convert to cupy array for parallel processing on GPU(s)
      cov_cpy = cupy.cov( np.transpose(df_cpy) )
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:{ORANGE}        (cupy) cov_cpy.shape     = {MIKADO}{cov_cpy.shape}{RESET}" )
      if DEBUG>999:        
        print( f"ANALYSEDATA:        INFO:{ORANGE}        (cupy) cov_cpy           = {MIKADO}{cov_cpy}{RESET}" )
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:{ORANGE}        about to convert cupy array to numpy array{RESET}" )
      cov_npy =  cupy.asnumpy(cov_cpy)
      if cov_npy.shape[1]==0:
        print( f"{RED}ANALYSEDATA:   FATAL:    covariance matrix is empty ... exiting now [384]{RESET}" )
        sys.exit(0)
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:{ORANGE}        about to convert numpy array to pandas dataframe{RESET}" )
      cov = pd.DataFrame( cov_npy )

      if cov.shape[1]<20:
        label_size=9  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.3f'
      elif cov.shape[1]<30:
        label_size=8  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.2f'
      elif cov.shape[1]<50:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.6)                
        fmt='.1f'
      elif cov.shape[1]<100:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.4)                
        fmt='.1f'
      else:
        label_size=4.5        
        do_annotate=False
        sns.set( font_scale = 0.2 )
        fmt='.1f'  

      if DEBUG>0:          
        print ( f"ANALYSEDATA:        INFO:{BLEU}        about to generate heatmap{RESET}")
      sns.heatmap(cov, cmap='coolwarm', annot=do_annotate, fmt='.1f')
      plt.xticks(range(cov.shape[1]), cov.columns, fontsize=text_size, rotation=90)
      plt.yticks(range(cov.shape[1]), cov.columns, fontsize=text_size)
      plt.title('Covariance Heatmap', fontsize=title_size)
      if DEBUG>0:
        print ( f"ANALYSEDATA:        INFO:{BLEU}        about to add figure to Tensorboard{RESET}")      
      writer.add_figure('Covariance Matrix', fig_11, 0)
      #plt.show()



    do_cpu_correlation='False'
    #  CPU version of Correlation ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
    if do_cpu_correlation=='True':
      fig_2 = plt.figure(figsize=(figure_dim, figure_dim))
      corr=df_sml.corr()
      if DEBUG>0:
        print( f"\n{YELLOW}ANALYSEDATA:        INFO:        corr                 = {MIKADO}{corr.shape}{RESET}" )       
        print( f"{YELLOW}ANALYSEDATA:        INFO:        corr                 = \n{MIKADO}{corr}{RESET}" )       
      if corr.shape[1]==0:
        print( f"{RED}ANALYSEDATA:   FATAL:    correlation matrix is empty ... exiting now [384]{RESET}" )
        sys.exit(0) 
 
 
      if corr.shape[1]<20:
        label_size=9  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.3f'
      elif corr.shape[1]<30:
        label_size=8  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.2f'
      elif corr.shape[1]<50:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.6)                
        fmt='.1f'
      elif corr.shape[1]<100:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.4)                
        fmt='.1f'
      else:
        label_size=4.5        
        do_annotate=False
        sns.set( font_scale = 0.2 )
        fmt='.1f'   
        
      sns.heatmap(corr, cmap='coolwarm', annot=do_annotate, fmt='.1f' )
      plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
      plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
      plt.title('Correlation Heatmap', fontsize=title_size)
      writer.add_figure('Correlation Matrix', fig_2, 0)
      #plt.show()
             

    do_gpu_correlation='True'
    # GPU version of correlation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if do_gpu_correlation=='True':
      if DEBUG>0:          
        print ( f"ANALYSEDATA:        INFO:{BOLD}        Calculating and Displaying Correlation Matrix (GPU version) {MIKADO}(COV_THRESHOLD={cov_threshold}{RESET}){BOLD} (GPU version){RESET}")            
      fig_22 = plt.figure(figsize=(figure_dim, figure_dim))       
      df_sml_npy = df_sml.to_numpy()
      df_cpy = cupy.asarray( df_sml_npy )                                                                                   # convert to cupy array for parallel processing on GPU(s)
      if DEBUG>9:
        print( f"ANALYSEDATA:        INFO:        {GREEN}type(df_cpy)           = {MIKADO}{type(df_cpy)}{RESET}" )  
      corr_cpy = cupy.corrcoef( cupy.transpose( df_cpy ) )
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:{ORANGE}        (cupy) corr_cpy.shape    = {MIKADO}{corr_cpy.shape}{RESET}" )
      if DEBUG>999:        
        print( f"ANALYSEDATA:        INFO:{ORANGE}        (cupy) corr_cpy          = {MIKADO}{corr_cpy}{RESET}" )
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:{ORANGE}        about to convert cupy array to numpy array{RESET}" )
      corr_npy =  cupy.asnumpy( corr_cpy )
      if corr_npy.shape[1]==0:
        print( f"{RED}ANALYSEDATA:   FATAL:    covariance matrix is empty ... exiting now [384]{RESET}" )
        sys.exit(0)

      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:{ORANGE}        about to convert numpy array to pandas dataframe{RESET}" )
      corr_pda = pd.DataFrame( corr_npy )


      if corr_pda.shape[1]<20:
        label_size=9  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.3f'
      elif corr_pda.shape[1]<30:
        label_size=8  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.2f'
      elif corr_pda.shape[1]<50:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.6)                
        fmt='.1f'
      elif corr_pda.shape[1]<100:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.4)                
        fmt='.1f'
      else:
        label_size=4.5        
        do_annotate=False
        sns.set( font_scale = 0.2 )
        fmt='.1f'  

      if DEBUG>0:          
        print ( f"ANALYSEDATA:        INFO:{BLEU}        about to generate Seaborn heatmap of Correlation Matrix{RESET}")
      sns.heatmap(corr_pda, cmap='coolwarm', annot=do_annotate, fmt='.1f')
      plt.xticks(range(corr_pda.shape[1]), corr_pda.columns, fontsize=text_size, rotation=90)
      plt.yticks(range(corr_pda.shape[1]), corr_pda.columns, fontsize=text_size)
      plt.title('Correlation Heatmap', fontsize=title_size)
      if DEBUG>0:
        print ( f"ANALYSEDATA:        INFO:{BLEU}        about to add Correlation Matrix figure to Tensorboard{RESET}")      
      writer.add_figure('Correlation Matrix', fig_22, 0)
      #plt.show() 
 
 
 
    select_hi_corr_genes='False'
    # select high correlation rows and columns ----------------------------------------------------------------------------------------------------------------------------------------------------------------   
    if select_hi_corr_genes=='True':
      if DEBUG>0:          
        print ( f"ANALYSEDATA:        INFO:{BOLD}        Selecting just genes with multiple high correlations {MIKADO}(UQ>{cov_uq_threshold}{RESET}){BOLD} for presentation(CPU version){RESET}")  
      fig_3 = plt.figure(figsize=(figure_dim, figure_dim))
      threshold=cov_uq_threshold
      corr_abs=np.abs(corr)
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_abs.shape           = {MIKADO}{corr_abs.shape}{RESET}" )
      if DEBUG>99:        
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_abs              = \n{MIKADO}{corr_abs}{RESET}" )        
      corr_hi = corr_abs.loc[(corr_abs.quantile(0.75, axis=1)>threshold), (corr_abs.quantile(0.75, axis=1)>threshold) ]
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_hi.shape            = {MIKADO}{corr_hi.shape}{RESET}" )
      if DEBUG>99:
        print( f"ANALYSEDATA:        INFO:       {GREEN} corr_hi               = \n{MIKADO}{corr_hi}{RESET}" )        

      if corr_hi.shape[1]<20:
        label_size=9  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.3f'
      elif corr_hi.shape[1]<30:
        label_size=8  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.2f'
      elif corr_hi.shape[1]<50:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.6)                
        fmt='.1f'
      elif corr_hi.shape[1]<100:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.4)                
        fmt='.1f'
      else:
        label_size=4.5
        do_annotate=False
        sns.set( font_scale = 0.2 )
        fmt='.1f'

      title = 'Just Genes with Multiple High Correlations'
      sns.heatmap(corr_hi, cmap='coolwarm', annot=do_annotate, fmt=fmt )
      plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
      plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
      plt.title(title, fontsize=title_size)
      writer.add_figure(title, fig_3, 0)
      plt.show()


    select_gpu_hi_corr_genes='True'
    # select high correlation rows and columns ----------------------------------------------------------------------------------------------------------------------------------------------------------------   
    if select_gpu_hi_corr_genes=='True':
      if DEBUG>0:          
        print ( f"ANALYSEDATA:        INFO:{BOLD}        Reducing Correlation Matrix to just genes with multiple high correlations and displaying {MIKADO}(COV_UQ_THRESHOLD>{cov_uq_threshold}{RESET}){BOLD} (GPU version){RESET}") 
      fig_33 = plt.figure(figsize=(figure_dim, figure_dim))
      threshold=cov_uq_threshold
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy.shape              = {MIKADO}{corr_cpy.shape}{RESET}" )
      if DEBUG>9:        
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy                    = \n{MIKADO}{corr_cpy}{RESET}" )      
      corr_abs=cupy.absolute(corr_cpy)
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_abs.shape              = {MIKADO}{corr_abs.shape}{RESET}" )
      if DEBUG>9:
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_abs                    = \n{MIKADO}{corr_abs}{RESET}" )
      if DEBUG>99:
        print( f"ANALYSEDATA:        INFO:        {GREEN}type(corr_cpy)              = {MIKADO}{type(corr_cpy)}{RESET}" )    
        print( f"ANALYSEDATA:        INFO:        {GREEN}type(corr_abs)              = {MIKADO}{type(corr_abs)}{RESET}" )
      upper_quartiles   = cupy.percentile (          corr_abs, 75, axis=1          )                       # make a row vector comprising  the upper quartile expression values of each of the genes (columns)
      logical_mask      = cupy.array      (  [ upper_quartiles>cov_uq_threshold ]  )                       # convert it to Boolean values (TRUE, FALSE)
      squeezed_mask     = cupy.squeeze    (           logical_mask                 )                       # get rid of the extra dimension that' for some reason is created in the last step
      integer_mask      = cupy.squeeze    (      squeezed_mask.astype(int)         )                       # change type from Boolean to Integer values (0,1) so we can use it as a mask
      if DEBUG>9:                                                                                          # make sure that there are at least SOME non-zero values in the mask or else we'll make an empty matrix in subsequent steps
        print( f"ANALYSEDATA:        INFO:       {PINK}integer_mask          = \n{MIKADO}{integer_mask}{RESET}" )      
      if cupy.sum( integer_mask, axis=0 )==0:
        print( f"{RED}ANALYSEDATA:        FATAL:    the value provided for COV_UQ_THRESHOLD ({MIKADO}{cov_uq_threshold}{RESET}{RED}) would filter out {UNDER}every{RESET}{RED} gene -- try a smaller vallue.  Exiting now [717]{RESET}" )
        sys.exit(0)
      non_zero_indices  = cupy.nonzero (   integer_mask  )                                                 # make a vector of indices corresponding to non-zero values in the mask 
      corr_reduced_cols = cupy.take ( corr_abs,   non_zero_indices, axis=1  )                      # take columns corresponding to the indices (i.e. delete the others)
      corr_hi           = cupy.take ( corr_reduced_cols,  non_zero_indices, axis=0  )                      # take rows    corresponding to the indices (i.e. delete the others)
      corr_hi           = cupy.squeeze( corr_hi )                                                          # get rid of the extra dimension that for some reason is created in the last step
      corr_hi           = cupy.asnumpy( corr_hi )                                                          # convert to numpy, as matplotlib can't use cupy arrays
      if DEBUG>0:
        print( f"ANALYSEDATA:        INFO:        {PINK}upper_quartiles.shape       = {MIKADO}{upper_quartiles.shape}{RESET}" )
        print( f"ANALYSEDATA:        INFO:        {PINK}logical_mask.shape          = {MIKADO}{logical_mask.shape}{RESET}" )
        print( f"ANALYSEDATA:        INFO:        {PINK}squeezed_mask.shape         = {MIKADO}{squeezed_mask.shape}{RESET}" )
        print( f"ANALYSEDATA:        INFO:        {PINK}integer_mask.shape          = {MIKADO}{integer_mask.shape}{RESET}" )
   #     print( f"ANALYSEDATA:        INFO:        {PINK}non_zero_indices.shape      = {MIKADO}{non_zero_indices.shape}{RESET}" ) 
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_reduced_cols.shape     = {MIKADO}{corr_reduced_cols.shape}{RESET}" )
        print( f"ANALYSEDATA:        INFO:        {GREEN}corr_hi .shape              = {MIKADO}{corr_hi.shape}{RESET}" )


      if corr_hi.shape[1]<20:
        label_size=9  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.3f'
      elif corr_hi.shape[1]<30:
        label_size=8  
        do_annotate=True
        sns.set(font_scale = 1.0)    
        fmt='.2f'
      elif corr_hi.shape[1]<50:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.6)                
        fmt='.1f'
      elif corr_hi.shape[1]<100:
        label_size=8  
        do_annotate=True 
        sns.set(font_scale = 0.4)                
        fmt='.1f'
      else:
        label_size=4.5
        do_annotate=False
        sns.set( font_scale = 0.2 )
        fmt='.1f'

      title = 'Just Genes with Multiple High Correlations'
      sns.heatmap(corr_hi, cmap='coolwarm', annot=do_annotate, fmt=fmt )
      plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
      plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
      plt.title(title, fontsize=title_size)
      writer.add_figure(title, fig_33, 0)
      plt.show()
      
      


    do_pca_dims='False'
    # PCA specifying number of dimensions ----------------------------------------------------------------------------------------------------------------------------------------------------------------   
    if do_pca_dims=='True':
      number_of_samples = np.min(df_sml.shape)
      start_at = int( 0.4 * number_of_samples)
      for n in range( start_at, number_of_samples ):
        print(f'ANALYSEDATA:        INFO: performing PCA for              {MIKADO}{n+1}{RESET} dimensions (out of {MIKADO}{number_of_samples}{RESET}):' )  
        pca                  = PCA(n_components=n+1)                                                         # create a PCA object                                   
        fitted_transform     = pca.fit_transform( df_sml )                                                   # perform PCA on df_sml
        pca_components       = pca.components_                                                               # have to do after 'fit_transform'
        explainable_variance = pca.explained_variance_ratio_
        if DEBUG>99:
          print(f'ANALYSEDATA:        INFO: principle components:\n{ORANGE}{pca_components}{RESET}'                       )
          print(f'ANALYSEDATA:        INFO: variance explained by {BOLD}each{RESET} principle component:\n{explainable_variance}'        )
        print(f'ANALYSEDATA:        INFO: total variance explained by all {MIKADO}{n+1}{RESET} principle components: {MAGENTA if np.sum(explainable_variance)>0.98 else GREEN if np.sum(explainable_variance)>0.95 else PALE_GREEN if np.sum(explainable_variance)>0.9 else WHITE} {np.sum(explainable_variance):>5.6}{RESET}', end='', flush=True)
        print(f'\033[2A')
        if np.sum(explainable_variance)>0.99:
          print(f'\033[1B')
          print(f'ANALYSEDATA:        INFO:   explainable variance exceeds 0.99 .. stopping'                       )
          break


    do_pca_target='False'
    # PCA specifying a target for the explainable variance -----------------------------------------------------------------------------------------------------------------------------------------------
    if do_pca_target=='True':
      for target_explainable_variance in ( 0.95, 0.99):
        print(f'\nANALYSEDATA:        INFO: performing PCA with target_explainable_variance = {MIKADO}{target_explainable_variance}{RESET})' )  
        pca                  = PCA( target_explainable_variance )                                            # create a PCA object                             
        fitted_transform     = pca.fit_transform( df_sml )                                                   # perform PCA on df_sml
        pca_components       = pca.components_                                                               # have to do after 'fit_transform'
        explainable_variance = pca.explained_variance_ratio_
        if DEBUG>0:
          print(f'ANALYSEDATA:        INFO: number of dimensions required: {MIKADO}{explainable_variance.shape[0]}{RESET} of {MIKADO}{df_sml.shape[1]}{RESET} original dimensions'                       )     
          print(f'ANALYSEDATA:        INFO: variance explained by {BOLD}each{RESET} principle component:\n{explainable_variance}'        )
          print(f'ANALYSEDATA:        INFO: total variance explained by all {MIKADO}{explainable_variance.shape[0]}{RESET} principle components: {MAGENTA if np.sum(explainable_variance)>0.98 else GREEN if np.sum(explainable_variance)>0.95 else PALE_GREEN if np.sum(explainable_variance)>0.9 else WHITE} {np.sum(explainable_variance):>5.6}{RESET}', flush=True)
        if DEBUG>99:
          print(f'ANALYSEDATA:        INFO: principle components:\n{ORANGE}{pca_components}{RESET}'                                      )              
    
    
    do_k_means='False'
    # K-means clustering  -----------------------------------------------------------------------------------------------------------------------------------------------
    if do_k_means=='True':
      number_of_samples = np.min (df_sml.shape )     
      for number_of_centroids in range ( 50, number_of_samples, 50 ):
        print(f'ANALYSEDATA:        INFO: performing K-means clustering with these numbers of centroids = {MIKADO}{number_of_centroids}{RESET})' )  
        model = KElbowVisualizer(KMeans(), k=number_of_centroids, metric='calinski_harabasz', timings=False, locate_elbow=False )
        model.fit( df_sml)
        model.show()

        # Reset matplotlib parameters, changed by elbow visualizer
        mpl.rcParams.update(mpl.rcParamsDefault)
        model = KMeans( n_clusters=number_of_centroids )
        model.fit( df_sml )
        all_predictions = model.predict( df_sml )
        centroids       = model.cluster_centers_
        
        plt.figure(figsize=( figure_dim, figure_dim ))
        plt.scatter(df_sml.iloc[:,0].values, df_sml.iloc[:,1].values, c=all_predictions)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='#0f0f0f')
        plt.xlabel('X label')
        plt.ylabel('Y label')
        plt.show()
    
 
    do_TSNE='False'
    # t SNE (t distributed Stochastic Neighbour Embedding) ----------------------------------------------------------------------------------------------------------------
    if do_TSNE=='True':
      if DEBUG>0:     
        print( f"ANALYSEDATA:        INFO:        df_sml                  = \n{PURPLE}{df_sml}{RESET}" )        
      number_of_samples = np.min (df_sml.shape )
      dims = 2
      df_sml_npy=df_sml.to_numpy()    
      for perplexity in range( 25, 50, 25  ):
        print(f'ANALYSEDATA:        INFO: run {MIKADO}{perplexity//5}{RESET} of t-SNE with perplexity = {MIKADO}{perplexity}{RESET}' )  
        result = TSNE( perplexity=perplexity, n_components=dims ).fit_transform( df_sml_npy )            
        if DEBUG>0:
          print( f"ANALYSEDATA:        INFO:       for perplexity={MIKADO}{perplexity}{RESET} TSNE result.shape               = {MIKADO}{result.shape}{RESET}" )
        if DEBUG>99:          
          print( f"ANALYSEDATA:        INFO:       for perplexity={MIKADO}{perplexity}{RESET} first few values of TSNE result = \n{MIKADO}{result[:40,:]}{RESET}" )      
        if DEBUG>0:    
          print( f"ANALYSEDATA:        INFO:        about to call plot with results.shape = {MIKADO}{result.shape}{RESET}" ) 

  
        # Create a scatter plot.
        fig_4 = plt.figure(figsize=(figure_dim, figure_dim))   
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter( result[:,0], result[:,1], lw=0, s=40)
      
        title = f"t-SNE: perplexity = {perplexity}"     
        writer.add_figure( title, fig_4, 0)



    print( f"\n\nANALYSEDATA:        INFO: {YELLOW}finished{RESET}" )
    hours   = round((time.time() - start_time) / 3600, 1  )
    minutes = round((time.time() - start_time) / 60,   1  )
    seconds = round((time.time() - start_time), 0  )
    #pprint.log_section('Job complete in {:} mins'.format( minutes ) )
  
    print(f'ANALYSEDATA:        INFO: took {minutes} mins ({seconds:.1f} secs)')
    
    writer.close()
    
    sys.exit(0)





####################################################################################

    pprint.set_logfiles( log_dir )
  
    pprint.log_section('Loading config.')
    cfg = loader.get_config( args.nn_mode, args.lr, args.batch_size )
    pprint.log_config(cfg)

    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    pprint.log_section('Loading dataset.')
    train_loader, test_loader = loader.get_data_loaders(args,
                                                        cfg,
                                                        batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.pct_test)
                                                        
    if just_test=='False':
      pprint.save_test_indices(test_loader.sampler.indices)

    model = PRECOMPRESS(cfg, nn_type, n_classes, n_genes, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, args.latent_dim, args.em_iters)
    
    model = model.to(device)

    pprint.log_section('Model specs.')
    pprint.log_model(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint.log_section('Training model.\n\n'\
                       'Epoch\t\tTrain x1 err\tTrain x2 err\tTrain l1\t'\
                       '\tTest x1 err\tTest x2 err\tTest l1')

    number_correct_max   = 0
    pct_correct_max      = 0
    test_loss_min        = 999999
    train_loss_min       = 999999
    
    #(10) Train/Test
    
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
    
    test_genes_loss_sum_ave_last           = 99999 
    test_lowest_genes_loss_observed        = 99999      
    test_lowest_genes_loss_observed_epoch  = 0 
        
                     
    print( "ANALYSEDATA:     INFO: \033[1m10 about to commence training loop, one iteration per epoch\033[m" )

    for epoch in range(1, args.n_epochs + 1):   

        print( f'{DIM_WHITE}ANALYSEDATA:     INFO:   {RESET}epoch: {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}, mode: {MIKADO}{input_mode}{RESET}, samples: {MIKADO}{n_samples}{RESET}, batch size: {MIKADO}{batch_size}{RESET}, tile: {MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide: {MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )


        train_loss_images_sum_ave, train_loss_genes_sum_ave, train_l1_loss_sum_ave, train_total_loss_sum_ave =\
                                           train (      args, epoch, train_loader, model, optimizer, writer, train_loss_min, batch_size )

  
        test_total_loss_sum_ave, test_l1_loss_sum_ave, test_loss_min                =\
                                           test ( cfg, args, epoch, test_loader,  model,  tile_size, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, nn_type, annotated_tiles, class_names, class_colours)

        if DEBUG>0:
          if ( (test_total_loss_sum_ave < (test_total_loss_sum_ave_last)) | (epoch==1) ):
            consecutive_test_loss_increases = 0
            last_epoch_loss_increased = False
          else:
            last_epoch_loss_increased = True
            
          print ( f"\
\033[2K\
{DIM_WHITE}ANALYSEDATA:     INFO:   {RESET}\
\r\033[27Cbatch():\
\r\033[73Cae_loss2_sum={GREEN}{test_total_loss_sum_ave:<11.3f}{DULL_WHITE}\
\r\033[98Cl1_loss={test_l1_loss_sum_ave:<11.3f}{DULL_WHITE}\
\r\033[124CBATCH AVE LOSS={GREEN if last_epoch_loss_increased==False else RED}{test_total_loss_sum_ave:<11.3f}\r\033[144C{UP_ARROW if last_epoch_loss_increased==True else DOWN_ARROW}{DULL_WHITE}\
\r\033[167Cmins: total: {test_lowest_total_loss_observed:<11.3f}@{ORANGE}e={test_lowest_total_loss_observed_epoch:<2d}{DULL_WHITE} | \
\r\033[220Cgenes:{test_lowest_genes_loss_observed:><11.3f}@{DULL_BLUE}e={test_lowest_genes_loss_observed_epoch:<2d}{RESET}\
\033[3B\
", end=''  )

          if last_epoch_loss_increased == True:
            consecutive_test_loss_increases +=1

            if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before, so that the most recent model for which the loss improved will be saved
                now = time.localtime(time.time())
                print(time.strftime("ANALYSEDATA:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
                sys.exit(0)
          
          if (last_epoch_loss_increased == False):
            print ('')
  
        test_total_loss_sum_ave_last = test_total_loss_sum_ave
        
        if DEBUG>9:
          print( f"{DIM_WHITE}ANALYSEDATA:     INFO:   test_lowest_total_loss_observed = {MIKADO}{test_lowest_total_loss_observed}{RESET}" )
          print( f"{DIM_WHITE}ANALYSEDATA:     INFO:   test_total_loss_sum_ave         = {MIKADO}{test_total_loss_sum_ave}{RESET}"         )
        
        if test_total_loss_sum_ave < test_lowest_total_loss_observed:
          test_lowest_total_loss_observed       = test_total_loss_sum_ave
          test_lowest_total_loss_observed_epoch = epoch
          if DEBUG>0:
            print( f"{DIM_WHITE}ANALYSEDATA:     INFO:   {GREEN}{ITALICS}new low total loss{RESET}" )

        if epoch % LOG_EVERY == 0:
            save_samples(args.log_dir, model, test_loader, cfg, epoch)
        if epoch % SAVE_MODEL_EVERY == 0:
            save_model(args.log_dir, model)
    
    writer.close()  

    print( "TRAINLENEJ:     INFO: \033[33;1mtraining complete\033[m" )
  
    hours   = round((time.time() - start_time) / 3600, 1  )
    minutes = round((time.time() - start_time) / 60,   1  )
    seconds = round((time.time() - start_time), 0  )
    #pprint.log_section('Job complete in {:} mins'.format( minutes ) )
  
    print(f'TRAINLENEJ:     INFO: run completed in {minutes} mins ({seconds:.1f} secs)')

    save_model(args.log_dir, model)
    pprint.log_section('Model saved.')

# ------------------------------------------------------------------------------

def train(args, epoch, train_loader, model, optimizer, writer, train_loss_min, batch_size  ):  
    """Train PCCA model and update parameters in batches of the whole train set.
    """
    model.train()

    ae_loss2_sum  = 0
    l1_loss_sum   = 0


    for i, (x2) in enumerate(train_loader):

        optimizer.zero_grad()

        x2 = x2.to(device)

        x2r = model.forward(x2)

        ae_loss2 = F.mse_loss(x2r, x2)
        l1_loss  = l1_penalty(model, args.l1_coef)
        #loss     = ae_loss1 + ae_loss2 + l1_loss
        loss     = ae_loss2 

        loss.backward()
        # Perform gradient clipping *before* calling `optimizer.step()`.
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        ae_loss2_sum += ae_loss2.item()
        l1_loss_sum  += l1_loss.item()
        #total_loss = ae_loss1_sum + ae_loss2_sum + l1_loss_sum
        total_loss = ae_loss2_sum
        
        if DEBUG>0:
          print ( f"\
\033[2K\
{DIM_WHITE}ANALYSEDATA:     INFO:{RESET}\
\r\033[27C{DULL_WHITE}train():\
\r\033[40Cn={i+1:>3d}\
\r\033[73Cae_loss2_sum={ ae_loss2:<11.3f}\
\r\033[98Cl1_loss_sum={l1_loss:<11.3f}\
\r\033[124C    BATCH LOSS=\r\033[{139+4*int((ae_loss2*10)//1) if ae_loss2<1 else 150+4*int((ae_loss2*2)//1) if ae_loss2<12 else 160}C{PALE_GREEN if ae_loss2<1 else GOLD if 1<=ae_loss2<2 else PALE_RED}{ae_loss2:11.3f}{RESET}" )
          print ( "\033[2A" )

    ae_loss2_sum  /= (i+1)
    l1_loss_sum   /= (i+1)
    train_msgs     = [ae_loss2_sum, l1_loss_sum]

    if ae_loss2_sum    <  train_loss_min:
      train_loss_min   =  ae_loss2_sum
       
    writer.add_scalar( 'loss_train',      ae_loss2_sum,  epoch )
    writer.add_scalar( 'loss_train_min',  train_loss_min,  epoch )   

    return ae_loss2_sum, l1_loss_sum, total_loss, train_loss_min

# ------------------------------------------------------------------------------

def test( cfg, args, epoch, test_loader, model, tile_size, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, nn_type, annotated_tiles, class_names, class_colours ):
  
    """Test model by computing the average loss on a held-out dataset. No
    parameter updates.
    """
    model.eval()

    ae_loss2_sum = 0
    l1_loss_sum  = 0

    for i, (x2) in enumerate(test_loader):

        x2 = x2.to(device)

        x2r = model.forward(x2)

        ae_loss2 = F.mse_loss(x2r, x2)
        l1_loss  = l1_penalty(model, args.l1_coef)

        ae_loss2_sum += ae_loss2.item()
        l1_loss_sum  += l1_loss.item()

        if i == 0 and epoch % LOG_EVERY == 0:
            cfg.save_comparison(args.log_dir, x2, x2r, epoch, is_x1=False)

        total_loss =  ae_loss2_sum # PGD 200715 - IGNORE IMAGE LOSS AT THE MOMENT 
        
        if DEBUG>0:
          if i==0:
            print ("")
          print ( f"\
\033[2K\
{DIM_WHITE}ANALYSEDATA:     INFO:{RESET}\
\r\033[27Ctest():\
\r\033[40C{DULL_WHITE}n={i+1:>3d}\
\r\033[73Cae_loss2_sum={ ae_loss2_sum:<11.3f}\
\r\033[98Cl1_loss_sum={l1_loss_sum:<11.3f}\
\r\033[124C    BATCH LOSS=\r\033[{139+4*int((ae_loss2*10)//1) if ae_loss2<1 else 150+4*int((ae_loss2*2)//1) if ae_loss2<12 else 160}C{GREEN if ae_loss2<1 else ORANGE if 1<=ae_loss2<2 else RED}{ae_loss2:<11.3f}{RESET}" )
        print ( "\033[2A" )
    
    print ("")

    ae_loss2_sum /= (i+1)
    l1_loss_sum  /= (i+1)

    if ae_loss2_sum    <  test_loss_min:
       test_loss_min   =  ae_loss2_sum
    
    if DEBUG>9:
      print ( f"ANALYSEDATA:     INFO:      test(): x2.shape  = {MIKADO}{x2.shape}{RESET}" )
      print ( f"ANALYSEDATA:     INFO:      test(): x2r.shape = {MIKADO}{x2r.shape}{RESET}" )
    
    if (epoch+1)%1==0:
      if DEBUG>0:
        number_to_display=28
        print ( f"{DIM_WHITE}ANALYSEDATA:     INFO:     {RESET}test(): original/reconstructed values for first {MIKADO}{number_to_display}{RESET} examples" )
        np.set_printoptions(formatter={'float': lambda x: "{:>8.2f}".format(x)})
        x2_nums  = x2.cpu().detach().numpy()  [12,0:number_to_display]                                               # the choice of sample 12 is arbitrary
        x2r_nums = x2r.cpu().detach().numpy() [12,0:number_to_display]
        
        print (  f"x2     = {x2_nums}"     )
        print (  f"x2r    = {x2r_nums}"     )
        errors = np.absolute( ( x2_nums - x2r_nums  ) )
        ratios= np.around(np.absolute( ( (x2_nums+.00001) / (x2r_nums+.00001)  ) ), decimals=2 )                      # to avoid divide by zero error
        np.set_printoptions(linewidth=600)   
        np.set_printoptions(edgeitems=600)
        np.set_printoptions(formatter={'float': lambda x: f"{GREEN if abs(x-1)<0.01 else PALE_GREEN if abs(x-1)<0.05 else GOLD if abs(x-1)<0.1 else PALE_RED}{x:^8.2f}"})     
        print (  f"errors = {errors}"     )
        print (  f"ratios = {ratios}"     )
        
    writer.add_scalar( 'loss_test',      ae_loss2_sum,   epoch )
    writer.add_scalar( 'loss_test_min',  test_loss_min,  epoch )    
    
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
      print( f"TRAINLENEJ:     INFO:   save_model(){DULL_YELLOW}{ITALICS}: new lowest loss on this epoch... saving model to {fpath}{RESET}" )      
    model_state = model.state_dict()
    torch.save(model_state, fpath)


# Support function for t-SNE ----------------------------------------------------------------------------------------------------------------------    
def plot( x, figure_dim, perplexity, writer):

  if DEBUG>99:    
    print( f"ANALYSEDATA:        INFO:                                      x.shape = {MIKADO}{x.shape}{RESET}" ) 

  
  # Create a scatter plot.
  fig_4 = plt.figure(figsize=(figure_dim, figure_dim))   
  ax = plt.subplot(aspect='equal')
  #sc = ax.scatter( np.abs(x[:,0]), np.abs(x[:,1]), lw=0, s=40, c=1)  
  sc = ax.scatter( np.abs(x[:,0]), np.abs(x[:,1]),  lw=0, s=40)

  title = f"t-SNE: perplexity = {perplexity}"     
  writer.add_figure( title, fig_4, 0)
  
  return



  
  
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
    p.add_argument('--wall_time',                      type=int,   default=24)
    p.add_argument('--seed',                           type=int,   default=0)
    p.add_argument('--nn_mode',                        type=str,   default='analyse_data')
    p.add_argument('--use_same_seed',                  type=str,   default='False')
    p.add_argument('--nn_type',             nargs="+", type=str,   default='VGG11')
    p.add_argument('--encoder_activation',  nargs="+", type=str,   default='sigmoid')                              # USED BY AEDENSE(), AEDENSEPOSITIVE()
    p.add_argument('--nn_dense_dropout_1',  nargs="+", type=float, default=0.0)                                    # USED BY DENSE()    
    p.add_argument('--nn_dense_dropout_2',  nargs="+", type=float, default=0.0)                                    # USED BY DENSE()
    p.add_argument('--dataset',                        type=str,   default='STAD')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--input_mode',                     type=str,   default='NONE')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--n_samples',           nargs="+", type=int,   default=101)                                    # USED BY generate()      
    p.add_argument('--n_tiles',             nargs="+", type=int,   default=100)                                    # USED BY generate() and all ...tiler() functions 
    p.add_argument('--supergrid_size',                 type=int,   default=1)                                      # USED BY main()
    p.add_argument('--patch_points_to_sample',         type=int,   default=1000)                                   # USED BY tiler()    
    p.add_argument('--tile_size',           nargs="+", type=int,   default=128)                                    # USED BY many
    p.add_argument('--gene_data_norm',      nargs="+", type=str,   default='NONE')                                 # USED BY generate()
    p.add_argument('--gene_data_transform', nargs="+", type=str,   default='NONE' )
    p.add_argument('--n_genes',                        type=int,   default=506)                                   # USED BY main() and generate()
    p.add_argument('--remove_unexpressed_genes',       type=str,   default='True' )                               # USED generate()
    p.add_argument('--remove_low_expression_genes',    type=str,   default='True' )                               # USED generate()
    p.add_argument('--low_expression_threshold',       type=float, default=0      )                               # USED generate()
    p.add_argument('--batch_size',         nargs="+",  type=int,   default=256)                                   # USED BY tiler() 
    p.add_argument('--learning_rate',      nargs="+",  type=float, default=.00082)                                # USED BY main()                               
    p.add_argument('--n_epochs',                       type=int,   default=10)
    p.add_argument('--pct_test',                       type=float, default=0.2)
    p.add_argument('--lr',                             type=float, default=0.0001)
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
    p.add_argument('--class_names',        nargs="+"                                  )                           # USED BY main()
    p.add_argument('--long_class_names',   nargs="+"                                  )                           # USED BY main()
    p.add_argument('--class_colours',      nargs="*"                                  )    
    p.add_argument('--target_tile_coords', nargs=2,    type=int, default=[2000,2000]       )                       # USED BY tiler_set_target()
    
    p.add_argument('cov_threshold',                    type=float, default=8.0)                                      # USED BY main()   
    p.add_argument('cov_uq_threshold',                 type=float, default=0.0)                                      # USED BY main()   
        
    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 12
    args.pin_memory = torch.cuda.is_available()

    main(args)
