"""=============================================================================
Code to support Data Analysis Mode
============================================================================="""

import os
import sys
import json
import math
import time
import cuda
import pprint
import argparse
import numpy as np
import cupy
import torch

import pandas    as pd
import seaborn   as sns
import missingno as msno

from   tabulate          import tabulate

from   matplotlib        import cm
from   matplotlib.colors import ListedColormap
from   matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import matplotlib
import matplotlib             as mpl
import matplotlib.patheffects as pe
import matplotlib.colors      as mcolors
import matplotlib.pyplot      as plt
import matplotlib.lines       as mlines
import matplotlib.patches     as mpatches
import matplotlib.gridspec    as gridspec
#from  matplotlib import figure
#from pytorch_memlab import profile

from torch                      import optim
from torch.nn.utils             import clip_grad_norm_
from torch.nn                   import functional
from torch.nn                   import DataParallel
from itertools                  import product, permutations
from PIL                        import Image

from   torchvision              import datasets, transforms
from   torch.utils.tensorboard  import SummaryWriter

import torchvision
import torch.utils.data

from sklearn.preprocessing      import StandardScaler
from sklearn.manifold           import TSNE 
from sklearn.datasets           import load_digits
from sklearn.decomposition      import PCA
from sklearn.cluster            import KMeans

from yellowbrick.cluster        import KElbowVisualizer

from data                       import loader
from data.pre_compress.generate import generate
from data.pre_compress.config   import pre_compressConfig
from data                       import loader
from models                     import LENETIMAGE
from models                     import PRECOMPRESS

from tiler                      import *
from tiler_scheduler            import *
from tiler_threader             import *
from tiler_set_target           import *



inline_rc = dict(mpl.rcParams)
pd.set_option('max_colwidth', 50)
#===========================================

np.set_printoptions(edgeitems=500)
np.set_printoptions(linewidth=400)

pd.set_option('display.max_rows',     128 )
pd.set_option('display.max_columns',  128 )
pd.set_option('display.width',        100 )
pd.set_option('display.max_colwidth',  99 )  

torch.backends.cudnn.enabled     = True                                                                     #for CUDA memory optimizations
# ------------------------------------------------------------------------------

LOG_EVERY        = 10
SAVE_MODEL_EVERY = 100

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

DEBUG=1

device = cuda.device()

pool = cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)
cupy.cuda.set_allocator(pool.malloc)

rows=38
cols=38

# ------------------------------------------------------------------------------

def main(args):

  """Main program: ...
  """
  
  
  os.system("taskset -p 0xffffffff %d" % os.getpid())

  now = time.localtime(time.time())
  print(time.strftime("\nANALYSEDATA:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
  start_time = time.time()
    
  print ( "ANALYSEDATA:     INFO:   torch       version =    {:}".format (  torch.__version__       )  )
  print ( "ANALYSEDATA:     INFO:   torchvision version =    {:}".format (  torchvision.__version__ )  )
  print ( "ANALYSEDATA:     INFO:   matplotlib version  =    {:}".format (  matplotlib.__version__ )   )   


  print( f"ANALYSEDATA:     INFO:  common args: \
dataset={MIKADO}{args.dataset}{RESET}, \
mode={MIKADO}{args.input_mode}{RESET}, \
samples={MIKADO}{args.n_samples[0]}{RESET}",
  flush=True )


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
  nn_type_rna                = args.nn_type_rna
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
  label_swap_pct         = args.label_swap_pct
  make_grey_pct          = args.make_grey_pct
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
  
  do_covariance               = args.do_covariance
  
  threshold                   = args.cov_threshold
  cutoff_percentile           = args.cutoff_percentile  

  do_correlation              = args.do_correlation
  use_cupy                    = args.a_d_use_cupy
  cov_uq_threshold            = args.cov_uq_threshold
  show_rows                   = args.show_rows
  show_cols                   = args.show_cols   

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
                            nn_type_rna  =   nn_type_rna,
                 nn_dense_dropout_1  =   nn_dense_dropout_1,
                 nn_dense_dropout_2  =   nn_dense_dropout_2,
                        nn_optimizer =  nn_optimizer,
                          stain_norm =  stain_norm,
                      gene_data_norm =  gene_data_norm, 
                 gene_data_transform =  gene_data_transform,                                                
                  label_swap_pct = [   0.0   ],
                   make_grey_pct = [   0.0   ],
                              jitter = [  [ 0.0, 0.0, 0.0, 0.0 ] ]  )

  param_values = [v for v in parameters.values()]

  if DEBUG>9999:
    print("\033[2Clr\r\033[14Cn_samples\r\033[26Cbatch_size\r\033[38Cn_tiles\r\033[48Ctile_size\r\033[59Crand_tiles\r\033[71Cnn_type_rna\r\033[90Cnn_drop_1\r\033[100Cnn_drop_2\r\033[110Coptimizer\r\033[120Cstain_norm\
\r\033[130Cg_norm\r\033[140Cg_xform\r\033[155Clabel_swap\r\033[170Cgreyscale\r\033[182Cjitter vector\033[m")
    for       lr,      n_samples,        batch_size,                 n_tiles,         tile_size,        rand_tiles,         nn_type_rna,          nn_dense_dropout_1, nn_dense_dropout_2,       nn_optimizer,          stain_norm, \
    gene_data_norm,    gene_data_transform,   label_swap_pct, make_grey_pct,   jitter in product(*param_values):
      print( f"\033[0C{MIKADO}{lr:9.6f} \r\033[14C{n_samples:<5d} \r\033[26C{batch_size:<5d} \r\033[38C{n_tiles:<5d} \r\033[48C{tile_size:<3d} \r\033[59C{rand_tiles:<5s} \r\033[71C{nn_type_rna:<8s} \r\033[90C{nn_dense_dropout_1:<5.2f}\
\r\033[100C{nn_dense_dropout_2:<5.2f} \r\033[110C{nn_optimizer:<8s} \r\033[120C{stain_norm:<10s} \r\033[130C{gene_data_norm:<10s} \r\033[140C{gene_data_transform:<10s} \r\033[155C{label_swap_pct:<6.1f}\
\r\033[170C{make_grey_pct:<5.1f}\r\033[182C{jitter:}{RESET}" )      


  
  # (B) RUN JOB LOOP

  run=0
  
  for lr, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type_rna, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_pct, make_grey_pct, jitter in product(*param_values): 
    
    run+=1


    #(1) set up Tensorboard
    
    print( "ANALYSEDATA:     INFO: \033[1mI  about to set up Tensorboard\033[m" )
    
    if input_mode=='image':
#      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type_rna}; opt={nn_optimizer}; n_samps={n_samples}; n_t={n_tiles}; t_sz={tile_size}; rnd={rand_tiles}; tot_tiles={n_tiles * n_samples}; n_epochs={n_epochs}; bat={batch_size}; stain={stain_norm};  uniques>{min_uniques}; grey>{greyness}; sd<{min_tile_sd}; lr={lr}; lbl_swp={label_swap_pct*100}%; greyscale={make_grey_pct*100}% jit={jitter}%' )
      writer = SummaryWriter(comment=f' NN={nn_type_rna}; n_smp={n_samples}; sg_sz={supergrid_size}; n_t={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; n_e={n_epochs}; b_sz={batch_size}' )
    elif input_mode=='rna':
      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; n_smp={n_samples}; n_g={n_genes}')
    elif input_mode=='image_rna':
      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type_rna}; opt={nn_optimizer}; n_smp={n_samples}; n_t={n_tiles}; t_sz={tile_size}; t_tot={n_tiles*n_samples}; n_g={n_genes}; gene_norm={gene_data_norm}; g_xform={gene_data_transform}; n_e={n_epochs}; b_sz={batch_size}; lr={lr}')
    else:
      print( f"{RED}ANALYSEDATA:   FATAL:    input mode of type '{MIKADO}{input_mode}{RESET}{RED}' is not supported [314]{RESET}" )
      sys.exit(0)

    print( "ANALYSEDATA:     INFO:     \033[3mTensorboard has been set up\033[m" ) 
      









    #(2) start user selected data analyses
    
    print( f"ANALYSEDATA:     INFO: {BOLD}II about to start user selected data analyses{RESET}" )

    # Global settings --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    genes_to_show      = 30  
    title_size         = 14
    text_size          = 12
    sns.set(font_scale = 1.0)
    np.set_printoptions(formatter={'float': lambda x: "{:>7.3f}".format(x)})    
    do_annotate=False

    if DEBUG>0:
      print( f"ANALYSEDATA:     INFO:      {ITALICS}figure_width{RESET}  =   {CYAN}{figure_width}{RESET}"   )
      print( f"ANALYSEDATA:     INFO:      {ITALICS}figure_height{RESET} =   {CYAN}{figure_height}{RESET}"  )
      print( f"ANALYSEDATA:     INFO:      {ITALICS}title_size{RESET}    =   {CYAN}{title_size}{RESET}"     )
      print( f"ANALYSEDATA:     INFO:      {ITALICS}text_size{RESET}     =   {CYAN}{text_size}{RESET}"      )
      print( f"ANALYSEDATA:     INFO:      {ITALICS}do_annotate{RESET}   =   {CYAN}{do_annotate}{RESET}"    )
  
      
   #pd.set_option( 'display.max_columns',    25 )
   #pd.set_option( 'display.max_categories', 24 )
   #pd.set_option( 'precision',               1 )
    pd.set_option( 'display.min_rows',    8     )
    pd.set_option( 'display.float_format', lambda x: '%6.3f' % x)    
    np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})

    # Load ENSG->gene name lookup table -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
    ENSG_reference_merged_file_name = f"{data_dir}/ENSG_reference_merged"
    if DEBUG>0:  
      print ( f"ANALYSEDATA:     INFO:      loading ENSG_reference_merged_file_name (containing ENSG->gene name mapping) from {MAGENTA}{ENSG_reference_merged_file_name}{RESET}", flush=True )      
    df_map = pd.read_csv( ENSG_reference_merged_file_name, sep='\t' )
    gene_names_table=df_map.iloc[:,1]                                                                      # select all rows for column 1
    if DEBUG>99:
      print ( f"ANALYSEDATA:     INFO:      pandas description of df_map: \n{CYAN}{df_map.describe}{RESET}", flush=True )  
    if DEBUG>99:
      print ( f"ANALYSEDATA:     INFO:      df_map.shape = {CYAN}{ df_map.shape}{RESET}", flush=True )  
      print ( f"ANALYSEDATA:     INFO:      start of df_map: \n{CYAN}{df_map.iloc[:,1]}{RESET}", flush=True )
    if DEBUG>99:
      print(tabulate(df_map, tablefmt='psql'))

    

    if use_cupy=='True':
      if DEBUG>0:
        print( f"{ORANGE}ANALYSEDATA:     INFO:      cupy mode has been selected (use_cupy='True'). cupy data structures (and not numpy data structures) will be established and used{RESET}" )       
        
      generate_file_name  = f'{base_dir}/dpcca/data/analyse_data/genes_cupy.pickle.npy'
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:      1. about to load pickled cupy dataframe file from file: '{MIKADO}{generate_file_name}{RESET}'", flush=True ) 
      try:
        df_cpy  = cupy.load( generate_file_name, mmap_mode='r+', allow_pickle='True')
      except Exception:
        print( f"{RED}ANALYSEDATA:     FATAL: file {MAGENTA}{generate_file_name}{RESET}{RED} doesn't exist. Try running again with {MIKADO}NN_MODE='pre_compress'{RESET}{RED} mode to create it...  Exiting now [277]{RESET}" )
        sys.exit(0)
        
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:           {DIM_WHITE}data.shape                     =  {MIKADO}{df_cpy.shape}{RESET}", flush=True   )
        print( f"ANALYSEDATA:     INFO:           {DIM_WHITE}loading complete{RESET}"                          )           

      if DEBUG>0:           
        print( f"ANALYSEDATA:     INFO:           {DIM_WHITE}df_cpy[0:{MIKADO}{rows}{RESET}{DIM_WHITE},0:{MIKADO}{cols}{RESET}{DIM_WHITE}] = \n{COQUELICOT}{df_cpy[0:rows,0:cols]}{RESET}"       ) 


      # Normalize -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:      2. about to normalize values to have a mean of zero and a standard deviation of one{RESET}", flush=True ) 
      df_cpy = df_cpy - cupy.expand_dims( cupy.mean ( df_cpy, axis=1 ), axis=1 )
      df_cpy = df_cpy / cupy.expand_dims( cupy.std  ( df_cpy, axis=1 ), axis=1 )
      if DEBUG>0:    
        print( f"ANALYSEDATA:     INFO:           {DIM_WHITE}normalized df_cpy.shape = {MIKADO}{df_cpy.shape}{RESET}", flush=True ) 

      if DEBUG>0:           
        print( f"ANALYSEDATA:     INFO:           {DIM_WHITE}df_cpy[0:{MIKADO}{rows}{RESET}{DIM_WHITE},0:{MIKADO}{cols}{RESET}{DIM_WHITE}] = \n{ASPARAGUS}{df_cpy[0:rows,0:cols]}{RESET}"       ) 


      # Add pseudo-index as the first row. Divide index by very large number so that it won't interfere with the covariance and correlation calculations ---------------------------------------------------   
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:      3. now adding an (encoded to be harmless) index as the first row{RESET}", flush=True ) 
      if DEBUG>0:          
        print( f"ANALYSEDATA:     INFO:           {DIM_WHITE}df_cpy.shape                   = {MIKADO}{df_cpy.shape}{RESET}", flush=True )
      scale_down=10000000
      index_col = cupy.transpose(cupy.expand_dims(cupy.asarray([ n/scale_down for n in range (0, df_cpy.shape[1])  ]), axis=1))

      index_of_cols = index_col
      if DEBUG>9:
        print( f"ANALYSEDATA:     INFO:        {AZURE}df_cpy.shape                   = {MIKADO}{df_cpy.shape}{RESET}", flush=True )
        print( f"ANALYSEDATA:     INFO:        {AZURE}index_of_cols.shape            = {MIKADO}{index_of_cols.shape}{RESET}", flush=True ) 
      if DEBUG>99:
        print( f"ANALYSEDATA:     INFO:        {AZURE}index_of_cols                    = {MIKADO}{index_col}{RESET}", flush=True )  
      df_cpy = cupy.vstack ( [ index_col, df_cpy ])
      if DEBUG>9:
        print( f"ANALYSEDATA:     INFO:        {AZURE}stacked df_cpy.shape           = {MIKADO}{df_cpy.shape}{RESET}", flush=True ) 
      if DEBUG>99:        
        np.set_printoptions(formatter={'float': lambda x: "{:>13.8f}".format(x)})
        print( f"ANALYSEDATA:     INFO:        {AZURE}stacked df_cpy (start)         = \n{MIKADO}{df_cpy[0:10,0:12],}{RESET}", flush=True )
        print( f"ANALYSEDATA:     INFO:        {AZURE}stacked df_cpy  (end)          = \n{MIKADO}{df_cpy[0:10,-12:-1],}{RESET}", flush=True )
        np.set_printoptions(formatter={'float': lambda x: "{:>13.2f}".format(x)})
      if DEBUG>0:           
        print( f"ANALYSEDATA:     INFO:           {DIM_WHITE}(cupy) (after) df_cpy[0:{MIKADO}{rows}{RESET},0:{MIKADO}{cols}] = \n{BLEU}{df_cpy[0:rows,0:cols]}{RESET}"       )



      # Remove genes with low expression values -----------------------------------------------------------------------------------------------------------------------------------------------------------   
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:       4. a) about to apply {CYAN}COV_THRESHOLD{RESET} to filter out genes that aren't very expressive across all samples (genes whose {MIKADO}{cutoff_percentile}%{RESET} percentile is less than the user provided {CYAN}COV_THRESHOLD{CYAN} = {MIKADO}{threshold}{RESET})", flush=True )    
      if DEBUG>9:
        print( f"ANALYSEDATA:     INFO:              {WHITE}df_cpy.shape (before)           = {MIKADO}{df_cpy.shape}{RESET}"                 )
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:       {WHITE}4. b) calculating percentiles for each column (gene){RESET}", flush=True )            
      percentiles  = cupy.percentile (   cupy.abs(df_cpy), cutoff_percentile, axis=0          )                                     # row vector "90% of values lie above ..."
      if DEBUG>9:
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}percentiles                   = {MIKADO}{percentiles}{RESET}" )        
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}percentiles.shape             = {MIKADO}{percentiles.shape}{RESET}" )        
      logical_mask      = cupy.array(  [ ( percentiles ) > threshold ]  )                                  # filter out genes that aren't very expressive across all samples
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}logical_mask.shape            = {MIKADO}{logical_mask.shape}{RESET}" )    # 
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}about to convert logical mask into a integer mask", flush=True )          
      integer_mask      = cupy.squeeze    (      logical_mask.astype(int)         )                        # change type from Boolean to Integer values (0,1) so we can use it as a mask
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}integer_mask.shape            = {MIKADO}{integer_mask.shape}{RESET}" )
      if DEBUG>9:                                                                                          # make sure that there are at least SOME non-zero values in the mask or else we'll make an empty matrix in subsequent steps
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}integer_mask          = \n{MIKADO}{integer_mask}{RESET}" )      
      if cupy.sum( integer_mask, axis=0 )==0:
        print( f"{RED}ANALYSEDATA: FATAL:  the value provided for {CYAN}COV_THRESHOLD{RESET} ({MIKADO}{threshold}{RESET}{RED}) would filter out {UNDER}every{RESET}{RED} gene -- try a smaller vallue.  Exiting now [755]{RESET}" )
        sys.exit(0)
      non_zero_indices  = cupy.nonzero (   integer_mask  )                                                 # make a vector of indices corresponding to non-zero values in the mask 

      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:       {WHITE}4. c) removing columns corresponding to low correlation genes" ) 
      df_cpy = cupy.take ( df_cpy,   non_zero_indices, axis=1  )                                           # take columns corresponding to the indices (i.e. delete the others)
      df_cpy = cupy.squeeze( df_cpy )                                                                      # get rid of the extra dimension that for some reason is created in the last step                                                         # convert to numpy, as matplotlib can't use cupy arrays
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:               {MIKADO}{logical_mask.shape[1]-df_cpy.shape[1]:,}{RESET}{DIM_WHITE} of {MIKADO}{logical_mask.shape[1]:,}{RESET} {DIM_WHITE}genes have been removed from consideration{RESET}" ) 

      
      if DEBUG>0:
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}df_cpy.shape (after)          = {MIKADO}{df_cpy.shape}{RESET}"                 )
      if DEBUG>0:           
        print( f"ANALYSEDATA:     INFO:               {DIM_WHITE}(cupy) (after removal of low expression genes) df_cpy[0:{MIKADO}{rows}{RESET},0:{MIKADO}{cols}{RESET}{DIM_WHITE}] = \n{AUREOLIN}{df_cpy[0:rows,0:cols]}{RESET}"       )




      if do_covariance=='True':
        if use_cupy=='True':
          if DEBUG>0:          
            print ( f"ANALYSEDATA:        INFO:{BOLD}        V1 a) calculating and Displaying Covariance Matrix (GPU version){RESET}")  
          if DEBUG>9: 
            print( f"ANALYSEDATA:        INFO:{WHITE}              (cupy) (before covariance algorithm) df_cpy{RESET}[0:{MIKADO}{rows}{RESET},0:{MIKADO}{cols}{RESET}] = \n{PURPLE}{df_cpy[0:rows,0:cols]}{RESET}"   )
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:{WHITE}        V1 b) about to perform covariance on df_cpy{RESET}" )                                                                                            # convert to cupy array for parallel processing on GPU(s)
          cov_cpy = cupy.cov( np.transpose(df_cpy) )
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:{DIM_WHITE}              (cupy) (after covariance algorithm) cov_cpy.shape       = {MIKADO}{cov_cpy.shape}{RESET}"       )
          if DEBUG>0:        
            
            print( f"ANALYSEDATA:        INFO:{DIM_WHITE}              (cupy) cov_cpy[0:{MIKADO}{rows}{RESET},0:{MIKADO}{cols}{RESET}]  = \n{COQUELICOT}{cov_cpy[0:rows,0:cols]}{RESET}"           )
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:{WHITE}      V2 about to convert cupy array to numpy array{RESET}"                                     )
            
          cov_npy =  cupy.asnumpy(cov_cpy)
          if DEBUG>9:
            print( f"ANALYSEDATA:        INFO:{DIM_WHITE}            (cupy) (after cupy.asnumpy)  cov_npy.shape          = {MIKADO}{cov_npy.shape}{RESET}"   )
          if DEBUG>9:        
            print( f"ANALYSEDATA:        INFO:{DIM_WHITE}            (cupy) (after cupy.asnumpy)  cov_npy[0:{MIKADO}{rows}{RESET},0:{MIKADO}{cols}{RESET}]         = \n{CARRIBEAN_GREEN}{cov_npy[0:rows,0:cols]}{RESET}"       )
            
          del cov_cpy
          if cov_npy.shape[1]==0:
            print( f"{RED}ANALYSEDATA:   FATAL:    covariance matrix is empty ... exiting now [384]{RESET}" )
            sys.exit(0)
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:{WHITE}      V3 about to convert numpy array to pandas dataframe so that it can be displayed using seaborn and tensorboard{RESET}" )
          cov = pd.DataFrame( cov_npy )
          if DEBUG>9:
            print( f"ANALYSEDATA:        INFO:{DIM_WHITE}            done{RESET}" )
          if DEBUG>9:
            print (cov)
      
      
          if np.max(cov.shape)<=12:
            sns.set(font_scale=1)    
            title_size  = 16
            label_size  = 11
            text_size   = 9            
            do_annotate=True
            fmt='.1f'
          if np.max(cov.shape)<=20:
            sns.set(font_scale=1)    
            title_size  = 16
            label_size  = 11
            text_size   = 9   
            do_annotate=True
            fmt='.3f'
          elif np.max(cov.shape)<=30:
            sns.set(font_scale=1)    
            title_size  = 16
            label_size  = 11
            text_size   = 9    
            do_annotate=True
            fmt='.2f'
          elif np.max(cov.shape)<=50:
            sns.set(font_scale=1)    
            label_size=9 
            do_annotate=True 
            fmt='.1f'
          elif np.max(cov.shape)<=125:
            sns.set(font_scale=0.5)    
            label_size=7  
            do_annotate=True 
            fmt='.1f'
          elif np.max(cov.shape)<=250:
            sns.set(font_scale=0.4)    
            label_size=7  
            do_annotate=False 
            fmt='.1f'
          else:
            sns.set(font_scale=0.25)    
            label_size=6
            do_annotate=False
            fmt='.1f' 
      
          if DEBUG>0:          
            print ( f"ANALYSEDATA:        INFO:{WHITE}      V4 about to generate heatmap{RESET}")

          fig_11 = plt.figure(figsize=(figure_width, figure_height))                                       # set up tensorboard figure

          sns.heatmap(cov, cmap='coolwarm', square=True, cbar=True, cbar_kws={"shrink": .77}, annot=do_annotate, annot_kws={"size": text_size}, fmt=fmt)
          # ~ sns.set(rc={'figure.figsize':(0.6,0.6)})
          # ~ plt.figure(figsize=(1, 1))
          plt.tight_layout()          
          plt.xticks(range(cov.shape[1]), cov.columns, fontsize=text_size, rotation=90)
          plt.yticks(range(cov.shape[1]), cov.columns, fontsize=text_size )
          plt.title('Covariance Heatmap', fontsize=title_size)
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:{WHITE}      V5 about to add figure to Tensorboard{RESET}" )            
          writer.add_figure('Covariance Matrix', fig_11, 0)
          #plt.show()



      
      if do_correlation=='True':
        # GPU version of correlation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        if use_cupy=='True':
          
          if DEBUG>0:          
            print ( f"\nANALYSEDATA:        INFO:{BOLD}        W1 Calculating and Displaying Correlation Matrix (GPU version){RESET}")            
          if DEBUG>9:
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}(cupy) df_cpy.shape               = {MIKADO}{df_cpy.shape}{RESET}" )  
          if DEBUG>9:
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}(cupy) (before) df_cpy[0:{MIKADO}{rows}{RESET}{DIM_WHITE}],0:{MIKADO}{cols}{RESET}{DIM_WHITE}] = \n{BLEU}{df_cpy[0:rows,0:cols]}{RESET}"       )
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})            
          if DEBUG>0:          
            print ( f"ANALYSEDATA:       INFO:           about to calculate ({MIKADO}{df_cpy.shape[1]} x {df_cpy.shape[1]}{RESET}) correlation coefficients matrix (this can take a long time if there are a large number of genes, as it's an outer product)", flush=True)            
          
          # Do correlation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
          col_i     = df_cpy[0,:]                                                                          # grab index row now as we will soon reinstate it                
          corr_cpy = cupy.corrcoef( cupy.transpose( df_cpy[1:,:]) )
          del  df_cpy
          if DEBUG>0:
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)}) 
            # ~ print( f"ANALYSEDATA:        INFO:          col_i     (index)            = \n{MIKADO}{col_i[0:12]}{RESET}", flush=True  )                
            # ~ print( f"ANALYSEDATA:        INFO:        post corr (body)             = \n{MIKADO}{corr_cpy[0:7,0:12]}{RESET}", flush=True  )
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}(post correlation algorithm) corr_cpy.shape = {MIKADO}{corr_cpy.shape}{RESET}" )
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}(post correlation algorithm) corr_cpy       = \n{CARRIBEAN_GREEN}{corr_cpy[0:rows,0:cols]}{RESET}" )
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
    
    
          # Reinstate gene (column) index ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:        {WHITE}W2 a) about to reinstate gene (column) index{RESET}", flush=True )  
          
          index_of_columns=col_i
          if DEBUG>0:
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}corr_cpy.shape                = {MIKADO}{corr_cpy.shape}{RESET}" )
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}index_of_columns.shape        = {MIKADO}{index_of_columns.shape}{RESET}" )
          if DEBUG>9:
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}index_of_columns              = \n{MIKADO}{index_of_columns[0:cols]}{RESET}", flush=True  )                
          corr_cpy = cupy.vstack ( [ index_of_columns, corr_cpy ])          
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}corr_cpy post vstack (index)           = \n{COTTON_CANDY}{corr_cpy[:cols,0]}{RESET}", flush=True )
            print( f"ANALYSEDATA:        INFO:          {DIM_WHITE}corr_cpy post vstack (all)             = \n{COTTON_CANDY}{corr_cpy[:cols,:cols]}{RESET}", flush=True )
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
  
          # Reinstate gene (row) index ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:        {WHITE}W2 b) about to reinstate gene (row) index{RESET}", flush=True )  
  
          index_of_rows=cupy.transpose(cupy.expand_dims( cupy.hstack(( [0,col_i] )), axis=0))                # use hstack to add an arbitrary value (0) to the start of col_i array, because corr_cpy now has an index row atop it
          if DEBUG>0:
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
            print( f"ANALYSEDATA:        INFO:                {DIM_WHITE}corr_cpy.shape                = {MIKADO}{corr_cpy.shape}{RESET}" )
            print( f"ANALYSEDATA:        INFO:                {DIM_WHITE}index_of_rows.shape           = {MIKADO}{index_of_rows.shape}{RESET}" )
          if DEBUG>9:
            print( f"ANALYSEDATA:        INFO:                {DIM_WHITE}index_of_rows                 = \n{MIKADO}{index_of_rows[0:rows]}{RESET}",  flush=True  )                
          corr_cpy = cupy.hstack ( [ index_of_rows, corr_cpy ])          
          if DEBUG>0:
            print( f"ANALYSEDATA:        INFO:                {DIM_WHITE}corr_cpy post hstack (index)           = \n{CAMEL}{corr_cpy[:rows,0]}{RESET}",      flush=True )
            print( f"ANALYSEDATA:        INFO:                {DIM_WHITE}corr_cpy post hstack (all)             = \n{CAMEL}{corr_cpy[:rows,:rows]}{RESET}",    flush=True )
            np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
  

          fig_22 = plt.figure(figsize=(figure_width, figure_height))                                                             # convert to cupy array for parallel processing on GPU(s)

  
          display_gpu_correlation='True'
          # GPU version of correlation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
          if display_gpu_correlation=='True':

            if DEBUG>9:
              print( f"ANALYSEDATA:        INFO:     {BRIGHT_GREEN}np.min(corr_cpy.shape)           = {np.min(corr_cpy.shape)}{RESET}",      flush=True )

            if np.min(corr_cpy.shape)<=20:
              sns.set(font_scale=1)  
              title_size  = 16
              label_size  = 11
              text_size   = 9  
              do_annotate=True
              fmt='.2f'
            elif np.min(corr_cpy.shape)<=30:
              sns.set(font_scale=1)    
              title_size  = 18
              label_size  = 12
              text_size   = 10  
              do_annotate=True
              fmt='.2f'
            elif np.min(corr_cpy.shape)<=50:
              sns.set(font_scale=1)    
              title_size  = 14
              label_size  = 8
              text_size   = 7  
              do_annotate=True
              fmt='.1f'
            elif np.min(corr_cpy.shape)<=125:
              sns.set(font_scale=1)    
              title_size  = 18
              label_size  = 7
              text_size   = 5  
              do_annotate=True
              fmt='.1f'
            elif np.min(corr_cpy.shape)<=250:
              sns.set(font_scale=0.4)    
              label_size=13  # works  
              do_annotate=False 
              fmt='.1f'
            else:
              sns.set(font_scale=0.3)    
              label_size=6
              do_annotate=False
              fmt='.1f' 
  
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:{WHITE}      W3 about to convert cupy array to numpy array{RESET}"   )
              
            
            corr_npy =  cupy.asnumpy( corr_cpy )
            if corr_npy.shape[1]==0:
              print( f"{RED}ANALYSEDATA:   FATAL:    correlation matrix is empty ... exiting now [384]{RESET}" )
              sys.exit(0)
      
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:{WHITE}      W4 about to convert numpy array to pandas dataframe so that it can be displayed using seaborn and tensorboard{RESET}" )
              
              
            corr_pda = pd.DataFrame( corr_npy )
            # ~ del corr_npy
      
            if DEBUG>0:          
              print ( f"ANALYSEDATA:        INFO:{WHITE}      W5 about to generate Seaborn heatmap of Correlation Matrix{RESET}")
              
            ax   = sns.heatmap(corr_npy[1:,1:], cmap='coolwarm', square=True, cbar=True, cbar_kws={"shrink": 0.33}, annot=do_annotate, annot_kws={"size": text_size}, fmt=fmt)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=14)
            plt.xticks(range(corr_pda.shape[1]), corr_pda.columns, fontsize=label_size, rotation=90)
            plt.yticks(range(corr_pda.shape[1]), corr_pda.columns, fontsize=label_size)
            plt.title('Correlation Heatmap', fontsize=title_size)

            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:{WHITE}      W6 about to add figure to Tensorboard{RESET}" )  
            writer.add_figure('Correlation Matrix', fig_22, 0)
            del corr_pda
            #plt.show () 
            
            
            
          
          if cov_uq_threshold !=0:
          # select high correlation rows and columns ----------------------------------------------------------------------------------------------------------------------------------------------------------------   
            if DEBUG>0:          
              print ( f"ANALYSEDATA:        INFO:{BOLD}      H1 Reducing Correlation Matrix to Just Highly Correlated Genes (COV_UQ_THRESHOLD>{MIKADO}{cov_uq_threshold}{RESET}){BOLD} (GPU version){RESET}") 
            threshold=cov_uq_threshold
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:         {DIM_WHITE}size of correlation matrix prior to reduction (corr_cpy.shape) = {MIKADO}{corr_cpy.shape}{RESET}" )
            if DEBUG>0:           
              print( f"ANALYSEDATA:        INFO:         {DIM_WHITE}(prior) (after) corr_cpy[0:{MIKADO}{rows}{RESET}{DIM_WHITE},0:{MIKADO}{cols}] = \n{BLEU}{corr_cpy[0:rows,0:cols]}{RESET}"       )


            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:         {WHITE}H1 a) about to create an absolute value version of the correlation matrix{RESET}", flush=True )                   
            corr_cpy=cupy.absolute(corr_cpy)
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {DIM_WHITE}(absolute) corr_cpy.shape                = {MIKADO}{corr_cpy.shape}{RESET}" )
            if DEBUG>0:           
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})          
              print( f"ANALYSEDATA:        INFO:                 {DIM_WHITE}(prior) (after) (absolute) corr_cpy[0:{MIKADO}{rows}{DIM_WHITE},0:{MIKADO}{cols}] = \n{PALE_GREEN}{corr_cpy[0:rows,0:cols]}{RESET}"       )
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})


            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:         {WHITE}H1 b) about to calculate percentiles for each column (gene){RESET}", flush=True )            
            percentiles       = cupy.percentile (   corr_cpy, 5, axis=1          )                         # make a row vector comprising  the percentiles expression values of each of the genes (columns)
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {PINK}percentiles.shape               = {MIKADO}{percentiles.shape}{RESET}" )        
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {PINK}about to apply {CYAN}COV_UQ_THRESHOLD{RESET}{PINK} = {MIKADO}{cov_uq_threshold}{RESET}{PINK} and {CYAN}CUTOOFF_PERCENTILE{RESET}{PINK} = {MIKADO}{cutoff_percentile}{RESET}{PINK}) to create a logical mask to use to elininate low correlation genes{RESET}", flush=True )    
            logical_mask      = cupy.array      (  [ percentiles>cov_uq_threshold ]  )                     # convert it to Boolean values (TRUE, FALSE)
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {PINK}logical_mask.shape              = {MIKADO}{logical_mask.shape}{RESET}" )        
            squeezed_mask     = cupy.squeeze    (           logical_mask                 )                 # get rid of the extra dimension that' for some reason is created in the last step
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {PINK}squeezed_mask.shape             = {MIKADO}{squeezed_mask.shape}{RESET}" )
              print( f"ANALYSEDATA:        INFO:                 {PINK}about to convert logical mask into an integer mask{RESET}", flush=True )          
            integer_mask      = cupy.squeeze    (      squeezed_mask.astype(int)         )                 # change type from Boolean to Integer values (0,1) so we can use it as a mask
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {PINK}integer_mask.shape              = {MIKADO}{integer_mask.shape}{RESET}" )
            if DEBUG>9:                                                                                    # make sure that there are at least SOME non-zero values in the mask or else we'll make an empty matrix in subsequent steps
              print( f"ANALYSEDATA:        INFO:                 {PINK}integer_mask                    = \n{MIKADO}{integer_mask}{RESET}" )      
            if cupy.sum( integer_mask, axis=0 )==0:
              print( f"{RED}ANALYSEDATA:        ERROR:    the value provided for {CYAN}COV_UQ_THRESHOLD{RESET} ({MIKADO}{cov_uq_threshold}{RESET}{RED}) would filter out {UNDER}every{RESET}{RED} gene -- try a smaller vallue.  No filtering will be performed. Continuing [717]{RESET}" )
              time.sleep(2)
              # ~ sys.exit(0)
            else:
              non_zero_indices  = (cupy.nonzero( integer_mask ))[0]                                          # make a vector of indices corresponding to non-zero values in the mask (confusingly, cupy.nonzero returns a tuple) 
              if DEBUG>9:
                print( f"ANALYSEDATA:        INFO:           {DIM_WHITE}len(non_zero_indices[0]        = {MIKADO}{len(non_zero_indices)}{RESET}" )   
              if DEBUG>9:
                print( f"ANALYSEDATA:        INFO:           {DIM_WHITE}non_zero_indices               = {MIKADO}{non_zero_indices}{RESET}" )   
              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:         {WHITE}H1 c) about to exclude columns corresponding to low correlation genes{RESET}" )
              corr_reduced = cupy.take ( corr_cpy[:,1:], non_zero_indices,  axis=1  )                        # take columns corresponding to the indices (i.e. delete the others)
              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:                {DIM_WHITE}corr_reduced.shape     = {MIKADO}{corr_reduced.shape}{RESET}" )
              
              corr_reduced           = cupy.squeeze( corr_reduced )                                                                # get rid of the extra dimension that for some reason is created in the last step
              corr_reduced_new       = cupy.ones( ( corr_reduced.shape[0], corr_reduced.shape[1]+1) , dtype=cupy.float32)          # one extra column, to hold the row index from corr_cpy
              corr_reduced_new[:,0]  = corr_cpy[:,0]
              corr_reduced_new[:,1:] = corr_reduced
              corr_cpy               = corr_reduced_new



            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {DIM_WHITE}(after reduction and squeeze) corr_cpy.shape                = {MIKADO}{corr_cpy.shape}{RESET}" )
            if DEBUG>0:           
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})          
              print( f"ANALYSEDATA:        INFO:                 {DIM_WHITE}(after redution and squeeze) corr_cpy[0:{MIKADO}{rows}{DIM_WHITE},0:{MIKADO}{cols}] = \n{PALE_RED}{corr_cpy[0:rows,0:cols]}{RESET}"       )
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
            
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:         {WHITE}H1 d) about to sort columns so that the most highly correlated genes get displayed most prominently{RESET}" )
    
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:             {DIM_WHITE}    about to calculate sum of the expression values of all genes (columns){RESET}", flush=True )            
            highest_corr_values        = cupy.sum ( corr_cpy, axis=0 )                                             # make a row vector comprising the sum of the expression values of all genes (columns)
            if DEBUG>9:
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)}) 
              print( f"ANALYSEDATA:        INFO:                 {DIM_WHITE}sum of genes' rna-seq values   = \n{MIKADO}{highest_corr_values}{RESET}" ) 
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})               
            if DEBUG>9:
              print( f"ANALYSEDATA:        INFO:         {DIM_WHITE}len(highest_corr_values)      = {MIKADO}{len(highest_corr_values)}{RESET}" )           
    
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:         {WHITE}H1 e) about to establish sorting indices", flush=True )    
            sorting_indices = cupy.argsort( highest_corr_values )
            if DEBUG>9:
              np.set_printoptions(formatter={'int': lambda x: "{:>6d}".format(x)})
              print( f"ANALYSEDATA:        INFO:               {DIM_WHITE}sorting_indices.shape      = {MIKADO}{sorting_indices.shape}{RESET}" )  
              print( f"ANALYSEDATA:        INFO:               {DIM_WHITE}sorting_indices            = \n{MIKADO}{sorting_indices}{RESET}" )               
            sorting_indices = cupy.flip ( sorting_indices, axis=0 )                                        # change order from low->high to high->low  
            if DEBUG>9:
              print( f"ANALYSEDATA:        INFO:              {DIM_WHITE}flipped sorting_indices     = \n{MIKADO}{sorting_indices}{RESET}" )
            if DEBUG>9:
              print( f"ANALYSEDATA:        INFO:              {DIM_WHITE}sorting_indices.shape       = {MIKADO}{sorting_indices.shape}{RESET}" )        
            for n in range(sorting_indices.shape[0]-1, 0, -1 ):  
              sorting_indices[n] = sorting_indices[n-1]
            sorting_indices[0] = 0                                                                         # don't move the index column
            if DEBUG>9:
              np.set_printoptions(formatter={'int': lambda x: "{:>6d}".format(x)})
              print( f"ANALYSEDATA:        INFO:              {DIM_WHITE}offset sorting_indices    = \n{MIKADO}{sorting_indices}{RESET}" )
              
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:         {WHITE}H1 f) about to populate sorted matrix", flush=True )
            corr_cpy = corr_cpy[:,sorting_indices] 
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:                 {DIM_WHITE}(after sort) corr_cpy.shape                = {MIKADO}{corr_cpy.shape}{RESET}" )
            if DEBUG>0:           
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})          
              print( f"ANALYSEDATA:        INFO:                 {DIM_WHITE}(after sort) corr_cpy[0:{MIKADO}{rows}{DIM_WHITE},0:{MIKADO}{cols}] = \n{PALE_ORANGE}{corr_cpy[0:rows,0:cols]}{RESET}"       )
              np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})
    
    
            if DEBUG>0:        
              print( f"ANALYSEDATA:        INFO:         {WHITE}H1 g) about to make numpy version of the now reduced and sorted cupy correlation matrix{RESET}" )
            corr_cpy           = cupy.asnumpy( corr_cpy )                                                  # convert to numpy, as matplotlib can't use cupy arrays
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:              {DIM_WHITE}corr_cpy.shape      (numpy)       = {MIKADO}{corr_cpy.shape}{RESET}" )
           
            limit_display='False'
            if limit_display=='True':
              corr_cpy = corr_cpy[ :show_rows, :show_cols ]
              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:              {DIM_WHITE}corr_cpy.shape (display shape)     = {MIKADO}{corr_cpy.shape}{RESET}" ) 
            
            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:         {WHITE}H1 h) about to display user selected views of the data (available versions: unsorted, sorted/rows, sorted/columns, sorted/both)" )        
                          


            if DEBUG>0:
              print( f"ANALYSEDATA:        INFO:             {WHITE}np.min(corr_cpy.shape)           = {BRIGHT_GREEN}{np.min(corr_cpy.shape)}{RESET}",      flush=True )

            if np.min(corr_cpy.shape)<=20:
              sns.set(font_scale=1)    
              title_size  = 14
              label_size  = 10
              text_size   = 8.5 
              do_annotate=True
              fmt='.2f'
            elif np.min(corr_cpy.shape)<=30:
              sns.set(font_scale=1)    
              title_size  = 14
              label_size  = 10
              text_size   = 10 
              do_annotate=True
              fmt='.1f'
            elif np.min(corr_cpy.shape)<=50:
              sns.set(font_scale=1)    
              title_size  = 14
              label_size  = 8
              text_size   = 7  
              do_annotate=True
              fmt='.1f'
            elif np.min(corr_cpy.shape)<=125:
              sns.set(font_scale=1)    
              title_size  = 14
              label_size  = 6
              text_size   = 5  
              do_annotate=True
              fmt='.1f'
            elif np.min(corr_cpy.shape)<=250:
              sns.set(font_scale=1)    
              title_size  = 14
              label_size  = 8
              text_size   = 7  
              do_annotate=True
              fmt='.1f'
            else:
              sns.set(font_scale=1)    
              title_size  = 14
              label_size  = 8
              text_size   = 7  
              do_annotate=True
              fmt='.1f' 
        
    
            show_heatmap_unsorted='True'
            # shows unsorted version of the CORRELATION heatmap -------------------------------------------------------------------------------------------------------------   
            if show_heatmap_unsorted=='True':          

              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:         {WHITE}      user has selected 'unsorted' view" )   
              if DEBUG>0:    
                print( f"ANALYSEDATA:        INFO:        {BRIGHT_GREEN}corr_cpy.shape  = {BRIGHT_GREEN}{corr_cpy.shape}{RESET}" ) 
              if DEBUG>0:
                np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)}) 
                print( f"ANALYSEDATA:        INFO:        {BRIGHT_GREEN}corr_cpy   = \n{BRIGHT_GREEN}{corr_cpy[0:rows,0:cols]}{RESET}" )
                np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)}) 
              if DEBUG>99:
                print ( f"ANALYSEDATA:       INFO:      df_map.shape = {PURPLE}{ df_map.shape}{RESET}", flush=True )  
                print ( f"ANALYSEDATA:       INFO:      start of df_map: \n{PURPLE}{df_map.iloc[:,[0,1]]}{RESET}", flush=True )
              
              corr_cpy_index_row              = 0
              df_map_gene_name_column_number  = 1
              col_gene_name_labels            = []
              row_gene_name_labels            = []
              col_indices=np.around(scale_down*corr_cpy[0,:], decimals=0).astype(int)                      # original index values reinstated from the row above the top of the correlation matrix, which is where we encoded them as small values
              row_indices=np.around(scale_down*corr_cpy[:,0], decimals=0).astype(int)                      # original index values reinstated from the row above the top of the correlation matrix, which is where we encoded them as small values
    
              if DEBUG>9:
                np.set_printoptions(formatter={'float': lambda x: "{:>12.2f}".format(x)})            
                print ( f"ANALYSEDATA:       INFO:      col_indices          = \n{PURPLE}{col_indices}{RESET}",     flush=True )  
                print ( f"ANALYSEDATA:       INFO:      col_indices.shape    = {PURPLE}{col_indices.shape}{RESET}", flush=True )  
                print ( f"ANALYSEDATA:       INFO:      row_indices          = \n{PURPLE}{row_indices}{RESET}",     flush=True )  
                print ( f"ANALYSEDATA:       INFO:      row_indices.shape    = {PURPLE}{row_indices.shape}{RESET}", flush=True )  
                np.set_printoptions(formatter={'float': lambda x: "{:>8.2f}".format(x)})    
    
              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:         {DIM_WHITE}      assembling gene labels for columns of the correlation matrix", flush=True )

              for col in range (0, corr_cpy.shape[1]):
                col_gene_name_labels.append( df_map.iloc[ col_indices[col]-2, df_map_gene_name_column_number] )   # subtract 2 because df_map has a header row
                
                if DEBUG>2:            
                  print ( f"\r\033[0Cgene name column= {MIKADO}{df_map_gene_name_column_number}{RESET} \r\033[25CFor column {MIKADO}{col}{RESET}: \r\033[45Cindex={MIKADO}{col_indices[col]}{RESET} \r\033[57Cand df_map.iloc[ {MIKADO}{col_indices[col]:5d}{RESET}-2], {df_map_gene_name_column_number} ] = {MIKADO}{df_map.iloc[ col_indices[col]-2, df_map_gene_name_column_number]}{RESET}"  )
      
              if DEBUG>0:
                np.set_printoptions(formatter={'float': lambda x: "{:>6.9f}".format(x)})
                print( f"ANALYSEDATA:        INFO:         {PINK}post corr_cpy.shape                      = {MIKADO}{corr_cpy.shape}{RESET}",      flush=True )
                print( f"ANALYSEDATA:        INFO:         {PINK}post corr_cpy (index)                    = \n{MIKADO}{corr_cpy[0,:12]}{RESET}",   flush=True )
                print( f"ANALYSEDATA:        INFO:         {PINK}post corr_cpy (inc. indexes)             = \n{MIKADO}{corr_cpy[:12,:12]}{RESET}", flush=True )
                np.set_printoptions(formatter={'float': lambda x: "{:>6.2f}".format(x)})

              col_df_labels = pd.DataFrame( col_gene_name_labels )
            
              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:         {WHITE}columns gene labels = {MIKADO}{col_df_labels}{RESET}", flush=True )
    
              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:         {DIM_WHITE}assembling gene labels for rows of the correlation matrix", flush=True )
                    
              for row in range (0, corr_cpy.shape[0]):

                row_gene_name_labels.append( df_map.iloc[ row_indices[row]-2, df_map_gene_name_column_number] )   # subtract 2 because df_map has a header row

                if DEBUG>2:            
                  print ( f"\r\033[0Cgene name column= {MIKADO}{df_map_gene_name_column_number}{RESET} \r\033[25CFor row {MIKADO}{row}{RESET}: \r\033[45Cindex={MIKADO}{row_indices[col]}{RESET} \r\033[57Cand df_map.iloc[ {MIKADO}{row_indices[row]:5d}-2{RESET}], {df_map_gene_name_column_number} ] = {MIKADO}{df_map.iloc[ row_indices[row]-2, df_map_gene_name_column_number]}{RESET}"  )

              row_df_labels = pd.DataFrame( row_gene_name_labels )
              
              if DEBUG>0:
                print( f"ANALYSEDATA:        INFO:         {WHITE}row    gene labels = {MIKADO}{row_df_labels}{RESET}", flush=True )              
              if DEBUG>0:
                print ( f"ANALYSEDATA:       INFO:      col_df_labels       = \n{PURPLE}{ col_df_labels.iloc[:,0] }{RESET}", flush=True ) 
                print ( f"ANALYSEDATA:       INFO:      row_df_labels       = \n{PURPLE}{ row_df_labels.iloc[:,0] }{RESET}", flush=True )          
              if DEBUG>99:      
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy   = \n{MIKADO}{corr_cpy[ 0:corr_cpy.shape[1], :   ]}{RESET}" )
              if DEBUG>0:    
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy.shape                     = {MIKADO}{corr_cpy.shape}{RESET}" ) 

              fig_33 = plt.figure(figsize=(figure_width, figure_height))        
              if DEBUG>0:          
                print ( f"ANALYSEDATA:        INFO:         {WHITE}H1 i) about to generate Seaborn heatmap of highly correlated genes (unsorted){RESET}")
    
              title = 'Just Highly Correlated Genes (Unsorted)'
    
              # don't show row 1 or column 1 because they hold (encoded) index values
              ax = sns.heatmap(corr_cpy[1:genes_to_show,1:genes_to_show], square=False, cmap='coolwarm', cbar=True, cbar_kws={"shrink": 0.33},  annot=do_annotate, annot_kws={"size": text_size}, xticklabels=col_df_labels.iloc[1:genes_to_show,0], yticklabels=row_df_labels.iloc[1:genes_to_show,0], fmt=fmt )
              cbar = ax.collections[0].colorbar
              cbar.ax.tick_params(labelsize=14)
              plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
              plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,    labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
              plt.title(title, fontsize=title_size)
              plt.tight_layout()
              if DEBUG>0:
                print ( f"ANALYSEDATA:        INFO:{BLEU}        about to add      Seaborn heatmap figure to Tensorboard (unsorted){RESET}")        
              writer.add_figure(title, fig_33, 0)
    
    

            # show a version of the heatmap which is sorted by rows (highest gene expression first)--------------------------------------------------------------------------------------------------------------       
            show_heatmap_sorted_by_rows='False'
            if show_heatmap_sorted_by_rows=='True':
              corr_cpy = cupy.sort(corr_cpy, axis=0 ) 
              corr_cpy = cupy.flip(corr_cpy, axis=0 )                     
              if DEBUG>99:      
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy   = \n{MIKADO}{corr_cpy[ 0:corr_cpy.shape[1], :   ]}{RESET}" )
              if DEBUG>0:    
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy.shape                     = {MIKADO}{corr_cpy.shape}{RESET}" ) 
      
              fig_34 = plt.figure(figsize=(figure_width, figure_height))        
              if DEBUG>0:          
                print ( f"ANALYSEDATA:        INFO:{BLEU}        about to generate Seaborn heatmap of highly correlated genes (sorted by rows) {RESET}")
              title = 'Just Highly Correlated Genes (sorted by rows)'
              sns.heatmap(corr_cpy, cmap='coolwarm', square=True, cbar=False, annot=do_annotate, fmt=fmt )
              plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
              plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
              plt.title(title, fontsize=title_size)
              if DEBUG>0:
                print ( f"ANALYSEDATA:        INFO:{BLEU}        about to add      Seaborn heatmap to Tensorboard (sorted by rows){RESET}")        
              writer.add_figure(title, fig_34, 0)
    
    
            show_heatmap_sorted_by_columns='False'
            # show a version of the heatmap which is sorted by columns (highest gene expression first)--------------------------------------------------------------------------------------------------------------   
            if show_heatmap_sorted_by_columns=='True':
              corr_cpy = cupy.sort(corr_cpy, axis=1 ) 
              corr_cpy = cupy.flip(corr_cpy, axis=1 )                     
              if DEBUG>99:      
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy   = \n{MIKADO}{corr_cpy[ 0:corr_cpy.shape[1], :   ]}{RESET}" )
              if DEBUG>0:    
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy.shape                     = {MIKADO}{corr_cpy.shape}{RESET}" )  
      
              fig_35 = plt.figure(figsize=(figure_width, figure_height))        
              if DEBUG>0:          
                print ( f"ANALYSEDATA:        INFO:{BLEU}        about to generate Seaborn heatmap of highly correlated genes (sorted by columns) {RESET}")
              title = 'Just Highly Correlated Genes (sorted by columns)'    
              sns.heatmap(corr_cpy, cmap='coolwarm', square=True, cbar=False, annot=do_annotate, fmt=fmt )
              plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
              plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
              plt.title(title, fontsize=title_size)
              if DEBUG>0:
                print ( f"ANALYSEDATA:        INFO:{BLEU}        about to add heatmap figure to Tensorboard (sorted by columns){RESET}")        
              writer.add_figure(title, fig_35, 0)
    
    
    
            # show a version of the heatmap which is sorted by both rows and columns (highest gene expression at left and top)--------------------------------------------------------------------------------------------------------------   
            show_heatmap_sorted_by_both='True'
            if show_heatmap_sorted_by_both=='True':
              corr_cpy = cupy.sort(corr_cpy, axis=0 ) 
              corr_cpy = cupy.flip(corr_cpy, axis=0 ) 
              corr_cpy = cupy.sort(corr_cpy, axis=1 ) 
              corr_cpy = cupy.flip(corr_cpy, axis=1 )                     
              if DEBUG>99:      
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy   = \n{MIKADO}{corr_cpy[ 0:corr_cpy.shape[1], :   ]}{RESET}" )
              if DEBUG>9:    
                print( f"ANALYSEDATA:        INFO:        {GREEN}corr_cpy.shape                    = {MIKADO}{corr_cpy.shape}{RESET}" ) 
    
              fig_36 = plt.figure(figsize=(figure_width, figure_height))         
              if DEBUG>0:          
                print ( f"ANALYSEDATA:        INFO:{BLEU}        about to generate Seaborn heatmap of highly correlated genes (sorted by both) {RESET}")
              title = f'Just Highly Correlated Genes, sorted by both rows and columns)'       
              sns.heatmap(corr_cpy, cmap='coolwarm', square=True, cbar=False, annot=do_annotate, fmt=fmt )
              plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
              plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
              plt.title(title, fontsize=title_size)
              if DEBUG>0:
                print ( f"ANALYSEDATA:        INFO:{BLEU}        about to add heatmap figure to Tensorboard (sorted by both){RESET}")        
              writer.add_figure(title, fig_36, 0)
    
  
  
  
  
  
  























    ########################## NUMPY VERSIONS BELOW THIS LINE
    if use_cupy=='False':
      if DEBUG>0:
        print( f"{ORANGE}ANALYSEDATA:        NOTE:    numpy mode has been selected (use_cupy='False').  numpy data structures (and not cupy data structures) will be used{RESET}" )      
      save_file_name  = f'{base_dir}/dpcca/data/{nn_mode}/genes_df_lo.pickle'                              # if it exists, just use it
      
      if os.path.isfile( save_file_name ):    
        if DEBUG>0:
          print( f"ANALYSEDATA:        INFO:    checking to see if saved file                 '{MAGENTA}{save_file_name}{RESET}' exists" )
        df_sml = pd.read_pickle(save_file_name)
        if DEBUG>0:
          print( f"ANALYSEDATA:        INFO:    saved dataframe                               '{MAGENTA}{save_file_name}{RESET}' exists ... will load and use the previously saved file" )      
      else:
        print( f"ANALYSEDATA:        INFO:      file                                            '{RED}{save_file_name}{RESET}' does not exist ... will create" )          
        generate_file_name  = f'{base_dir}/dpcca/data/{nn_mode}/genes.pickle'
        print( f"ANALYSEDATA:        INFO:      about to load pickled pandas dataframe file   '{MIKADO}{generate_file_name}{RESET}'" ) 
        df  = pd.read_pickle(generate_file_name)
        if DEBUG>0:
          print( f"ANALYSEDATA:        INFO:      data.shape =  {MIKADO}{df.shape}{RESET}",  flush=True  )
          print( f"ANALYSEDATA:        INFO:      loading complete",                         flush=True  )     

      if DEBUG>0:          
        print ( f"ANALYSEDATA:        INFO:{BOLD}      Removing genes with low rna-exp values ({CYAN}COV_THRESHOLD{RESET}<{MIKADO}{threshold}{RESET}{BOLD}) across all samples{RESET}") 
      df_sml = df.loc[:, (df>=threshold).all()]
      if DEBUG>9:
        print( f"ANALYSEDATA:        INFO:        {YELLOW}df_sml = df.loc[:, (df>threshold).any(axis=0)].shape = \n{MIKADO}{df_sml.shape}{RESET}" )
      if DEBUG>0:                  
        print( f"ANALYSEDATA:        INFO:        about to save pandas file as {MIKADO}{save_file_name}{RESET}"   )
      df_sml.to_pickle(save_file_name)
      if DEBUG>0:     
        print( f"ANALYSEDATA:        INFO:        {PINK}df_sml.shape                = {MIKADO}{df_sml.shape}{RESET}" )    
      if DEBUG>99:     
        print( f"ANALYSEDATA:        INFO:        {PINK}df_sml                      = \n{MIKADO}{df_sml}{RESET}" ) 
      if DEBUG>90:     
        print( f"ANALYSEDATA:        INFO:        df_sm l.columns.tolist()           = \n{MIKADO}{df_sml.columns.tolist()}{RESET}" )    
 
      # Normalize -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
      df_sml = pd.DataFrame( StandardScaler().fit_transform(df_sml), index=df_sml.index, columns=df_sml.columns )    
      if DEBUG>0:    
        print( f"ANALYSEDATA:        INFO:        {PINK}normalized df_sml.shape     = {MIKADO}{df_sml.shape}{RESET}" ) 
      if DEBUG>99:        
        print( f"ANALYSEDATA:        INFO:        {PINK}normalized df_sml            = \n{MIKADO}{df_sml}{RESET}" )      
    
      summarize_data='False'
      # CPU version of coveriance --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      if summarize_data=='True':        
        print ( f"ANALYSEDATA:        INFO:      summarising data",                                                 flush=True )        
        print ( f"ANALYSEDATA:        INFO:      summary description of data =  \n{MIKADO}{df_sml.describe()}{RESET}",  flush=True )    
    
      do_cpu_covariance='True'
      # CPU version of coveriance ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
      if do_cpu_covariance=='True':
        if DEBUG>0:          
          print ( f"ANALYSEDATA:        INFO:{BOLD}    Calculating and Displaying  Covariance Matrix (CPU version){RESET}")  
        fig_1 = plt.figure(figsize=(figure_width, figure_height))
        cov=df_sml.cov()
        if DEBUG>0:
          print( f"\n{YELLOW}ANALYSEDATA:        INFO:        cov                 = {MIKADO}{cov.shape}{RESET}" )       
          print( f"{YELLOW}ANALYSEDATA:        INFO:        cov                 = \n{MIKADO}{cov}{RESET}" )         
        if cov.shape[1]==0:
          print( f"{RED}ANALYSEDATA:   FATAL:    covariance matrix is empty ... exiting now [980]{RESET}" )
          sys.exit(0)
  
        if cov.shape[1]<=20:
          label_size=9  
          do_annotate=True
          sns.set(font_scale = 1.0)    
          fmt='.3f'
        elif cov.shape[1]<=30:
          label_size=8  
          do_annotate=True
          sns.set(font_scale = 1.0)    
          fmt='.2f'
        elif cov.shape[1]<=50:
          label_size=7  
          do_annotate=True 
          sns.set(font_scale = 0.6)                
          fmt='.1f'
        elif cov.shape[1]<=126:
          label_size=6  
          do_annotate=True 
          sns.set(font_scale = 0.4)                
          fmt='.1f'
        elif cov.shape[1]<=250:
          label_size=6  
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


      do_cpu_correlation='True'
      #  CPU version of Correlation ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
      if do_cpu_correlation=='True':
        if DEBUG>0:          
          print ( f"ANALYSEDATA:        INFO:{BOLD}    Calculating and Displaying Correlation Matrix (CPU version){RESET}")    
        fig_2 = plt.figure(figsize=(figure_width, figure_height))
        corr=df_sml.corr()
        if DEBUG>0:
          print( f"\n{YELLOW}ANALYSEDATA:        INFO:        corr                 = {MIKADO}{corr.shape}{RESET}" )       
          print( f"{YELLOW}ANALYSEDATA:        INFO:        corr                 = \n{MIKADO}{corr}{RESET}" )       
        if corr.shape[1]==0:
          print( f"{RED}ANALYSEDATA:   FATAL:    correlation matrix is empty ... exiting now [384]{RESET}" )
          sys.exit(0) 
   
        if corr.shape[1]<=20:
          label_size=9  
          do_annotate=True
          sns.set(font_scale = 1.0)    
          fmt='.3f'
        elif corr.shape[1]<=30:
          label_size=8  
          do_annotate=True
          sns.set(font_scale = 1.0)    
          fmt='.2f'
        elif corr.shape[1]<=50:
          label_size=7  
          do_annotate=True 
          sns.set(font_scale = 0.6)                
          fmt='.1f'
        elif corr.shape[1]<=125:
          label_size=6  
          do_annotate=True 
          sns.set(font_scale = 0.4)                
          fmt='.1f'
        elif corr.shape[1]<=250:
          label_size=6  
          do_annotate=True 
          sns.set(font_scale = 0.4)                
          fmt='.1f'
        else:
          label_size=4.5        
          do_annotate=False
          sns.set( font_scale = 1.0 )
          fmt='.1f'   
          
        sns.heatmap(corr, cmap='coolwarm', annot=do_annotate, fmt='.1f' )
        plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
        plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
        plt.title('Correlation Heatmap', fontsize=title_size)
        writer.add_figure('Correlation Matrix', fig_2, 0)
        #plt.show()
           
      select_hi_corr_genes='True'
      # select high correlation rows and columns ----------------------------------------------------------------------------------------------------------------------------------------------------------------   
      if select_hi_corr_genes=='True':
        if DEBUG>0:          
          print ( f"ANALYSEDATA:        INFO:{BOLD}      Reducing Correlation Matrix to Just Highly Correlated Genes (COV_UQ_THRESHOLD>{MIKADO}{cov_uq_threshold}{RESET}){BOLD} (CPU version){RESET}")
        fig_3 = plt.figure(figsize=(figure_width, figure_height))
        threshold=cov_uq_threshold
        corr_abs=np.abs(corr)
        if DEBUG>0:
          print( f"ANALYSEDATA:        INFO:        {GREEN}corr_abs.shape           = {MIKADO}{corr_abs.shape}{RESET}", flush=True  )
        if DEBUG>99:        
          print( f"ANALYSEDATA:        INFO:        {GREEN}corr_abs              = \n{MIKADO}{corr_abs}{RESET}", flush=True  )
        if DEBUG>0:
          print( f"ANALYSEDATA:        INFO:        about to calculate quantiles" )                 
        corr_hi = corr_abs.loc[(corr_abs.quantile(0.75, axis=1)>threshold), (corr_abs.quantile(0.75, axis=1)>threshold) ]
        if DEBUG>0:
          print( f"ANALYSEDATA:        INFO:        {GREEN}corr_hi.shape            = {MIKADO}{corr_hi.shape}{RESET}", flush=True  )
        if DEBUG>99:
          print( f"ANALYSEDATA:        INFO:       {GREEN} corr_hi               = \n{MIKADO}{corr_hi}{RESET}", flush=True  )        
  
        if corr.shape[1]<=20:
          label_size=9  
          do_annotate=True
          sns.set(font_scale = 1.0)    
          fmt='.3f'
        elif corr.shape[1]<=30:
          label_size=8  
          do_annotate=True
          sns.set(font_scale = 1.0)    
          fmt='.2f'
        elif corr.shape[1]<=50:
          label_size=7  
          do_annotate=True 
          sns.set(font_scale = 0.6)                
          fmt='.1f'
        elif corr.shape[1]<=126:
          label_size=6  
          do_annotate=True 
          sns.set(font_scale = 0.4)                
          fmt='.1f'
        elif corr.shape[1]<=250:
          label_size=6  
          do_annotate=True 
          sns.set(font_scale = 0.4)                
          fmt='.1f'
        else:
          label_size=4.5        
          do_annotate=False
          sns.set( font_scale = 0.2 )
          fmt='.1f'
  
        title = 'Just Highly Correlated Genes'
        sns.heatmap(corr_hi, cmap='coolwarm', annot=do_annotate, fmt=fmt )
        plt.tick_params(axis='x', top='on',    labeltop='off',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=90 )    
        plt.tick_params(axis='y', left='on',   labelleft='on',   which='major',  color='lightgrey',  labelsize=label_size,  labelcolor='dimgrey',  width=1, length=6,  direction = 'out', rotation=0  )
        plt.title(title, fontsize=title_size)
        writer.add_figure(title, fig_3, 0)
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
          print(f'ANALYSEDATA:        INFO: total variance explained by all {MIKADO}{n+1}{RESET} principle components: {MAGENTA if np.sum(explainable_variance)>0.98 else GREEN if np.sum(explainable_variance)>0.95 else PALE_GREEN if np.sum(explainable_variance)>0.9 else WHITE} {np.sum(explainable_variance):>5.6}{RESET}', end='', flush=True )
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
          
          plt.figure(figsize=( figure_width, figure_height ))
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
          fig_4 = plt.figure(figsize=(figure_width, figure_height))   
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




# Support function for t-SNE ----------------------------------------------------------------------------------------------------------------------    
def plot( x, figure_width, perplexity, writer):

  if DEBUG>99:    
    print( f"ANALYSEDATA:        INFO:                                      x.shape = {MIKADO}{x.shape}{RESET}" ) 

  
  # Create a scatter plot.
  fig_4 = plt.figure(figsize=(figure_width, figure_width))   
  ax = plt.subplot(aspect='equal')
  #sc = ax.scatter( np.abs(x[:,0]), np.abs(x[:,1]), lw=0, s=40, c=1)  
  sc = ax.scatter( np.abs(x[:,0]), np.abs(x[:,1]),  lw=0, s=40)

  title = f"t-SNE: perplexity = {perplexity}"     
  writer.add_figure( title, fig_4, 0)
  
  return



  
  
if __name__ == '__main__':

  def str2bool(v):
      if isinstance(v, bool):
          return v
      if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
      elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
      else:
          raise argparse.ArgumentTypeError('Boolean value expected for this input parameter')

  p = argparse.ArgumentParser()

  p.add_argument('--skip_tiling',                    type=str,   default='False')
  p.add_argument('--skip_generation',                type=str,   default='False')
  p.add_argument('--log_dir',                        type=str,   default='data/pre_compress/logs')
  p.add_argument('--base_dir',                       type=str,   default='/home/peter/git/pipeline')             # NOT CURRENTLY USED
  p.add_argument('--data_dir',                       type=str,   default='/home/peter/git/pipeline/dataset')
  p.add_argument('--save_model_name',                type=str,   default='model.pt')
  p.add_argument('--save_model_every',               type=int,   default=10)
  p.add_argument('--rna_file_name',                  type=str,   default='rna.npy')
  p.add_argument('--rna_file_suffix',                type=str,   default='*FPKM-UQ.txt' )
  p.add_argument('--use_unfiltered_data',            type=str2bool, nargs='?', const=True, default=True, help="If true, don't filter the genes, but rather use all of them")
  p.add_argument('--rna_file_reduced_suffix',        type=str,   default='_reduced')
  p.add_argument('--class_numpy_file_name',          type=str,   default='class.npy')
  p.add_argument('--wall_time',                      type=int,   default=24)
  p.add_argument('--seed',                           type=int,   default=0)
  p.add_argument('--nn_mode',                        type=str,   default='analyse_data')
  p.add_argument('--use_same_seed',                  type=str,   default='False')
  p.add_argument('--nn_type_img',                                       nargs="+",  type=str,    default='VGG11'                           )
  p.add_argument('--nn_type_rna',                                       nargs="+",  type=str,    default='DENSE'                           )
  p.add_argument('--encoder_activation',  nargs="+", type=str,   default='sigmoid')
  p.add_argument('--nn_dense_dropout_1',  nargs="+", type=float, default=0.0)   
  p.add_argument('--nn_dense_dropout_2',  nargs="+", type=float, default=0.0)
  p.add_argument('--dataset',                        type=str,   default='STAD')                                 # taken in as an argument so that it can be used as a label in Tensorboard
  p.add_argument('--input_mode',                     type=str,   default='NONE')                                 # taken in as an argument so that it can be used as a label in Tensorboard
  p.add_argument('--n_samples',           nargs="+", type=int,   default=101)
  p.add_argument('--n_tiles',             nargs="+", type=int,   default=100)
  p.add_argument('--supergrid_size',                 type=int,   default=1)
  p.add_argument('--patch_points_to_sample',         type=int,   default=1000)
  p.add_argument('--tile_size',           nargs="+", type=int,   default=128)
  p.add_argument('--gene_data_norm',      nargs="+", type=str,   default='NONE')
  p.add_argument('--gene_data_transform', nargs="+", type=str,   default='NONE' )
  p.add_argument('--n_genes',                        type=int,   default=506)
  p.add_argument('--batch_size',         nargs="+",  type=int,   default=256)
  p.add_argument('--learning_rate',      nargs="+",  type=float, default=.00082)
  p.add_argument('--n_epochs',                       type=int,   default=10)
  p.add_argument('--pct_test',                       type=float, default=0.2)
  p.add_argument('--lr',                             type=float, default=0.0001)
  p.add_argument('--latent_dim',                     type=int,   default=100)
  p.add_argument('--l1_coef',                        type=float, default=0.1)
  p.add_argument('--em_iters',                       type=int,   default=1)
  p.add_argument('--clip',                           type=float, default=1)
  p.add_argument('--max_consecutive_losses',         type=int,   default=7771)
  p.add_argument('--optimizer',          nargs="+",  type=str,   default='ADAM')
  p.add_argument('--label_swap_pct',             type=float,   default=0.0)                                    
  p.add_argument('--make_grey_pct',              type=float, default=0.0) 
  p.add_argument('--figure_width',                   type=float, default=14)                                  
  p.add_argument('--figure_height',                  type=float, default=14)
  p.add_argument('--annotated_tiles',                type=str,   default='True')
  p.add_argument('--scattergram',                    type=str,   default='True')
  p.add_argument('--probs_matrix',                   type=str,   default='True')
  p.add_argument('--probs_matrix_interpolation',     type=str,   default='none')
  p.add_argument('--show_patch_images',              type=str,   default='True')
  p.add_argument('--regenerate',                     type=str,   default='True')
  p.add_argument('--just_profile',                   type=str,   default='False')
  p.add_argument('--just_test',                      type=str,   default='False')
  p.add_argument('--rand_tiles',                     type=str,   default='True')
  p.add_argument('--points_to_sample',               type=int,   default=100)
  p.add_argument('--min_uniques',                    type=int,   default=0)
  p.add_argument('--min_tile_sd',                    type=float, default=3)
  p.add_argument('--greyness',                       type=int,   default=0          )
  p.add_argument('--stain_norm',         nargs="+",  type=str,   default='NONE'     )
  p.add_argument('--stain_norm_target',              type=str,   default='NONE'     )
  p.add_argument('--use_tiler',                      type=str,   default='external' )
  p.add_argument('--cancer_type',                    type=str,   default='NONE'     )
  p.add_argument('--cancer_type_long',               type=str,   default='NONE'     )
  p.add_argument('--class_names',        nargs="+"                                  )
  p.add_argument('--long_class_names',   nargs="+"                                  )
  p.add_argument('--class_colours',      nargs="*"                                  )    
  p.add_argument('--target_tile_coords', nargs=2,    type=int, default=[2000,2000]  )

  p.add_argument('--do_covariance',                  type=str,   default='False'    )
  p.add_argument('--do_correlation',                 type=str,   default='False'    )
  p.add_argument('--a_d_use_cupy',                   type=str,   default='True'     )
  p.add_argument('--cov_threshold',                  type=float, default=8.0        )
  p.add_argument('--cutoff_percentile',              type=float, default=0.05       )
  p.add_argument('--cov_uq_threshold',               type=float, default=0.0        )
  
  p.add_argument('--show_rows',                      type=int,   default=500        )
  p.add_argument('--show_cols',                      type=int,   default=100        )
      
  args, _ = p.parse_known_args()

  is_local = args.log_dir == 'experiments/example'

  # ~ print ( f"figure_width =   {args.figure_width}"  )
  # ~ print ( f"figure_height =  {args.figure_height}" )

  args.n_workers  = 0 if is_local else 12
  args.pin_memory = torch.cuda.is_available()

  main(args)
