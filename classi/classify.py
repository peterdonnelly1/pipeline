"""============================================================================= 
Main program file
============================================================================="""

import sys
import math
import decimal
import time
import torch
import cuda
import pplog
import argparse
import datetime
import matplotlib
import torchvision
import torch.utils.data


import scipy
import sklearn

import numpy                 as np
import pandas                as pd
import seaborn               as sns
import matplotlib.pyplot     as plt
import matplotlib.lines      as mlines
import matplotlib.patches    as mpatches
import matplotlib.gridspec   as gridspec

# ~ from matplotlib import rc
# ~ rc('text', usetex=True)

from   select                       import select
from   IPython.display              import display
from   pathlib                      import Path
from   random                       import randint
from   matplotlib.colors            import ListedColormap
from   matplotlib                   import cm
from   matplotlib.ticker            import (AutoMinorLocator, MultipleLocator)
from   sklearn                      import metrics 
from   pandas.plotting              import table
from   tabulate                     import tabulate

from   torch                        import optim
from   torch.nn.utils               import clip_grad_norm_
from   torch.nn                     import functional
from   torch.nn                     import DataParallel
from   itertools                    import product, permutations
from   PIL                          import Image
from   torch.utils.tensorboard      import SummaryWriter
from   torchvision                  import datasets, transforms

from modes                          import loader
from models.ttvae                   import vae_loss
from modes.classify.config          import classifyConfig
from modes.classify.generate        import generate
from   models                       import COMMON
from   tiler_scheduler              import *
from   tiler_threader               import *
from   tiler_set_target             import *
from   tiler                        import *
from   _dbscan                      import _dbscan
from   h_dbscan                     import h_dbscan
from   o_tsne                       import o_tsne
from   cuda_tsne                    import cuda_tsne
from   sk_tsne                      import sk_tsne
from   sk_agglom                    import sk_agglom
from   sk_spectral                  import sk_spectral

from constants  import *

last_stain_norm='NULL'
last_gene_norm='NULL'

np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=300)

#torch.backends.cudnn.benchmark   = False                                                                  #for CUDA memory optimization
torch.backends.cudnn.enabled     = True                                                                    #for CUDA memory optimization

pd.set_option('display.max_rows',     99 )
pd.set_option('display.max_columns',  99 )
pd.set_option('display.width',        80 )
pd.set_option('display.max_colwidth',  8 ) 
pd.set_option('display.float_format', lambda x: '%6.2f' % x)

# ------------------------------------------------------------------------------


DEBUG   = 1

pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]


device = cuda.device()
num_cpus = multiprocessing.cpu_count()
start_column = 112
start_row    = 60-num_cpus

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)

global global_batch_count


run_level_total_correct             = []

global_batch_count    = 0
total_runs_in_job     = 0
final_test_batch_size = 0


# ---------------------------------------------------------------------------------------------------------

#@profile
def main(args):
  
  
  if DEBUG>0:
    print  ( f"pid = {os.getpid()}" )
  os.system("taskset -p 0xfffff %d" % os.getpid())
  
  now = time.localtime(time.time())
  print(time.strftime( f"CLASSI:         INFO:  start time = %Y-%m-%d %H:%M:%S %Z", now ))
  start_time = time.time() 

  if DEBUG>0:
    print ( f"\nCLASSI:         INFO:     torch         version =  {MIKADO}{torch.__version__}{RESET}",      flush=True    )
    print ( f"CLASSI:         INFO:     torchvision   version =  {MIKADO}{torchvision.__version__}{RESET}",  flush=True    )
    print ( f"CLASSI:         INFO:     scipy         version =  {MIKADO}{scipy.version.version}{RESET}",    flush=True    )
    print ( f"CLASSI:         INFO:     sklearn       version =  {MIKADO}{sklearn.__version__}{RESET}",      flush=True    )
    print ( f"CLASSI:         INFO:     matplotlib    version =  {MIKADO}{matplotlib.__version__}{RESET}",   flush=True    ) 
    print ( f"CLASSI:         INFO:     seaborn       version =  {MIKADO}{sns.__version__}{RESET}",          flush=True    )
    print ( f"CLASSI:         INFO:     pandas        version =  {MIKADO}{pd.__version__}{RESET}",           flush=True    )  
    print ( f"CLASSI:         INFO:     numpy         version =  {MIKADO}{np.version.version}{RESET}",       flush=True    )  
    print ( f"CLASSI:         INFO:     cuda toolkit  version =  {MIKADO}{torch.version.cuda}{RESET}\n",     flush=True    )
    print ( f"CLASSI:         INFO:     cuda          version via os command = \n{MIKADO}",                  flush=True    )  
    print ( f"{os.system('/usr/local/cuda/bin/nvcc --version')}{RESET}\n",                                   flush=True    )
    print ( f"CLASSI:         INFO:     cuda driver  version via os command = \n{MIKADO}",                   flush=True    )  
    print ( f"{os.system('cat /proc/driver/nvidia/version')}{RESET}\n",                                      flush=True    )
  
  

  mode = 'TRAIN' if args.just_test!='True' else 'TEST'

  print( f"{GREY_BACKGROUND}CLASSI:         INFO:  common args:  \
{WHITE}mode={AUREOLIN}{mode if args.clustering=='NONE' else 'CLUSTERING' }{WHITE}, \
input={AUREOLIN}{args.input_mode}{WHITE}, \
network={AUREOLIN}{args.mode}{WHITE}, \
multimode={AUREOLIN}{args.multimode}{WHITE}, \
cases={CARRIBEAN_GREEN}{args.cases}{WHITE}, \
dataset={BITTER_SWEET}{args.dataset}{WHITE}, \
n_samples={AUREOLIN}{args.n_samples}{WHITE}, \
pct_test={AUREOLIN}{args.pct_test}{WHITE}, \
epochs={AUREOLIN}{args.n_epochs}{WHITE}, \
nn_optimizer={AUREOLIN}{args.optimizer}{WHITE}, \
batch_size={AUREOLIN}{args.batch_size}{WHITE}, \
learning_rate(s)={AUREOLIN}{args.learning_rate}{WHITE}, \
max_consec_losses={AUREOLIN}{args.max_consecutive_losses}{WHITE} \
                        {RESET}"\
, flush=True )

  
  if args.input_mode=='image':
    print( f"{GREY_BACKGROUND}CLASSI:         INFO:  image  args:  \
{WHITE}nn_type_img={AUREOLIN}{args.nn_type_img}{WHITE}, \
h_class={AUREOLIN}{args.highest_class_number}{WHITE}, \
n_tiles={AUREOLIN}{args.n_tiles}{WHITE}, \
tile_size={AUREOLIN}{args.tile_size}{WHITE}, \
rand_tiles={AUREOLIN}{args.rand_tiles}{WHITE}, \
greyness<{AUREOLIN}{args.greyness}{WHITE}, \
sd<{AUREOLIN}{args.min_tile_sd}{WHITE}, \
min_uniques>{AUREOLIN}{args.min_uniques}{WHITE}, \
latent_dim={AUREOLIN}{args.latent_dim}{WHITE}, \
label_swap={AUREOLIN}{args.label_swap_pct}{WHITE}, \
make_grey={AUREOLIN}{args.make_grey_pct}{WHITE}, \
stain_norm={AUREOLIN}{args.stain_norm}{WHITE}, \
annotated_tiles={AUREOLIN}{args.annotated_tiles}{WHITE}, \
probs_matrix_interpolation={AUREOLIN}{args.probs_matrix_interpolation}{WHITE} \
        {RESET}"
, flush=True )

  elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
    print( f"{GREY_BACKGROUND}CLASSI:         INFO:  rna-seq args: \
nn_type_rna={CYAN}{args.nn_type_rna}{WHITE}, \
hidden_layer_neurons={YELLOW}{args.hidden_layer_neurons if args.nn_type_rna[0]=='DENSE' else args.hidden_layer_neurons  if args.nn_type_rna[0]=='AEDENSE' else ' N/A' }{WHITE}, \
topology={YELLOW}{args.hidden_layer_encoder_topology  if args.nn_type_rna[0]=='DEEPDENSE' else args.hidden_layer_encoder_topology  if args.nn_type_rna[0]=='AEDEEPDENSE' else ' N/A' }{WHITE}, \
embedding_dimensions={YELLOW}{args.embedding_dimensions if args.nn_type_rna[0]=='AEDENSE' else args.embedding_dimensions if args.nn_type_rna[0]=='AEDEEPDENSE' else ' N/A' }{WHITE}, \
dropout_1={YELLOW}{args.nn_dense_dropout_1}{WHITE}, \
dropout_2={YELLOW}{args.nn_dense_dropout_2}{WHITE}, \
gene_norm={YELLOW if not args.gene_data_norm[0]=='NONE'    else YELLOW if len(args.gene_data_norm)>1       else YELLOW}{args.gene_data_norm}{WHITE}, \
g_xform={YELLOW if not args.gene_data_transform[0]=='NONE' else YELLOW if len(args.gene_data_transform)>1  else YELLOW}{args.gene_data_transform}{WHITE} \
                                                                                  {RESET}"
, flush=True )


  if args.clustering != "NONE":
    if args.input_mode=='image':
      print( f"{GREY_BACKGROUND}CLASSI:         INFO:  additional: \
  {BOLD}{WHITE}clustering algorithm={CARRIBEAN_GREEN}{args.clustering}{WHITE} \
                  {RESET}"
  , flush=True )

  repeat                        = args.repeat
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
  use_unfiltered_data           = args.use_unfiltered_data
  cancer_type                   = args.cancer_type
  cancer_type_long              = args.cancer_type_long    
  class_colours                 = args.class_colours
  colour_map                    = args.colour_map
  input_mode                    = args.input_mode
  multimode                     = args.multimode
  mode                          = args.mode
  nn_type_img                   = args.nn_type_img
  nn_type_rna                   = args.nn_type_rna
  use_same_seed                 = args.use_same_seed
  hidden_layer_neurons          = args.hidden_layer_neurons
  low_expression_threshold      = args.low_expression_threshold
  cutoff_percentile             = args.cutoff_percentile
  embedding_dimensions          = args.embedding_dimensions
  hidden_layer_encoder_topology = args.hidden_layer_encoder_topology
  dropout_1                     = args.nn_dense_dropout_1
  dropout_2                     = args.nn_dense_dropout_2
  label_swap_pct                = args.label_swap_pct
  nn_optimizer                  = args.optimizer
  n_samples                     = args.n_samples
  n_tiles                       = args.n_tiles
  make_balanced                 = args.make_balanced
  n_iterations                  = args.n_iterations
  pct_test                      = args.pct_test
  batch_size                    = args.batch_size
  lr                            = args.learning_rate
  tsne_lr                       = args.tsne_learning_rate
  rand_tiles                    = args.rand_tiles
  zoom_out_mags                 = args.zoom_out_mags
  zoom_out_prob                 = args.zoom_out_prob
  n_genes                       = args.n_genes
  gene_data_norm                = args.gene_data_norm 
  gene_data_transform           = args.gene_data_transform    
  n_epochs                      = args.n_epochs
  greyness                      = args.greyness
  min_tile_sd                   = args.min_tile_sd
  min_uniques                   = args.min_uniques  
  make_grey_pct                 = args.make_grey_pct
  peer_noise_pct                = args.peer_noise_pct
  stain_norm                    = args.stain_norm
  stain_norm_target             = args.stain_norm_target
  annotated_tiles               = args.annotated_tiles
  figure_width                  = args.figure_width
  figure_height                 = args.figure_height  
  probs_matrix_interpolation    = args.probs_matrix_interpolation
  max_consecutive_losses        = args.max_consecutive_losses
  target_tile_coords            = args.target_tile_coords
  base_dir                      = args.base_dir
  application_dir               = args.application_dir
  data_dir                      = args.data_dir
  log_dir                       = args.log_dir
  tile_size                     = args.tile_size
  rna_file_name                 = args.rna_file_name
  class_numpy_file_name         = args.class_numpy_file_name
  highest_class_number          = args.highest_class_number
  regenerate                    = args.regenerate
  just_profile                  = args.just_profile
  just_test                     = args.just_test
  save_model_name               = args.save_model_name
  save_model_every              = args.save_model_every
  supergrid_size                = args.supergrid_size
  minimum_job_size              = args.minimum_job_size
  box_plot                      = args.box_plot
  box_plot_show                 = args.box_plot_show
  bar_chart_x_labels            = args.bar_chart_x_labels
  bar_chart_sort_hi_lo          = args.bar_chart_sort_hi_lo
  remove_unexpressed_genes      = args.remove_unexpressed_genes
  encoder_activation            = args.encoder_activation
  hidden_layer_neurons          = args.hidden_layer_neurons
  embedding_dimensions          = args.embedding_dimensions
  use_autoencoder_output        = args.use_autoencoder_output  
  ae_add_noise                  = args.ae_add_noise  
  clustering                    = args.clustering  
  metric                        = args.metric  
  perplexity                    = args.perplexity  
  momentum                      = args.momentum  
  epsilon                       = args.epsilon
  min_cluster_size              = args.min_cluster_size
  names_column                  = args.names_column
  case_column                   = args.case_column
  class_column                  = args.class_column


  def expand_args( parm, bash_name, highlight_colour ):
    
    COL = highlight_colour
    c=57
    
    
    if len(parm)>=3:
      if parm[2]<0:
        lo=parm[0]
        hi=parm[1]
        num=abs(int(parm[2]))
        RANDOM_CHOICES = False
        if len(parm)>=4:
          if parm[3]<0:
            RANDOM_CHOICES = True
            if bash_name != 'ZOOM_OUT_PROB':
              if abs(parm[2])>=(abs(hi-lo)):
                num = int(abs(hi-lo)-1)
              if num<0:
                num=1
              if (hi-lo)<0:
                hi-lo+1
        if RANDOM_CHOICES!=True:
          if isinstance( parm[0], int ):
            parm_new = [ lo + int( ( n / (num+1) )*(hi-lo) ) for n in range (0, num+2) ]
            print( f"{PINK}CLASSI:         INFO: 3rd &/or 4th value in {CYAN}{bash_name}{RESET}{COL} \r\033[{c}C are negative: {BOLD_MIKADO}{parm}{RESET}{COL}  \r\033[{c+40}C therefore {BOLD_MIKADO}{abs(parm[2]):2d}{RESET}{COL} additional {CARRIBEAN_GREEN}EQUIDISTANT{RESET}{COL} values will be generated, in between and in addition to {BOLD_MIKADO}{abs(parm[0]):5d}{RESET}{COL} and {BOLD_MIKADO}{abs(parm[1]):5d}{RESET}{COL}. \r\033[{c+150}C New value of {CYAN}{bash_name}{RESET}{COL} \r\033[{c+183}C= {BOLD_MIKADO}{parm_new}{RESET}{COL}{RESET}" )
          elif isinstance( parm[0], float ):
            if bash_name == 'ZOOM_OUT_PROB':
              parm_new = [ round(1/(num+2),3) for n in range( 0, num+2) ]              
            elif bash_name == 'ZOOM_OUT_MAGS':
              parm_new = [ round(lo +  ( n / (num+1) )*(hi-lo), 3) for n in range (0, num+2) ]
            else:
              parm_new = [ lo +  ( n / (num+1) )*(hi-lo) for n in range (0, num+2) ]
            np.set_printoptions(formatter={'float': lambda x: "{:>5.3f}".format(x)})
            print( f"{COL}CLASSI:         INFO: 3rd &/or 4th value in {CYAN}{bash_name}{RESET}{COL} \r\033[{c}C are negative: {BOLD_MIKADO}{parm}{RESET}{COL}  \r\033[{c+40}C therefore {BOLD_MIKADO}{abs(parm[2]):2.0f}{RESET}{COL} additional {AMETHYST}RANDOM     {RESET}{COL} values will be generated, in between and in addition to {BOLD_MIKADO}{abs(parm[0]):5.3f}{RESET}{COL} and {BOLD_MIKADO}{abs(parm[1]):5.5f}{RESET}{COL}. \r\033[{c+150}C New value of {CYAN}{bash_name}{RESET}{COL} \r\033[{c+183}C= {BOLD_MIKADO}{np.array(parm_new)}{RESET}{COL}{RESET}" )
        else:
          if isinstance( parm[0], int ):
            parm_new = random.sample( range(lo+1, hi), num)
            parm_new.insert(0, lo)
            parm_new.append(hi)
            print( f"{COL}CLASSI:         INFO: 3rd &/or 4th value in {CYAN}{bash_name}{RESET}{COL} \r\033[{c}C are negative: {BOLD_MIKADO}{parm}{RESET}{COL}  \r\033[{c+40}C therefore {BOLD_MIKADO}{abs(parm[2]):2d}{RESET}{COL} additional {AMETHYST}RANDOM     {RESET}{COL} values will be generated, in between and in addition to {BOLD_MIKADO}{abs(parm[0]):5d}{RESET}{COL} and {BOLD_MIKADO}{abs(parm[1]):5d}{RESET}{COL}. \r\033[{c+150}C New value of {CYAN}{bash_name}{RESET}{COL} \r\033[{c+183}C= {BOLD_MIKADO}{parm_new}{RESET}{COL}{RESET}" )
          elif isinstance( parm[0], float ):
            if bash_name == 'ZOOM_OUT_PROB':
              parm_new = [ random.uniform( 0.0, 1.0 ) for el in range (0, num+2) ]
              parm_new = [ (parm_new[n] / sum(parm_new)) for n in range (0, len(parm_new)) ]
              parm_new[0] = 1-sum(parm_new[1:])
            else:
              parm_new = [ random.uniform(  lo,  hi ) for el in range (0, num) ]
              parm_new.insert(0, lo)
              parm_new.append(hi)
            if bash_name == 'ZOOM_OUT_MAGS':
              parm_new = [ round(el,3) for el in parm_new ]
            np.set_printoptions(formatter={'float': lambda x: "{:>5.3f}".format(x)})
            print( f"{COL}CLASSI:         INFO: 3rd &/or 4th value in {CYAN}{bash_name}{RESET}{COL} \r\033[{c}C are negative: {BOLD_MIKADO}{parm}{RESET}{COL}  \r\033[{c+40}C therefore {BOLD_MIKADO}{abs(parm[2]):2.0f}{RESET}{COL} additional {AMETHYST}RANDOM     {RESET}{COL} values will be generated, in between and in addition to {BOLD_MIKADO}{abs(parm[0]):5.3f}{RESET}{COL} and {BOLD_MIKADO}{abs(parm[1]):5.3f}{RESET}{COL}. \r\033[{c+150}C New value of {CYAN}{bash_name}{RESET}{COL} \r\033[{c+183}C= {BOLD_MIKADO}{np.round(np.array(parm_new),3)}{RESET}{COL}{RESET}" )
            
      else:
        return parm
        
      parm = parm_new
      
    return parm


  args.n_tiles        = expand_args( args.n_tiles,       "N_TILES",         CAMEL  )
  args.tile_size      = expand_args( args.tile_size,     "TILE_SIZE",       CAMEL  )
  args.batch_size     = expand_args( args.batch_size,    "BATCH_SIZE",      CAMEL  )
  args.n_samples      = expand_args( args.n_samples,     "N_SAMPLES",       CAMEL  )
  args.pct_test       = expand_args( args.pct_test,      "PCT_TEST",        CAMEL  )
  args.zoom_out_mags  = expand_args( args.zoom_out_mags, "ZOOM_OUT_MAGS",   BITTER_SWEET  )
  args.zoom_out_prob  = expand_args( args.zoom_out_prob, "ZOOM_OUT_PROB",   MAGENTA  )
  
  n_tiles       = args.n_tiles
  tile_size     = args.tile_size
  batch_size    = args.batch_size
  n_samples     = args.n_samples
  pct_test      = args.pct_test
  zoom_out_mags = args.zoom_out_mags
  zoom_out_prob = args.zoom_out_prob

  (args.n_samples).sort  ( reverse=True )                                                                  # to minimise retiling and regenerating in multi-run jobs, move from largest to smallest value of n_samples 
  n_samples.sort         ( reverse=True )                                                                  # to minimise retiling and regenerating in multi-run jobs, move from largest to smallest value of n_samples 
  (args.batch_size).sort ( reverse=True )                                                                  # ditto
  batch_size.sort        ( reverse=True )                                                                  # ditto
  (args.n_tiles).sort    ( reverse=True )                                                                  # ditto
  n_tiles.sort           ( reverse=True )                                                                  # ditto
  (args.tile_size).sort  ( reverse=True )                                                                  # ditto
  tile_size.sort         ( reverse=True )                                                                  # ditto
                          
  # Need to remember these across all runs in a job
  global top_up_factors_train
  global top_up_factors_test  
  
  global last_stain_norm                                                                                   
  global last_gene_norm                                                                                    
  global run_level_classifications_matrix
  global run_level_classifications_matrix_acc
  global job_level_classifications_matrix 
  global aggregate_tile_probabilities_matrix
  global aggregate_tile_level_winners_matrix
  global patches_true_classes
  global patches_case_id  
  
  global probabilities_matrix                                                                              # same, but for rna
  global true_classes                                                                                      # same, but for rna
  global rna_case_id                                                                                       # same, but for rna
  
  global descriptor
  global class_colors

  multimode_case_count = unimode_case_matched_count = unimode_case_unmatched_count = unimode_case____image_count = unimode_case____image_test_count = unimode_case____rna_count = unimode_case____rna_test_count = 0

  if (just_test!='True'):
    if any(i >= 1.  for i in pct_test):
      print ( f"{BOLD}{RED}CLASSI:       FATAL: in training mode, but {CYAN}pct_test>={MIKADO}{pct_test}{RESET}{BOLD}{RED}, which would result in there being no examples for training.{RESET}",        flush=True  )                                        
      print ( f"{BOLD}{RED}CLASSI:       FATAL: cannot continue - halting now{RESET}" )                 
      sys.exit(0)
    if any(i >= 0.9 for i in pct_test):
      print ( f"{BOLD}{ORANGE}CLASSI:         WARNG: in training mode, and {CYAN}pct_test={MIKADO}{pct_test}{RESET}{BOLD}{ORANGE}. One of more of these values are greater than or equal to 0.9 which seems unusual.{RESET}",        flush=True  )                                        
      print ( f"{BOLD}{ORANGE}CLASSI:         WARNG: proceeding, but be aware that for these values, fewer than 10% of the examples will be used for training{RESET}",             flush=True  )
      time.sleep(5)       
  
  we_are_autoencoding=False
  if ( (input_mode=='image') &  ('AE' in nn_type_img[0]) )  |  ( (input_mode=='rna') & ('AE' in nn_type_rna[0]) ):
    we_are_autoencoding=True

  allowable_input_modes = [ 'image', 'rna', 'image_rna' ]
  
  if not ( args.input_mode in allowable_input_modes ):
    print ( f"{RED}CLASSI:        FATAL:  input_mode={CYAN}{args.input_mode}{RESET}{RED} is not recognised{RESET}",        flush=True  )                                        
    print ( f"{RED}CLASSI:        FATAL:  these are the recognised input modes are {CYAN}{allowable_input_modes}{RESET}",  flush=True  )                 
    print ( f"{RED}CLASSI:        FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)       

  if ( any(i>1. for i in pct_test) ) | ( any(i<0. for i in pct_test) ) :
    print ( f"{RED}CLASSI:        FATAL:  {CYAN}pct_test{RESET}{RED} must be between 0.0 and 1.0{RESET}",                 flush=True )                                        
    print ( f"{RED}CLASSI:        FATAL:  cannot continue - halting now{RESET}",                                          flush=True )                 
    sys.exit(0)   

  # extract subtype names from the applicable master clinical data spreadsheet
  
  subtype_specific_global_data_location   = f"{global_data}"
  master_spreadsheet_found=False
  for f in os.listdir( subtype_specific_global_data_location ):
    if f.endswith(f"MASTER.csv"):
      master_spreadsheet_found = True
      master_spreadsheet_name  = f
      break
    
  if master_spreadsheet_found==False:
    print ( f"{RED}CLASSI:        FATAL:  could not find the '{CYAN}{dataset}{RESET}{RED}' master clinical data spreadsheet in {MAGENTA}{subtype_specific_global_data_location}{RESET}" )
    print ( f"{RED}CLASSI:        FATAL:  remedy: ensure there's a valid master clinical data spreadsheet (named {MAGENTA}{dataset}_mapping_file_MASTER.csv{RESET}{RED}) in {MAGENTA}{subtype_specific_global_data_location}{RESET}" )
    print ( f"{RED}CLASSI:        FATAL:          the master clinical data spreadsheet is created with a command line similar to this: {CYAN}python create_master_mapping_file.py  --dataset <DATASET>{RESET}{RED}{RESET}" )                                        
    print ( f"{RED}CLASSI:        FATAL:          or with the convenience shell script: {CYAN}./create_master.sh <DATASET>{RESET}{RED}{RESET}" )                                        
    print ( f"{RED}CLASSI:        FATAL:          detailed instructions on how to construct a master spreadsheet can be found in the comments section at the start of {CYAN}create_master_mapping_file.py{RESET}{RED}){RESET}" )                                        
    print ( f"{RED}CLASSI:        FATAL:  cannot continue - halting now{RESET}" )                 
    sys.exit(0)       


  fqn = f"{subtype_specific_global_data_location}/{master_spreadsheet_name}"

  if DEBUG>0:
    print ( f"CLASSI:         INFO:  extracting {CYAN}{dataset}{RESET} subtype names from {MAGENTA}{fqn}{RESET}'" )


  subtype_names           = pd.read_csv( fqn, usecols=[names_column], sep=',').dropna()                    # use pandas to extract data, dropping all empty cells
  subtype_names_as_list   = list( subtype_names[names_column][2:] )                                        # convert everything from row 2 onward into a python list. row 2 is where the subtype names are supposed to start


  if DEBUG>99:
    print ( f"CLASSI:         INFO:  highest_class_number                          = {MIKADO}{highest_class_number}{RESET}" )
    print ( f"CLASSI:         INFO:  len(subtype_names_as_list)                    = {MIKADO}{len(subtype_names_as_list)}{RESET}" )
    print ( f"CLASSI:         INFO:  subtype_names_as_list                         = {CYAN}{subtype_names_as_list}{RESET}" )
    print ( f"CLASSI:         INFO:  subtype_names_as_list[0:highest_class_number] = {CYAN}{subtype_names_as_list[0:highest_class_number+1]}{RESET}" )

  if highest_class_number > len(subtype_names_as_list)-1:
    print( f"{BOLD}{ORANGE}CLASSI:         WARNG: config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{BOLD}{ORANGE}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{BOLD}{ORANGE}') = \
{MIKADO}{highest_class_number}{RESET}{BOLD}{ORANGE}, which is greater than the highest class (subtype) in the dataset ({MIKADO}{len(subtype_names_as_list)-1}{RESET}{BOLD}{ORANGE}) (note that class numbers start at zero){RESET}", flush=True)
    print( f"{ORANGE}CLASSI:         WARNG:   therefore this config setting will be ignored. Continuing ...{RESET}", flush=True)
  
  if highest_class_number < 2:
    print( f"{BOLD}{RED}CLASSI:         FATAL: config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{BOLD}{RED}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{BOLD}{RED}') = \
{MIKADO}{highest_class_number}{RESET}{BOLD}{RED}, but there must be at least two classes (cancer subtypes) for classification to be meaningful", flush=True)
    print( f"{BOLD}{RED}CLASSI:         FATAL: cannot continue ... halting{RESET}", flush=True)
    time.sleep(10)
    sys.exit(0)

  class_specific_global_data_location   = f"{base_dir}/{dataset}_global"

  if len(subtype_names_as_list) <2:
    print( f"{BOLD}{RED}CLASSI:         FATAL:  only '{CYAN}{len(subtype_names_as_list)}{RESET}{BOLD}{RED}' classes (subtypes) were detected but there must be at least two class names (cancer subtype names) \
for classification to be meaningful",                                                           flush=True   )
    print( f"{BOLD}{RED}CLASSI:         FATAL:  further information: review the applicable MASTER mapping file ({MAGENTA}{class_specific_global_data_location}/{args.mapping_file_name}{RESET}{BOLD}{RED}). \
Ensure that at leat two subtypes are listed in the leftmost column, and that the first of these is in row 4",                                                                                flush=True   ) 
    print( f"{BOLD}{RED}CLASSI:         FATAL:  cannot continue ... halting{RESET}",                                                                                                         flush=True   ) 
    time.sleep(10)
    sys.exit(0)

  class_names  =  subtype_names_as_list if highest_class_number>=len(subtype_names_as_list) else subtype_names_as_list[0:highest_class_number+1]

  if DEBUG>0:
    print ( f"CLASSI:         INFO:  subtype names  = {CYAN}{class_names}{RESET}" )


  if ( input_mode=='image' ):
  
    if stain_norm[0]!='spcn':
      
      # make sure there are enough samples available to cover the user's requested 'n_samples' - svs case
    
      svs_file_count         = 0
      has_image_flag_count   = 0
    
      for dir_path, dirs, files in os.walk( args.data_dir ):                                               
    
        if not (dir_path==args.data_dir):                                                                  # the top level directory (dataset) has be skipped because it only contains sub-directories, not data      
          
          for f in files:
           
            if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
              svs_file_count +=1


      if svs_file_count==0:
        print ( f"{BOLD}{RED}\n\nCLASSI:         FATAL:  there are no image files at all in the working data directory{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:  possible cause: perhaps you changed to a different cancer type or input type but did not regenerate the dataset?{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:                  if so, use the {CYAN}-r {RESET}{BOLD}{RED}option ('{BOLD}{CYAN}REGEN{RESET}{BOLD}{RED}') to force the dataset to be regenerated into the working directory{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:                  e.g. '{CYAN}./do_all.sh -d <cancer type code> -i image ... {CHARTREUSE}-r True{RESET}{BOLD}{RED}'{RESET}" )
        print(  f"{BOLD}{RED}CLASSI:         FATAL: ... halting now{RESET}\n\n" )
        time.sleep(10)                                    
        sys.exit(0)
        
      if svs_file_count<10:
        print ( f"{BOLD}{ORANGE}\n\nCLASSI:         WARNG:  there are fewer than 10 image files in the working data directory, which seems odd{RESET}" )
        print ( f"{BOLD}{ORANGE}CLASSI:         WARNG:          consider using the the {CYAN}-r {RESET}{BOLD}{ORANGE}option ('{CYAN}REGEN{RESET}{BOLD}{ORANGE}') to force the dataset to be regenerated{RESET}" )
        print ( f"{BOLD}{ORANGE}CLASSI:         WARNG:          e.g. '{CYAN}./do_all.sh -d <cancer type code> -i image ... {CHARTREUSE}-r True{RESET}{BOLD}{ORANGE}'{RESET}" )
        print ( f"{BOLD}{ORANGE}CLASSI:         WARNG: ... continuing, but it's kind of pointless{RESET}\n\n\n" )
        time.sleep(5)                           


      if just_test != True:
        if svs_file_count<np.max(args.n_samples):
          print( f"{BOLD}{ORANGE}CLASSI:         WARNG: there aren't enough samples. A file count reveals a total of {MIKADO}{svs_file_count}{RESET}{BOLD}{ORANGE} SVS and TIF files in {MAGENTA}{args.data_dir}{RESET}{BOLD}{ORANGE}, whereas the largest value in user configuation parameter '{CYAN}N_SAMPLES[]{RESET}{BOLD}{ORANGE}' = {MIKADO}{np.max(args.n_samples)}{RESET})" ) 
          print( f"{ORANGE}CLASSI:         WARNG:   changing values of '{CYAN}N_SAMPLES{RESET}{ORANGE} that are greater than {RESET}{MIKADO}{svs_file_count}{RESET}{ORANGE} to exactly {MIKADO}{svs_file_count}{RESET}{ORANGE} and continuing{RESET}" )
          args.n_samples = [  el if el<=svs_file_count else svs_file_count for el in args.n_samples   ]
          n_samples = args.n_samples
        else:
          print( f"CLASSI:         INFO:  {WHITE}a file count shows there is a total of {MIKADO}{svs_file_count}{RESET} SVS and TIF files in {MAGENTA}{args.data_dir}{RESET}, which may be sufficient to perform all requested runs (configured value of'{CYAN}N_SAMPLES{RESET}' depending on the case subset used.{RESET})" )
      else:
        min_required = int(np.max(args.n_samples) * pct_test  )
        if svs_file_count< min_required:
          print( f"{BOLD}{ORANGE}CLASSI:         WARNG: there aren't enough samples. A file count reveals a total of {MIKADO}{svs_file_count}{RESET}{BOLD}{ORANGE} SVS and TIF files in {MAGENTA}{args.data_dir}{RESET}{BOLD}{ORANGE}, whereas the absolute minimum number required for this test run is {MIKADO}{min_required}{RESET}" ) 
          print( f"{ORANGE}CLASSI:         WARNG: changing values of '{CYAN}N_SAMPLES{RESET}{ORANGE} that are greater than {RESET}{MIKADO}{svs_file_count}{RESET}{ORANGE} to exactly {MIKADO}{svs_file_count}{RESET}{ORANGE} and continuing{RESET}" )
          args.n_samples = [  el if el<=svs_file_count else svs_file_count for el in args.n_samples   ]
          n_samples = args.n_samples
        else:
          print( f"CLASSI:         INFO:  {WHITE}a file count shows there is a total of {MIKADO}{svs_file_count}{RESET} SVS and TIF files in {MAGENTA}{args.data_dir}{RESET}, which may be sufficient to perform all requested runs (configured value of'{CYAN}N_SAMPLES{RESET}{BOLD}{ORANGE}' depending on the case subset used = {MIKADO}{np.max(args.n_samples)}{RESET})" )


    if stain_norm[0]=='spcn':
      
      # make sure there are enough samples available to cover the user's requested 'n_samples' - spcn case
    
      spcn_file_count   = 0
    
      for dir_path, dirs, files in os.walk( args.data_dir ):                                               
    
        if not (dir_path==args.data_dir):                                                                  # the top level directory (dataset) has be skipped because it only contains sub-directories, not data      
          
          for f in files:
           
            if ( f.endswith( 'spcn' )):
              spcn_file_count +=1
            
      if spcn_file_count<np.max(args.n_samples):
        print( f"{BOLD}{ORANGE}CLASSI:         WARNG:  there aren't enough samples. A file count reveals a total of {MIKADO}{spcn_file_count}{RESET} {BOLD}{MAGENTA}spcn{RESET}{BOLD_ORANGE} files in {BOLD}{MAGENTA}{args.data_dir}{RESET}{BOLD}{ORANGE}, whereas (the largest value in) user configuation parameter '{CYAN}N_SAMPLES[]{RESET}{BOLD}{ORANGE}' = {MIKADO}{np.max(args.n_samples)}{RESET})" ) 
        print( f"{BOLD}{ORANGE}CLASSI:         WARNG:  changing values of '{CYAN}N_SAMPLES{RESET}{BOLD_ORANGE} that are greater than {RESET}{BOLD}{MIKADO}{spcn_file_count}{RESET}{BOLD}{ORANGE} to exactly {MIKADO}{spcn_file_count}{RESET}{BOLD}{ORANGE} and continuing{RESET}" )
        args.n_samples = [  el if el<=spcn_file_count else spcn_file_count for el in args.n_samples   ]
        n_samples = args.n_samples
      else:
        print( f"CLASSI:         INFO:  {WHITE}a file count shows there is a total of {MIKADO}{spcn_file_count}{RESET} spcn files in {MAGENTA}{args.data_dir}{RESET}, which is sufficient to perform all requested runs (configured value of'{CYAN}N_SAMPLES{RESET}' = {MIKADO}{np.max(args.n_samples)}{RESET})" )


    # if tiling is to be skipped, make sure there tiling has been previously conducted (i.e. tile files exist)
  
    if skip_tiling=='True':
  
      png_file_count   = 0
    
      for dir_path, dirs, files in os.walk( args.data_dir ):                                                 # each iteration takes us to a new directory under data_dir
    
        if not (dir_path==args.data_dir):                                                                    # the top level directory (dataset) has be skipped because it only contains sub-directories, not data      
          
          for f in files:
           
            if (   ( f.endswith( 'png' ))    ):
              png_file_count +=1
            
      if png_file_count<20:
        print( f"{BOLD}{RED}CLASSI:         FATAL:  a count just now reveals a total of {MIKADO}{png_file_count}{RESET}{BOLD}{RED} tiles (png files) in {MAGENTA}{args.data_dir}{RESET}{BOLD}{RED} !!!{RESET}" ) 
        print( f"{BOLD}{RED}CLASSI:         FATAL:  possible remedy: do not use either the '{CYAN}SKIP_TILING{RESET}{BOLD}{RED}' flag ({CYAN}-s {RESET}{BOLD}{RED}) or the '{CYAN}SKIP_GENERATION{RESET}{BOLD}{RED}' flag ({CYAN}-g {RESET}{BOLD}{RED}), so that tiling and dataset generation can occur. After you've tiled and generated once, it's OK to used these flags, which save a lot of time{RESET}"      ) 
        time.sleep(10)
        sys.exit(0)



  if  ( stain_norm[0]=='spcn' ):
    print( f"{MAGENTA}{BOLD}CLASSI:         INFO:  '{CYAN}{BOLD}stain_norm{RESET}{MAGENTA}'{BOLD} option '{CYAN}{BOLD}spcn{RESET}{MAGENTA}{BOLD}' is set. The spcn slide set will be used and the svs side set will be ignored{RESET}", flush=True)


  if  any( el<0.05 for el in pct_test ):
    print( f"{BOLD_RED}CLASSI:         INFO:  pct_test = {MIKADO}{pct_test}{BOLD_RED}. At least one of these is less than 0.05 (5%){RESET}{BOLD_RED}'. This is such a low percentage for hold out testing that it might be unintended.{RESET}", flush=True)
    print( f"{BOLD_RED}CLASSI:         INFO:    further information: correct if necessary by changing the percent test option: Bash long form: {BOLD_CYAN}PCT_TEST{BOLD_RED}; Bash short form '{CYAN}-1'{BOLD_ORANGE}; python {BOLD_CYAN}--pct_test{RESET}", flush=True)
    print( f"{BOLD_RED}CLASSI:         INFO:    not halting ... resuming in 7 seconds", flush=True)
    time.sleep(5)
  elif  any(el<0.1 for el in pct_test):
    np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})
    print( f"{BOLD_ORANGE}CLASSI:         INFO:  pct_test = {MIKADO}{np.array(pct_test)}{BOLD_ORANGE}. At least one of these is less than 0.1 (10%){RESET}{BOLD_ORANGE}'. Is this intended?{RESET}", flush=True)
    print( f"{BOLD_ORANGE}CLASSI:         INFO:    further information: correct if necessary by changing the percent test option: Bash long form: {BOLD_CYAN}PCT_TEST{BOLD_ORANGE}; Bash short form '{CYAN}-1'{BOLD_ORANGE}; python {BOLD_CYAN}--pct_test{RESET}", flush=True)
    print( f"{BOLD_ORANGE}CLASSI:         INFO:    not halting ... resuming in 7 seconds", flush=True)
    time.sleep(5)

  if DEBUG>1:
    if  0 in highest_class_number:
      print( f"{RED}CLASSI:         FATAL:  config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{RED}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{RED}') is not permitted to have the value {MIKADO}0{RESET}", flush=True)
      print( f"{RED}CLASSI:         FATAL: ... halting now{RESET}" )
      time.sleep(4)
  
    if  1 in highest_class_number:
      print( f"\n{CHARTREUSE}CLASSI:         WARNG:  config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{CHARTREUSE}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{CHARTREUSE}') contains the value {MIKADO}1{RESET}{CHARTREUSE}, which seems very odd", flush=True)
      print( f"{CHARTREUSE}CLASSI:         WARNG: ... continuing{RESET}" )
      time.sleep(4)
  
    if  2 in highest_class_number:
      print( f"\n{CHARTREUSE}CLASSI:         WARNG:  config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{CHARTREUSE}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{CHARTREUSE}') contains the value {MIKADO}2{RESET}{CHARTREUSE}, which is very low. Was this intentional?", flush=True)
      print( f"{CHARTREUSE}CLASSI:         WARNG: ... continuing{RESET}" )
      time.sleep(4)

  # ~ if len(args.zoom_out_prob)==1:
    # ~ args.zoom_out_prob[0]=1.
    # ~ zoom_out_prob[0]=1.

  # ~ if math.fsum(args.zoom_out_prob) != 1.0:
   # ~ print( f"\r{RESET}{BOLD}{RED}CLASSI:         FATAL: probabilities in configuration vector '{CYAN}zoom_out_prob{RESET}{BOLD}{RED}' add up to more than {MIKADO}1.0{RESET}{BOLD}{RED} (FYI they add up to {MIKADO}{sum(args.zoom_out_prob)}{RESET}{BOLD}{RED})", flush=True)
   # ~ print( f"\r{RESET}{BOLD}{RED}CLASSI:         FATAL: can't continue ... halting now{RESET}" )   
   # ~ sys.exit(0)
   
   
     

  if args.clustering == 'NONE':
    if  'VGG' in nn_type_img[0]:
      if  min(args.tile_size)<32:
        print( f"{RED}CLASSI:         FATAL:  for the VGG models, tile size ('{CYAN}TILE_SIZE{RESET}{RED}' corresponding to python argument '{CYAN}--tile_size{RESET}{RED}') is not permitted to be less than {MIKADO}32{RESET}", flush=True)
        print( f"{RED}CLASSI:         FATAL: ... halting now{RESET}" )
        sys.exit(0)

  if  ( pretrain=='True' ) & ( input_mode=='image' ):
    print( f"{COTTON_CANDY}CLASSI:         INFO:  {CYAN}PRETRAIN{RESET}{COTTON_CANDY} option ({CYAN}-p True{RESET}{COTTON_CANDY}) (corresponding to python argument '{CYAN}--pretrain True{RESET}{COTTON_CANDY}') has been selected{RESET}", flush=True)

  if  ( pretrain=='True' ) & ( input_mode!='image' ):
    print( f"{RED}CLASSI:         FATAL: the {CYAN}PRETRAIN{RESET}{RED} option ({CYAN}-p True{RESET}{RED}) (corresponding to python argument '{CYAN}--pretrain True{RESET}{RED}') is only supported in image mode{RESET}", flush=True)
    print( f"{RED}CLASSI:         FATAL: ... halting now{RESET}" )
    sys.exit(0)

  if just_test!='True':
    if  not (  ( args.cases=='ALL_ELIGIBLE_CASES' ) | ( args.cases=='UNIMODE_CASE' ) | ( args.cases=='MULTIMODE____TEST' )  ):
      print( f"{RED}CLASSI:         FATAL: user option  {CYAN}-c ('cases')  {RESET}{RED} = '{CYAN}{args.cases}{RESET}{RED}' is not supported{RESET}" )
      print( f"{RED}CLASSI:         FATAL: explanation:  in training mode the following options are supported: '{CYAN}ALL_ELGIBLE_CASES{RESET}{RED}', '{CYAN}MULTIMODE____TEST{RESET}{RED}', '{CYAN}UNIMODE_CASE{RESET}{RED}'" )
      print( f"{RED}CLASSI:         FATAL: ... halting now{RESET}" )
      sys.exit(0)
  else:
    if pretrain=='True':
      print( f"{RED}CLASSI:         FATAL: the {CYAN}PRETRAIN{RESET}{RED} option ({CYAN}-p True{RESET}{RED}) corresponding to python argument {CYAN}--pretrain True{RESET}{RED} is not supported in test mode (because it makes no sense){RESET}", flush=True)
      print( f"{RED}CLASSI:         FATAL: ... halting now{RESET}" )
      sys.exit(0)
    if we_are_autoencoding != True:                                                                        # it's ok to use ALL_ELIGIBLE_CASES in test mode if we are generating embeddings, but not otherwise
      if args.cases=='ALL_ELIGIBLE_CASES':
        print( f"{RED}CLASSI:         FATAL: in test mode '{RESET}{CYAN}-c ALL_ELIGIBLE_CASES{RESET}{RED}' is not supported{RESET}" )
        print( f"{RED}CLASSI:         FATAL:   explanation:  The '{CYAN}CASES{RESET}{RED}' subset '{CYAN}ALL_ELIGIBLE_CASES{RESET}{RED}' includes tiles from examples (cases) which were used to train the model that is about to be used. Therefore, the results would be meaningless{RESET}" )
        print( f"{RED}CLASSI:         FATAL:   explanation:  in test mode the following case subsets are supported: ''{CYAN}UNIMODE_CASE{RESET}{RED}', '{CYAN}MULTIMODE____TEST{RESET}{RED}'" )
        print( f"{RED}CLASSI:         FATAL:   ... halting now{RESET}" )
        sys.exit(0)
      elif  not ( ( args.cases=='UNIMODE_CASE' ) | ( args.cases=='MULTIMODE____TEST' )  ):
        print( f"{RED}CLASSI:         FATAL: unknown case subset: {CYAN}-c ('cases')  {RESET}{RED} = '{CYAN}{args.cases}{RESET}{RED}'{RESET}" )
        print( f"{RED}CLASSI:         FATAL:   ... halting now{RESET}" )
        sys.exit(0)
  
      

  if  ( args.cases!='ALL_ELIGIBLE_CASES' ) & ( args.divide_cases == 'False' ):
    print( f"{ORANGE}CLASSI:         INFO:  user option {CYAN}-v ('divide_cases') {RESET}{ORANGE} = {CYAN}False{RESET}{ORANGE}, however option {CYAN}-c ('cases'){RESET}{ORANGE} is NOT '{CYAN}ALL_ELIGIBLE_CASES{RESET}{ORANGE}'.  The requested subset of cases may or may not already exist{RESET}" )
    print( f"{ORANGE}CLASSI:         INFO:    this will cause problems if the requested subset ({RESET}{ORANGE}'{CYAN}{args.cases}{RESET}{ORANGE}') does not exist (in {RESET}{ORANGE}'{CYAN}{args.data_dir}{RESET}{ORANGE}') from a previous run with {CYAN}-v {'divide_cases'}{RESET}{ORANGE} flag set. NOT halting, but if CLASSI crashes, you'll know why.{RESET}" )
      
  c_m = f"plt.cm.{eval('colour_map')}"                                                                     # the 'eval' is so that the user input string will be treated as a variable
  class_colors = [ eval(c_m)(i) for i in range(len(class_names))]                                          # makes an array of colours by calling the user defined colour map (which is a function, not a variable)
  if DEBUG>555:
    print (f"CLASSI:         INFO:  class_colors = \n{MIKADO}{class_colors}{RESET}" )


  if ( input_mode=='image' ): 
    if 1 in batch_size:
      print ( f"{RED}CLASSI:         INFO: sorry - parameter '{CYAN}BATCH_SIZE{RESET}{RED}' (currently '{MIKADO}{batch_size}{RESET}{RED}' cannot include a value <2 for images{RESET}" )
      print ( f"{RED}CLASSI:         INFO: halting now{RESET}" )      
      sys.exit(0) 
 
  
  # ~ if  (mode=='classify') & (args.clustering=='NONE'):
    # ~ if  'AE' in nn_type_img[0]:
      # ~ print( f"{RED}CLASSI:         FATAL: the network model must not be an autoencoder if mode='{MIKADO}{mode}{RESET}{RED}' (you have NN_TYPE_IMG='{MIKADO}{nn_type_img[0]}{RESET}{RED}', which is an autoencoder) ... halting now{RESET}" )
      # ~ sys.exit(0)
    # ~ if  'AE' in nn_type_rna[0]:
      # ~ print( f"{RED}CLASSI:         FATAL: the network model must {UNDER}not{RESET}{RED} be an autoencoder if mode='{MIKADO}{mode}{RESET}{RED}' (you have NN_TYPE_RNA='{MIKADO}{nn_type_rna[0]}{RESET}{RED}', which is an autoencoder) ... halting now{RESET}" )
      # ~ sys.exit(0)
      

  if clustering=='NONE':
    if  ( use_autoencoder_output=='True' ):
      if  ( input_mode=='image' ) and not  ( 'AE' in nn_type_img[0] ) :
        print( f"{RED}CLASSI:         FATAL: the network model must be an autoencoder if flag '{CYAN}USE_AUTOENCODER_OUTPUT{RESET}{RED}=='{MIKADO}{True}{RESET}{RED}' (you have NN_TYPE_IMG='{CYAN}{nn_type_img[0]}{RESET}{RED}', which is not an autoencoder) ... halting now{RESET}" )
        sys.exit(0)
      if  ( input_mode=='rna'   ) and not  ( 'AE' in nn_type_rna[0] ):
        print( f"{RED}CLASSI:         FATAL: the network model must be an autoencoder if flag '{CYAN}USE_AUTOENCODER_OUTPUT{RESET}{RED}=='{MIKADO}{True}{RESET}{RED}' (you have NN_TYPE_RNA='{CYAN}{nn_type_rna[0]}{RESET}{RED}', which is not an autoencoder) ... halting now{RESET}" )
        sys.exit(0)
    
  if supergrid_size<1:
    print( f"{RED}CLASSI:         FATAL:    parameter 'supergrid_size' (current value {supergrid_size}) must be an integer greater than zero ... halting now{RESET}" )
    sys.exit(0)

  if ( args.cases=='MULTIMODE____TEST' ):                                                                           
    if DEBUG>0:
      print( f"{CHARTREUSE}CLASSI:         INFO:  '{CYAN}args.cases{RESET}{CHARTREUSE}' = {MAGENTA}{args.cases}{RESET}{CHARTREUSE}' Therefore '{CYAN}N_SAMPLES{RESET}{CHARTREUSE}' (currently {MIKADO}{n_samples[0]}{RESET}{CHARTREUSE}) will be changed to the value of '{CYAN}CASES_RESERVED_FOR_IMAGE_RNA{RESET}{CHARTREUSE} ({MIKADO}{args.cases_reserved_for_image_rna}{RESET}{CHARTREUSE}){RESET}" ) 
    args.n_samples[0] = cases_reserved_for_image_rna
    n_samples         = args.n_samples
  
  n_samples_max  = np.max(n_samples)
  tile_size_max  = np.max(tile_size)  
  n_tiles_max    = np.max(n_tiles)
  n_tiles_last   = 0                                                                                       # used to trigger regeneration of tiles if a run requires more tiles that the preceeding run 
  n_samples_last = 0
  tile_size_last = 0                                                                                       # also used to trigger regeneration of tiles if a run requires a different file size than the preceeding run 
  n_classes      = len(class_names)
  

  if just_test=='True':
    print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set. No training will be performed.  Saved model (which must exist from previous training run) will be loaded.{RESET}" )
    if n_epochs>1:
      print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set, so {CYAN}n_epochs{RESET}{ORANGE} (currently {MIKADO}{n_epochs}{RESET}{ORANGE}) has been set to {MIKADO}1{RESET}{ORANGE} for this run{RESET}" ) 
      n_epochs=1
      args.n_epochs=1
    if ( multimode!='image_rna' ) & ( input_mode!='image_rna' ):
      print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set. Only one thread will be used for processing to ensure patch tiles are processed in the correct sequence{RESET}" )
      if len(args.hidden_layer_neurons)>1:
        print( f"{RED}CLASSI:         INFO:  in test mode, ({CYAN}JUST_TEST=\"True\"{RESET}{RED}), only one value is allowed for the parameter '{CYAN}HIDDEN_LAYER_NEURONS{RESET}{RED}'. At the moment it has {MIKADO}{len(args.hidden_layer_neurons)}{RESET}{RED} values ... halting{RESET}" )
        sys.exit(0)        
      if input_mode=='image':
        if not tile_size_max**0.5 == int(tile_size_max**0.5):
          print( f"{RED}CLASSI:         INFO:  in test_mode, '{CYAN}TILE_SIZE ('-T'){RESET}{RED}' (currently {MIKADO}{tile_size[0]}{RESET}{RED}) must be a perfect square (eg. 49, 64, 144, 256 ) ... halting [1586]{RESET}" )
          sys.exit(0)
      if len(batch_size)>1:
        print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'   flag is set but but '{CYAN}BATCH_SIZE{RESET}{ORANGE}' has {MIKADO}{len(batch_size)}{RESET}{ORANGE} values ({MIKADO}{batch_size}{RESET}{ORANGE}). Only the first value ({MIKADO}{batch_size[0]}{ORANGE}) will be used{RESET}" )
        del batch_size[1:]       
      if len(n_tiles)>1:
        print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'   flag is set but but '{CYAN}N_TILES{RESET}{ORANGE}'    has {MIKADO}{len(n_tiles)}{RESET}{ORANGE} values ({MIKADO}{n_tiles}{RESET}{ORANGE}). Only the first value ({MIKADO}{n_tiles[0]}{RESET}{ORANGE}) will be used{RESET}" )
        del n_tiles[1:] 
      n_tiles[0] = supergrid_size**2 * batch_size[0]
      print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set, therefore '{CYAN}N_TILES{RESET}{ORANGE}' has been set to '{CYAN}SUPERGRID_SIZE^2 * BATCH_SIZE{RESET}{ORANGE}' ({MIKADO}{supergrid_size} * {supergrid_size} * {batch_size} =  {n_tiles}{RESET} {ORANGE}) for this job{RESET}" )          
    else:
      print( f"{BOLD}{CHARTREUSE}CLASSI:         INFO:   user argument  '{BOLD}{CYAN}MULTIMODE'{RESET}{BOLD}{CHARTREUSE} = '{multimode}'. Embeddings will be generated.{RESET}"   )      
  else:
    if ( input_mode=='image' ) &  ( pretrain!='True' ):
      if not tile_size_max**0.5 == int(tile_size_max**0.5):
        print( f"{ORANGE}CLASSI:         WARNG: '{CYAN}TILE_SIZE{RESET}{ORANGE}' ({BOLD}{MIKADO}{tile_size_max}{RESET}{ORANGE}) isn't a perfect square, which is fine for training, but will mean you won't be able to use test mode on the model you train here{RESET}" )
      if supergrid_size>1:
        if DEBUG>99:
          print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'  flag is NOT set, so supergrid_size (currently {MIKADO}{supergrid_size}{RESET}{ORANGE}) will be ignored{RESET}" )
        args.supergrid_size=1

           

  if ( ( just_test=='True')  & ( multimode!='image_rna' ) ):
    
    if ( rand_tiles=='True'):
      print ( f"\r{BOLD}{ORANGE}CLASSI:         WARNG: {CYAN}( just_test=={MIKADO}'True'{RESET}{CYAN})  & ( multimode!={MIKADO}'image_rna'{RESET}{CYAN}){BOLD}{ORANGE} but user argument {CYAN}rand_tiles=={MIKADO}'True'{RESET}{BOLD}{ORANGE}. It will  be changed to {MIKADO}'False'{RESET}{BOLD}{ORANGE} since test mode requires sequentially generated tiles{RESET}\n\n" )

      args.rand_tiles = 'False'
      rand_tiles      = 'False'



  if use_same_seed=='True':
    print( f"{ORANGE}CLASSI:         WARNG: '{CYAN}USE_SAME_SEED{RESET}{ORANGE}' flag is set. The same seed will be used for all runs in this job{RESET}" )
    torch.manual_seed(0.223124)    

  if ( input_mode=='rna' ): 
    
    # make sure there are enough samples available to cover the user's requested "n_samples"
  
    rna_file_count   = 0
  
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                      # each iteration takes us to a new directory under data_dir
  
      if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data      
        
        for f in files:
         
          if f=='rna.npy':
            rna_file_count +=1
          
    if rna_file_count<np.max(args.n_samples):
      print( f"{BOLD}{ORANGE}CLASSI:         WARNG: there are not {MIKADO}{np.max(args.n_samples)}{BOLD}{ORANGE} RNA-Seq examples available to be used. A file count reveals a total of {MIKADO}{rna_file_count}{RESET}{BOLD}{ORANGE} rna files in {MAGENTA}{args.data_dir}{RESET}{BOLD}{ORANGE}, whereas (the largest value in) user configuation parameter '{CYAN}N_SAMPLES{RESET}{BOLD}{ORANGE}' = {MIKADO}{np.max(args.n_samples)}{RESET}" ) 
      print( f"{BOLD}{ORANGE}CLASSI:         WARNG: changing all values in the user configuration parameter '{CYAN}N_SAMPLES{RESET}{BOLD}{ORANGE}' that are greater than {RESET}{BOLD}{MIKADO}{rna_file_count}{RESET}{BOLD}{ORANGE} to exactly {MIKADO}{rna_file_count}{RESET}{BOLD}{ORANGE}{RESET}" )
      args.n_samples = [  el if el<=rna_file_count else rna_file_count for el in args.n_samples   ]
      n_samples      = args.n_samples

    else:
      print( f"CLASSI:         INFO:  {WHITE}a file count shows there is a total of {MIKADO}{rna_file_count}{RESET} rna files in {MAGENTA}{args.data_dir}{RESET}, which is sufficient to perform all requested runs (configured value of'{CYAN}N_SAMPLES{RESET}' = {MIKADO}{np.max(args.n_samples)}{RESET})" )


  if (DEBUG>8):
    print ( f"CLASSI:         INFO:  highest_class_number = {MIKADO}{highest_class_number}{RESET}",    flush=True)
    print ( f"CLASSI:         INFO:  n_samples            = {MIKADO}{n_samples}{RESET}",               flush=True)
    if ( input_mode=='image' ):
      print ( f"CLASSI:         INFO:  n_tiles              = {MIKADO}{n_tiles}{RESET}",                 flush=True)
      print ( f"CLASSI:         INFO:  tile_size            = {MIKADO}{tile_size}{RESET}",               flush=True)


  # (A)  SET UP JOB LOOP

  already_tiled=False
  already_generated=False
  total_tiles_required_train = 456789
  total_tiles_required_test  = 123456
  top_up_factors_train        = np.zeros( n_classes, dtype=int )
  top_up_factors_test         = np.zeros( n_classes, dtype=int )  
                          
  repeater = [ 1 for r in range( 0, repeat) ] 
  
  parameters = dict( 
                            repeater =   repeater,
                          stain_norm =   stain_norm,
                          tile_size  =   tile_size,
                                 lr  =   lr,
                           pct_test  =   pct_test,
                          n_samples  =   n_samples,
                         batch_size  =   batch_size,
                            n_tiles  =   n_tiles,
                         rand_tiles  =   [ rand_tiles ],
                        nn_type_img  =   nn_type_img,
                        nn_type_rna  =   nn_type_rna,
               hidden_layer_neurons  =   hidden_layer_neurons,
           low_expression_threshold  =   low_expression_threshold,
                  cutoff_percentile  =   cutoff_percentile,
               embedding_dimensions  =   embedding_dimensions,
                          dropout_1  =   dropout_1,
                          dropout_2  =   dropout_2,
                        nn_optimizer =   nn_optimizer,
                      gene_data_norm =   gene_data_norm, 
                 gene_data_transform =   gene_data_transform,                                                
                      label_swap_pct =   [   0.0   ],
                       make_grey_pct =   [   0.0   ],
                              jitter =   [  [ 0.0, 0.0, 0.0, 0.0 ] ]  )


  param_keys   = [v for v in parameters.keys()]
  param_values = [v for v in parameters.values()]
  
  if DEBUG>2:
    print ( f"\n\n\nCLASSI:         INFO:  parameters      =  \n{BOLD}{HOT_PINK}{parameters}{RESET}"  )
  if DEBUG>99:
    print ( f"CLASSI:         INFO:  param_keys      =  \n{BOLD}{HOT_PINK}{param_keys}{RESET}"  )
    print ( f"CLASSI:         INFO:  param_values    =  \n{BOLD}{HOT_PINK}{param_values}{RESET}"  )

  start_column  = 0
  offset        = 12
  second_offset = 12

  total_runs_in_job = len(list(product(*param_values)))
  
  if skip_tiling=='True':

    if (total_runs_in_job==1) & (args.make_balanced=='True'):

      print( f"{SAVE_CURSOR}\033[77;0H{BOLD}{ORANGE}CLASSI:         WARNG:  skip tiling flag is set ({CYAN}-s True{RESET}{BOLD}{ORANGE}), but cannot skip tiling if there is only one run in a job and {CYAN}MAKE_BALANCED=True{RESET}{BOLD}{ORANGE}, as {CYAN}top_up_factors{RESET}{BOLD}{ORANGE}, which are necessary to adjust tiles per subtype per slide, would not be calculated{RESET}" ) 
      print( f"\033[78;0H{BOLD}{ORANGE}CLASSI:         WARNG:  ignoring skip tiling flag tiling will be performed{RESET}{RESTORE_CURSOR}"      ) 
      skip_tiling='False'
      args.skip_tiling='False'
      time.sleep(1)

    
  # establish and initialise some variables
  n_classes = len(class_names)
  run_level_classifications_matrix     =  np.zeros( (n_classes, n_classes), dtype=int )
  job_level_classifications_matrix     =  np.zeros( (n_classes, n_classes), dtype=int )
  run_level_classifications_matrix_acc =  np.zeros( ( total_runs_in_job, n_classes, n_classes ), dtype=int )     
  
  
  if DEBUG>0:
    print ( f"CLASSI:         INFO:  total_runs_in_job    =  {CARRIBEAN_GREEN}{total_runs_in_job}{RESET}"  )

  image_headings =\
f"\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Cpct_test\
\r\033[{start_column+2*offset}Cexamples\
\r\033[{start_column+3*offset}Cbatch_size\
\r\033[{start_column+4*offset}Ctiles/image\
\r\033[{start_column+5*offset}Cnum_classes\
\r\033[{start_column+6*offset}Ctile_size\
\r\033[{start_column+7*offset}Crand_tiles\
\r\033[{start_column+8*offset}Cnet_img\
\r\033[{start_column+9*offset}Coptimizer\
\r\033[{start_column+10*offset}Cstain_norm\
\r\033[{start_column+11*offset}Clabel_swap\
\r\033[{start_column+12*offset}Cgreyscale\
\r\033[{start_column+13*offset}Cextraction dimensions (multiples of base tile size)\
\r\033[{start_column+14*offset+52}Cprobability for each of the extraction dimensions\
\r\033[{start_column+15*offset+105}Cjitter vector\
"

  rna_headings =\
f"\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Cpct_test\
\r\033[{start_column+2*offset}Csamples\
\r\033[{start_column+3*offset}Cbatch_size\
\r\033[{start_column+4*offset}Cnetwork\
\r\033[{start_column+5*offset}Chidden\
\r\033[{start_column+6*offset}CFPKM percentile/threshold\
\r\033[{start_column+8*offset}Cembedded\
\r\033[{start_column+9*offset}Cdropout_1\
\r\033[{start_column+10*offset}Cdropout_2\
\r\033[{start_column+11*offset}Coptimizer\
\r\033[{start_column+12*offset}Cnormalisation\
\r\033[{start_column+13*offset+3}Ctransform\
\r\033[{start_column+14*offset+3}Clabel_swap\
"
  
  if DEBUG>0:
    if input_mode=='image':
      print(f"\n{UNDER}JOB LIST:{RESET}")
      print(f"\r\033[155C ------------ tile extraction parameters (all tiles will be saved at base tile size ({MIKADO}{tile_size[0]}x{tile_size[0]}{RESET}) -------------- {RESET}")      
      print(f"\r\033[2C{image_headings}{RESET}")      
      for repeater, stain_norm, tile_size, lr, pct_test, n_samples, batch_size, n_tiles, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, low_expression_threshold, cutoff_percentile, embedding_dimensions, dropout_1, dropout_2, nn_optimizer, gene_data_norm, gene_data_transform, label_swap_pct, make_grey_pct, jitter in product(*param_values):    

        print( f"{CARRIBEAN_GREEN}\
\r\033[2C\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{pct_test:<9.6f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{n_tiles:<5d}\
\r\033[{start_column+5*offset}C{n_classes:<2d}\
\r\033[{start_column+6*offset}C{tile_size:<3d}\
\r\033[{start_column+7*offset}C{rand_tiles:<5s}\
\r\033[{start_column+8*offset}C{nn_type_img:<10s}\
\r\033[{start_column+9*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+10*offset}C{stain_norm:<10s}\
\r\033[{start_column+11*offset}C{label_swap_pct:<6.1f}\
\r\033[{start_column+12*offset}C{make_grey_pct:<5.1f}\
\r\033[{start_column+13*offset}C{zoom_out_mags:}\
\r\033[{start_column+14*offset+55}C{np.round(np.array(zoom_out_prob),3):}\
\r\033[{start_column+15*offset+105}C{jitter:}\
{RESET}" )



    elif input_mode=='rna':
      print(f"\n{UNDER}JOB LIST:{RESET}")
      print(f"\033[2C\{rna_headings}{RESET}")
      
      for repeater, stain_norm, tile_size, lr, pct_test, n_samples, batch_size, n_tiles, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, low_expression_threshold, cutoff_percentile, embedding_dimensions, dropout_1, dropout_2, nn_optimizer, gene_data_norm, gene_data_transform, label_swap_pct, make_grey_pct, jitter in product(*param_values):    

        print( f"{CARRIBEAN_GREEN}\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{100*pct_test:<9.0f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{nn_type_rna:<10s}\
\r\033[{start_column+5*offset}C{hidden_layer_neurons:<5d}\
\r\033[{start_column+6*offset}C{cutoff_percentile:<4.0f}\
\r\033[{start_column+7*offset}C{low_expression_threshold:<9.6f}\
\r\033[{start_column+8*offset}C{embedding_dimensions:<5d}\
\r\033[{start_column+9*offset}C{dropout_1:<5.2f}\
\r\033[{start_column+10*offset}C{dropout_2:<5.2f}\
\r\033[{start_column+11*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+12*offset}C{gene_data_norm:<10s}\
\r\033[{start_column+13*offset+3}C{gene_data_transform:<10s}\
\r\033[{start_column+14*offset+3}C{label_swap_pct:<6.1f}\
{RESET}" )
  

  # ~ if total_runs_in_job>1:
    # ~ while(True):
      # ~ key = input("review job list, then press any key to continue ...")
      # ~ if(len(key) >= 0):
        # ~ break

  if total_runs_in_job>1:
    print ( "Execution will continue in 20 seconds, or press any key to continue now ..." )
    timeout = 20
    rlist, wlist, xlist = select([sys.stdin], [], [], timeout)


  if (just_test=='True') & (input_mode=='image') & (multimode!= 'image_rna'):   
    if not ( batch_size == int( math.sqrt(batch_size) + 0.5) ** 2 ):
      print( f"{RED}CLASSI:         FATAL:  in test mode {CYAN}BATCH_SIZE ('-b') {RESET}{RED}(currently {MIKADO}{batch_size}{RESET}{RED}) must be a perfect square (4, 9, 16, 25 ...) to permit selection of a a 2D contiguous patch. Halting [2989].\033[m" )
      sys.exit(0)      


  # (B) RUN JOB LOOP

  run=0
  
  for repeater, stain_norm, tile_size, lr, pct_test, n_samples, batch_size, n_tiles, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, low_expression_threshold, cutoff_percentile, embedding_dimensions, dropout_1, dropout_2, nn_optimizer, gene_data_norm, gene_data_transform, label_swap_pct, make_grey_pct, jitter in product(*param_values): 
 
    if ( divide_cases == 'True' ):
      
      if just_test=='False':                                                                      
        multimode_case_count, unimode_case_matched_count, unimode_case_unmatched_count, unimode_case____image_count, unimode_case____image_test_count, unimode_case____rna_count, unimode_case____rna_test_count =  \
                    segment_cases( args, n_classes, class_names, n_tiles, pct_test )  # boils down to setting flags in the directories of certain cases, esp. 'MULTIMODE_CASE_FLAG'

      else:
        print( f"{RED}CLASSI:         FATAL: user option  {CYAN}-v ('args.cases'){RESET}{RED} is not allowed in test mode ({CYAN}JUST_TEST=True{RESET}, {CYAN}--just_test 'True'{RESET}){RED}{RESET}" )
        print( f"{RED}CLASSI:         FATAL: explanation:  it will resegment the cases, meaning there is every chance cases you've trained on will end up in the test set{RESET}" )
        print( f"{RED}CLASSI:         FATAL: ... halting now{RESET}" )
        sys.exit(0)
 

    if input_mode=='image':                                                                                # in the case of images, the HAS_IMAGE flag MUST be present, so we check.  Can't automate, as segment_cases should only be run one time, after regeneration of the dataset. 
      
      has_image_flag_count   = 0
    
      for dir_path, dirs, files in os.walk( args.data_dir ):                                               
    
        if not (dir_path==args.data_dir):                                                                  # the top level directory (dataset) has be skipped because it only contains sub-directories, not data      
          
          for f in files:
           
            if (   ( f.endswith( 'HAS_IMAGE' ))   ):
              has_image_flag_count +=1
    
      if has_image_flag_count==0:
        print ( f"{BOLD}{RED}\n\nCLASSI:         FATAL:  although there are {MIKADO}{svs_file_count}{RESET}{BOLD}{RED} cases with images in the working dataset, none is currently flagged as having an image{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:  this suggests that the case division process, which is mandatory for image inputs ({CYAN}-i image{RESET}{BOLD}{RED}), has not been carried out{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:          remedy: re-run the experiment with option {CYAN}-v {RESET}{BOLD}{RED} set to {CYAN}True{RESET}{BOLD}{RED} to have cases divided and flagged{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:                  i.e. '{CYAN}./do_all.sh -d <cancer type code> -i image ... {CHARTREUSE}-v True{RESET}{BOLD}{RED}'{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:          further information: this only needs to be one time, following dataset regeneration{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:          further information: in general, it should not be done more than one time following dataset regeneration{RESET}" )
        print ( f"{BOLD}{RED}CLASSI:         FATAL:          further information: in particular, for multimode image+rna classification, NEVER perform case division more than one time, since each repetition would flag different subsets of the examples for hold-out testing, and these need to be strictly separated{RESET}" )
        print(  f"{BOLD}{RED}CLASSI:         FATAL: ... halting now{RESET}\n\n" )                           
        sys.exit(0)
    


    if use_unfiltered_data == True:
      args.rna_genes_tranche  = f"EVERY_GENE"
      rna_genes_tranche       = f"EVERY_GENE"
    else:
      rna_genes_tranche       = os.path.basename(target_genes_reference_file)    
      args.rna_genes_tranche  = os.path.basename(target_genes_reference_file)    
    

    mags = ("_".join(str(z) for z in zoom_out_mags))
    prob = ("_".join(str(z) for z in zoom_out_prob))


    if (input_mode=='image') & (nn_type_img=='INCEPT3') &  ( ( tile_size!=299 ) ):
      print( f"{RED}CLASSI:         FATAL:  for Inception 3 ('{CYAN}NN_TYPE_IMG={MIKADO}{nn_type_img}{RESET}{RED}' corresponding to python argument '{CYAN}--nn_type_img{RESET}{RED}') the only permitted tile size is {MIKADO}299{RESET}{RED}, however the tile size parameter ('{CYAN}TILE_SIZE{RESET}'{RED}) is currently {MIKADO}{tile_size}{RESET}{RED}", flush=True)
      print( f"{RED}CLASSI:         FATAL: ... halting now{RESET}" )
      sys.exit(0)


    if (input_mode=='image') & (nn_type_img[0:3]=='VGG') &  ( float(int( (tile_size/32) ) ) !=  (tile_size/32)  ):
      print( f"{RED}CLASSI:         FATAL:  for network type '{CYAN}VGGNN{RESET}'{RED}, tile size (currently '{MIKADO}{tile_size}{RESET}{RED}') must be a multiple of 32{RESET}", flush=True )
      print( f"{RED}CLASSI:         FATAL:  examples of acceptable tile sizes include: {MIKADO}32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600{RESET}{RED}",  flush=True )
      print( f"{RED}CLASSI:         FATAL:    ... cannot continue, halting now{RESET}" )
      sys.exit(0)

    if (input_mode=='image') & (nn_type_img[0:5]=='INCEPT') & (tile_size<75):
      print( f"{RED}CLASSI:         FATAL:  for network type '{CYAN}INCEPT3{RESET}'{RED} and '{CYAN}INCEPT4{RESET}'{RED}, tile size (currently '{MIKADO}{tile_size}{RESET}{RED}') must be greater than or equal to 75  (i.e. 75x75){RESET}", flush=True )
      print( f"{RED}CLASSI:         FATAL:    ... cannot continue, halting now{RESET}" )
      sys.exit(0)

    balanced     ="BALANCED"
    not_balanced ="NOTBLNCD"
    if input_mode=='image':
      descriptor = f"_{run+1:02d}_OF_{total_runs_in_job:03d}_{args.dataset.upper()}_{input_mode.lower():_<9s}_{balanced if make_balanced=='True' else not_balanced}_{args.cases[0:20]:_<20s}_{nn_type_img:_<15s}_{stain_norm:_<4s}_{nn_optimizer:_<8s}_e_{args.n_epochs:03d}_N_{n_samples:04d}\
_hi_{n_classes:02d}_bat_{batch_size:03d}_test_{int(100*pct_test):03d}_lr_{lr:09.6f}_tiles_{n_tiles:04d}_tlsz_{tile_size:04d}__mag_{mags}__prob_{prob:_<20s}"
      descriptor = descriptor[0:200]

      descriptor_2 = f"Cancer type={args.cancer_type_long}   Cancer Classes={highest_class_number+1:d}   Autoencoder={nn_type_img}   Training Epochs={args.n_epochs:d}  Tiles/Slide={n_tiles:d}   Tile size={tile_size}x{tile_size}\n\
Magnif'n vector={mags}   Stain Norm={stain_norm}   Peer Noise Pct={peer_noise_pct}   Grey Scale Pct={make_grey_pct}   Batch Size={batch_size:d}   Held Out={int(100*pct_test):03d}%   Learning Rate={lr:<09.6f}   Selected from cases subset: {args.cases[0:50]}"

      descriptor_clustering = f'{args.dataset.upper()}_HighClass_{highest_class_number:d}_Encoder_{nn_type_img}_e_{args.n_epochs:d}_tiles_{n_tiles:d}_tsz_{tile_size:d}x{tile_size:d}_\
Mags_{mags}_Stain_Norm_{stain_norm}_Peer_Noise_{peer_noise_pct}_Grey_Pct_{make_grey_pct}_Batch_Size{batch_size:03d}_Pct_Test_{int(100*pct_test):03d}_lr_{lr:<9.6f}_N_{n_samples:d}_Cases_{args.cases[0:50]}'


    else:
      topology_length = len( f"{hidden_layer_encoder_topology}" )
      if DEBUG>999:
        print ( f"-------------------------------------------------------------------------------------------------------------> {hidden_layer_encoder_topology}"          ) 
        print ( f"-------------------------------------------------------------------------------------------------------------> {topology_length}"                        ) 
    
      topology_as_whitespace_free_string = '-'.join(map(str, hidden_layer_encoder_topology))
     
      descriptor = f"_{run+1:02d}_OF_{total_runs_in_job:03d}_{args.dataset.upper()}_{input_mode.lower():_<9s}_{args.cases[0:10]:_<10s}__{rna_genes_tranche[0:10].upper():_<10s}__{nn_type_rna:_<15s}_{nn_optimizer[0:8]:_<8s}\
_e_{args.n_epochs:03d}_N_{n_samples:04d}_hi_{n_classes:02d}_bat_{batch_size:03d}_test_{int(100*pct_test):03d}_lr_{lr:09.6f}_hid_{hidden_layer_neurons:04d}_lo_{low_expression_threshold:<02.2e}_low_{cutoff_percentile:04.0f}\
_dr_{100*dropout_1:4.1f}_xfrm_{gene_data_transform:_<8s}_shape_{topology_as_whitespace_free_string}"                                                                                                # need to abbreviate everything because the long topology string will make the file name too long and it will crash
      if topology_length > 14:
        descriptor = descriptor[0:200-topology_length]

      descriptor_2 = f"Cancer type={args.cancer_type_long}   Cancer Classes={n_classes:d}   Autoencoder={nn_type_img}   Training Epochs={args.n_epochs:d}\n\
Batch Size={batch_size:d}   Held Out={int(100*pct_test):d}%   Learning Rate={lr:<9.6f}   Cases from subset: {args.cases[0:50]} Genes subset: {rna_genes_tranche}"

      descriptor_clustering = f"_{args.dataset.upper()}_{input_mode.upper():_<9s}_{args.cases[0:10]:_<10s}__{rna_genes_tranche[0:10].upper():_<10s}__{nn_type_rna:_<9s}_{nn_optimizer[0:8]:_<8s}\
_e_{args.n_epochs:03d}_N_{n_samples:04d}_hicls_{n_classes:02d}_bat_{batch_size:03d}_test_{int(100*pct_test):03d}_lr_{lr:09.6f}_hid_{hidden_layer_neurons:04d}"


    # ~ if just_test=='True':
        # ~ print( f"{ORANGE}CLASSI:         INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set, so n_samples (currently {MIKADO}{n_samples}{RESET}{ORANGE}) has been set to {MIKADO}1{RESET}{ORANGE} for this run{RESET}" ) 
        # ~ n_samples = int(pct_test * n_samples )


    now  = datetime.datetime.now()
    pplog.set_logfiles( log_dir, descriptor, now )
    pplog.log_section(f"run = {now:%y-%m-%d %H:%M}   parameters = {descriptor}")
    zoom_out_mags_string = " ".join([str(i) for i in np.around( np.array(zoom_out_mags), 3)])
    zoom_out_prob_string = " ".join([str(i) for i in zoom_out_prob])
    bash_command = f"cls; ./do_all.sh -d {args.dataset}  -i {input_mode}   -S {n_samples}  -A {highest_class_number}  -f {n_tiles}   -T {tile_size}  -b {batch_size}  -o {n_epochs}  -1 {pct_test}  -a {nn_type_img}  -c {args.cases}   -0 {stain_norm}  -U '{zoom_out_mags_string}' -Q '{zoom_out_prob_string}'  "
    print ( f"\033[79;0H{bash_command}" )
    time.sleep(1)

    pplog.log_section(f"{bash_command}" )
    pplog.log_section(f"      zoom_out_mags = {np.around(np.array(zoom_out_mags),3)}")
    pplog.log_section(f"      zoom_out_mags = {np.around(np.array(zoom_out_prob),3)}")

    pplog.log_section(f"      run args      = {sys.argv}")
    

    run+=1

    # accumulator
    if just_test!='True':
      aggregate_tile_probabilities_matrix =  np.zeros     ( ( n_samples, n_classes ),     dtype=float               )
      aggregate_tile_level_winners_matrix =  np.full_like ( aggregate_tile_probabilities_matrix, 0                  )
      patches_true_classes                        =  np.zeros     ( ( n_samples            ),     dtype=int         )
      patches_case_id                             =  np.zeros     ( ( n_samples            ),     dtype=int         )    
      
      probabilities_matrix                        =  np.zeros     ( ( n_samples, n_classes ),     dtype=float       )              # same, but for rna        
      true_classes                                =  np.zeros     ( ( n_samples            ),     dtype=int         )              # same, but for rna 
      rna_case_id                                 =  np.zeros     ( ( n_samples            ),     dtype=int         )              # same, but for rna 
    else:
      aggregate_tile_probabilities_matrix =  np.zeros     ( ( n_samples, n_classes ),     dtype=float               )
      aggregate_tile_level_winners_matrix =  np.full_like ( aggregate_tile_probabilities_matrix, 0                  )
      patches_true_classes                        =  np.zeros     ( ( n_samples            ),     dtype=int         )
      patches_case_id                             =  np.zeros     ( ( n_samples            ),     dtype=int         )    
      
      probabilities_matrix                        =  np.zeros     ( ( n_samples, n_classes ),     dtype=float       )              # same, but for rna        
      true_classes                                =  np.zeros     ( ( n_samples            ),     dtype=int         )              # same, but for rna 
      rna_case_id                                 =  np.zeros     ( ( n_samples            ),     dtype=int         )              # same, but for rna 

      if DEBUG>9:
        print ( f"\n\n" )
        print ( f"CLASSI:         INFO:      test: n_samples                        = {PALE_GREEN}{n_samples}{RESET}"                     )
        print ( f"CLASSI:         INFO:      test: n_classes                        = {PALE_GREEN}{n_classes}{RESET}"                     )
        print ( f"CLASSI:         INFO:      test: probabilities_matrix.shape       = {PALE_GREEN}{probabilities_matrix.shape}{RESET}"    )                                    


    if DEBUG>0:
      if input_mode=='image':
        if run !=1:
          print( f"\033[12B")                                                                              # cursor to bottom of screen
        print( f"\n\n{UNDER}{BOLD}RUN: {run} of {total_runs_in_job}{RESET}")
        print( f"\033[2C{image_headings}{RESET}") 
        print( f"{BITTER_SWEET}\
\r\033[2C\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{pct_test:<9.6f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{n_tiles:<5d}\
\r\033[{start_column+5*offset}C{n_classes:<2d}\
\r\033[{start_column+6*offset}C{tile_size}x{tile_size}\
\r\033[{start_column+7*offset}C{rand_tiles:<5s}\
\r\033[{start_column+8*offset}C{nn_type_img:<10s}\
\r\033[{start_column+9*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+10*offset}C{stain_norm:<10s}\
\r\033[{start_column+11*offset}C{label_swap_pct:<6.1f}\
\r\033[{start_column+12*offset}C{make_grey_pct:<5.1f}\
\r\033[{start_column+13*offset}C{jitter:}\
{RESET}" )  

      elif input_mode=='rna':
        if run !=1: 
          print( f"\033[12B")                                                                              # cursor to bottom of screen
        print( f"\n\n{UNDER}{BOLD}RUN: {run} of {total_runs_in_job}{RESET}")
        print(f"\033[2C\{rna_headings}{RESET}")
        print( f"{CARRIBEAN_GREEN}\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{pct_test:<9.2f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{nn_type_rna:<10s}\
\r\033[{start_column+5*offset}C{hidden_layer_neurons:<5d}\
\r\033[{start_column+6*offset}C{low_expression_threshold:<02.2e}\
\r\033[{start_column+7*offset}C{cutoff_percentile:<4.0f}\
\r\033[{start_column+8*offset}C{embedding_dimensions:<5d}\
\r\033[{start_column+9*offset}C{dropout_1:<5.2f}\
\r\033[{start_column+10*offset}C{dropout_2:<5.2f}\
\r\033[{start_column+11*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+12*offset}C{gene_data_norm:<10s}\
\r\033[{start_column+13*offset+3}C{gene_data_transform:<10s}\
\r\033[{start_column+14*offset+3}C{label_swap_pct:<6.1f}\
{RESET}" ) 

      if DEBUG>0:
        print ("")
    
    
    final_test_batch_size =   int(n_samples * n_tiles * pct_test)
    
    if DEBUG>99:
      print( f"CLASSI:         INFO:          requested FINAL_TEST_BATCH_SIZE = {MIKADO}{int(args.final_test_batch_size)}{RESET}" )      
      print( f"CLASSI:         INFO:          N_SAMPLES (notional)            = {MIKADO}{n_samples}{RESET}" )
      print( f"CLASSI:         INFO:          N_TILES (per sample)            = {MIKADO}{n_tiles}{RESET}" )
      print( f"CLASSI:         INFO:          PCT_TEST                        = {MIKADO}{pct_test}{RESET}" )
      print( f"CLASSI:         INFO:          hence available test tiles      = {MIKADO}{int(final_test_batch_size)}{RESET}" )
    if args.final_test_batch_size > final_test_batch_size:
      args.final_test_batch_size = final_test_batch_size
      if (input_mode=='image'):
        print ( f"{ORANGE}CLASSI:         WARNG: there aren't enough test tiles to support a {CYAN}FINAL_TEST_BATCH_SIZE{RESET}{ORANGE} of {MIKADO}{args.final_test_batch_size}{RESET}{ORANGE} for this run{RESET}", flush=True )                
        print ( f"{ORANGE}CLASSI:         WARNG: the number of test tiles available is {CYAN}N_SAMPLES{RESET} x {CYAN}N_TILES{RESET} x {CYAN}PCT_TEST{RESET}  = {MIKADO}{n_samples}{RESET} x {MIKADO}{n_tiles}{RESET} x {MIKADO}{pct_test}{RESET} = {MIKADO}{int(final_test_batch_size)}{RESET}{ORANGE}{RESET}", flush=True )                
        print ( f"{ORANGE}CLASSI:         WARNG: {CYAN}FINAL_TEST_BATCH_SIZE{RESET}{ORANGE} has accordingly been set to {MIKADO}{int(final_test_batch_size)}{RESET} {ORANGE}for this run {RESET}", flush=True )
      else:
        print ( f"{ORANGE}CLASSI:         WARNG: there aren't enough examples to support a {CYAN}FINAL_TEST_BATCH_SIZE{RESET}{ORANGE} of {MIKADO}{args.final_test_batch_size}{RESET}{ORANGE} for this run{RESET}", flush=True )                
        print ( f"{ORANGE}CLASSI:         WARNG: {CYAN}FINAL_TEST_BATCH_SIZE{RESET}{ORANGE} has accordingly been set to {MIKADO}{int(final_test_batch_size)}{RESET} {ORANGE}for this run {RESET}", flush=True )

    # (1) set up Tensorboard
    
    if DEBUG>1:    
      print( "CLASSI:         INFO: \033[1m1 about to set up Tensorboard\033[m" )
    
    writer = SummaryWriter(comment=f'_{randint(100, 999)}_{descriptor}' )


    #print ( f"\033[36B",  flush=True )
    if DEBUG>1:    
      print( "CLASSI:         INFO:   \033[3mTensorboard has been set up\033[m" )





    # (2) Maybe schedule and run tiler threads

    # ~ if (input_mode=='image') & (multimode!='image_rna'):
    if (input_mode=='image'):
      
      if skip_tiling=='False':
                  
        # need to re-tile if certain parameters have eiher INCREASED ('n_tiles' or 'n_samples') or simply CHANGED ( 'stain_norm' or 'tile_size') since the last run
        if ( ( already_tiled==True ) & ( ( stain_norm==last_stain_norm ) | (last_stain_norm=="NULL") ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) ):
          if DEBUG>0:
            print( f"CLASSI:         INFO: {BOLD}!! no need to perform tiling again: existing tiles are of the correct size and there are sufficient of them for the next run configured in this job{RESET}" )
          pass                                                                                             # no need to re-tile 
                                                                       
        else:                                                                                              # must re-tile
          if (total_runs_in_job==1) | (run==1):                                                                  # if this is the first or only run in the job, always regenerate
            print( f"CLASSI:         INFO: {BOLD}1  first or only run in job - will perform tiling{RESET}" )  
            pass
          else:
            print( f"\033[79;0HCLASSI:         INFO: {BOLD}2  will re-tile, for the following reason(s):{RESET}" )            
            if n_tiles>n_tiles_last:
              print( f"CLASSI:         INFO:           -- value of n_tiles    ({MIKADO}{n_tiles}{RESET})        \r\033[60Chas increased since last run (was {MIKADO}{n_tiles_last}){RESET}" )
            if ( (n_samples!=0) & (n_samples>n_samples_last)):
              print( f"CLASSI:         INFO:           -- value of n_samples  ({MIKADO}{n_samples_last}{RESET}) \r\033[60Chas increased since last run (was {MIKADO}{n_samples_last}){RESET}")
            if tile_size_last!=tile_size:
              print( f"CLASSI:         INFO:           -- value of tile_size  ({MIKADO}{tile_size}{RESET})      \r\033[60Chas changed   since last run (was {MIKADO}{tile_size_last}){RESET}")
            if stain_norm!=last_stain_norm:
              print( f"CLASSI:         INFO:           -- value of stain_norm ({MIKADO}{stain_norm}{RESET})     \r\033[60Chas changed   since last run (was {MIKADO}{last_stain_norm}){RESET}")

          delete_selected( data_dir, "png" )
          last_stain_norm=stain_norm
          already_tiled=True

          if DEBUG>999:
            print( f"CLASSI:           INFO:   n_samples_max                   = {MIKADO}{n_samples_max}{RESET}")
            print( f"CLASSI:           INFO:   n_tiles_max                     = {MIKADO}{n_tiles_max}{RESET}")
  
          if stain_norm!="reinhard":                                                                           # we are NOT going to stain normalize ...
          # ~ if stain_norm=="NONE":                                                                       
            norm_method='NONE'
          # ~ else:                                                                                        # we ARE going to stain normalize ...
          elif stain_norm=="reinhard":                                                                     # <---------------------------------------------- 'spcn' is now handled by the standalone process 'normalise_stain' 
            if DEBUG>0:
              print( f"CLASSI:           INFO: {BOLD}about to set up stain normalization target{RESET}" )
            if stain_norm_target.endswith(".svs"):                                                       # ... then grab the user provided target
              norm_method = tiler_set_target( args, n_tiles, tile_size, stain_norm, stain_norm_target, writer )
            else:                                                                                        # ... and there MUST be a target
              print( f"CLASSI:         FATAL:    for {MIKADO}{stain_norm}{RESET} an SVS file must be provided from which the stain normalization target will be extracted" )
              sys.exit(0)

          print ( f"{SAVE_CURSOR}" )
            
          if just_test=='True':

              try:
                fqn = f"{args.data_dir}/SUFFICIENT_SLIDES_TILED"
                os.remove( fqn )
              except:
                pass

              if (  args.cases == 'UNIMODE_CASE' ):
                
                flag  = 'UNIMODE_CASE____IMAGE_TEST'
                total_slides_counted_test, total_tiles_required_test, top_up_factors_test  = determine_top_up_factors ( args, n_classes, class_names, n_tiles, flag )

                if DEBUG>1:
                  print( f"{SAVE_CURSOR}\r\033[{num_cpus}B{WHITE}CLASSI:         INFO: about to call tiler_threader with flag = {BLEU}{flag}{RESET}; total_slides_counted_test = {MIKADO}{total_slides_counted_test:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles = {MIKADO}{n_tiles}{RESET}{RESTORE_CURSOR}", flush=True )

                slides_tiled_count = tiler_threader( args, flag, total_slides_counted_test, n_samples, n_tiles, top_up_factors_test, tile_size, batch_size, stain_norm, norm_method )

              if (  args.cases == 'MULTIMODE____TEST' ):
                
                flag  = 'MULTIMODE____TEST'
                total_slides_counted_test, total_tiles_required_test, top_up_factors_test  = determine_top_up_factors ( args, n_classes, class_names, n_tiles, flag )
                count = cases_reserved_for_image_rna

                if DEBUG>1:
                  print( f"{SAVE_CURSOR}\r\033[{num_cpus}B{WHITE}CLASSI:         INFO: about to call tiler_threader with flag = {BLEU}{flag}{RESET}; count = {MIKADO}{count:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles = {MIKADO}{n_tiles}{RESET}{RESTORE_CURSOR}", flush=True )

                slides_tiled_count = tiler_threader( args, flag, total_slides_counted_test, n_samples, n_tiles, top_up_factors_test, tile_size, batch_size, stain_norm, norm_method )

              if (  args.cases == 'ALL_ELIGIBLE_CASES' ):
                
                slides_to_be_tiled = n_samples
  
                flag  = 'HAS_IMAGE'
                total_slides_counted_test, total_tiles_required_test, top_up_factors_test  = determine_top_up_factors ( args, n_classes, class_names, n_tiles, flag )
              
                if DEBUG>1:
                  print( f"{SAVE_CURSOR}\r\033[{num_cpus+1}B{WHITE}CLASSI:         INFO: about to call tiler_threader with flag = {BLEU}{flag}{RESET}; slides_to_be_tiled = {MIKADO}{slides_to_be_tiled:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles_max = {MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )

                slides_tiled_count = tiler_threader( args, flag, slides_to_be_tiled, n_samples, n_tiles_max, top_up_factors_test, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job


          else:

            if (  args.cases == 'ALL_ELIGIBLE_CASES' ):
              
              slides_to_be_tiled = n_samples

              try:
                fqn = f"{args.data_dir}/SUFFICIENT_SLIDES_TILED"
                os.remove( fqn )
              except:
                pass

              flag  = 'HAS_IMAGE'
              total_slides_counted_train, total_tiles_required_train, top_up_factors_train  = determine_top_up_factors ( args, n_classes, class_names, n_tiles, flag )
            
              if DEBUG>1:
                print( f"{SAVE_CURSOR}\r\033[{num_cpus+1}B{WHITE}CLASSI:         INFO: about to call tiler_threader with flag = {CYAN}{flag}{RESET}; slides_to_be_tiled = {MIKADO}{slides_to_be_tiled:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles_max = {MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )
              slides_tiled_count = tiler_threader( args, flag, slides_to_be_tiled, n_samples, n_tiles_max, top_up_factors_train, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job

              
            if (  args.cases == 'UNIMODE_CASE' ):

              try:
                fqn = f"{args.data_dir}/SUFFICIENT_SLIDES_TILED"
                os.remove( fqn )
              except:
                pass

              flag  = 'UNIMODE_CASE____IMAGE'
              total_slides_counted_train, total_tiles_required_train, top_up_factors_train  = determine_top_up_factors ( args, n_classes, class_names, n_tiles, flag )

              if DEBUG>1:
                np.set_printoptions(formatter={'float': lambda x: "{:>.2f}".format(x)})
                print( f"\r{WHITE}CLASSI:         INFO: about to call {MAGENTA}tiler_threader{RESET}: flag={CYAN}{flag}{RESET}; train_count={MIKADO}{total_slides_counted_train:3d}{RESET}; top_up_factors_train  = {MIKADO}{top_up_factors_train}{RESET}; %_test={MIKADO}{pct_test:2.2f}{RESET}; n_samples={MIKADO}{n_samples_max:3d}{RESET}; n_tiles={MIKADO}{n_tiles_max}{RESET}", flush=True )

              slides_tiled_count = tiler_threader( args, flag, total_slides_counted_train, n_samples, n_tiles_max, top_up_factors_train, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job

              flag  = 'UNIMODE_CASE____IMAGE_TEST'
              total_slides_counted_test, total_tiles_required_test, top_up_factors_test  = determine_top_up_factors ( args, n_classes, class_names, n_tiles, flag )

              if DEBUG>1:
                np.set_printoptions(formatter={'float': lambda x: "{:>.2f}".format(x)})
                print( f"\r{WHITE}CLASSI:         INFO: about to call {MAGENTA}tiler_threader{RESET}: flag={BLEU}{flag}{RESET}; test_count={PINK}{total_slides_counted_test:3d}{RESET}; top_up_factors_test  = {PINK}{top_up_factors_test}{RESET}; %_test={PINK}{pct_test:2.2f}{RESET}; n_samples={PINK}{n_samples_max:3d}{RESET}; n_tiles={PINK}{n_tiles_max}{RESET}", flush=True )
              slides_tiled_count = tiler_threader( args, flag, total_slides_counted_test, n_samples, n_tiles_max, top_up_factors_train, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job
              

          if just_profile=='True':                                                                         # then we are all done
            sys.exit(0)



    # (3) Maybe Regenerate Torch '.pt' file

    if  (input_mode=='image') & ( skip_generation!='True' ):
      
      if ( ( already_tiled==True ) & (n_tiles==n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) & ( stain_norm==last_stain_norm ) ):    # all have to be true, or else we must regenerate the .pt file
        if DEBUG>0:
          print( f"CLASSI:         INFO: {BOLD}!! no need to re-generate the pytorch dataset. the existing dataset contains sufficient tiles of the correct size{RESET}" )
        pass

      else:
        if (total_runs_in_job==1) | (run==1):                                                                  # if this is the first or only run in the job, always regenerate
          print( f"CLASSI:         INFO: {BOLD}2  first or only run in job  - will generate torch '.pt' file from files{RESET}" )  
          pass
        else:
          print( f"CLASSI:         INFO: {BOLD}2  will regenerate torch '.pt' file from files, for the following reason(s):{RESET}" )            
          if n_tiles>n_tiles_last:
            print( f"CLASSI:         INFO:           -- value of n_tiles    ({MIKADO}{n_tiles}{RESET})        \r\033[60Chas increased since last run (was {MIKADO}{n_tiles_last}){RESET}" )
          if ( (n_samples!=0) & (n_samples>n_samples_last)):
            print( f"CLASSI:         INFO:           -- value of n_samples  ({MIKADO}{n_samples_last}{RESET}) \r\033[60Chas increased since last run (was {MIKADO}{n_samples_last}){RESET}")
          if not tile_size_last==tile_size:
            print( f"CLASSI:         INFO:           -- value of tile_size  ({MIKADO}{tile_size}{RESET})      \r\033[60Chas changed   since last run (was {MIKADO}{tile_size_last}){RESET}")
          if not stain_norm==last_stain_norm:
            print( f"CLASSI:         INFO:           -- value of stain_norm ({MIKADO}{stain_norm}{RESET})     \r\033[60Chas changed   since last run (was {MIKADO}{last_stain_norm}){RESET}")

        if DEBUG>8:
          print( f"CLASSI:         INFO: n_samples               = {MAGENTA}{n_samples}{RESET}",        flush=True  )
          print( f"CLASSI:         INFO: args.n_samples          = {MAGENTA}{args.n_samples}{RESET}",   flush=True  )
          print( f"CLASSI:         INFO: batch_size              = {MAGENTA}{batch_size}{RESET}",       flush=True  )
          print( f"CLASSI:         INFO: args.batch_size         = {MAGENTA}{args.batch_size}{RESET}",  flush=True  )
          print( f"CLASSI:         INFO: n_classes               = {MAGENTA}{n_classes}{RESET}",        flush=True  )
          print( f"CLASSI:         INFO: n_tiles                 = {MAGENTA}{n_tiles}{RESET}",          flush=True  )
          print( f"CLASSI:         INFO: args.n_tiles            = {MAGENTA}{args.n_tiles}{RESET}",     flush=True  )
          print( f"CLASSI:         INFO: n_genes                 = {MAGENTA}{n_genes}{RESET}",          flush=True  )
          print( f"CLASSI:         INFO: args.n_genes            = {MAGENTA}{args.n_genes}{RESET}",     flush=True  )
          print( f"CLASSI:         INFO: gene_data_norm          = {MAGENTA}{gene_data_norm}{RESET}",   flush=True  )            
  
        highest_class_number = n_classes-1
        _, _,  _ = generate( args, class_names, n_samples, total_slides_counted_train, total_slides_counted_test, total_tiles_required_train, total_tiles_required_test, batch_size, highest_class_number, multimode_case_count, unimode_case_matched_count, unimode_case_unmatched_count, 
                             unimode_case____image_count, unimode_case____image_test_count, unimode_case____rna_count, unimode_case____rna_test_count, pct_test, n_tiles, top_up_factors_train, top_up_factors_test, tile_size, 
                             low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform  
                           ) 
  
        if DEBUG>8:
          print( f"CLASSI:         INFO: n_samples               = {BLEU}{n_samples}{RESET}"       )
          print( f"CLASSI:         INFO: args.n_samples          = {BLEU}{args.n_samples}{RESET}"  )
          print( f"CLASSI:         INFO: batch_size              = {BLEU}{batch_size}{RESET}"      )
          print( f"CLASSI:         INFO: args.batch_size         = {BLEU}{args.batch_size}{RESET}" )
          print( f"CLASSI:         INFO: n_tiles                 = {BLEU}{n_tiles}{RESET}"         )
          print( f"CLASSI:         INFO: args.n_tiles            = {BLEU}{args.n_tiles}{RESET}"    )
          print( f"CLASSI:         INFO: n_genes                 = {BLEU}{n_genes}{RESET}"         )
          print( f"CLASSI:         INFO: args.n_genes            = {MAGENTA}{args.n_genes}{RESET}" )
          print( f"CLASSI:         INFO: gene_data_norm          = {BLEU}{gene_data_norm}{RESET}"  )            
          
        n_tiles_last   = n_tiles                                                                           # for the next run
        n_samples_last = n_samples                                                                         # for the next run
        tile_size_last = tile_size                                                                         # for the next run
  
        
      if ( input_mode=='rna' ) | ( input_mode=='image_rna' ) :
        
        top_up_factors = 0
        
        highest_class_number = n_classes-1
        
        n_genes, n_samples, batch_size = generate( args, class_names, n_samples, total_slides_counted_train, total_slides_counted_test, total_tiles_required_train, total_tiles_required_test, batch_size, highest_class_number, multimode_case_count, unimode_case_matched_count, unimode_case_unmatched_count, 
                                                    unimode_case____image_count, unimode_case____image_test_count, unimode_case____rna_count, unimode_case____rna_test_count, pct_test, n_tiles, top_up_factors_train, top_up_factors_test, tile_size, 
                                                    low_expression_threshold, cutoff_percentile, gene_data_norm, gene_data_transform  
                                                 )
  
        if DEBUG>0:
          print( f"CLASSI:         INFO:    n_genes/embed length (calculated)  = {MIKADO}{n_genes:,}{RESET}",     flush=True     )
          print( f"CLASSI:         INFO:    n_samples   (determined)           = {MIKADO}{n_samples:,}{RESET}",   flush=True     )
          print( f"CLASSI:         INFO:    batch_size  (determined)           = {MIKADO}{batch_size:,}{RESET}",  flush=True     )


    if input_mode=='image_rna':
      print( f"{BOLD}{CHARTREUSE}CLASSI:         INFO:   input = '{BOLD}{CYAN}{input_mode}{RESET}{BOLD}{CHARTREUSE}'. Concatentated image_rna embeddings will be generated.{RESET}"  )


    if clustering!='NONE':
       if args.input_mode == 'rna':
        print ( f"{BOLD}{ORANGE}CLASSI:         WARNG:  there are almost certainly not enough data points to do meaningful clustering on rna gene expression values{RESET}",   flush=True )
        print ( f"{BOLD}{ORANGE}CLASSI:         WARNG:  continuing, but don't be surprised if the clustering algorithm crashes{RESET}",                                        flush=True )
     
    if clustering=='o_tsne':
      o_tsne   ( args, class_names, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'CLASSI:          INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)

    elif clustering=='cuda_tsne':
      cuda_tsne(  args, class_names, pct_test, descriptor_2, descriptor_clustering)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'CLASSI:          INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)
      
    elif clustering=='sk_tsne':
      sk_tsne(  args, class_names, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'CLASSI:          INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)

    elif clustering=='sk_agglom':
      sk_agglom(  args, class_names, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'CLASSI:          INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)
      
    elif clustering=='sk_spectral':
      sk_spectral(  args, class_names, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'CLASSI:          INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)
      
    elif clustering=='dbscan':
      _dbscan ( args, class_names, pct_test, epsilon )
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'CLASSI:          INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)
      
    elif clustering=='h_dbscan':
      h_dbscan ( args, class_names, pct_test, min_cluster_size )
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'CLASSI:           INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)

    elif clustering!='NONE':
      print ( f"{RED}CLASSI:         FATAL:    there's no such clustering option as '{CYAN}{clustering}{RESET}'", flush=True)
      print ( f"{RED}CLASSI:         FATAL:    supported clustering algorithms are cuda-tsne ('{CYAN}cuda_tsne{RESET}{RED}'), scikit-learn tsne ('{CYAN}sk_tsne{RESET}{RED}'), agglomerative clustering ('{CYAN}sk_agglom{RESET}{RED}') and spectral clustering ('{CYAN}sk_spectral{RESET}{RED}'); , open tsne ('{CYAN}otsne{RESET}{RED}'), DBSCAN ('{CYAN}dbscan{RESET}{RED}'), HDBSCAN ('{CYAN}h_dbscan{RESET}{RED}'){RESET}", flush=True)
      print ( f"{RED}CLASSI:         FATAL:    halting now...{RESET}", flush=True)      
      sys.exit(0)

      

    # (4) Load experiment config.  (NOTE: Almost all configurable parameters are now provided via user arguments rather than this config file)
    
    if DEBUG>1:    
      print( f"CLASSI:         INFO: {BOLD}4 about to load experiment config{RESET}" )
    cfg = loader.get_config( mode, lr, batch_size )                                                        #################################################################### change to just using args at some point
    classifyConfig.MAKE_GREY          = make_grey_pct                                                      # modify config class variable to take into account user preference
    classifyConfig.JITTER             = jitter                                                             # modify config class variable to take into account user preference
#          if args.input_mode=='rna':  pplog.log_config(cfg) 

    # ~ pplog.log_section('Loading script arguments.')
    # ~ pplog.log_args(args)

    if DEBUG>1:      
      print( f"CLASSI:         INFO:   {ITALICS}experiment config has been loaded{RESET}" )
   



    # (5) Load neural network

    if DEBUG>100:                                                                                                        
      print( f"CLASSI:         INFO: {BOLD}5 about to load network {MIKADO}{nn_type_img}{RESET}{BOLD} or {MIKADO}{nn_type_rna}{RESET}" )
      
    model = COMMON( args, cfg, input_mode, nn_type_img, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, dropout_1, dropout_2, tile_size, args.latent_dim, args.em_iters  )

    if DEBUG>100: 
      print( f"CLASSI:         INFO:    {ITALICS}network loaded{RESET}" )



    # (6) maybe load existing models (two cases where this happens: (i) test mode and (ii) pretrain option selected )

    fqn_pretrained = f"{log_dir}/model_pretrained.pt"
    fqn_image      = f"{log_dir}/model_image.pt"
    fqn_rna        = f"{log_dir}/model_rna.pt"
    fqn_image_rna  = f"{log_dir}/model_image_rna.pt"
    
    if pretrain=='True':                                                                                   # then load the last pretrained (as defined) model

      try:
        model.load_state_dict(torch.load(fqn_pretrained))
        print( f"{ORANGE}CLASSI:         INFO:  pre-trained model named {CYAN}{fqn_pretrained}{RESET}{ORANGE} exists.  Will load and use pre-trained model{RESET}", flush=True)
      except Exception as e:
        print( f"{ORANGE}CLASSI:         INFO:  no pre-trained model named {CYAN}{fqn_pretrained}{RESET}{ORANGE} exists.  Will attempt to used model {CYAN}{fqn_image}{RESET}{ORANGE}, if it exists{RESET}", flush=True)
        try:
          model.load_state_dict(torch.load(fqn_image))
          print( f"{ORANGE}CLASSI:         INFO:  model named {CYAN}{fqn_image}{RESET}{ORANGE} exists.  Will load and use this network model as the starting point for training{RESET}", flush=True)
        except Exception as e:
          print( f"{RED}CLASSI:         INFO:  mo model named {CYAN}{fqn_image}{RESET}{RED} exists.  Cannot continue{RESET}", flush=True)
          time.sleep(4)
          sys.exit(0)

    elif just_test=='True':                                                                                  # then load the already trained model

      if args.input_mode == 'image':
        fqn = fqn_image
      elif args.input_mode == 'rna':
        fqn = fqn_rna
      elif args.input_mode == 'image_rna':
        fqn = fqn_image_rna

      if DEBUG>0:
        print( f"{ORANGE}CLASSI:         INFO:  'just_test' flag is set.  About to load model state dictionary {MAGENTA}{fqn}{RESET}" )
        
      try:
        model.load_state_dict(torch.load(fqn))       
      except Exception as e:
        print ( f"{RED}CLASSI:         FATAL:  error when trying to load model {MAGENTA}'{fqn}'{RESET}", flush=True)    
        if args.input_mode == 'image':
          print ( f"{RED}CLASSI:         FATAL:    explanation 1: this is a test run. ({CYAN}JUST_TEST==TRUE{RESET}{RED} (shell) or {CYAN}--just_test=='True'{RESET}{RED} (python user argument). Did you forget to train a model ?{RESET}", flush=True)
          print ( f"{RED}CLASSI:         FATAL:    explanation 2: perhaps you're using a different tile size ({CYAN}'TILE_SIZE'{RESET}{RED})than than the saved model uses{RESET}", flush=True)
        if args.input_mode == 'rna':
          print ( f"{RED}CLASSI:         FATAL:    explanation: this is a test run. ({CYAN}JUST_TEST==TRUE{RESET}{RED} (shell) or {CYAN}'just_test'=='True'{RESET}{RED} (python user argument). Did you forget to train a model ?{RESET}", flush=True)
        print ( f"{RED}CLASSI:         FATAL:    halting now...{RESET}", flush=True)      
        time.sleep(4)
        sys.exit(0)
                                            


    #(7) Send model to GPU(s)
    
    if DEBUG>1:    
      print( f"CLASSI:         INFO: {BOLD}6 about to send model to device{RESET}" )   
    model = model.to(device)
    if DEBUG>1:
      print( f"CLASSI:         INFO:     {ITALICS}model sent to device{RESET}" ) 
  
    #pplog.log_section('Model specs.')
    #pplog.log_model(model)
     
    
    if DEBUG>9:
      print( f"CLASSI:         INFO:   pytorch Model = {MIKADO}{model}{RESET}" )



    #(8) Fetch data loaders
    
    gpu        = 0
    world_size = 0
    rank       = 0
    

    if DEBUG>1: 
      print( f"CLASSI:         INFO: {BOLD}7 about to call dataset loader" )
    train_loader, test_loader, final_test_batch_size, final_test_loader = loader.get_data_loaders( args,
                                                         gpu,
                                                         cfg,
                                                         world_size,
                                                         rank,
                                                         batch_size,
                                                         n_samples,
                                                         args.n_workers,
                                                         args.pin_memory,                                                       
                                                         pct_test,
                                                         writer
                                                        )
    if DEBUG>1:
      print( "CLASSI:         INFO:   \033[3mdataset loaded\033[m" )
  
    #if just_test=='False':                                                                                # c.f. loader() Sequential'SequentialSampler' doesn't return indices
    #  pplog.save_test_indices(test_loader.sampler.indices)





    #(9) Select and configure optimizer


    if nn_optimizer=='ADAM':
      optimizer = optim.Adam       ( model.parameters(),  lr=lr,  weight_decay=0,  betas=(0.9, 0.999),  eps=1e-08,               amsgrad=False                                    )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAMAX':
      optimizer = optim.Adamax     ( model.parameters(),  lr=lr,  weight_decay=0,  betas=(0.9, 0.999),  eps=1e-08                                                                 )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mAdamax optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAMW':                                                                            # Decoupled Weight Decay Regularization (https://arxiv.org/abs/1711.05101)
      optimizer = optim.AdamW     ( model.parameters(),  lr=lr,  weight_decay=0.01,  betas=(0.9, 0.999),  eps=1e-08,   amsgrad=False                                              )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mAdamax optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAMW_AMSGRAD':                                                                            # Decoupled Weight Decay Regularization (https://arxiv.org/abs/1711.05101)
      optimizer = optim.AdamW     ( model.parameters(),  lr=lr,  weight_decay=0.01,  betas=(0.9, 0.999),  eps=1e-08,   amsgrad=True                                               )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mAdamax optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAGRAD':
      optimizer = optim.Adagrad    ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-10,               lr_decay=0, initial_accumulator_value=0          )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='SPARSEADAM':
      optimizer = optim.SparseAdam ( model.parameters(),  lr=lr,                   betas=(0.9, 0.999),  eps=1e-08                                                                 )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mSparseAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADADELTA':
      optimizer = optim.Adadelta   ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-06, rho=0.9                                                        )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mAdagrad optimizer selected and configured\033[m" )
    elif nn_optimizer=='ASGD':
      optimizer = optim.ASGD       ( model.parameters(),  lr=lr,  weight_decay=0,                                               alpha=0.75, lambd=0.0001, t0=1000000.0            )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mAveraged Stochastic Gradient Descent optimizer selected and configured\033[m" )
    elif   nn_optimizer=='RMSPROP':
      optimizer = optim.RMSprop    ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-08,  momentum=0,  alpha=0.99, centered=False                       )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mRMSProp optimizer selected and configured\033[m" )
    elif   nn_optimizer=='RPROP':
      optimizer = optim.Rprop      ( model.parameters(),  lr=lr,                                                                etas=(0.5, 1.2), step_sizes=(1e-06, 50)           )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mResilient backpropagation algorithm optimizer selected and configured\033[m" )
    elif nn_optimizer=='SGD':
      optimizer = optim.SGD        ( model.parameters(),  lr=lr,  weight_decay=0,                                   momentum=0.9, dampening=0, nesterov=True                      )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mStochastic Gradient Descent optimizer selected and configured\033[m" )
    elif nn_optimizer=='LBFGS':
      optimizer = optim.LBFGS      ( model.parameters(),  lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None  )
      if DEBUG>1:
        print( "CLASSI:         INFO:   \033[3mL-BFGS optimizer selected and configured\033[m" )
    else:
      print( f"{BOLD}{RED}CLASSI:         FATAL: optimizer '{MIKADO}{nn_optimizer}{RESET}{BOLD}{RED}' not supported", flush=True )
      print( f"{BOLD}{RED}CLASSI:         FATAL: can't continue ... halting now{RESET}" )
      sys.exit(0)
 
 
 
 
         
    # (10) Select Loss function
    
    loss_type=''
    
    if  ( input_mode!='image' ) & ( nn_type_rna in [ 'AELINEAR', 'AEDENSE', 'AEDENSEPOSITIVE', 'DCGANAE128' ] ):    # autoencoders
 
      loss_function = torch.nn.MSELoss()
      loss_type     = 'mean_squared_error'                                                                         
      # ~ ae_loss2 = functional.mse_loss( x2r, x2.squeeze())    

    elif  ( input_mode!='image' ) & ( nn_type_rna in [ 'TTVAE' ] ):

      if DEBUG>0:
        print ( f"CLASSI:         INFO:      train: x2 [0:12,0:12] = {MIKADO}{x2[0:12,0:12]}{RESET}" ) 
        print ( f"CLASSI:         INFO:      train: x2r[0:12,0:12] = {MIKADO}{x2r[0:12,0:12]}{RESET}" )

      bce_loss       = False
      loss_reduction = 'sum'
      loss_function = torch.nn.BCELoss() if bce_loss else torch.nn.MSELoss()  
      # ~ loss_function = BCELoss( reduction=loss_reduction ).cuda(gpu) if bce_loss else MSELoss( reduction=loss_reduction ).cuda(gpu)      # Have to use Binary cross entropy loss for TTVAE (and VAEs generally)
      ae_loss2, reconstruction_loss, kl_loss = vae_loss( x2r, x2, mean, logvar, loss_fn, epoch, kl_warm_up=0, beta=1. )
      del mean
      del logvar      
      
    elif ( input_mode=='image')  & ( nn_type_img in [ 'AE3LAYERCONV2D', 'AEDCECCAE_3', 'AEDCECCAE_5', 'AEVGG16' ] ):

      loss_function = torch.nn.MSELoss()
      loss_type     = 'mean_squared_error'                                                                                                                                                 
      # ~ ae_loss2 = functional.mse_loss( x2r, x2.squeeze())
      
    else:
      
      loss_function = torch.nn.CrossEntropyLoss()
      
    if DEBUG>0:
      print( f"CLASSI:         INFO:   loss function = {CYAN}{loss_function}{RESET}" )  
        
        
        
        
#    show,  via Tensorboard, what the samples look like
#    images, labels = next(iter(train_loader))                                                              # PGD 200129 -
#    images = images.to(device)
#    labels = labels.to (device)
  
#    show,  via Tensorboard, what the samples look like
#    grid = torchvision.utils.make_grid( images, nrow=16 )                                                  # PGD 200129 - 
#    writer.add_image('images', grid, 0)                                                                    # PGD 200129 - 
#    writer.add_graph(model, images)                                                                        # PGD 200129 -  
  
    
    #pplog.log_section('Training model.\n\n'\
    #                   'Epoch\t\tTrain x1 err\tTrain x2 err\tTrain l1\t'\
    #                   '\tTest x1 err\tTest x2 err\tTest l1')
   
   
   
   
   
   
    # (11) Train/Test
    
    if DEBUG>1:
      print( f"CLASSI:         INFO:  {BOLD}about to commence main loop, one iteration per epoch{RESET}",  end='' )

    global_correct_prediction_count = 0
    global_number_tested            = 0
    max_correct_predictions         = 0
    max_percent_correct             = 0
    
    test_loss_min           = 999999
    train_loss_min          = 999999  
   
    consecutive_training_loss_increases    = 0
    consecutive_test_loss_increases        = 0
    

    last_epoch_loss_increased              = True

    train_total_loss_sum_ave_last                 = 99999                                                  # used to determine whether total loss is increasing or decreasing
    train_lowest_total_loss_observed_so_far       = 99999                                                  # keeps ongiing track of the lowest total training loss ...
    train_lowest_total_loss_observed_so_far_epoch = 0                                                      # ... and the epoch it occurred at

    test_total_loss_sum_ave_last                  = 99999
    test_lowest_total_loss_observed_so_far        = 99999
    test_lowest_total_loss_observed_so_far_epoch  = 0


    # (12) Prep for embeddings accumulation (Autoencoder only) 
      
    if (just_test=='True') & (we_are_autoencoding==True):
      
      if DEBUG>10:
        print( f"CLASSI:         INFO:        batch_size     =  {MAGENTA}{batch_size}{RESET}",                       flush=True )
        print( f"CLASSI:         INFO:        embedding_dimensions =  {MAGENTA}{embedding_dimensions}{RESET}",       flush=True )
                                                                                            
      embeddings_accum = torch.zeros( [ 0, embedding_dimensions ],  requires_grad=False,  dtype=torch.float )
      
      if DEBUG>10:
        print( f"CLASSI:         INFO:        embeddings_accum.size  =  {MAGENTA}{embeddings_accum.size() }{RESET}",  flush=True )
                  
      labels_accum     = torch.zeros( [ 0                       ],  requires_grad=False,  dtype=torch.int64 ) 

    else:
      embeddings_accum = 0
      labels_accum     = 0



    # (13) Main loop

    for epoch in range(1, n_epochs+1):
  
        if   args.input_mode=='image':
          print( f'\nCLASSI:         INFO:  {CARRIBEAN_GREEN}(RUN {run} of {total_runs_in_job}){RESET} in epoch {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}  input:{CYAN}{input_mode}{RESET}  network:{BOLD}{CYAN}{nn_type_img}{RESET}  stain norm:{MAGENTA if stain_norm=="spcn" else CYAN}{stain_norm}{RESET}  lr:{MIKADO}{lr:<9.6f}{RESET}  samples:{MIKADO}{n_samples}{RESET}  batch size:{MIKADO}{batch_size}{RESET}  tile size:{MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide:{MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {BOLD_MAGENTA}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          print( f'\nCLASSI:         INFO:  {CARRIBEAN_GREEN}(RUN {run} of {total_runs_in_job}){RESET} in epoch {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}  input:{MIKADO}{input_mode}{RESET} lr:{MIKADO}{lr:<9.6f}{RESET} samples:{MIKADO}{n_samples}{RESET} batch size:{MIKADO}{batch_size}{RESET} hidden layer neurons:{MIKADO}{hidden_layer_neurons}{RESET} embedded dimensions:{MIKADO}{batch_size if args.use_autoencoder_output==True  else "N/A" }{RESET}.  {DULL_WHITE}will halt if test loss increases for {BOLD_MAGENTA}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )
        else:
          print( f'\nCLASSI:         INFO:  {CARRIBEAN_GREEN}(RUN {run} of {total_runs_in_job}){RESET} in epoch {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}  input:{MIKADO}{input_mode}{RESET} lr:{MIKADO}{lr:<9.6f}{RESET} samples:{MIKADO}{n_samples}{RESET} batch size:{MIKADO}{batch_size}{RESET} tile size:{MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide:{MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {BOLD_MAGENTA}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )

    
        if just_test=='True':                                                                              # skip training in 'test mode'
          pass
        
        # DO TRAINING (AS MANY BATCHES AS ARE NECESSARY TO WORK THROUGH EVERY EXAMPLE)
        
        else:
    
          train_loss_images_sum_ave, train_loss_genes_sum_ave, train_l1_loss_sum_ave, train_total_loss_sum_ave =\
                                                                                                       train ( args, epoch, train_loader, model, optimizer, loss_function, loss_type, writer, train_loss_min, batch_size )
    
          if train_total_loss_sum_ave < train_lowest_total_loss_observed_so_far:
            train_lowest_total_loss_observed_so_far       = train_total_loss_sum_ave
            train_lowest_total_loss_observed_so_far_epoch = epoch
    
          if ( (train_total_loss_sum_ave < train_total_loss_sum_ave_last) | (epoch==1) ):
            consecutive_training_loss_increases = 0
            last_epoch_loss_increased = False
          else:
            last_epoch_loss_increased = True

          if DEBUG>0:
            if ( input_mode=='image' ):
              print ( f"\
  \r\033[1C{CLEAR_LINE}{DULL_WHITE}\
  \r\033[27Ctrain:\
  \r\033[49Craw loss_images={train_loss_images_sum_ave:5.2f}\
  \r\033[120CBATCH AVE LOSS OVER EPOCH (LOSS PER 1000 TILES) = {PALE_GREEN if last_epoch_loss_increased==False else PALE_RED}{train_total_loss_sum_ave*1000/batch_size:6.1f}{DULL_WHITE}\
  \r\033[250C{BLACK if epoch<2 else WHITE}min loss: {train_lowest_total_loss_observed_so_far*100/batch_size:>6.2f} at epoch {train_lowest_total_loss_observed_so_far_epoch+1:<2d}"
  , end=''  )
            elif ( input_mode=='rna' ):
              print ( f"\
  \r\033[1C{CLEAR_LINE}{DULL_WHITE}\
  \r\033[27Ctrain:\
  \r\033[73Craw loss_rna={train_loss_genes_sum_ave:5.2f}\
  \r\033[120CBATCH AVE LOSS OVER EPOCH (LOSS PER 1000 EXAMPLES) = {PALE_GREEN if last_epoch_loss_increased==False else PALE_RED}{train_total_loss_sum_ave*1000/batch_size:6.1f}{DULL_WHITE}\
  \r\033[250C{BLACK if epoch<2 else WHITE}min loss: {train_lowest_total_loss_observed_so_far*100/batch_size:>6.2f} at epoch {train_lowest_total_loss_observed_so_far_epoch+1:<2d}"
  , end=''  )
  
  
            if last_epoch_loss_increased == True:
              consecutive_training_loss_increases +=1
              if consecutive_training_loss_increases == 1:
                print ( f"\r\033[280C{DARK_RED} < train loss increased{RESET}", end='' )
              else:
                print ( f"\r\033[280C{DARK_RED} < {consecutive_training_loss_increases} {DARK_RED}consec increases !{RESET}", end='' )
              print ( "" )
    
            if (last_epoch_loss_increased == False):
              print ('')
    
          train_total_loss_sum_ave_last = train_total_loss_sum_ave
  

#        if (just_test=='True') & (multimode=='image_rna'):                                                # skip testing in Test mode if multimode is True 
        if (just_test=='True') & (multimode=='image_rnaxxx'):                                              # skip testing in Test mode if multimode is True 
          pass  # <---- This will never happen
            
            
            
        # DO TESTING
        else:  
    
          show_all_test_examples=False
          
          embeddings_accum, labels_accum, test_loss_images_sum_ave, test_loss_genes_sum_ave, test_l1_loss_sum_ave, test_total_loss_sum_ave, correct_predictions, number_tested, max_correct_predictions, max_percent_correct, test_loss_min, embedding     =\
                        test ( cfg, args, parameters, embeddings_accum, labels_accum, epoch, test_loader,  model,  tile_size, loss_function, loss_type, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                      test_loss_min, show_all_test_examples, batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours)
  
          global_correct_prediction_count += correct_predictions
          global_number_tested            += number_tested
          
          if DEBUG>99:
            print( f"CLASSI:           INFO:   global_correct_prediction_count   = {MIKADO}{global_correct_prediction_count:>}{RESET}")        
            print( f"CLASSI:           INFO:   global_number_tested              = {MIKADO}{global_number_tested}{RESET:>}")
            print( f"CLASSI:           INFO:   global_percent_correct            = {MIKADO}{global_correct_prediction_count/global_number_tested*100:<3.0f}%{RESET}")                    
          
          if ( (test_total_loss_sum_ave <= ( test_total_loss_sum_ave_last )) | (epoch==1) ):               # if this epoch had a lower average per batch ...
            consecutive_test_loss_increases = 0
            last_epoch_loss_increased = False
          else:
            last_epoch_loss_increased = True
            
          if ( input_mode=='image' ):
            print ( f"\
  \033[5A\
  \r\033[1C\033[2K{DULL_WHITE}\
  \r\033[27Ctest:\
  \r\033[49Craw loss_images={CARRIBEAN_GREEN}{test_loss_images_sum_ave:5.2f}{DULL_WHITE}\
  \r\033[120CBATCH AVE LOSS OVER EPOCH (LOSS PER 1000 TILES) = {GREEN if last_epoch_loss_increased==False else RED}{test_total_loss_sum_ave*1000/batch_size:6.1f}{DULL_WHITE}\
  \r\033[250C{BLACK if epoch<2 else WHITE}min loss: {test_lowest_total_loss_observed_so_far*100/batch_size:6.2f} at epoch {test_lowest_total_loss_observed_so_far_epoch+1:<2d}{DULL_WHITE}\
  \033[5B\
  ", end=''  )
          elif ( input_mode=='rna' ):
            print ( f"\
  \033[5A\
  \r\033[1C\033[2K{DULL_WHITE}\
  \r\033[27Ctest:\
  \r\033[73Craw loss_rna={BITTER_SWEET}{test_loss_genes_sum_ave:5.2f}{DULL_WHITE}\
  \r\033[120CBATCH AVE LOSS OVER EPOCH (LOSS PER 100 EXAMPLES) = {GREEN if last_epoch_loss_increased==False else RED}{test_total_loss_sum_ave*100/batch_size:6.2f}{DULL_WHITE}\
  \r\033[250C{BLACK if epoch<2 else WHITE}min loss: {test_lowest_total_loss_observed_so_far*100/batch_size:6.2f} at epoch {test_lowest_total_loss_observed_so_far_epoch+1:<2d}{DULL_WHITE} \
  \033[5B\
  ", end=''  )
  

          if last_epoch_loss_increased == True:
            consecutive_test_loss_increases +=1
            if consecutive_test_loss_increases == 1:
              print ( "\033[5A", end='' )
              print ( f"\r\033[280C{PALE_RED} < test loss increased{RESET}", end='' )
              print ( "\033[5B", end=''  )
            else:
              print ( "\033[5A", end='' )
              print ( f"\r\033[280C{RED} < {consecutive_test_loss_increases} consec increases !{RESET}", end='' )
              print ( "\033[5B", end=''  )
              
            if consecutive_test_loss_increases>args.max_consecutive_losses:                                # Stop one before, so that the most recent model for which the loss improved will be saved
                now = time.localtime(time.time())
                print(time.strftime("CLASSI:         INFO: %Y-%m-%d %H:%M:%S %Z", now))
                break
          else:
            print ( "\033[5A", end='' )
            print ( f"\r\033[280C{PALE_GREEN} < test loss decreased{RESET}", end='' )
            print ( "\033[5B", end=''  )
          
        
  
          test_total_loss_sum_ave_last = test_total_loss_sum_ave
          
          if test_total_loss_sum_ave < test_lowest_total_loss_observed_so_far:
            test_lowest_total_loss_observed_so_far       = test_total_loss_sum_ave
            test_lowest_total_loss_observed_so_far_epoch = epoch
            if DEBUG>0:
              print ( "\033[5A", end='' )
              print ( f"\r\033[280C\033[0K{BRIGHT_GREEN} < new global low/saving model{RESET}", end='' )
              print ( "\033[5B", end='' )
            
            if just_test=='False':
              save_model(args.log_dir, model) 

          if args.input_mode=='rna':
            print ( "\033[8A", end='' )
          else:
            print ( "\033[8A", end='' )       


    #  ^^^^^^^^  THE MAIN LOOP FINISHES HERE ^^^^^^^^





    # (A)  MAYBE SAVE THE AUTOENCODER GENERATED EMBEDDINGS (IF THIS WAS AN AUTOENCODER TEST RUN)

    if loss_type=='mean_squared_error':

      if args.just_test=='True':
        
        #  save embeddings to file
        
        fqn = f"{args.log_dir}/ae_output_features.pt"
  
        if DEBUG>0:
          print( f"\n\n\n\n{CLEAR_LINE}{BOLD}{CHARTREUSE}CLASSI:         INFO:        about to save autoencoder generated embeddings and labels as torch dictionary to {MAGENTA}{fqn}{RESET}" )
          print( f"{CLEAR_LINE}{BOLD}{CHARTREUSE}CLASSI:         INFO:        embeddings_accum              .size     = {CARRIBEAN_GREEN}{embeddings_accum.size() }{RESET}", flush=True )        
          print( f"{CLEAR_LINE}{BOLD}{CHARTREUSE}CLASSI:         INFO:        labels_accum                  .size     = {CARRIBEAN_GREEN}{labels_accum.size() }{RESET}",     flush=True )    
         
        if args.input_mode=='image':
            
          try:          
            torch.save({
                'embeddings': embeddings_accum,
                'labels':     labels_accum,
            }, fqn )
          except Exception as e:
            print( f"{RED}CLASSI:         FATAL:  couldn't save embeddings to file '{MAGENTA}{fqn}{RESET}{RED}'{RESET}" )  
            time.sleep(5)          
        
        if args.input_mode=='rna':            
          torch.save({
              'embeddings': embeddings_accum,
              'labels':     labels_accum,
          }, fqn )
          
        if DEBUG>0:
          print( f"CLASSI:         INFO:        embeddings have been saved" )          


      # This is all we have to do in the case of Autoencoding, so we close up and end
  
      writer.close()        
    
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      #pplog.log_section('Job complete in {:} mins'.format( minutes ) )
      
      print( f'\n\n\n\nCLASSI:          INFO: Job complete {BOLD}{CHARTREUSE}(Autoencoder ending).{RESET} The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      now = time.localtime(time.time())
      print(time.strftime( f"CLASSI:          INFO:  end time = %Y-%m-%d %H:%M:%S %Z", now ))
      start_time = time.time() 
                
      #pplog.log_section('Model saved.')
      
      sys.exit()

  
    # (B)  MAYBE CLASSIFY FINAL_TEST_BATCH_SIZE TEST SAMPLES USING THE BEST MODEL SAVED DURING THIS RUN
  
    if final_test_batch_size>0:
    
      if ( ( args.just_test!='True') &  (args.input_mode!='image_rna') )   |   ( (args.just_test=='True')  &  (args.input_mode=='image_rna') & (args.multimode=='image_rna')      ):
      #       ------------------ unimode training -------------------              ------------------------------------ multimode testing ------------------------------------                                                       
      
        if DEBUG>0:
          print ( "\033[8B" )        
          print ( f"CLASSI:         INFO:      test: {BOLD}about to classify {MIKADO}{final_test_batch_size}{RESET}{BOLD} test samples through the best model this run produced"        )
        
        pplog.log ( f"\nCLASSI:         INFO:  test: about to classify {final_test_batch_size} test samples through the best model this run produced"                                 )


        if args.input_mode == 'image':
          fqn = '%s/model_image.pt'     % log_dir
        elif args.input_mode == 'rna':
          fqn = '%s/model_rna.pt'       % log_dir
        elif args.input_mode == 'image_rna':
          fqn = '%s/model_image_rna.pt' % log_dir
    
          if DEBUG>0:
            print( f"CLASSI:         INFO:  about to load model state dictionary for best model (from {MIKADO}{fqn}{RESET})" )
  
          try:
            model.load_state_dict(torch.load(fqn))
            model = model.to(device)
          except Exception as e:
            print ( f"{RED}GENERATE:             FATAL: error when trying to load model {MAGENTA}'{fqn}'{RESET}", flush=True)    
            print ( f"{RED}GENERATE:                    reported error was: '{e}'{RESET}", flush=True)
            print ( f"{RED}GENERATE:                    halting now{RESET}", flush=True)      
            time.sleep(2)
            pass
    
        show_all_test_examples=True
        
        if DEBUG>1:
          print ( f"CLASSI:         INFO:      test: final_test_batch_size = {MIKADO}{final_test_batch_size}{RESET}" )
          
        # note that we pass 'final_test_loader' to test
        embeddings_accum, labels_accum, test_loss_images_sum_ave, test_loss_genes_sum_ave, test_l1_loss_sum_ave, test_total_loss_sum_ave, correct_predictions, number_tested, max_correct_predictions, max_percent_correct, test_loss_min, embedding     =\
                          test ( cfg, args, parameters, embeddings_accum, labels_accum, epoch, final_test_loader,  model,  tile_size, loss_function, loss_type, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                                           test_loss_min, show_all_test_examples, final_test_batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours )    
    
      job_level_classifications_matrix               += run_level_classifications_matrix                     # accumulate for the job level stats. Has to be just after call to 'test'    




    # (C)  MAYBE CREATE AND SAVE EMBEDDINGS FOR ALL TEST SAMPLES (IN TEST MODE, SO THE OPTIMUM MODEL HAS ALREADY BEEN LOADED AT STEP 5 ABOVE)
    
    if (just_test=='True') & (multimode=="image_rna"):

      if DEBUG>0:
        print( f"{BOLD}{CHARTREUSE}\r\033[7BCLASSI:         INFO:      test: about to generate and save embeddings for all test samples{RESET}", flush=True )

      model.eval()                                                                                         # set model to evaluation mode

      embedding_count = 0
        
      for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in  enumerate( test_loader ):
          
        batch_images = batch_images.to(device)
        batch_genes  = batch_genes.to (device)
        image_labels = image_labels.to(device)

        if DEBUG>2:
          print( f"\nCLASSI:         INFO:      test: embeddings: embedding_count         = {MIKADO}{embedding_count+1}{RESET}",              flush=True )
          print( f"CLASSI:         INFO:      test: embeddings: batch count             = {MIKADO}{i+1}{RESET}",                        flush=True )
          if args.input_mode=='image': 
            print( f"CLASSI:         INFO:      test: embeddings: batch_images size       = {BLEU}{batch_images.size()}{RESET}                                                     {MAGENTA}<<<<< Note: don't use dropout in test runs{RESET}", flush=True)
          if ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
            print( f"CLASSI:         INFO:      test: embeddings: batch_genes size        = {BLEU}{batch_genes.size()}{RESET}                                                     {MAGENTA}<<<<< Note: don't use dropout in test runs{RESET}", flush=True)
          print( f"CLASSI:         INFO:      test: embeddings: batch_fnames size       = {BLEU}{batch_fnames.size()}{RESET}",          flush=True)
        if DEBUG>888:
          print( f"CLASSI:         INFO:      test: embeddings: batch_fnames            = {PURPLE}{batch_fnames.cpu().numpy()}{RESET}", flush=True )

        gpu                = 0                                                                             # not currently used
        encoder_activation = args.encoder_activation
        
        if args.input_mode=='image':
          with torch.no_grad(): 
            y1_hat, y2_hat, embedding = model.forward( [ batch_images, 0            , batch_fnames] , gpu, args  )          # y1_hat = image outputs
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          with torch.no_grad(): 
            y1_hat, y2_hat, embedding = model.forward( [ 0,            batch_genes  , batch_fnames], gpu, args )            # y2_hat = rna outputs
                # ~ x2r, mean, logvar = model.forward( args, x2, args.input_mode,                    gpu, args )

        if DEBUG>2:
          print( f"CLASSI:         INFO:      test: embeddings: returned embedding size = {ARYLIDE}{embedding.size()}{RESET}",          flush=True )
  
        batch_fnames_npy = batch_fnames.numpy()                                                            # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details

        if DEBUG>2:
          np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
          print ( f"CLASSI:         INFO:      test: embeddings: batch_fnames_npy.shape  = {batch_fnames_npy.shape}",       flush=True )        
          print ( f"CLASSI:         INFO:      test: embeddings: batch_fnames_npy        = {batch_fnames_npy}",             flush=True )


        if DEBUG>2:
          fq_link       = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"                                    # convert the saved integer to the matching file name
          print( f"CLASSI:         INFO:      test: e.g. batch_fnames_npy[0]                 = {MAGENTA}{fq_link}{RESET}",      flush=True )
                  
  
        # batch will only sometimes be complete
        
        # save each embedding in its associated case directory using a randomly generated name
        if just_test=='True':                                                                              #  in test mode we are pushing inputs through the optimised model, which was saved during training mode

          for n in range( 0, batch_fnames_npy.shape[0] ):                                                    
  
            if batch_fnames_npy[n] == 0:
              break
            if args.input_mode=='image': 
              fq_link       = f"{args.data_dir}/{batch_fnames_npy[n]}.fqln"                                # where to save the embedding (which case directory to save it to)
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"CLASSI:         INFO:      test: embeddings:   batch_fnames_npy[{MIKADO}{n}{RESET}]   = {PINK}{batch_fnames_npy[n]}{RESET}",              flush=True )
                print ( f"CLASSI:         INFO:      test: embeddings:   fq_link                = {PINK}{fq_link}{RESET}",                                          flush=True )
              save_path     =  os.path.dirname(os.readlink(fq_link))
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"CLASSI:         INFO:      test: embeddings:   save_path              = {PINK}{save_path}{RESET}",                                        flush=True )
              random_name   = f"_{randint(10000000, 99999999)}_image_rna_matched___image"
              save_fqn      = f"{save_path}/{random_name}"
              if DEBUG>8:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"CLASSI:         INFO:      test: embeddings:   save_fqn               = {PINK}{save_fqn}{RESET}",                                         flush=True )
              np.save( save_fqn, embedding.cpu().numpy()[n] )
  
            if ( args.input_mode=='rna' ):
              fq_link       = f"{args.data_dir}/{batch_fnames_npy[n]}.fqln"
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"CLASSI:         INFO:      test: embeddings:   batch_fnames_npy[{MIKADO}{n}{RESET}]   = {PINK}{batch_fnames_npy[n]}{RESET}",              flush=True )
                print ( f"CLASSI:         INFO:      test: embeddings:   fq_link                = {BLEU}{fq_link}{RESET}",                                          flush=True )
              save_path     =   os.readlink(fq_link)                                                       # link is to the case directory for rna_seq (for tiles, it's to the patch file within the case directory)
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"CLASSI:         INFO:      test: embeddings:   save_path              = {BLEU}{save_path}{RESET}",                                        flush=True )
              random_name   = f"_image_rna_matched___rna"
              save_fqn      = f"{save_path}/{random_name}"
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"CLASSI:         INFO:      test: embeddings:   save_fqn               = {BLEU}{save_fqn}{RESET}",                                         flush=True )
              np.save( save_fqn, embedding.cpu().numpy()[n] )

            
        
          if DEBUG>88:
            np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
            print ( f"CLASSI:         INFO:      test: embeddings: embedding [{MIKADO}{n},0:10{RESET}]     = {PINK}{embedding.cpu().numpy()[n,0:10]}{RESET}",  flush=True )
            print ( f"CLASSI:         INFO:      test: embeddings: fq_link [{MIKADO}{n}{RESET}]            = {PINK}{fq_link}{RESET}",                          flush=True )
            print ( f"CLASSI:         INFO:      test: embeddings: random name [{MIKADO}{n}{RESET}]        = {PINK}{ranndom_name}{RESET}",                     flush=True )
           #print ( f"CLASSI:         INFO:      test: embeddings: points to                               = {PINK}{os.readlink(fq_link)}{RESET}",             flush=True )
            print ( f"CLASSI:         INFO:      test: embeddings: save path                               = {BLEU}{save_path}{RESET}",                        flush=True )
            print ( f"CLASSI:         INFO:      test: embeddings: save fqn                                = {BLEU}{save_fqn}{RESET}",                         flush=True )
    
          embedding_count+=1


    if args.input_mode=='rna':
      print ( "\033[8A", end='' )
    # ~ else:
      # ~ print ( "\033[8A", end='' )  



    # (D)  ALWAYS DISPLAY & SAVE BAR CHARTS

    if (just_test=='True') & (multimode!="image_rna"):                                                     # don't currently produce bar-charts for embedded outputs ('image_rna')


      # case image:
        
      if input_mode=='image':
        
        pd.set_option('display.max_columns',  300)
        pd.set_option('display.max_colwidth', 300)      
        pd.set_option('display.width',       2000)
        
        if DEBUG>1:
          print ( f"\nCLASSI:         INFO:      patches_true_classes                                        = {BITTER_SWEET}{patches_true_classes}{RESET}", flush=True )
          print ( f"CLASSI:         INFO:      patches_case_id                                             = {BITTER_SWEET}{patches_case_id}{RESET}",     flush=True )        

        if args.cases=='MULTIMODE____TEST':
          upper_bound_of_indices_to_plot_image = cases_reserved_for_image_rna
        else:  # correct for UNIMODE_CASE
          upper_bound_of_indices_to_plot_image = n_samples


        # case image- 1: PREDICTED - AGGREGATE probabilities
        
        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:      aggregate_tile_probabilities_matrix                 = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix}{RESET}", flush=True )

        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:      class_names                 = \n{CHARTREUSE}{class_names}{RESET}", flush=True )
          
  
        figure_width  = 20
        figure_height = 10
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
        ax.set_title ( args.cancer_type_long )
        plt.xticks( rotation=90 )
        plt.ylim  ( 0, n_tiles  )     
        #sns.set_theme(style="whitegrid")
        pd_aggregate_tile_probabilities_matrix                    = pd.DataFrame( aggregate_tile_probabilities_matrix )   [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix.columns            = class_names
        pd_aggregate_tile_probabilities_matrix[ 'agg_prob'     ]  = np.sum(aggregate_tile_probabilities_matrix,   axis=1 )[0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ]  = pd_aggregate_tile_probabilities_matrix.max   (axis=1) [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix[ 'pred_class'   ]  = pd_aggregate_tile_probabilities_matrix.idxmax(axis=1) [0:upper_bound_of_indices_to_plot_image]  # grab class (which is the column index with the highest value in each row) and save as a new column vector at the end, to using for coloring 
        pd_aggregate_tile_probabilities_matrix[ 'true_class'   ]  = patches_true_classes                                  [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix[ 'n_classes'    ]  = len(class_names) 
        pd_aggregate_tile_probabilities_matrix[ 'case_id'      ]  = patches_case_id                                       [0:upper_bound_of_indices_to_plot_image]
        # ~ pd_aggregate_tile_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )
        #fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"
        

        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>3d}"})
          print ( f"\nCLASSI:         INFO:      upper_bound_of_indices_to_plot_image                              = {CHARTREUSE}{upper_bound_of_indices_to_plot_image}{RESET}",     flush=True      ) 
          print ( f"\nCLASSI:         INFO:      pd_aggregate_tile_probabilities_matrix[ 'case_id' ]         = \n{CHARTREUSE}{pd_aggregate_tile_probabilities_matrix[ 'case_id' ]}{RESET}",     flush=True      ) 
          print ( f"\nCLASSI:         INFO:      pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ]    = \n{CHARTREUSE}{pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ]}{RESET}",     flush=True )            
  
        if bar_chart_x_labels=='case_id':
          c_id = pd_aggregate_tile_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_aggregate_tile_probabilities_matrix.shape[0])]

        if DEBUG>1:
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:       (extended) pd_aggregate_tile_probabilities_matrix = \n{CHARTREUSE}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True )
          # ~ print ( f"\nCLASSI:         INFO:       (extended) aggregate_tile_probabilities_matrix    = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix}{RESET}", flush=True )
       
        if DEBUG>88:          
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:                                             aggregate_tile_probabilities_matrix = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix}{RESET}", flush=True )
        if DEBUG>88:          
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:          aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image]  = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image]}{RESET}", flush=True )
          print ( f"\nCLASSI:         INFO: np.argmax(aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image] = \n{CHARTREUSE}{np.argmax(aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image], axis=1)}{RESET}", flush=True )
          
        x_labels = [  str(el) for el in c_id ]
        cols     = [ class_colors[el] for el in np.argmax(aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image], axis=1)  ]
                  
        if DEBUG>88:
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:                                                     cols = \n{CHARTREUSE}{cols}{RESET}", flush=True )
          print ( f"\nCLASSI:         INFO:                                                len(cols) = \n{CHARTREUSE}{len(cols)}{RESET}", flush=True )
          
        # ~ if DEBUG>0:
          # ~ print ( f"\nCLASSI:         INFO:      cols                = {MIKADO}{cols}{RESET}", flush=True )        
        
        p1 = plt.bar( x=x_labels, height=pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ], color=cols ) 
              
        # ~ ax = sns.barplot( x=c_id,  y=pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ], hue=pd_aggregate_tile_probabilities_matrix['pred_class'], palette=class_colors, dodge=False )                  # in pandas, 'index' means row index
        ax.set_title   ("Score of Predicted Subtype (sum of tile-level probabilities)",  fontsize=16 )
        ax.set_xlabel  ("Case (Patch)",                                                  fontsize=14 )
        ax.set_ylabel  ("Aggregate Probabilities",                                       fontsize=14 )
        ax.tick_params (axis='x', labelsize=12,  labelcolor='black')
        # ~ ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
        # ~ plt.legend( class_names, loc=2, prop={'size': 14} )
        
        # ~ patch0 = mpatches.Patch(color=cols[1], label=class_names[0])
        # ~ patch1 = mpatches.Patch(color=cols[2], label=class_names[1])
        # ~ patch2 = mpatches.Patch(color=cols[3], label=class_names[2])
        # ~ patch3 = mpatches.Patch(color=cols[4], label=class_names[3])
        # ~ patch4 = mpatches.Patch(color=cols[5], label=class_names[4])
        # ~ patch5 = mpatches.Patch(color=cols[0], label=class_names[5])
        
        # ~ plt.legend( handles=[patch0, patch1, patch2, patch3, patch4, patch5 ], loc=2, prop={'size': 14} )
                
        correct_count = 0
        i=0
        for p in ax.patches:
          #ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',  fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
          if not np.isnan(p.get_height()):                                                                   # if it's a number, then it will be a height (y value)
            for index, row in pd_aggregate_tile_probabilities_matrix.iterrows():
              if DEBUG>888:
                print ( f"CLASSI:         INFO:      row['max_agg_prob']                       = {CHARTREUSE}{row['max_agg_prob']}{RESET}", flush=True )            
                print ( f"CLASSI:         INFO:      p.get_height()                            = {CHARTREUSE}{p.get_height()}{RESET}", flush=True )
                print ( f"CLASSI:         INFO:      patches_true_classes[{MIKADO}{i}{RESET}]  = {CHARTREUSE}{patches_true_classes[i]}{RESET}", flush=True ) 
              if row['max_agg_prob'] == p.get_height():                                                      # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>888:
                    print ( f"CLASSI:         INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}class_names[row['true_class']]  = {RESET}{MIKADO}{class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}correct_count                        = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=14, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  correct_count+=1
            i+=1 
  
        if DEBUG>1:
          print ( f"\nCLASSI:         INFO:      number correct (pd_aggregate_tile_probabilities_matrix) = {CHARTREUSE}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        if args.bar_chart_show_all=='True':
          writer.add_figure('images___aggregate_tile_level_probabs_matrix', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
        
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_bar_chart_images___agg_tile_level_raw____probs"
        fqn = f"{fqn[0:255]}.png"
        fig.savefig(fqn)
        
        
            

        # case image-2: PREDICTED - WINNER TAKE ALL probabilities
        
        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:      aggregate_tile_level_winners_matrix                = \n{AMETHYST}{aggregate_tile_level_winners_matrix}{RESET}", flush=True )
  
        figure_width  = 20
        figure_height = 10
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
        ax.set_title ( args.cancer_type_long )
        
        plt.xticks( rotation=90 )
        plt.ylim  ( 0, n_tiles  )     
        #sns.set_theme(style="whitegrid")
        pd_aggregate_tile_level_winners_matrix                      = pd.DataFrame( aggregate_tile_level_winners_matrix )    [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_level_winners_matrix.columns              = class_names
        pd_aggregate_tile_level_winners_matrix[ 'max_tile_count' ]  = pd_aggregate_tile_level_winners_matrix.max   (axis=1)  [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_level_winners_matrix[ 'pred_class']       = pd_aggregate_tile_level_winners_matrix.idxmax(axis=1)  [0:upper_bound_of_indices_to_plot_image]  # grab class (which is the column index with the highest value in each row) and save as a new column vector at the end, to using for coloring 
        pd_aggregate_tile_level_winners_matrix[ 'true_class' ]      = patches_true_classes                                   [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_level_winners_matrix[ 'case_id' ]         = patches_case_id                                        [0:upper_bound_of_indices_to_plot_image]
        # ~ pd_aggregate_tile_level_winners_matrix.sort_values( by='max_tile_count', ascending=False, ignore_index=True, inplace=True )
        #fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"

        if bar_chart_x_labels=='case_id':
          c_id = pd_aggregate_tile_level_winners_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_aggregate_tile_probabilities_matrix.shape[0])]

        if DEBUG>1:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:       (extended) pd_aggregate_tile_level_winners_matrix  = \n{BLEU}{pd_aggregate_tile_level_winners_matrix}{RESET}", flush=True )  
          

        if DEBUG>88:
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:                                             aggregate_tile_level_winners_matrix = \n{AMETHYST}{aggregate_tile_level_winners_matrix}{RESET}", flush=True )
          print ( f"\nCLASSI:         INFO:          aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image]  = \n{AMETHYST}{aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image]}{RESET}", flush=True )
          print ( f"\nCLASSI:         INFO: np.argmax(aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image] = \n{AMETHYST}{np.argmax(aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image], axis=1)}{RESET}", flush=True )
          
        x_labels = [  str(el) for el in c_id ]
        cols     = [ class_colors[el] for el in np.argmax(aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image], axis=1)  ]
                
        p1 = plt.bar( x=x_labels, height=pd_aggregate_tile_level_winners_matrix[ 'max_tile_count' ], color=cols  )   
        
        # ~ ax = sns.barplot( x=c_id, y=pd_aggregate_tile_level_winners_matrix[ 'max_tile_count' ], hue=pd_aggregate_tile_level_winners_matrix['pred_class'], palette=class_colors, dodge=False )                  # in pandas, 'index' means ROW index
        #ax.tick_params(axis='x', bottom='on', which='major',  color='lightgrey', labelsize=9,  labelcolor='lightgrey', width=1, length=6, direction = 'out')
        ax.set_title  ("Score of Predicted Subtype ('tile-winner-take-all' scoring)",  fontsize=16 )
        ax.set_xlabel ("Case (Patch)",                                              fontsize=14 )
        ax.set_ylabel ("Number of Winning Tiles",                                       fontsize=14 )
        ax.tick_params(axis='x', labelsize=12,  labelcolor='black')
        ax.tick_params(axis='y', labelsize=14,  labelcolor='black') 
        # ~ plt.legend( class_names,loc=2, prop={'size': 14} )
                
        correct_count=0
        i=0
        for p in ax.patches:
          #ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',  fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
          if not np.isnan(p.get_height()):
            for index, row in pd_aggregate_tile_level_winners_matrix.iterrows():
              if DEBUG>888:
                print ( f"CLASSI:         INFO:      row['max_tile_count']                     = {MIKADO}{row['max_tile_count']}{RESET}", flush=True )            
                print ( f"CLASSI:         INFO:      p.get_height()                            = {MIKADO}{p.get_height()}{RESET}", flush=True )
                print ( f"CLASSI:         INFO:      patches_true_classes[{MIKADO}{i}{RESET}]  = {MIKADO}{patches_true_classes[i]}{RESET}", flush=True ) 
              if row['max_tile_count'] == p.get_height():                                                    # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>888 :
                    print ( f"CLASSI:         INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}class_names[row['true_class']]  = {RESET}{MIKADO}{class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}correct_count   max_tilmax                     = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=14, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  correct_count+=1
            i+=1 
  
        if DEBUG>88:
          print ( f"\nCLASSI:         INFO:      number correct (pd_aggregate_tile_level_winners_matrix) = {MIKADO}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
        
        plt.tight_layout()
        
        if args.bar_chart_show_all=='True':        
          writer.add_figure('images___aggregate_tile_level_winners_matrix', fig, 0 )

        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_bar_chart_images___agg_tile_level_winner_probs"
        fqn = f"{fqn[0:255]}.png"
        fig.savefig(fqn)
        
        


        # Case image-3: probabilities assigned to TRUE classes 

        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        true_class_prob = aggregate_tile_probabilities_matrix[ range(0, patches_true_classes.shape[0]), patches_true_classes ]   # 'patches_true_classes' was established during test run
        pred_class_idx  = np.argmax ( aggregate_tile_probabilities_matrix, axis=1   )
        correct_count   = np.sum    (    patches_true_classes == pred_class_idx     )

        pd_aggregate_tile_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx [0:upper_bound_of_indices_to_plot_image]   # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c MULTIMODE____TEST')
        pd_aggregate_tile_probabilities_matrix[ 'true_class_prob' ]  = true_class_prob[0:upper_bound_of_indices_to_plot_image]   # same
        # ~ pd_aggregate_tile_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )

        if DEBUG>1:
          print ( f"\nCLASSI:         INFO:      pd_aggregate_tile_probabilities_matrix {CYAN}image{RESET} = \n{COTTON_CANDY}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True )    
                
                
        if bar_chart_x_labels=='case_id':                                                                  # user wants case ids as labels
          c_id = pd_aggregate_tile_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_aggregate_tile_probabilities_matrix.shape[0])]
          
        for i in range ( 0, aggregate_tile_probabilities_matrix.shape[0] ):
          agg_prob = pd_aggregate_tile_probabilities_matrix[ 'agg_prob'][i]
          arg_max  = np.argmax( aggregate_tile_probabilities_matrix[i,:] )
          if DEBUG>88:
            print ( f"CLASSI:         INFO:      i                                                                       = {COTTON_CANDY}{i}{RESET}", flush=True ) 
            print ( f"CLASSI:         INFO:      str(c_id[i])                                                            = {COTTON_CANDY}{str(c_id[i])}{RESET}", flush=True ) 
            print ( f"CLASSI:         INFO:      arg_max                                                                 = {COTTON_CANDY}{arg_max}{RESET}", flush=True ) 
            print ( f"CLASSI:         INFO:      class_names[ arg_max ]                                                  = {COTTON_CANDY}{class_names[ arg_max ]}{RESET}", flush=True ) 
            print ( f"CLASSI:         INFO:      height = [ aggregate_tile_probabilities_matrix[i,arg_max] / agg_prob ]  = {COTTON_CANDY}{[ aggregate_tile_probabilities_matrix[i,arg_max] / agg_prob ]}{RESET}", flush=True ) 
          plt.bar( x=[ str(c_id[i]) ],   height=[ aggregate_tile_probabilities_matrix[i,arg_max] / agg_prob ],  color=class_colors[ arg_max ], label=class_names[ arg_max ] )  # just plots the maximum value


        plt.title   ("Input Data = Slide Image Tiles;  Bar Height = Probability Assigned to **TRUE** Cancer Sub-type",            fontsize=16 )
        plt.xlabel  ("Case ID",                                                     fontsize=14 )
        plt.ylabel  ("Probability Assigned by Network",                             fontsize=14 )
        plt.ylim    (0.0, 1.0)
        plt.tick_params (axis='x', labelsize=8,   labelcolor='black')
        plt.tick_params (axis='y', labelsize=14,  labelcolor='black')
        plt.xticks  ( rotation=90 )

        plt.legend( class_names, loc=2, prop={'size': 14} )
            
        pct_correct = correct_count/ n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        writer.add_figure('images___probs_assigned_to_TRUE_classes', fig, 0 )
        
        # save version to logs directory
        now = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_bar_chart_images___probs_assigned_to_TRUE_classes"
        fqn = f"{fqn[0:255]}.png"
        
        fig.savefig(fqn)
          



        # Case image-4:  graph aggregate probabilities for ALL classses

        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        if DEBUG>55:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:       pd_aggregate_tile_probabilities_matrix = \n{BLEU}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True ) 

        true_class_prob = aggregate_tile_probabilities_matrix[ range(0, patches_true_classes.shape[0]), patches_true_classes ]
        pred_class_idx  = np.argmax( aggregate_tile_probabilities_matrix, axis=1   )
        correct_count   = np.sum( patches_true_classes == pred_class_idx )

        if DEBUG>88:
          print ( f"\033[16B" )
          print ( f"\nCLASSI:         INFO:      patches_case_id                                = \n{ASPARAGUS}{patches_case_id}{RESET}",                              flush=True )  
          print ( f"\nCLASSI:         INFO:      pd_aggregate_tile_probabilities_matrix.shape   = {ASPARAGUS}{pd_aggregate_tile_probabilities_matrix.shape}{RESET}",  flush=True )                
          print ( f"\nCLASSI:         INFO:      true_class_prob                                = \n{ASPARAGUS}{true_class_prob}{RESET}",                               flush=True )
          print ( f"\nCLASSI:         INFO:      pred_class_idx                                 = \n{ASPARAGUS}{pred_class_idx}{RESET}",                               flush=True )
          print ( f"\nCLASSI:         INFO:      patches_true_classes                           = \n{ASPARAGUS}{patches_true_classes}{RESET}",                                 flush=True )
  
        plt.xticks( rotation=90 )
        pd_aggregate_tile_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx                                        [0:upper_bound_of_indices_to_plot_image]   # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c MULTIMODE____TEST')
        pd_aggregate_tile_probabilities_matrix[ 'true_class_prob' ]  = true_class_prob                                       [0:upper_bound_of_indices_to_plot_image]   # same
        # ~ pd_aggregate_tile_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )
        
        df = pd_aggregate_tile_probabilities_matrix
        
        class_data = [d for d in pd_aggregate_tile_probabilities_matrix]
   
        if bar_chart_x_labels=='case_id':
          c_id = pd_aggregate_tile_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_aggregate_tile_probabilities_matrix.shape[0])]
                  
        x_labels = [  str(el) for el in c_id ]

        bottom_bars = [0] * (len(class_names)+1)
        top_bars    = [0] * (len(class_names)+1)
        
        for i in range ( 0, len(class_names) ):
          top_bars    [i]   = pd_aggregate_tile_probabilities_matrix.iloc[:,i] / agg_prob
          bottom_bars [i+1] = bottom_bars[i]+top_bars[i]
          plt.bar( x=x_labels, height=top_bars[i],   bottom=bottom_bars[i], color=class_colors[i] )          
        
        ax.set_title   ("Input Data = Slide Image Tiles;  Bar Heights = Probabilities Assigned to *EACH* Cancer Sub-type",            fontsize=16 )
        ax.set_xlabel  ("Case ID",                                                     fontsize=14 )
        ax.set_ylabel  ("Probability Assigned by Network",                             fontsize=14 )
        ax.tick_params (axis='x', labelsize=12,   labelcolor='black')
        ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
        plt.legend( class_names,loc=2, prop={'size': 14} )

  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
      
        writer.add_figure('images___probs_assigned_to_ALL__classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_bar_chart_images___probs_assigned_to_ALL__classes"
        fqn = f"{fqn[0:255]}.png"
        
        fig.savefig(fqn)



        fqn = f"{args.log_dir}/probabilities_dataframe_image.csv"
        try:
          pd_aggregate_tile_probabilities_matrix.to_csv ( fqn, sep='\t' )
          if DEBUG>88:
            print ( f"CLASSI:         INFO:     now saving  probabilities dataframe {ASPARAGUS}(image){RESET} to   {MAGENTA}{fqn}{RESET}"  )
        except Exception as e:
          print ( f"{ORANGE}CLASSI:         WARNING:     could not save file   = {ORANGE}{fqn}{RESET}"  )
          # ~ print ( f"{ORANGE}CLASSI:         WARNING:     error was: {e}{RESET}" )
          




      # Case rna: 
    
      elif input_mode=='rna':
        
        pd.set_option('display.max_columns',  300 )
        pd.set_option('display.max_rows',     600 )
        pd.set_option('display.max_colwidth', 300 )
        pd.set_option('display.width',        300 )
        pd.set_option("display.precision",      8 )
                          
        if DEBUG>55:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:      probabilities_matrix                 = \n{CAMEL}{probabilities_matrix}{RESET}", flush=True )
        if DEBUG>0:
          print ( f"\nCLASSI:         INFO:      probabilities_matrix.shape                   = {MIKADO}{probabilities_matrix.shape}{RESET}", flush=True )


        figure_width  = 20
        figure_height = 10

        if args.just_test!='True':        
          if args.cases!='ALL_ELIGIBLE_CASES':
            upper_bound_of_indices_to_plot_rna = n_samples
          elif args.cases!='MULTIMODE____TEST':
            upper_bound_of_indices_to_plot_rna = cases_reserved_for_image_rna
          else:
            upper_bound_of_indices_to_plot_rna = n_samples
        else:
          upper_bound_of_indices_to_plot_rna   = n_samples

        if DEBUG>8:
          print ( f"\nCLASSI:         INFO:                                 n_samples                                    = {MIKADO}{n_samples}{RESET}",                                        flush=True )
          print ( f"\nCLASSI:         INFO:                                 cases_reserved_for_image_rna                 = {MIKADO}{cases_reserved_for_image_rna}{RESET}",                     flush=True )
          print ( f"\nCLASSI:         INFO:                                 upper_bound_of_indices_to_plot_rna           = {MIKADO}{upper_bound_of_indices_to_plot_rna}{RESET}\n\n\n\n\n\n  ", flush=True )
        


        # Case rna-1:  bar chart showing probability assigned to PREDICTED classes
           
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        if DEBUG>55:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:       probabilities_matrix = \n{CAMEL}{probabilities_matrix}{RESET}", flush=True )

        true_class_prob = probabilities_matrix[ range(0, true_classes.shape[0]), true_classes ]
        pred_class_idx  = np.argmax ( probabilities_matrix, axis=1   )
        correct_count   = np.sum    ( true_classes == pred_class_idx )

        plt.xticks( rotation=90 )
        probabilities_matrix=probabilities_matrix[0:n_samples,:]                                  # possibly truncate rows because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c MULTIMODE____TEST')
        pd_probabilities_matrix                       = pd.DataFrame( probabilities_matrix )
        pd_probabilities_matrix.columns               = class_names
        pd_probabilities_matrix[ 'agg_prob'        ]  = np.sum(probabilities_matrix,   axis=1 )  [0:upper_bound_of_indices_to_plot_rna]
        pd_probabilities_matrix[ 'max_agg_prob'    ]  = pd_probabilities_matrix.max   (axis=1)   [0:upper_bound_of_indices_to_plot_rna]
        pd_probabilities_matrix[ 'pred_class'      ]  = pd_probabilities_matrix.idxmax(axis=1)   [0:upper_bound_of_indices_to_plot_rna]    # grab class (which is the column index with the highest value in each row) and save as a new column vector at the end, to using for coloring 
        pd_probabilities_matrix[ 'true_class'      ]  = true_classes                             [0:upper_bound_of_indices_to_plot_rna]    # same
        pd_probabilities_matrix[ 'n_classes'       ]  = len(class_names) 
        pd_probabilities_matrix[ 'case_id'         ]  = rna_case_id                              [0:upper_bound_of_indices_to_plot_rna]    # same
        pd_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx                           [0:upper_bound_of_indices_to_plot_rna]    # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c MULTIMODE____TEST')
        pd_probabilities_matrix[ 'true_class_prob' ]  = true_class_prob                          [0:upper_bound_of_indices_to_plot_rna]    # same
        # ~ pd_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )
 
        if DEBUG>0: ##################DON'T DELETE
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:       (extended) pd_probabilities_matrix {CYAN}(rna){RESET} = \n{ARYLIDE}{pd_probabilities_matrix[0:upper_bound_of_indices_to_plot_rna]}{RESET}", flush=True ) 
  
        if bar_chart_x_labels=='case_id':
          c_id = pd_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_probabilities_matrix.shape[0])]


        x_labels = [  str(el) for el in c_id ]
        cols     = [ class_colors[el] for el in  pd_probabilities_matrix[ 'pred_class_idx']  ]
        
        p1 = plt.bar( x=x_labels, height=pd_probabilities_matrix[ 'max_agg_prob' ], color=cols )  

        # ~ ax = sns.barplot( x=c_id,  y=pd_probabilities_matrix[ 'max_agg_prob' ], hue=pd_probabilities_matrix['pred_class'], palette=class_colors, dodge=False )                  # in pandas, 'index' means row index
        ax.set_title   ("Input Data = RNA-Seq UQ FPKM Values;  Bar Height = Probability Assigned to *PREDICTED* Cancer Sub-type",            fontsize=16 )
        ax.set_xlabel  ("Case ID",                                                     fontsize=14 )
        ax.set_ylabel  ("Probability Assigned by Network",                             fontsize=14 )
        ax.tick_params (axis='x', labelsize=12,   labelcolor='black')
        ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
        # ~ plt.legend( class_names,loc=2, prop={'size': 14} )        
        
        i=0
        for p in ax.patches:
          if not np.isnan(p.get_height()):                                                                   # if it's a number, then it will be a height (y value)
            for index, row in pd_probabilities_matrix.iterrows():
              if DEBUG>555:
                print ( f"CLASSI:         INFO:      row['max_agg_prob']                       = {CAMEL}{row['max_agg_prob']}{RESET}", flush=True )            
                print ( f"CLASSI:         INFO:      p.get_height()                            = {CAMEL}{p.get_height()}{RESET}", flush=True )
                print ( f"CLASSI:         INFO:      true_classes[{MIKADO}{i}{RESET}]  = {AMETHYST}{true_classes[i]}{RESET}", flush=True ) 
              if row['max_agg_prob'] == p.get_height():                                                      # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>555:
                    print ( f"CLASSI:         INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}class_names[row['true_class']]  = {RESET}{MIKADO}{class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}correct_count                        = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  pass
            i+=1 
  
  
        if DEBUG>8:
          print ( f"\nCLASSI:         INFO:      number correct (rna_seq_probabs_matrix) = {CHARTREUSE}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        # ~ writer.add_figure('rna_seq__probs_assigned_to_PREDICTED_classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_bar_chart_rna_seq__probs_assigned_to_PREDICTED_classes"
        fqn = f"{fqn[0:255]}.png"
        fig.savefig(fqn)
  
  
  
  
        # case rna-2:  bar chart showing probability assigned to TRUE classses
           
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
        # ~ ax.set_title ( args.cancer_type_long )
              
        plt.xticks( rotation=90 )
        
        if bar_chart_x_labels=='case_id':
          c_id = pd_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_probabilities_matrix.shape[0])]

        if DEBUG>88:
          print ( f"CLASSI:         INFO:      probabilities_matrix {CYAN}(rna){RESET}  = \n{HOT_PINK}{probabilities_matrix}{RESET}", flush=True ) 

        for i in range ( 0, probabilities_matrix.shape[0] ):
          agg_prob = pd_probabilities_matrix[ 'agg_prob'][i]
          arg_max  = np.argmax( probabilities_matrix[i,:] )
          if DEBUG>88:
            print ( f"CLASSI:         INFO:      arg_max                   = {COTTON_CANDY}{arg_max}{RESET}", flush=True ) 
            print ( f"CLASSI:         INFO:      class_names[ arg_max ]    = {COTTON_CANDY}{class_names[ arg_max ]}{RESET}", flush=True ) 
          plt.bar( x=[ str(c_id[i]) ],   height=[ probabilities_matrix[i,arg_max] / agg_prob ],  color=class_colors[ arg_max ], label=class_names[ arg_max ] )


        # ~ ax = sns.barplot( x=c_id,  y=pd_probabilities_matrix[ 'true_class_prob' ], hue=pd_probabilities_matrix['pred_class'], palette=class_colors, dodge=False )                  # in pandas, 'index' means row index
        ax.set_title   ("Input Data = RNA-Seq UQ FPKM Values;  Bar Height = Probability Assigned to *TRUE* Cancer Sub-type",            fontsize=16 )
        ax.set_xlabel  ("Case ID",                                                     fontsize=14 )
        ax.set_ylabel  ("Probability Assigned by Network",                             fontsize=14 )
        ax.tick_params (axis='x', labelsize=8,   labelcolor='black')
        ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
        plt.ylim        (0.0, 1.0)
        # ~ plt.legend( class_names,loc=2, prop={'size': 14} )
        
        i=0
        for p in ax.patches:
          if not np.isnan(p.get_height()):                                                                   # if it's a number, then it will be a height (y value)
            for index, row in pd_probabilities_matrix.iterrows():
              if DEBUG>555:
                print ( f"CLASSI:         INFO:      row['max_agg_prob']                       = {COQUELICOT}{row['max_agg_prob']}{RESET}", flush=True )            
                print ( f"CLASSI:         INFO:      p.get_height()                            = {COQUELICOT}{p.get_height()}{RESET}", flush=True )
                print ( f"CLASSI:         INFO:      true_classes[{MIKADO}{i}{RESET}]  = {COQUELICOT}{true_classes[i]}{RESET}", flush=True ) 
              if row['max_agg_prob'] == p.get_height():                                                      # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>555:
                    print ( f"CLASSI:         INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"CLASSI:         INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}class_names[row['true_class']]  = {RESET}{MIKADO}{class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"CLASSI:         INFO:      {GREEN}correct_count                        = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  pass
            i+=1 
  
        if DEBUG>8:
          print ( f"\nCLASSI:         INFO:      number correct (rna_seq_probabs_matrix) = {COQUELICOT}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        writer.add_figure('rna_seq__probs_assigned_to_TRUE_classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_bar_chart_rna_seq__probs_assigned_to_TRUE_classes"
        fqn = f"{fqn[0:255]}.png"
        fig.savefig(fqn)
  
  
  
    
        # case rna-3:  bar chart showing probabilities assigned to ALL classses

        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        if DEBUG>55:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nCLASSI:         INFO:       probabilities_matrix = \n{BLEU}{pd_probabilities_matrix}{RESET}", flush=True )
  
        plt.xticks( rotation=90 )
        pd_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx  [0:n_samples]                      # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c MULTIMODE____TEST')
        pd_probabilities_matrix[ 'true_class_prob' ]  = true_class_prob [0:n_samples]                      # same
        # ~ pd_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )

        
        if bar_chart_x_labels=='case_id':
          c_id = pd_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_probabilities_matrix.shape[0])]
                  
        x_labels    = [   str(el) for el in c_id  ]
        top_bars    = [0] * (len(class_names) + 1 )
        bottom_bars = [0] * (len(class_names) + 1 )

        for i in range ( 0, len(class_names) ):
          top_bars    [i]   = pd_probabilities_matrix.iloc[:,i] / agg_prob
          bottom_bars [i+1] = bottom_bars[i]+top_bars[i]
          plt.bar( x=x_labels, height=top_bars[i],   bottom=bottom_bars[i], color=class_colors[i] ) 
        
        # ~ ax = pd_probabilities_matrix.iloc[0:6,0:6].plot(kind='bar', stacked=True)
        # ~ ax = sns.barplot( x=c_id,  y=pd_probabilities_matrix, hue=pd_probabilities_matrix['pred_class'], palette=class_colors, dodge=False )                  # in pandas, 'index' means row index
        ax.set_title   ("Input Data = RNA-Seq UQ FPKM Values;  Bar Heights = Probabilities Assigned to *EACH* Cancer Sub-type",            fontsize=16 )
        ax.set_xlabel  ("Case ID",                                                     fontsize=14 )
        ax.set_ylabel  ("Probability Assigned by Network",                             fontsize=14 )
        ax.tick_params (axis='x', labelsize=12,   labelcolor='black')
        ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
        plt.legend( class_names,loc=2, prop={'size': 14} )

        if DEBUG>0:
          print ( f"\nCLASSI:         INFO:      number correct (pd_probabilities_matrix) = {COQUELICOT}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
      
        writer.add_figure('rna_seq__probs_assigned_to_ALL__classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_bar_chart_rna_seq__probs_assigned_to_ALL__classes"
        fqn = f"{fqn[0:255]}.png"        
        fig.savefig(fqn)
        
  

        fqn = f"{args.log_dir}/probabilities_dataframe_rna.csv"
        try:
          pd_probabilities_matrix.to_csv ( fqn, sep='\t' )
          if DEBUG>0:
            print ( f"CLASSI:         INFO:     now saving  probabilities dataframe {COQUELICOT}(rna){RESET}   to   {MAGENTA}{fqn}{RESET}"  )
        except Exception as e:
          print ( f"{ORANGE}CLASSI:         WARNING:     could not save file   = {ORANGE}{fqn}{RESET}"  )
          # ~ print ( f"{ORANGE}CLASSI:         WARNING:     error was: {e}{RESET}" )     
          
  
 
        
        
        
        
        # case multimode:

        if DEBUG>0:
          print ( f"CLASSI:         INFO:     now loading probabilities dataframe {CYAN}(image){RESET} from {MAGENTA}{fqn}{RESET} if it exists from an earlier run"  ) 
          
        image_dataframe_file_exists=False
        fqn = f"{args.log_dir}/probabilities_dataframe_image.csv"
        try:
          pd_aggregate_tile_probabilities_matrix = pd.read_csv( fqn, sep='\t'  )
          image_dataframe_file_exists=True
        except Exception as e:
          print ( f"{ORANGE}CLASSI:         INFO:     could not open file  {MAGENTA}{fqn}{RESET}{ORANGE} - it probably doesn't exist"  )
          print ( f"{ORANGE}CLASSI:         INFO:     explanation: if you want the bar chart which combines image and rna probabilities, you need to have performed both an image and an rna run. {RESET}" )                
          print ( f"{ORANGE}CLASSI:         INFO:     e.g. perform the following sequence of runs:{RESET}" )                 
          print ( f"{ORANGE}CLASSI:         INFO:          {CYAN}./do_all.sh     -d <cancer type code> -i image -c UNIMODE_CASE____MATCHED    -v true{RESET}" )                 
          print ( f"{ORANGE}CLASSI:         INFO:          {CYAN}./just_test.sh  -d <cancer type code> -i image -c UNIMODE_CASE____MATCHED{RESET}" )                 
          print ( f"{ORANGE}CLASSI:         INFO:          {CYAN}./do_all.sh     -d <cancer type code> -i rna   -c UNIMODE_CASE____MATCHED{RESET}" )                 
          print ( f"{ORANGE}CLASSI:         INFO:          {CYAN}./just_test.sh  -d <cancer type code> -i rna   -c UNIMODE_CASE____MATCHED{RESET}" )   
          print ( f"{ORANGE}CLASSI:         INFO:     continuing...{RESET}" ) 

        if image_dataframe_file_exists:

          upper_bound_of_indices_to_plot_image = len(pd_aggregate_tile_probabilities_matrix.index)
          
          if DEBUG>0:
            print ( f"\nCLASSI:         INFO:      upper_bound_of_indices_to_plot_image = {COQUELICOT}{upper_bound_of_indices_to_plot_image}{RESET}", flush=True )
                      
          if upper_bound_of_indices_to_plot_image  !=   upper_bound_of_indices_to_plot_rna:
            print ( f"{ORANGE}CLASSI:         INFO:     for some reason the numbers of image examples and the number of rna examples to be plotted differ{RESET}"      ) 
            print ( f"{ORANGE}CLASSI:         INFO:        upper_bound_of_indices_to_plot_image = {MIKADO}{upper_bound_of_indices_to_plot_image}{RESET}"  ) 
            print ( f"{ORANGE}CLASSI:         INFO:        upper_bound_of_indices_to_plot_rna   = {MIKADO}{upper_bound_of_indices_to_plot_rna}{RESET}"  ) 
            print ( f"{ORANGE}CLASSI:         INFO:     possible explanation: one or both of the {CYAN}N_SAMPLES{RESET}{ORANGE} config settings is too small to have captured sufficient of the {CYAN}{args.cases}{RESET}{ORANGE} cases"      ) 
            print ( f"{ORANGE}CLASSI:         INFO:     skipping combined image+rna porbabilities plot that would otherwise have been generated{RESET}"      ) 
            print ( f"{ORANGE}CLASSI:         INFO:     continuing ...{RESET}"      ) 
            

          else:
            
            if DEBUG>0:
              np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
              print ( f"\nCLASSI:         INFO:     pd_aggregate_tile_probabilities_matrix {CYAN}(image){RESET} (from {MAGENTA}{fqn}{RESET}) = \n{COTTON_CANDY}{pd_aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_rna]}{RESET}", flush=True )   
              
            pd_aggregate_tile_probabilities_matrix[ 'true_class_prob' ] /= pd_aggregate_tile_probabilities_matrix[ 'agg_prob' ]   # image case only: normalize by dividing by number of tiles in the patch (which was saved as field 'agg_prob')
      
            if DEBUG>0:
              np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
              print ( f"\nCLASSI:         INFO:       pd_aggregate_tile_probabilities_matrix {CYAN}(image){RESET} normalized probabilities (from {MAGENTA}{fqn}{RESET}) = \n{COTTON_CANDY}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True )  
              
            
            if DEBUG>0:
              print ( f"\nCLASSI:         INFO:     n me {CYAN}(rna){RESET} from {MAGENTA}{fqn}{RESET} if it exists from an earlier or the current run"  )  
         
            rna_dataframe_file_exists=False             
            fqn = f"{args.log_dir}/probabilities_dataframe_rna.csv"
            try:
              pd_probabilities_matrix = pd.read_csv(  fqn, sep='\t'  )
              rna_dataframe_file_exists=True
              if DEBUG>0:
                np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
                print ( f"\nCLASSI:         INFO:     pd_probabilities_matrix {CYAN}(rna){RESET} (from {MAGENTA}{fqn}{RESET}) = \n{ARYLIDE}{pd_probabilities_matrix}{RESET}", flush=True )  
            except Exception as e:
              print ( f"{ORANGE}CLASSI:         INFO:     could not open file  = {ORANGE}{fqn}{RESET}{ORANGE} - it probably doesn't exist"  )
              print ( f"{ORANGE}CLASSI:         INFO:     if you want the bar chart which combines image and rna probabilities, you need to have performed both an image and an rna run. {RESET}" )                
              print ( f"{ORANGE}CLASSI:         INFO:     e.g. perform the following sequence of runs:{RESET}" )                 
              print ( f"{ORANGE}CLASSI:         INFO:              {CYAN}./do_all.sh     -d <cancer type code> -i image -c UNIMODE_CASE____MATCHED -v true{RESET}{ORANGE}'{RESET}" )                 
              print ( f"{ORANGE}CLASSI:         INFO:              {CYAN}./just_test.sh  -d <cancer type code> -i image -c UNIMODE_CASE____MATCHED{RESET}" )                 
              print ( f"{ORANGE}CLASSI:         INFO:              {CYAN}./do_all.sh     -d <cancer type code> -i rna   -c UNIMODE_CASE____MATCHED{RESET}" )                 
              print ( f"{ORANGE}CLASSI:         INFO:              {CYAN}./just_test.sh  -d <cancer type code> -i rna   -c UNIMODE_CASE____MATCHED{RESET}" )   
              print ( f"{ORANGE}CLASSI:         INFO:     continuing...{RESET}" ) 
    
                        
      
            if image_dataframe_file_exists & rna_dataframe_file_exists:                                        # then it will be possible to do the multimode plot
      
              # case multimode_1:  multimode image+rns - TRUE classses (this is the only case for multimode)
                 
              fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
              
              if bar_chart_x_labels=='case_id':                                                                # user choice for the x_lables 
                c_id = pd_probabilities_matrix[ 'case_id' ]
              else:
                c_id = [i for i in range(pd_probabilities_matrix.shape[0])]
      
              x_labels = [  str(el) for el in c_id ]
    
      
              set1 =                pd_probabilities_matrix[ 'true_class_prob' ][0:upper_bound_of_indices_to_plot_rna]                               # rna
              set2 = pd_aggregate_tile_probabilities_matrix[ 'true_class_prob' ][0:upper_bound_of_indices_to_plot_rna]                               # image
      
              if bar_chart_x_labels=='case_id':
                c_id = pd_aggregate_tile_probabilities_matrix[ 'case_id' ]
              else:
                c_id = [i for i in range(pd_aggregate_tile_probabilities_matrix.shape[0])]    
                        
              x_labels = [  str(el) for el in c_id ][0:upper_bound_of_indices_to_plot_rna]                                                         
    
              col0     = plt.cm.tab20b(0)
              col1     = plt.cm.Accent(7)   
    
              if DEBUG>0: 
                print ( f"\nCLASSI:         INFO:      upper_bound_of_indices_to_plot_rna                                   = {ARYLIDE}{upper_bound_of_indices_to_plot_rna}{RESET}", flush=True )
                print ( f"\nCLASSI:         INFO:      x_labels                                                             = \n{ARYLIDE}{x_labels}{RESET}", flush=True )
                print ( f"\nCLASSI:         INFO:      {CYAN}(rna){RESET} pd_probabilities_matrix                [ 'true_class_prob' ]   = \n{ARYLIDE}{set1}{RESET}", flush=True )
                print ( f"\nCLASSI:         INFO:      {CYAN}(img){RESET} pd_aggregate_tile_probabilities_matrix [ 'true_class_prob' ]   = \n{COTTON_CANDY}{set2}{RESET}", flush=True )
    
              
              p1 = plt.bar( x=x_labels, height=set1,               color=col0 )
              p2 = plt.bar( x=x_labels, height=set2, bottom=set1,  color=col1 )
             
              ax.set_title   ("Input Data = Imaga Tiles; RNA-Seq FPKM UQ;  Bar Height = Composite (Image + RNA-Seq) Probability Assigned to *TRUE* Cancer Sub-types",  fontsize=16 )
              ax.set_xlabel  ("Case ID",                                                     fontsize=14 )
              ax.set_ylabel  ("Probability Assigned by Network",                             fontsize=14 )
              ax.tick_params (axis='x', labelsize=8,   labelcolor='black')
              ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
              # ~ plt.legend( class_names,loc=2, prop={'size': 14} )
              plt.xticks( rotation=90 )     
      
        
              if DEBUG>0:
                print ( f"\nCLASSI:         INFO:      number correct (image+rna) = {CHARTREUSE}{correct_count}{RESET}", flush=True )
        
              pct_correct = correct_count/n_samples
              stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
              plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
        
              plt.tight_layout()
                        
              writer.add_figure('z_multimode__probs_assigned_to_TRUE_classes', fig, 0 )         
            
  
  

   
    # (E)  MAYBE PROCESS AND DISPLAY RUN LEVEL CONFUSION MATRICES   
    
    if ( args.just_test!='True') | ( (args.just_test=='true')  &  (args.input_mode=='image_rna') & (args.multimode=='image_rna') ):
    
      #np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE if x==0 else WHITE if x<=5 else CARRIBEAN_GREEN} {x:>15d}"})  
      #print ( f"CLASSI:         INFO:  {ORANGE}run_level{RESET}_classifications_matrix (all test samples, using the best model that was saved during this run =\n" )
      #print ( f"         ", end='' ) 
      #print ( [ f"{name:.50s}" for name in class_names ] )    
      #print ( f"\n{run_level_classifications_matrix}{RESET}" )
  

      run_level_classifications_matrix_acc[run-1,:,:] = run_level_classifications_matrix[:,:]              # accumulate run_level_classifications_matrices
  
      if DEBUG>4:
        print ( f"\n{run_level_classifications_matrix}" )
                 
   
      if DEBUG>9:
        print ( f"\n{run_level_classifications_matrix_acc[run-1,:,:]}" )    
  
      if DEBUG>4:    
        print(  '\033[13B' )
        print( f"CLASSI:           INFO:    {BITTER_SWEET}Test predictions produced during training for this run{RESET}"         )
        print( f"CLASSI:           INFO:    {BITTER_SWEET}======================================================{RESET}"  )
        print( f"CLASSI:           INFO:                                                                                      "  )  
    
      total_correct, total_examples  = show_classifications_matrix( writer, total_runs_in_job, pct_test, epoch, run_level_classifications_matrix, class_names, level='run' )
  

      if DEBUG>4:  
        print( f"CLASSI:           INFO:    correct / examples  =  {BITTER_SWEET}{np.sum(total_correct, axis=0)} / {np.sum(run_level_classifications_matrix, axis=None)}{WHITE}  ({BITTER_SWEET}{100 * np.sum(total_correct, axis=0) / np.sum(run_level_classifications_matrix):3.1f}%){RESET}")
  
      for i in range( 0, len( run_level_classifications_matrix) ):                                         # reset for the next run   
        run_level_classifications_matrix[i] = 0  
    
  
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time),     0       )
      #pplog.log_section('run complete in {:} mins'.format( minutes ) )
  
      print( f'CLASSI:         INFO:  elapsed time since job started: {MIKADO}{minutes}{RESET} mins ({MIKADO}{seconds:.1f}{RESET} secs)')
  
      print ( "\033[6A" )
            
    #  ^^^  JOB FINISHES HERE ^^^
  
  
  
  
  
    # (F)  PROCESS AND GENERATE AND SAVE (AND MAYBE DISPLAY) JOB LEVEL CONFUSION MATRIX

    if (args.just_test!='True') & (run==total_runs_in_job):
    
      print(  '\033[6B' )      
      print( f'CLASSI:         INFO:'                                                                                    )
      print( f"CLASSI:         INFO:    {CARRIBEAN_GREEN}Test predictions produced during training for this job{RESET}"     )
      print( f"CLASSI:         INFO:    {CARRIBEAN_GREEN}======================================================{RESET}"  )  
      print( f'CLASSI:         INFO:'                                                                                    )      
    
    
      # (i) generate and save job level classification matrix
      
      total_correct, total_examples  = show_classifications_matrix( writer, total_runs_in_job, pct_test, epoch, job_level_classifications_matrix, class_names, level='job' )
    
      np.set_printoptions(edgeitems=1000)
      np.set_printoptions(linewidth=1000)
      
      np.seterr( invalid='ignore', divide='ignore' )
      print( f"\n" )
      print( f'CLASSI:         INFO:    number of runs in this job                   = {CHARTREUSE}{total_runs_in_job}{RESET}')
      print( f"CLASSI:         INFO:    total for ALL {BOLD}test{RESET} examples over ALL runs    = {CHARTREUSE}{np.sum(total_correct, axis=0)} / {np.sum(job_level_classifications_matrix, axis=None)}  ({CHARTREUSE}{100 * np.sum(total_correct, axis=0) / np.sum(job_level_classifications_matrix):3.1f}%){RESET}")
      np.set_printoptions(formatter={'int': lambda x: f"{CHARTREUSE}{x:>6d}"})
      print( f'CLASSI:         INFO:    total correct per subtype over all runs      = { total_correct }{RESET}')
      np.set_printoptions(formatter={'float': lambda x: f"{CHARTREUSE}{x:>6.1f}"})
      print( f'CLASSI:         INFO:     %    correct per subtype over all runs      = { 100 * np.divide( total_correct, total_examples) }{RESET}')
      np.seterr(divide='warn', invalid='warn')  
      
      if DEBUG>9:
        np.set_printoptions(formatter={'int': lambda x: f"{CHARTREUSE}{x:>6d}    "})    
        print ( f"CLASSI:        INFO:    run_level_classifications_matrix_acc[0:total_runs_in_job,:,:] = \n{run_level_classifications_matrix_acc[0:total_runs_in_job,:,:] }{RESET}",  flush=True  )
      if DEBUG>9:
        print ( f"CLASSI:           INFO:  run_level_classifications_matrix_acc        = \n{CHARTREUSE}{run_level_classifications_matrix_acc[ 0:total_runs_in_job, : ] }{RESET}",      flush=True      )


      # (ii) generate and save per-subtype confusion matrices and associated statistics
      
      if DEBUG>0:
        print ( f"CLASSI:           INFO:  job_level_classifications_matrix.shape       = {CHARTREUSE}{job_level_classifications_matrix.shape}{RESET}",  flush=True      )
        print ( f"CLASSI:           INFO:  job_level_classifications_matrix             = \n{CHARTREUSE}{job_level_classifications_matrix}{RESET}",  flush=True      )
      
        df = pd.DataFrame( columns=[ 'Per Subtype Confusion Matrices', '',  '', '',  '',  '', '' ]  )
              
      total_predictions = np.sum( job_level_classifications_matrix )
      for i in range ( 0, job_level_classifications_matrix.shape[1] ):                                                                                 # for each row (subtype)

        true_positives  =         job_level_classifications_matrix[ i, i ]                                                                             # the element on the diagonal
        false_positives = np.sum( job_level_classifications_matrix[ i, : ] )    -  true_positives                                                      # every item in the same the row    minus the diagonal element
        false_negatives = np.sum( job_level_classifications_matrix[ :, i ] )    -  true_positives                                                      # every item in the same the column minus the diagonal element
        true_negatives  = total_predictions - true_positives - false_positives  - false_negatives                                                      # everything else
        precision       = round ( ( true_positives / ( true_positives + false_positives )        if ( true_positives + false_positives ) !=0    else 0 ), 3)
        recall          = round ( ( true_positives / ( true_positives + false_negatives )        if ( true_positives + false_negatives ) !=0    else 0 ), 3)
        F1              = round ( ( ( 2 * precision * recall) / ( precision + recall )           if ( precision + recall               ) !=0    else 0 ), 3)
        accuracy        = round ( ( ( true_positives + true_negatives ) / total_predictions      if ( total_predictions                ) !=0    else 0 ), 3)
        specificity     = round ( ( true_negatives / ( true_negatives + false_positives )        if ( true_negatives + false_positives ) !=0    else 0 ), 3)
        total           = true_positives + true_negatives + false_positives + false_negatives

        true_positives_pct  = round ( (100 * true_positives  / total                            if ( total                            ) !=0    else 0 ), 2)
        false_positives_pct = round ( (100 * false_positives / total                            if ( total                            ) !=0    else 0 ), 2)
        false_negatives_pct = round ( (100 * false_negatives / total                            if ( total                            ) !=0    else 0 ), 2)
        true_negatives_pct  = round ( (100 * true_negatives  / total                            if ( total                            ) !=0    else 0 ), 2)

        if DEBUG>2:
          print ( f"\n",                                                                                                                                                                          flush=True  ) 
          print ( f"CLASSI:           INFO:  class/subtype name             [{CHARTREUSE}{i}{RESET}] = {COTTON_CANDY}{ class_names[i] }{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  total predictions              [{CHARTREUSE}{i}{RESET}] = {BLEU}{ total_predictions }{RESET}",                                                       flush=True  ) 
          print ( f"CLASSI:           INFO:  true  positives                [{CHARTREUSE}{i}{RESET}] = {CHARTREUSE}{  true_positives  }{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  true  negatives                [{CHARTREUSE}{i}{RESET}] = {CHARTREUSE}{  true_negatives  }{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  false positives                [{CHARTREUSE}{i}{RESET}] = {CHARTREUSE}{  false_positives }{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  false negatives                [{CHARTREUSE}{i}{RESET}] = {CHARTREUSE}{  false_negatives }{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  checksum                       [{CHARTREUSE}{i}{RESET}] = {BLEU}{  total }{RESET}",                                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  {BOLD}precision{RESET}         [{CHARTREUSE}{i}{RESET}] = {COQUELICOT}{ precision    :.3f}{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  {BOLD}recall{RESET}            [{CHARTREUSE}{i}{RESET}] = {COQUELICOT}{ recall       :.3f}{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  {BOLD}accuracy{RESET}          [{CHARTREUSE}{i}{RESET}] = {COQUELICOT}{ accuracy     :.3f}{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  {BOLD}specificity{RESET}       [{CHARTREUSE}{i}{RESET}] = {COQUELICOT}{ specificity  :.3f}{RESET}",                                                  flush=True  ) 
          print ( f"CLASSI:           INFO:  {BOLD}F1{RESET}                [{CHARTREUSE}{i}{RESET}] = {COQUELICOT}{ F1           :.3f}{RESET}",                                                  flush=True  ) 

        
        # use pandas to save as a csv file
        
        class_name = class_names[i][:25]
        row_1_string   = f"Subtype: {class_name: <30s}"
        row_2_string   = f".                              Predicted Positive Count"
        row_3_string   = f".                              Predicted Negative Count"
        row_4_string   = f".                                        Total Samples:"
        row_5_string   = f".                 Precision              TP / (TP+FP) :"
        row_6_string   = f".                 Recall                 TP / (TP+FN) :"
        row_7_string   = f".                 F1 Score            2*P*R / (P + R) :"
        row_8_string   = f".                 Accuracy    (TP+TN) / (TP+TN+FP+FN) :"
        row_9_string   = f".                 Specificity            TP / (TN+FP) :"
      
        
        # ~ df = pd.DataFrame( index=[ 'Actual Positives', 'Actual Negatives', 'blank row' ], columns=[ 'Subtype', 'Predicted Positives', 'Predicted Negatives'] )

        # ~ df.at['Actual Positives', 'Subtype'] = class_name
        # ~ df.at['Actual Negatives', 'Subtype'] = class_name

        # ~ df.at['Actual Positives', 'Predicted Positives'] = true_positives
        # ~ df.at['Actual Positives', 'Predicted Negatives'] = false_negatives
        # ~ df.at['Actual Negatives', 'Predicted Positives'] = false_positives
        # ~ df.at['Actual Negatives', 'Predicted Negatives'] = true_negatives

        df.loc[len(df.index)] = [  row_1_string,     '',                         'True Positive Count',     'True Negative Count',    '                    ',              'True Negative Percent',     'True Negative Percent' ]        
        df.loc[len(df.index)] = [  row_2_string,     '',                          true_positives,            false_positives,         '                    ',               true_positives_pct,         false_positives_pct     ]
        df.loc[len(df.index)] = [  row_3_string,     '',                          false_negatives,           true_negatives,          '                    ',               false_negatives_pct,        true_negatives_pct      ]
        df.loc[len(df.index)] = [  '.',               '--- Statistics ---',       '',                        '',                      '                    ',               '',                         ''                      ]
        df.loc[len(df.index)] = [  row_4_string,     total,                       '',                        '',                      '                    ',               '',                         ''                      ]
        df.loc[len(df.index)] = [  row_5_string,     precision,                   '',                        '',                      '                    ',               '',                         ''                      ]
        df.loc[len(df.index)] = [  row_6_string,     recall,                      '',                        '',                      '                    ',               '',                         ''                      ]
        df.loc[len(df.index)] = [  row_7_string,     F1,                          '',                        '',                      '                    ',               '',                         ''                      ]
        df.loc[len(df.index)] = [  row_8_string,     accuracy,                    '',                        '',                      '                    ',               '',                         ''                      ]
        df.loc[len(df.index)] = [  row_9_string,     specificity,                 '',                        '',                      '                    ',               '',                         ''                      ]
        df.loc[len(df.index)] = [  '.',               '',                         '',                        '',                      '                    ',               '',                         ''                      ]
        
        # ~ display( df)
      
        now = datetime.datetime.now()
        fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_conf_matrices_per_subtype"
        fqn = f"{fqn[0:255]}.tsv"

        df.to_csv ( fqn, sep='\t' )
        

      if DEBUG>0:
        print(tabulate( df, headers='keys', tablefmt = 'fancy_grid' ) )
      
      print ( f"\n" )


    # (G) MAYBE PROCESS AND GENERATE AND SAVE (AND MAYBE DISPLAY) BOX PLOTS
    

    if ( args.box_plot=='True' ) & (run==total_runs_in_job):      

        box_plot_by_subtype( args, class_names, n_genes, start_time, parameters, zoom_out_mags, zoom_out_prob, writer, total_runs_in_job, pct_test, run_level_classifications_matrix_acc )


  # (H)  CLOSE UP AND END
  
  writer.close()        

  hours   = round( (time.time() - start_time) / 3600,  1   )
  minutes = round( (time.time() - start_time) /   60,  1   )
  seconds = round( (time.time() - start_time)       ,  0   )
  #pplog.log_section('Job complete in {:} mins'.format( minutes ) )

  print( f'\033[18B')
  if ( args.just_test=='True') & ( args.input_mode=='rna' ):
    print( f'\033[12B')  
  
  print( f'CLASSI:          INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
  now = time.localtime(time.time())
  print(time.strftime( f"CLASSI:          INFO:  end time = %Y-%m-%d %H:%M:%S %Z", now ))
  start_time = time.time() 

  #pplog.log_section('Model saved.')
  

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def train( args, epoch, train_loader, model, optimizer, loss_function, loss_type, writer, train_loss_min, batch_size  ):

    """
    Train Model for one epoch (= every training example in however many batches are necessary to process every example)
    """
    
    model.train()                                                                                          # set model to training mode

    loss_images_sum  = 0
    loss_genes_sum   = 0
    l1_loss_sum      = 0
    total_loss_sum   = 0


    if DEBUG>9:
      print( "CLASSI:         INFO:     train: about to enumerate over dataset" )
    
    for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in enumerate( train_loader ):
        
        if DEBUG>88:
          print( f"CLASSI:         INFO:     train: len(batch_images) = \033[33;1m{len(batch_images)}\033[m" )
          print( f"CLASSI:         INFO:     train: len(image_labels) = \033[33;1m{len(image_labels)}\033[m" )
          print( f"CLASSI:         INFO:     train: len(rna_labels)   = \033[33;1m{len(rna_labels)}\033[m" )
        if DEBUG>888:
          print ( "\033[6B" )
          print( f"{ image_labels.cpu().detach().numpy()},  ", flush=True, end="" )
          print( f"{   rna_labels.cpu().detach().numpy()},  ", flush=True, end="" )    
          print ( "\033[6A" )
                            
        if DEBUG>888:
          print( f"CLASSI:         INFO:     train: about to call {CYAN}optimizer.zero_grad(){RESET}" )

        # from: https://stackoverflow.com/questions/44732217/why-do-we-need-to-explicitly-call-zero-grad
        # We explicitly need to call zero_grad() because, after loss.backward() (when gradients are computed), we need to use optimizer.step() to proceed gradient descent. More specifically, the gradients are not automatically zeroed because these two operations, loss.backward() and optimizer.step(), are separated, and optimizer.step() requires the just computed gradients.
        optimizer.zero_grad()

        batch_images = batch_images.to ( device )                                                          # send to GPU
        batch_genes  = batch_genes.to  ( device )                                                          # send to GPU
        image_labels = image_labels.to ( device )                                                          # send to GPU
        rna_labels   = rna_labels.to   ( device )                                                          # send to GPU

        if DEBUG>99:
          print ( f"CLASSI:         INFO:     train: batch_images[0]                    = {MIKADO}\n{batch_images[0] }{RESET}", flush=True   )

        if DEBUG>99:
          print ( f"CLASSI:         INFO:     train: type(batch_images)                 = {MIKADO}{type(batch_images)}{RESET}",  flush=True  )
          print ( f"CLASSI:         INFO:     train: batch_images.size()                = {MIKADO}{batch_images.size()}{RESET}", flush=True  )


        if DEBUG>2:
          print( f"CLASSI:         INFO:      train: about to call {MAGENTA}model.forward(){RESET}" )

        gpu                = 0                                                                             # not currently used
        encoder_activation = args.encoder_activation
        
        if args.input_mode=='image':
          y1_hat, y2_hat, embedding = model.forward( [ batch_images, 0          ,  batch_fnames] , gpu, args  )          # perform a step. y1_hat = image outputs; y2_hat = rna outputs
          
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          if DEBUG>9:
            print ( f"CLASSI:         INFO:     train: batch_genes.size()                = {batch_genes.size}" )
          y1_hat, y2_hat, embedding = model.forward( [0,             batch_genes,  batch_fnames],  gpu, args )           # perform a step. y1_hat = image outputs; y2_hat = rna outputs


        if (args.input_mode=='image'):
          
          if DEBUG>0:
            np.set_printoptions(formatter={'float': lambda x:   "{:>6.2f}".format(x)})
            image_labels_numpy = (image_labels .cpu() .data) .numpy()
            y1_hat_numpy       = (y1_hat       .cpu() .data) .numpy()
            batch_fnames_npy   = (batch_fnames .cpu() .data) .numpy()
            random_pick        = random.randint( 0, y1_hat_numpy.shape[0]-1 )
            if DEBUG>2:            
              print ( f"CLASSI:         INFO:      test:        y1_hat_numpy       [{random_pick:3d}]  {ORANGE}(Predictions){RESET}     = {MIKADO}{y1_hat_numpy[random_pick]}{RESET}"     )            
              print ( f"CLASSI:         INFO:      test:        image_labels_numpy [{random_pick:3d}]  {GREEN}(Truth)      {RESET}     = {MIKADO}{image_labels_numpy[random_pick]}{RESET}"     )            
              print ( f"CLASSI:         INFO:      test:        predicted class    [{random_pick:3d}]                    = {RED if image_labels_numpy[random_pick]!=np.argmax(y1_hat_numpy[random_pick]) else GREEN}{np.argmax(y1_hat_numpy[random_pick])}{RESET}"     )
              print ( f"CLASSI:         INFO:      test:        image_labels_numpy (all)  {GREEN}(Truth)       {RESET}     = {MIKADO}{image_labels_numpy}{RESET}"     )            
            if DEBUG>100:    
              print ( f"CLASSI:         INFO:      test:        y1_hat_numpy.shape                     {ORANGE}(Predictions){RESET}     = {MIKADO}{y1_hat_numpy.shape}{RESET}"     )
              print ( f"CLASSI:         INFO:      test:        y1_hat_numpy                           {ORANGE}(Predictions){RESET}     = \n{MIKADO}{y1_hat_numpy}{RESET}"         )
              print ( f"CLASSI:         INFO:      test:        image_labels_numpy.shape               {GREEN}(Truth)       {RESET}     = {MIKADO}{image_labels_numpy.shape}{RESET}"         )
              print ( f"CLASSI:         INFO:      test:        image_labels_numpy                     {GREEN}(Truth)       {RESET}     = \n{MIKADO}{image_labels_numpy}{RESET}"         )
            if DEBUG>100:            
              print ( f"CLASSI:         INFO:      test:        fq_link            [{random_pick:3d}]                                   = {MIKADO}{args.data_dir}/{batch_fnames_npy[random_pick]}.fqln{RESET}"     )            
            
          loss_images       = loss_function( y1_hat, image_labels )
          loss_images_value = loss_images.item()                                                           # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories

          
          if DEBUG>2:
            print ( f"CLASSI:         INFO:      test: {MAGENTA}loss_images{RESET} (for this mini-batch)  = {PURPLE}{loss_images_value:6.3f}{RESET}" )
        
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          if DEBUG>9:
            np.set_printoptions(formatter={'int': lambda x:   "{:>4d}".format(x)})
            rna_labels_numpy = (rna_labels.cpu().data).numpy()
            print ( "CLASSI:         INFO:      test:       rna_labels_numpy                = \n{:}".format( image_labels_numpy  ) )
          if DEBUG>9:
            np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
            y2_hat_numpy = (y2_hat.cpu().data).numpy()
            print ( "CLASSI:         INFO:      test:       y2_hat_numpy                      = \n{:}".format( y2_hat_numpy) )

          if loss_type == 'mean_squared_error':                                                            # autoencoders use mean squared error. The function needs to be provided with both the input and the output to calculate mean_squared_error 
            loss_genes        = loss_function( y2_hat, batch_genes.squeeze() )            
          else:
            loss_genes        = loss_function( y2_hat, rna_labels            )
        
          loss_genes_value  = loss_genes.item()                                                            # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories

        # ~ l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0

        if (args.input_mode=='image'):
          total_loss        = loss_images_value + l1_loss
          TL=loss_images_value
        elif (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          total_loss        = loss_genes_value + l1_loss
          TL=loss_genes_value
        
        if DEBUG>0:
          if ( args.input_mode=='image' ):
            offset=162
            print ( f"\
\033[2K\r\033[27C{DULL_WHITE}train:\
\r\033[40Cn={i+1:>3d}{CLEAR_LINE}\
\r\033[49Craw loss_images={ loss_images_value:5.2f}\
\r\033[120CBATCH LOSS                (LOSS PER 1000 TILES) = \r\033[\
{offset+10*int((TL*5)//1) if TL<1 else offset+16*int((TL*1)//1) if TL<12 else 250}C{PALE_GREEN if TL<1 else PALE_ORANGE if 1<=TL<2 else PALE_RED}{TL*1000/batch_size:6.1f}{RESET}" )
            print ( "\033[2A" )
          elif (args.input_mode=='rna') | (args.input_mode=='image_rna'):
            print ( f"\
\033[2K\r\033[27C{DULL_WHITE}train:\
\r\033[40Cn={i+1:>3d}{CLEAR_LINE}\
\r\033[73Craw loss_rna={loss_genes_value:5.2f}\
\r\033[120CBATCH LOSS                (LOSS PER 1000 EXAMPLES) = \r\033[\
{offset+5*int((TL*5)//1) if TL<1 else offset+6*int((TL*1)//1) if TL<12 else 250}C{PALE_GREEN if TL<1 else PALE_ORANGE if 1<=TL<2 else PALE_RED}{TL*1000/batch_size:6.1f}{RESET}" )
            print ( "\033[2A" )          


        if (args.input_mode=='image'):
          loss_images.backward()
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):          
          loss_genes.backward()

        # Perform gradient clipping *before* calling `optimizer.step()`.
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        
        if (args.input_mode=='image'):
          loss_images_sum      +=  loss_images_value
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          loss_genes_sum       +=  loss_genes_value
        l1_loss_sum            +=  l1_loss
        total_loss_sum         +=  total_loss

        if (args.input_mode=='image'):
          del y1_hat                 
          del loss_images
          del image_labels          
        if ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          del y2_hat   
          del loss_genes
          del rna_labels

        torch.cuda.empty_cache()

        if DEBUG>99:
          print ( "CLASSI:         INFO:      train:       type(loss_images_sum)                      = {:}".format( type(loss_images_sum)       ) )
          
    
    #    ^^^^^^^^^  epoch complete  ^^^^^^^^^^^^^^^
    
    
    
    
    loss_images_sum_ave = loss_images_sum / (i+1)                                                          # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    loss_genes_sum_ave  = loss_genes_sum  / (i+1)                                                          # average genes loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    l1_loss_sum_ave     = l1_loss_sum     / (i+1)                                                          # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    total_loss_sum_ave  = total_loss_sum  / (i+1)                                                          # average total loss for the entire epoch (divide cumulative loss by number of batches in the epoch)

    if total_loss_sum < train_loss_min:
      train_loss_min = total_loss_sum

    if args.just_test=='False':                                                                            # don't record stats in test mode because it's only one epoch and is of no interest
      writer.add_scalar( 'loss_train',      total_loss_sum, epoch )
      writer.add_scalar( 'loss_train_min',  train_loss_min, epoch )

    return loss_images_sum_ave, loss_genes_sum_ave, l1_loss_sum_ave, total_loss_sum_ave








# ------------------------------------------------------------------------------
def test( cfg, args, parameters, embeddings_accum, labels_accum, epoch, test_loader,  model,  tile_size, loss_function, loss_type, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                                        test_loss_min, show_all_test_examples, batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours ): 

    """
    Test model by pushing one or more held-out batches through the network
    """

    global class_colors 
    global descriptor
    global global_batch_count
    global run_level_total_correct    
    global run_level_classifications_matrix
    global job_level_classifications_matrix

    
    model.eval()                                                                                           # set model to evaluation mode

    loss_images_sum     = 0
    loss_genes_sum      = 0
    l1_loss_sum         = 0
    total_loss_sum      = 0
    # ~ random_pick        = random.randint( 0, batch_size-1 )
    random_pick        = 2
      
      
    for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in  enumerate( test_loader ):
  
        batch_images = batch_images.to(device)
        batch_genes  = batch_genes .to(device)
        image_labels = image_labels.to(device)
        rna_labels   = rna_labels  .to(device)        

        gpu                = 0                                                                             # not currently used
        encoder_activation = args.encoder_activation

        if args.input_mode=='image':
          with torch.no_grad():                                                                            # don't need gradients for testing
            y1_hat, y2_hat, embedding = model.forward( [ batch_images, 0            , batch_fnames], gpu, args  )          # perform a step. y1_hat = image outputs; y2_hat = rna outputs

          if DEBUG>9:
            np.set_printoptions(formatter={'float': lambda x:   "{:>6.2f}".format(x)})
            image_labels_numpy = (image_labels .cpu() .data) .numpy()
            y1_hat_numpy       = (y1_hat       .cpu() .data) .numpy()
            batch_fnames_npy   = (batch_fnames .cpu() .data) .numpy()
            print ( f"CLASSI:         INFO:      test:        fq_link            [{random_pick:3d}]                                     = {CAMEL}{args.data_dir}/{batch_fnames_npy[random_pick]}.fqln{RESET}"     )            
            print ( f"CLASSI:         INFO:      test:        image_labels_numpy [{random_pick:3d}]      {GREEN}(Truth){RESET}          = {MIKADO}{image_labels_numpy[random_pick]}{RESET}"     )            
            print ( f"CLASSI:         INFO:      test:        y1_hat_numpy       [{random_pick:3d}]      {ORANGE}(Predictions){RESET}   = {MIKADO}{y1_hat_numpy[random_pick]}{RESET}"     )
            print ( f"CLASSI:         INFO:      test:        predicted class    [{random_pick:3d}]                                     = {RED if image_labels_numpy[random_pick]!=np.argmax(y1_hat_numpy[random_pick]) else GREEN}{np.argmax(y1_hat_numpy[random_pick])}{RESET}"     )
          if DEBUG>99:
            print ( f"CLASSI:         INFO:      test:        y1_hat_numpy.shape          {ORANGE}(Predictions){RESET}     = {MIKADO}{y1_hat_numpy.shape}{RESET}"     )
            print ( f"CLASSI:         INFO:      test:        y1_hat_numpy                {ORANGE}(Predictions){RESET}     = \n{MIKADO}{y1_hat_numpy}{RESET}"         )



        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):

          if DEBUG>99:
            print ( f"CLASSI:         INFO:       train: batch_genes.size() = {MIKADO}{batch_genes.size()}{RESET}" )
            
          with torch.no_grad():                                                                            # don't need gradients for testing
            y1_hat, y2_hat, embedding = model.forward( [ 0,            batch_genes  , batch_fnames], gpu, args )


          if loss_type=='mean_squared_error':
            
            if args.just_test=='True':                                                                       # In test mode (only), the embeddings are the reduced dimensionality features that we want to save for use with NN models
              
              if DEBUG>2:   
                print( f"CLASSI:         INFO:     about to push x2 through the autoencoder to obtain the reduced dimensionality features using the best model generated by the last training run{RESET}" )
  
              embeddings  = model.encode  ( batch_genes, args.input_mode, gpu, args )             
  
              embeddings_accum = torch.cat( (embeddings_accum, embeddings.cpu().squeeze()   ), dim=0, out=None ) 

              if DEBUG>2:
                print( f"CLASSI:         INFO:        sanity check: embeddings_accum.size     = {ASPARAGUS}{embeddings_accum.size()}{RESET}",         flush=True )
                print( f"CLASSI:         INFO:        sanity check: labels_accum    .size     = {ASPARAGUS}{labels_accum.size()}{RESET}",             flush=True )
                print( f"CLASSI:         INFO:        sanity check: embeddings      .size     = {ASPARAGUS}{embeddings.size()}{RESET}",               flush=True )
                print( f"CLASSI:         INFO:        sanity check: embeddings      .dtype    = {ASPARAGUS}{embeddings.dtype}{RESET}",                flush=True )
                print( f"CLASSI:         INFO:        sanity check: image_labels    .size     = {ASPARAGUS}{image_labels.size()}{RESET}",             flush=True )
                print( f"CLASSI:         INFO:        sanity check: rna_labels      .dtype    = {ASPARAGUS}{rna_labels.dtype}{RESET}",              flush=True )


              if args.input_mode=="image":
                labels_accum     = torch.cat( (labels_accum,     image_labels.cpu() ), dim=0, out=None )
              if args.input_mode=="rna":
                if DEBUG>99:
                  print( f"CLASSI:         INFO:       sanity check: labels_accum.size   = {AMETHYST}{labels_accum.size()}{RESET}",          flush=True )
                labels_accum     = torch.cat( (labels_accum,     rna_labels  .cpu() ), dim=0, out=None ) 
                if DEBUG>2: 
                  print( f"CLASSI:         INFO:       sanity check: labels_accum.size   = {ASPARAGUS}{labels_accum.size()}{RESET}",         flush=True )
 


        image_labels_values   =   image_labels.cpu().detach().numpy()
        rna_labels_values     =   rna_labels  .cpu().detach().numpy()
        batch_fnames_npy      =   batch_fnames.cpu().detach().numpy()        


        if loss_type!='mean_squared_error':                                                                # autoencoders don't produce predictions,so ignore

          if   ( args.input_mode=='image' ):
            
            preds, p_full_softmax_matrix, p_highest, p_2nd_highest, p_true_class = analyse_probs( y1_hat, image_labels_values )          
            
          if ( args.input_mode=='image' ) & ( args.just_test=='True' ):
            
            if args.scattergram=='True':
              if DEBUG>0:
                  print ( f"CLASSI:         INFO:      test:           global_batch_count {DIM_WHITE}(super-patch number){RESET} = {global_batch_count+1:5d}  {DIM_WHITE}({((global_batch_count+1)/(args.supergrid_size**2)):04.2f}){RESET}" )
                        
            if global_batch_count%(args.supergrid_size**2)==0:                                                                                 # establish grid arrays on the FIRST batch of each grid
              grid_images                = batch_images.cpu().numpy()
              grid_labels                = image_labels.cpu().numpy()
              grid_preds                 = preds
              grid_p_highest             = p_highest
              grid_p_2nd_highest         = p_2nd_highest
              grid_p_true_class          = p_true_class
              grid_p_full_softmax_matrix = p_full_softmax_matrix
              image_tile_width           = args.supergrid_size**2*batch_size

              if DEBUG>0:
                print ( f"CLASSI:         INFO:      test:             supergrid_size                                                 = {BLEU}{args.supergrid_size}{RESET} hence image dimensions (tiles*tiles*batch_size = {args.supergrid_size}*{args.supergrid_size}*{batch_size}={image_tile_width}) will be {BLEU}{image_tile_width}x{image_tile_width}{RESET}{CLEAR_LINE}"  )
              if DEBUG>90:
                print ( f"CLASSI:         INFO:      test:             i                                                              = {BLEU}{i}{RESET}{CLEAR_LINE}"                                    )
                print ( f"CLASSI:         INFO:      test:             grid_labels.shape                                              = {BLEU}{grid_labels.shape}{RESET}{CLEAR_LINE}"                    )
                print ( f"CLASSI:         INFO:      test:             grid_preds.shape                                               = {BLEU}{grid_preds.shape}{RESET}{CLEAR_LINE}"                     )
                print ( f"CLASSI:         INFO:      test:             grid_p_highest.shape                                           = {BLEU}{grid_p_highest.shape}{RESET}{CLEAR_LINE}"                 )            
                print ( f"CLASSI:         INFO:      test:             grid_p_2nd_highest.shape                                       = {BLEU}{grid_p_2nd_highest.shape}{RESET}{CLEAR_LINE}"             )
                print ( f"CLASSI:         INFO:      test:             grid_p_true_class.shape                                        = {BLEU}{grid_p_true_class.shape}{RESET}{CLEAR_LINE}"              )
                print ( f"CLASSI:         INFO:      test:             grid_p_full_softmax_matrix.shape                               = {BLEU}{grid_p_full_softmax_matrix.shape}{RESET}{CLEAR_LINE}"     )
  
            else:                                                                                                                              # ... accumulate for subsequent batches that will go into the same grid 
              grid_images                = np.append( grid_images,                batch_images.cpu().numpy(), axis=0 )
              grid_labels                = np.append( grid_labels,                image_labels.cpu().numpy(), axis=0 )
              grid_preds                 = np.append( grid_preds,                 preds,                      axis=0 )
              grid_p_highest             = np.append( grid_p_highest,             p_highest,                  axis=0 )
              grid_p_2nd_highest         = np.append( grid_p_2nd_highest,         p_2nd_highest,              axis=0 )
              grid_p_true_class          = np.append( grid_p_true_class,          p_true_class,               axis=0 )
              grid_p_full_softmax_matrix = np.append( grid_p_full_softmax_matrix, p_full_softmax_matrix,      axis=0 )


              if DEBUG>90:
                print ( f"CLASSI:         INFO:      test:             i                                                              = {MIKADO}{i}{RESET}"                                                   )
                print ( f"CLASSI:         INFO:      test:             supergrid_size                                                 = {MIKADO}{args.supergrid_size}{RESET} hence image dimensions (tiles*tiles*batch_size = {args.supergrid_size}*{args.supergrid_size}*{batch_size}={image_tile_width}) will be {BLEU}{image_tile_width}x{image_tile_width}{RESET}{CLEAR_LINE}"  )
                print ( f"CLASSI:         INFO:      test:             grid_labels.shape                                              = {MIKADO}{grid_labels.shape}{RESET}"                                   )
                print ( f"CLASSI:         INFO:      test:             grid_preds.shape                                               = {MIKADO}{grid_preds.shape}{RESET}"                                    )
                print ( f"CLASSI:         INFO:      test:             grid_p_highest.shape                                           = {MIKADO}{grid_p_highest.shape}{RESET}"                                )            
                print ( f"CLASSI:         INFO:      test:             grid_p_2nd_highest.shape                                       = {MIKADO}{grid_p_2nd_highest.shape}{RESET}"                            )
                print ( f"CLASSI:         INFO:      test:             grid_p_true_class.shape                                        = {MIKADO}{grid_p_true_class.shape}{RESET}"                             )
                print ( f"CLASSI:         INFO:      test:             grid_p_full_softmax_matrix.shape                               = {MIKADO}{grid_p_full_softmax_matrix.shape}{RESET}"                    )


              if global_batch_count%(args.supergrid_size**2)==(args.supergrid_size**2)-1:                                                      # if it is the last batch to go into this grid (super-patch)
    
                index                                      = int(i/(args.supergrid_size**2))                                                   # the entry we will update. (We aren't accumulating on every i'th batch, but rather on every args.supergrid_size**2-1'th batch (one time per grid))

                batch_fnames_npy = batch_fnames.numpy()                                                                                        # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details
                fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"

                if DEBUG>0:
                  # ~ print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             i                                                              = {ASPARAGUS}{i} (last batch for this epoch and therefore for the batches that make up this super-image{RESET})"                                                   )
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             (image) index                                                  = {ASPARAGUS}{index}{RESET}"                                 )
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             supergrid_size                                                 = {ASPARAGUS}{args.supergrid_size}{RESET} hence image dimensions (tiles*tiles*batch_size = {args.supergrid_size}*{args.supergrid_size}*{batch_size}={image_tile_width}) will be {BLEU}{image_tile_width}x{image_tile_width}{RESET}{CLEAR_LINE}"  )
                  # ~ print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             n_samples                                                      = {ASPARAGUS}{args.n_samples[0]}{RESET}" )
                if DEBUG>10:
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             len(patches_true_classes)                                      = {ASPARAGUS}{len(patches_true_classes)}{RESET}" )
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             patches_true_classes                                           = {ASPARAGUS}{patches_true_classes}{RESET}" )
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             batch_fnames_npy                                               = {ASPARAGUS}{batch_fnames_npy}{RESET}" )
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             fq_link                                                        = {ASPARAGUS}{fq_link}{RESET}" )

                patches_true_classes[index]                = image_labels.cpu().detach().numpy()[0]                                            # all tiles in a patch belong to the same case, so we can chose any of them - we choose the zero'th
                patches_case_id     [index]                = batch_fnames_npy[0]                                                               # all tiles in a patch belong to the same case, so we can chose any of them - we choose the zero'th
                grid_tile_probabs_totals_by_class          = np.transpose (np.expand_dims( grid_p_full_softmax_matrix.sum( axis=0 ), axis=1 )) # this is where we sum the totals across all tiles
                binary_matrix                              = np.zeros_like (grid_p_full_softmax_matrix)                                        # new matrix same shape as grid_p_full_softmax_matrix, with all values set to zero
                binary_matrix[ np.arange( len(grid_p_full_softmax_matrix)), grid_p_full_softmax_matrix.argmax(1) ] = 1                         # set the max value in each row to 1, all others left at zero
                grid_tile_winners_totals_by_class          = np.transpose(np.expand_dims( binary_matrix.sum( axis=0 ), axis=1 ))               # same, but 'winner take all' at the tile level
                aggregate_tile_probabilities_matrix[index] = grid_tile_probabs_totals_by_class
                aggregate_tile_level_winners_matrix[index] = grid_tile_winners_totals_by_class + random.uniform( 0.001, 0.01)                  # necessary to make all the tile totals unique when we go looking for them later. ugly but it works

                if DEBUG>0:
                  np.set_printoptions(formatter={'float': lambda x:   "{:>6.2f}".format(x)})
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             aggregate_tile_probabilities_matrix[{ASPARAGUS}{index}{RESET}]                         = {BOLD}{ASPARAGUS}{aggregate_tile_probabilities_matrix[index]}{RESET}", flush=True  )
                  print ( f"{ASPARAGUS}CLASSI:         INFO:      test:             aggregate_tile_level_winners_matrix[{ASPARAGUS}{index}{RESET}]                         = {BOLD}{ASPARAGUS}{aggregate_tile_level_winners_matrix[index]}{RESET}",  flush=True  )
                if DEBUG>2:
                  np.set_printoptions(formatter={'float': lambda x:   "{:>6.2f}".format(x)})
                  print ( f"CLASSI:         INFO:      test:             grid_p_full_softmax_matrix{ASPARAGUS}{RESET}                                     = \n{BOLD}{ASPARAGUS}{grid_p_full_softmax_matrix}{RESET}", flush=True  )
                  print ( f"CLASSI:         INFO:      test:             aggregate_tile_probabilities_matrix{ASPARAGUS}{RESET}                            = \n{BOLD}{ASPARAGUS}{aggregate_tile_probabilities_matrix}{RESET}", flush=True  )


            global_batch_count+=1
          
            if global_batch_count%(args.supergrid_size**2)==0:
              
              if args.input_mode=='image':
                print("")
                
                if args.annotated_tiles=='True':
                  
                  fig=plot_classes_preds(args, model, tile_size, grid_images, grid_labels, 0,  grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_full_softmax_matrix, class_names, class_colours )
                  writer.add_figure('1 annotated tiles', fig, epoch)
                  plt.close(fig)
  
                batch_fnames_npy = batch_fnames.numpy()                                                      # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details
                fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"
                
                try:
                  background_image = np.load(f"{fq_link}")
                  if DEBUG>99:
                    print ( f"CLASSI:         INFO:      test:        fq_link                                              = {MIKADO}{fq_link}{RESET}" )
                  
                except Exception as e:
                  print ( f"{RED}CLASSI:         FATAL:  '{e}'{RESET}" )
                  print ( f"{RED}CLASSI:         FATAL:     explanation: a required {MAGENTA}entire_patch.npy{RESET}{RED} file doesn't exist. (Probably none exist). These contain the background images used for the scattergram. {RESET}" )                
                  print ( f"{RED}CLASSI:         FATAL:     if you used {CYAN}./just_test_dont_tile.sh{RESET}{RED} without first running {CYAN}./just_test.sh{RESET}{RED}' then tiling and patch generation will have been skipped ({CYAN}--skip_tiling = {MIKADO}'True'{RESET}{RED} in that script{RESET}{RED}){RESET}" )
                  print ( f"{RED}CLASSI:         FATAL:     if so, run '{CYAN}./just_test.sh -d <cancer type code> -i <INPUT_MODE>{RESET}{RED}' at least one time so that these files will be generated{RESET}" )                 
                  print ( f"{RED}CLASSI:         FATAL:     halting now ...{RESET}" )                 
                  sys.exit(0)              

                  
                if args.scattergram=='True':
                  
                  plot_scatter( args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_preds, p_full_softmax_matrix, show_patch_images='True')
                  # ~ plot_scatter(args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_preds, p_full_softmax_matrix, show_patch_images='False')
  
                if (args.probs_matrix=='True') & (args.multimode!='image_rna'):
                  
                  # ~ # without interpolation
                  # ~ matrix_types = [ 'margin_1st_2nd', 'confidence_RIGHTS', 'p_std_dev' ]
                  # ~ for n, matrix_type in enumerate(matrix_types):
                    # ~ plot_matrix (matrix_type, args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_p_full_softmax_matrix, grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_true_class, 'none' )    # always display without probs_matrix_interpolation 
                  # with  interpolation
                  matrix_types = [ 'probs_true' ]
                  for n, matrix_type in enumerate(matrix_types): 
                    plot_matrix ( matrix_type, args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_p_full_softmax_matrix, grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_true_class, args.probs_matrix_interpolation )
           
  




        if loss_type!='mean_squared_error':        

          # move to a separate function ----------------------------------------------------------------------------------------------
          if ( args.input_mode=='rna' ) & ( args.just_test=='True' ):
            
            preds, p_full_softmax_matrix, p_highest, p_2nd_highest, p_true_class = analyse_probs( y2_hat, rna_labels_values )
        
            batch_index_lo = i*batch_size
            batch_index_hi = batch_index_lo + batch_size
                                  
            if DEBUG>88:
              print ( f"\n\n" )
              print ( f"CLASSI:         INFO:      test: batch                            = {BLEU}{i+1}{RESET}"                           )
              print ( f"CLASSI:         INFO:      test: count                            = {BLEU}{(i+1)*batch_size}{RESET}"              ) 
              print ( f"CLASSI:         INFO:      test: probabilities_matrix.shape       = {BLEU}{probabilities_matrix.shape}{RESET}"    )                                    
              print ( f"CLASSI:         INFO:      test: p_full_softmax_matrix.shape      = {BLEU}{p_full_softmax_matrix.shape}{RESET}"   )                                    
              print ( f"CLASSI:         INFO:      test: batch_index_lo                   = {BLEU}{batch_index_lo}{RESET}"                )                                    
              print ( f"CLASSI:         INFO:      test: batch_index_hi                   = {BLEU}{batch_index_hi}{RESET}"                )                                    
            
            probabilities_matrix [batch_index_lo:batch_index_hi] = p_full_softmax_matrix # + random.uniform( 0.001, 0.01)                      # 'p_full_softmax_matrix' contains probs for an entire mini-batch; 'probabilities_matrix' has enough room for all cases
            true_classes         [batch_index_lo:batch_index_hi] = rna_labels_values
            rna_case_id          [batch_index_lo:batch_index_hi] = batch_fnames_npy[0:batch_size]
  
            if DEBUG>55:
              show_last=16
              np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})       
              print ( f"CLASSI:         INFO:      test: last {AMETHYST}{show_last}{RESET} entries in probabilities_matrix[{MIKADO}{batch_index_lo}{RESET}:{MIKADO}{batch_index_hi}{RESET}]     = \n{AMETHYST}{probabilities_matrix [args.n_samples[0]-show_last:args.n_samples[0]]}{RESET}"                       ) 
              np.set_printoptions(formatter={'int': lambda x: "{:^7d}".format(x)})   
              print ( f"CLASSI:         INFO:      test: true_classes                       [{MIKADO}{batch_index_lo}{RESET}:{MIKADO}{batch_index_hi}{RESET}] =   {AMETHYST}{true_classes         [batch_index_lo          :batch_index_hi]}{RESET}"        )           
              print ( f"CLASSI:         INFO:      test: rna_case_id                        [{MIKADO}{batch_index_lo}{RESET}:{MIKADO}{batch_index_hi}{RESET}] =   {AMETHYST}{rna_case_id          [batch_index_lo          :batch_index_hi]}{RESET}"        )   
  
           # move to a separate function ----------------------------------------------------------------------------------------------
  
        




        if (args.input_mode=='image'):
          loss_images       = loss_function(y1_hat, image_labels)
          loss_images_value = loss_images.item()                                                             # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories
 
          if DEBUG>2:
            print ( f"CLASSI:         INFO:      test: {COQUELICOT}loss_images{RESET} (for this mini-batch)  = {PURPLE}{loss_images_value:6.3f}{RESET}" )
            time.sleep(.25)
             
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):

          if loss_type == 'mean_squared_error':                                                                           # autoencoders use mean squared error. The function needs to be provided with both the input and the output to calculate mean_squared_error 
            loss_genes        = loss_function( y2_hat, batch_genes.squeeze() )            
          else:
            loss_genes        = loss_function( y2_hat, rna_labels            )

          loss_genes_value  = loss_genes.item()                                                              # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories


        # ~ l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0
        
        if (args.input_mode=='image'):
          total_loss        = loss_images_value + l1_loss
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          total_loss        = loss_genes_value + l1_loss    
        


        if DEBUG>0:
          if ( args.input_mode=='image' ):
            offset=162
            print ( f"\
\033[2K\r\033[27Ctest:\
\r\033[40C{DULL_WHITE}n={i+1:>3d}{CLEAR_LINE}\
\r\033[49Craw loss_images={loss_images_value:5.2f}\
\r\033[120CBATCH LOSS                (LOSS PER 1000 TILES) = \r\033[{offset+10*int((total_loss*5)//1) if total_loss<1 else offset+16*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss*1000/batch_size:6.1f}{RESET}" )
            print ( "\033[2A" )
          elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
            print ( f"\
\033[2K\r\033[27Ctest:\
\r\033[40C{DULL_WHITE}n={i+1:>3d}{CLEAR_LINE}\
\r\033[73Craw loss_rna={loss_genes_value:5.2f}\
\r\033[120CBATCH LOSS                (LOSS PER 1000 EXAMPLES) = \r\033[{offset+10*int((total_loss*5)//1) if total_loss<1 else offset+16*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss*1000/batch_size:6.1f}{RESET}" )
            print ( "\033[2A" )


        if args.input_mode=='image':
          loss_images_sum      += loss_images_value
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          loss_genes_sum       += loss_genes_value
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss    

        if ( args.input_mode=='image'):   
          del loss_images
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):       
          del loss_genes
        torch.cuda.empty_cache()

    #    ^^^^^^^^^  epoch complete  ^^^^^^^^^^^^^^^



    
    if loss_type=='mean_squared_error':
      print ( "" )

    if loss_type == 'mean_squared_error':
                      
      print ( "\n\n\n" )
      
    else:                                                                                                  # the following only make sense if we are classifying, so skip for autoencoders                  

      if epoch % 1 == 0:                                                                                   # every ... epochs, do an analysis of the test results and display same
        
        if args.input_mode=='image':      
          y1_hat_values             = y1_hat.cpu().detach().numpy()
          y1_hat_values_max_indices = np.argmax( np.transpose(y1_hat_values), axis=0 )                     # indices of the highest values of y1_hat = highest probability class
  
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):      
          y2_hat_values             = y2_hat.cpu().detach().numpy()
          y2_hat_values_max_indices = np.argmax( np.transpose(y2_hat_values), axis=0 )                     # indices of the highest values of y2_hat = highest probability class
      
        
        image_labels_values       = image_labels.cpu().detach().numpy()
        rna_labels_values         =   rna_labels.cpu().detach().numpy()
  
        torch.cuda.empty_cache()
        
        if DEBUG>2:
          print ( "CLASSI:         INFO:      test:        y1_hat.shape                      = {:}".format( y1_hat.shape                     ) )
          print ( "CLASSI:         INFO:      test:        y1_hat_values_max_indices.shape   = {:}".format( y1_hat_values_max_indices.shape  ) )
          print ( "CLASSI:         INFO:      test:        image_labels_values.shape         = {:}".format( image_labels_values.shape        ) )
          print ( "CLASSI:         INFO:      test:        rna_labels_values.shape           = {:}".format(   rna_labels_values.shape        ) )
        
        number_to_display= 9 if args.dataset=='tcl' else batch_size
        np.set_printoptions(linewidth=10000)   
        np.set_printoptions(edgeitems=10000)
        np.set_printoptions(threshold=10000)      
        print ( "" )
        
        if args.input_mode=='image':          
          correct=np.sum( np.equal(y1_hat_values_max_indices, image_labels_values) )
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):         
          correct=np.sum( np.equal(y2_hat_values_max_indices, rna_labels_values)   )
              
        pct=100*correct/batch_size if batch_size>0 else 0
        global_pct = 100*(global_correct_prediction_count+correct) / (global_number_tested+batch_size) 
        if show_all_test_examples==False:
          print ( f"{CLEAR_LINE}                           test: truth/prediction for first {MIKADO}{number_to_display}{RESET} examples from the most recent test batch \
    ( number correct this batch: {correct}/{batch_size} \
    = {MAGENTA if pct>=90 else BRIGHT_GREEN if pct>=80 else PALE_GREEN if pct>=70 else ORANGE if pct>=60 else WHITE if pct>=50 else DULL_WHITE}{pct:>3.0f}%{RESET} )  \
    ( number correct overall: {global_correct_prediction_count+correct}/{global_number_tested+batch_size} \
    = {MAGENTA if global_pct>=90 else BRIGHT_GREEN if global_pct>=80 else PALE_GREEN if global_pct>=70 else ORANGE if global_pct>=60 else WHITE if global_pct>=50 else DULL_WHITE}{global_pct:>3.0f}%{RESET} {DIM_WHITE}(number tested this run = epochs x test batches x batch size){RESET}" )
        else:
          run_level_total_correct.append( correct )
          print ( f"{CLEAR_LINE}                           test: truth/prediction for {MIKADO}{number_to_display}{RESET} test examples \
    ( number correct  - all test examples - this run: {correct}/{batch_size} \
    = {MAGENTA if pct>=90 else PALE_GREEN if pct>=80 else ORANGE if pct>=70 else GOLD if pct>=60 else WHITE if pct>=50 else DIM_WHITE}{pct:>3.0f}%{RESET} )  \
    ( number correct  - all test examples - cumulative over all runs: {global_correct_prediction_count+correct}/{global_number_tested}  \
    = {MAGENTA if global_pct>=90 else PALE_GREEN if global_pct>=80 else ORANGE if global_pct>=70 else GOLD if global_pct>=60 else BLEU if global_pct>=50 else DULL_WHITE}{global_pct:>3.0f}%{RESET} )" )
  
        if args.input_mode=='image':   
          labs   = image_labels_values       [0:number_to_display] 
          preds  = y1_hat_values_max_indices [0:number_to_display]
          delta  = np.abs(preds - labs)
          if len(class_names)<10:
            np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>1d}{RESET}"})
          else:
            np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>02d}{RESET}"})
          print (  f"truth = {CLEAR_LINE}{labs}{RESET}",  flush=True   )
          degenerate_result = np.all(preds == preds[0])
          if len(class_names)<10:
            np.set_printoptions(formatter={'int': lambda x: f"{BOLD_MAGENTA if degenerate_result else DIM_WHITE}{x:>1d}{RESET}"})
          else:
            np.set_printoptions(formatter={'int': lambda x: f"{BOLD_MAGENTA if degenerate_result else BOLD_MAGENTA}{x:>02d}{RESET}"})
          if DEBUG>190:
            print ( f"{SAVE_CURSOR}\033[0;0H{PINK}preds[0] = {preds[0]}; degenerate_result = np.all(preds==preds[0]) = {np.all(preds==preds[0])}            {RESET}{RESTORE_CURSOR}", end='', flush=True )
          print (  f"preds = {CLEAR_LINE}{preds}{BOLD_ORANGE if degenerate_result else BLACK}   <<< warning !!! degenerate result{RESET}", flush=True   )
          if len(class_names)<10:
            GAP=' '
            np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else BLACK}{x:>1d}{RESET}"}) 
          else:
             np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else BLACK}{x:>2d}{RESET}"})
             GAP='  '
          print (  f"{CLEAR_LINE}         ",  end='', flush=True  )        
          for i in range( 0, len(delta) ):                                                                 # should have been able to do this with a fancy list comprehension but I couldn't get it to work
            if delta[i]==0:                                                                                   
              print (  f"{GREEN}\u2713{GAP}", end='', flush=True  )
            else:
              print (  f"{RED}\u2717{GAP}",   end='', flush=True  )          
          print ( f"{RESET}" )
  
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):   
          labs   = rna_labels_values         [0:number_to_display]
          preds  = y2_hat_values_max_indices [0:number_to_display]
          delta  = np.abs(preds - labs)
          if len(class_names)<10:
            np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>1d}{RESET}"})
            GAP=' '
          else:
            np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>02d}{RESET}"})
            GAP='  '
          print (  f"truth = {CLEAR_LINE}{labs}",  flush=True   )
          print (  f"preds = {CLEAR_LINE}{preds}", flush=True   )
          if len(class_names)<10:
            np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else BLACK}{x:>1d}{RESET}"}) 
          else:
             np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else BLACK}{x:>2d}{RESET}"}) 
          
          print (  f"{CLEAR_LINE}         ",  end='', flush=True  )        
          for i in range( 0, len(delta) ):                                                                 # should have been able to do this with a fancy list comprehension but I couldn't get it to work
            if delta[i]==0:                                                                                   
              print (  f"{BRIGHT_GREEN}\u2713{GAP}", end='', flush=True  )
            else:
              print (  f"{RED}\u2717{GAP}",   end='', flush=True  )          
          print ( f"{RESET}" )
  
  
        # ~ if ( args.just_test!='True') | ( (args.just_test=='true')  &  (args.input_mode=='image_rna') & (args.multimode=='image_rna') ):
         # grab test stats produced during training
        for i in range(0, len(preds) ):
          run_level_classifications_matrix[ labs[i], preds[i] ] +=1
  
        if DEBUG>8:
          print ( run_level_classifications_matrix, flush=True )
          #time.sleep(3)
  
        if DEBUG>9:
          print ( "CLASSI:         INFO:      test:       y1_hat.shape                     = {:}".format( y1_hat_values.shape          ) )
          np.set_printoptions(formatter={'float': lambda x: "{0:10.2e}".format(x)})
          print (  "{:}".format( (np.transpose(y1_hat_values))[:,:number_to_display] )  )
          np.set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})
  
        if DEBUG>2:
          number_to_display=16  
          print ( "CLASSI:         INFO:      test:       FIRST  GROUP BELOW: y1_hat"                                                                      ) 
          print ( "CLASSI:         INFO:      test:       SECOND GROUP BELOW: y1_hat_values_max_indices (prediction)"                                      )
          print ( "CLASSI:         INFO:      test:       THIRD  GROUP BELOW: image_labels_values (truth)"                                                 )
          np.set_printoptions(formatter={'float': '{: >6.2f}'.format}        )
          print ( f"{(np.transpose(y1_hat_values)) [:,:number_to_display] }" )
          np.set_printoptions(formatter={'int': '{: >6d}'.format}            )
          print ( " {:}".format( y1_hat_values_max_indices    [:number_to_display]        ) )
          print ( " {:}".format( image_labels_values          [:number_to_display]        ) )
  
  
        pplog.log(f"epoch = {epoch}" )
        pplog.log(f"test: truth/prediction for first {number_to_display} examples from the most recent test batch ( number correct this batch: {correct}/{batch_size} = {pct:>3.0f}%  )  ( number correct overall: {global_correct_prediction_count+correct}/{global_number_tested+batch_size} = {global_pct:>3.0f}% (number tested this run = epochs x test batches x batch size)" )
        pplog.log(f"{CLEAR_LINE}        truth = {labs}"  )
        pplog.log(f"{CLEAR_LINE}        preds = {preds}" )
        pplog.log(f"{CLEAR_LINE}        delta = {delta}" ) 
        if degenerate_result:
         pplog.log(f"{CLEAR_LINE}        warning !!! degenerate result")  
 

    if args.input_mode=='image':   
      y1_hat_values               = y1_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y1_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y1_hat_values_max_indices   = np.argmax( np.transpose(y1_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
    elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
      y2_hat_values               = y2_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y2_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y2_hat_values_max_indices   = np.argmax( np.transpose(y2_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
      
    
    image_labels_values         = image_labels.cpu().detach().numpy()                                      # these are the true      classes corresponding to batch_images


    if loss_type != 'mean_squared_error':                                                                                 # the following analysis only make sense if we are classifying, so skip for autoencoders 
      if args.input_mode=='image':
        correct_predictions              = np.sum( y1_hat_values_max_indices == image_labels_values )
      elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
        correct_predictions              = np.sum( y2_hat_values_max_indices == rna_labels_values )
  
  
      pct_correct                 = correct_predictions / batch_size * 100
  
    loss_images_sum_ave = loss_images_sum / (i+1)                                                          # average batch loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    loss_genes_sum_ave  = loss_genes_sum  / (i+1)                                                          # average genes loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    l1_loss_sum_ave     = l1_loss_sum     / (i+1)                                                          # average l1    loss for the entire epoch (divide cumulative loss by number of batches in the epoch)
    total_loss_sum_ave  = total_loss_sum  / (i+1)                                                          # average total loss per batch for the entire epoch (divide cumulative loss by number of batches in the epoch)

    if total_loss_sum    <  test_loss_min:
       test_loss_min     =  total_loss_sum

    test_loss_trunc_10            = total_loss_sum_ave   if (total_loss_sum_ave < 10 )    else 10          # get rid of the spike that usually occurs on the first epoch or two
    test_loss_trunc_1             = total_loss_sum_ave   if (total_loss_sum_ave < 1  )     else 1          # ditto; to make it easy to get a close up look of low values on the chart
    test_loss_trunc_01            = total_loss_sum_ave   if (total_loss_sum_ave < 0.1)     else 0.1        # ditto; to make it easy to get a close up look of low values on the chart
    normalised_test_loss          = total_loss_sum_ave * 1000 / batch_size                                 # dividing by the batch_size makes it loss per tile.  Multiplying by 1000 makes it loss per 1000 tiles
    normalised_test_loss_trunc_10 = normalised_test_loss if (normalised_test_loss<10)  else 10
    normalised_test_loss_peak_10  = normalised_test_loss if (normalised_test_loss<10)  else 0
    normalised_test_loss_trunc_1  = normalised_test_loss if (normalised_test_loss<1)   else 1
    normalised_test_loss_peak_1   = normalised_test_loss if (normalised_test_loss<1)   else 0
    pct_correct_over_40           = pct_correct          if (pct_correct>=40)          else 0
    pct_correct_over_50           = pct_correct          if (pct_correct>=50)          else 0

    if loss_type != 'mean_squared_error':                                                                  # the following only make sense if we are classifying, so skip for autoencoders 

      if correct_predictions    >  max_correct_predictions:
        max_correct_predictions =  correct_predictions
  
      if pct_correct       >  max_percent_correct:
        max_percent_correct    =  pct_correct
            
      writer.add_scalar( '1a_ave_batch_test_loss',                                 total_loss_sum_ave,               epoch )
      writer.add_scalar( '1b_ave_batch_test_loss_trunc_10',                        test_loss_trunc_10,               epoch )
      writer.add_scalar( '1c_ave_batch_test_loss_trunc_1',                         test_loss_trunc_1,                epoch )
      writer.add_scalar( '1d_ave_batch_test_loss_trunc_01',                        test_loss_trunc_01,               epoch )
      writer.add_scalar( '1e_ave_batch_test_loss_per_1000_tiles',                  normalised_test_loss,             epoch )
      writer.add_scalar( '1f_ave_batch_test_loss_per_1000_tiles_trunc_10',         normalised_test_loss_trunc_10,    epoch )
      writer.add_scalar( '1g_ave_batch_test_loss_per_1000_tiles_peak_10',          normalised_test_loss_peak_10,     epoch )
      writer.add_scalar( '1h_ave_batch_test_loss_per_1000_tiles_trunc_1',          normalised_test_loss_trunc_1,     epoch )
      writer.add_scalar( '1i_ave_batch_test_loss_per_1000_tiles_peak_1',           normalised_test_loss_peak_1,      epoch )
      writer.add_scalar( '1j_ave_batch_test_loss____minimums',                     test_loss_min/(i+1),              epoch )    
      writer.add_scalar( '1k_num_correct',                                         correct_predictions,              epoch )
      writer.add_scalar( '1l_num_correct_max',                                     max_correct_predictions,          epoch )
      writer.add_scalar( '1m_pct_correct',                                         pct_correct,                      epoch ) 
      writer.add_scalar( '1n_just_over_40_pct_correct',                            pct_correct_over_40,              epoch ) 
      writer.add_scalar( '1o_just_over_50_pct_correct',                            pct_correct_over_50,              epoch ) 
      writer.add_scalar( '1p_max_percent_correct',                                 max_percent_correct,              epoch ) 
    
    else:                                                                                                  # only these learning curves are relevant for autoencoders
      writer.add_scalar( '1a_ave_batch_test_loss',                                 total_loss_sum_ave,               epoch )
      writer.add_scalar( '1b_ave_batch_test_loss_trunc_10',                        test_loss_trunc_10,               epoch )
      writer.add_scalar( '1c_ave_batch_test_loss_trunc_1',                         test_loss_trunc_1,                epoch )
      writer.add_scalar( '1d_ave_batch_test_loss_trunc_01',                        test_loss_trunc_01,               epoch )
      writer.add_scalar( '1e_ave_batch_test_loss_per_1000_tiles',                  normalised_test_loss,             epoch )
      writer.add_scalar( '1f_ave_batch_test_loss_per_1000_tiles_trunc_10',         normalised_test_loss_trunc_10,    epoch )
      writer.add_scalar( '1g_ave_batch_test_loss_per_1000_tiles_peak_10',          normalised_test_loss_peak_10,     epoch )
      writer.add_scalar( '1h_ave_batch_test_loss_per_1000_tiles_trunc_1',          normalised_test_loss_trunc_1,     epoch )
      writer.add_scalar( '1i_ave_batch_test_loss_per_1000_tiles_peak_1',           normalised_test_loss_peak_1,      epoch )
      writer.add_scalar( '1j_ave_batch_test_loss____minimums',                     test_loss_min/(i+1),              epoch )   
      
    
    if DEBUG>9:
      print ( "CLASSI:         INFO:      test:             batch_images.shape                       = {:}".format( batch_images.shape ) )
      print ( "CLASSI:         INFO:      test:             image_labels.shape                       = {:}".format( image_labels.shape ) )
      
#    if not args.just_test=='True':
#      if args.input_mode=='image':
#        writer.add_figure('Predictions v Truth', plot_classes_preds(args, model, tile_size, batch_images, image_labels, preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours), epoch)

    if loss_type != 'mean_squared_error':                                                                                 # the following only make sense if we are classifying, so skip for autoencoders 
  
      if args.just_test=='False':                                                                          # This call to plot_classes_preds() is for use by test during training, and not for use in "just_test" mode (the latter needs support for supergrids)
        if args.annotated_tiles=='True':
          fig=plot_classes_preds(args, model, tile_size, batch_images.cpu().numpy(), image_labels.cpu().numpy(), batch_fnames.cpu().numpy(), preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours)
          writer.add_figure('Predictions v Truth', fig, epoch)
          plt.close(fig)

    if (args.input_mode=='image'):
      del batch_images
      del image_labels

    
#    if args.just_test=='True':
#      if args.input_mode=='image':
#        it=list(permutations( range(0, batch_size)  ) )
#        writer.add_figure('Predictions v Truth', plot_classes_preds(args, model, tile_size, batch_images, image_labels, preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours), epoch)

    if args.multimode!="image_rna":
      embedding               = 0

    if loss_type == 'mean_squared_error':                                                                                 # these weren't applicable for mean squared error
      correct_predictions     = 0
      max_correct_predictions = 0
      max_percent_correct     = 0
      
      
    return embeddings_accum, labels_accum, loss_images_sum_ave, loss_genes_sum_ave, l1_loss_sum_ave, total_loss_sum_ave, correct_predictions, batch_size, max_correct_predictions, max_percent_correct, test_loss_min, embedding



# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def determine_top_up_factors ( args, n_classes, class_names, n_tiles, case_designation_flag ):

  #  Count the number images per cancer subtype in the dataset. This will be used in generate() to balance tiling so that all subtypes will be represented by the same number of tiles
  #  The balancing is performed in 'generate()', however the counts must be performed now, before tiling, because the results are used to adjust the number of tiles extracted
  #  Specifically, we count the number of images per subtype for the chosen subset (e.g. UNIMODE_CASE) and from this calculate 'top up factors' which are used in generate() 
  #  to increase the number of tiles extracted for subtypes which have fewer images than the subtype with the most number of cases (images)  

  class_counts = determine_class_counts ( args, n_classes, class_names, n_tiles, case_designation_flag )
  
  if args.make_balanced!='True':                                                                           # cater for the default case

    top_up_factors           = np.ones(len(class_names) )                                                  # top_up_factors are all 1 if we aren't going to balance the dataset
    tiles_needed_per_example = n_tiles
    total_slides_counted     = np.sum(class_counts)                                     
    total_tiles_required     = total_slides_counted * n_tiles

    if case_designation_flag!='UNIMODE_CASE____IMAGE_TEST':
      row    = 0
      colour = BLEU
      col    = 100
    else:
      row    = 0
      col    = 200
      colour = ASPARAGUS
    
    if args.just_test=='True':
      row    = 0
      col    = 200
      colour = CAMEL  
    
    if DEBUG>0:
      np.set_printoptions(formatter={'int':   lambda x: "{:>6d}".format(x)})
      print( f"\033[{row+1};{col}f{CLEAR_LINE}INFO:         {colour}{case_designation_flag}{RESET}", flush=True  )
      print( f"\033[{row+2};{col}f{CLEAR_LINE}INFO:           final class_counts           = {colour}{class_counts}{RESET}",                               flush=True  )
      print( f"\033[{row+3};{col}f{CLEAR_LINE}INFO:           total slides counted         = {colour}{total_slides_counted}{RESET}",                       flush=True  )      
      np.set_printoptions(formatter={'float': lambda x: "{:6.2f}".format(x)})
      print( f"\033[{row+4};{col}f{CLEAR_LINE}{BOLD}   INFO:           top up factors               = {colour}{top_up_factors}{RESET}  ",                             flush=True  )
      print( f"\033[{row+5};{col}f{CLEAR_LINE}INFO:                                         {ORANGE}^^^ note that {CYAN}{BOLD}MAKE_BALANCED{RESET}{ORANGE} is disabled ^^^{RESET}  ",                             flush=True  )
      np.set_printoptions(formatter={'int':   lambda x: "{:>6d}".format(x)})
      print( f"\033[{row+6};{col}f{CLEAR_LINE}INFO:           tiles_needed_per_example     = {colour}{tiles_needed_per_example}{RESET}",                   flush=True  )
      print( f"\033[{row+7};{col}f{CLEAR_LINE}INFO:           hence tiles per subtype      = {colour}{tiles_needed_per_example * class_counts}{RESET}",                   flush=True  )
      print( f"\033[{row+8};{col}f{CLEAR_LINE}INFO:           total_tiles_required         = {colour}{total_tiles_required:,}{RESET}",                    flush=True  )

  else:

    col=0
    if case_designation_flag!='UNIMODE_CASE____IMAGE_TEST':
      row    = 0
      colour = AMETHYST
      leader = "CLASSI: (level up info)      "
    else:
      row    = 0
      col    = col+150
      colour = ORANGE
      leader = "  " 
    if args.just_test=='True':
      row    = 0
      col    = col+150
      colour = CHARTREUSE
      leader = "  " 
  
      
    if DEBUG>0:
      np.set_printoptions(formatter={'int':   lambda x: "{:>6d}".format(x)})
      print( f"\033[{row+1};{col}f{CLEAR_LINE}{leader}{colour}{case_designation_flag}{RESET}",                                                       flush=True  )
      print( f"\033[{row+2};{col}f{CLEAR_LINE}{leader}final class_counts           = {colour}{class_counts}{RESET}",                               flush=True  )
      print( f"\033[{row+3};{col}f{CLEAR_LINE}{leader}total slides counted         = {colour}{np.sum(class_counts)}{RESET}",                       flush=True  )
  
    if np.any( class_counts < 1):
        print ( f"{BOLD}{RED}\033[75;0HCLASSI:       FATAL: one of the subtypes has no examples{CLEAR_LINE}",                                                                                                                              flush=True  )                                        
        print ( f"{BOLD}{RED}CLASSI:       FATAL: {CYAN}class_counts{RESET}{BOLD}{RED} are {MIKADO}{class_counts}{BOLD}{RED} for class names (subtypes) {MIKADO}{class_names}{BOLD}{RED} respectively{RESET}{CLEAR_LINE}",                 flush=True  )                                        
        print ( f"{BOLD}{RED}CLASSI:       FATAL: possible remedy (i):  it could be that all cases were allocated to just the training or just the test set. Re-run the experiment with option {CYAN}-v {RESET}{BOLD}{RED} set to {CYAN}True{RESET}{BOLD}{RED} to have cases re-divided and flagged{RESET}{CLEAR_LINE}",       flush=True  )                                        
        print ( f"{BOLD}{RED}CLASSI:       FATAL: possible remedy (ii): if al else failes, remove any class or classes that has only a tiny number of examples from the applicable master spreadsheet",                                    flush=True  )                                        
        print ( f"{BOLD}{RED}CLASSI:       FATAL: cannot continue - halting now{RESET}{CLEAR_LINE}" )                 
        sys.exit(0)
  
    relative_ratios = class_counts/np.max(class_counts)
  
    if DEBUG>0:
      np.set_printoptions(formatter={'float': lambda x: "{:6.2f}".format(x)})
      print( f"\033[{row+4};{col}f{CLEAR_LINE}{leader}relative class ratios        = {colour}{relative_ratios}{RESET}",                            flush=True  )
  
    top_up_factors           = np.divide(1,relative_ratios)
    # ~ tiles_needed_per_example = (top_up_factors*n_tiles).astype(int) + 1                                    # add one extra to be safe 
    tiles_needed_per_example = (top_up_factors*n_tiles).astype(int) 
    total_slides_counted     = np.sum(class_counts)                                     
    tiles_needed_per_example = (top_up_factors*n_tiles).astype(int) 
    tiles_needed_per_subtype = tiles_needed_per_example * class_counts
    total_tiles_required     = np.sum(tiles_needed_per_subtype)
  
    if DEBUG>0:
      np.set_printoptions(formatter={'float': lambda x: "{:6.2f}".format(x)})
      print( f"\033[{row+5};{col}f{CLEAR_LINE}{leader}top up factors               = {colour}{top_up_factors}{RESET}",                             flush=True  )
      np.set_printoptions(formatter={'int':   lambda x: "{:>6d}".format(x)})
      print( f"\033[{row+6};{col}f{CLEAR_LINE}{leader}tiles_needed_per_example     = {colour}{tiles_needed_per_example}{RESET}",                   flush=True  )
      print( f"\033[{row+7};{col}f{CLEAR_LINE}{leader}tiles_needed_per_subtype     = {colour}{tiles_needed_per_subtype}{RESET}",    flush=True  )
      print( f"\033[{row+8};{col}f{CLEAR_LINE}{leader}total_tiles_required         = {colour}{total_tiles_required:,}{RESET}",                    flush=True  )

  return total_slides_counted, total_tiles_required, top_up_factors


# ---------------------------------------------------------------------------------------------------------
def determine_class_counts ( args, n_classes, class_names, n_tiles, case_designation_flag ):

  #  Count the number images per cancer subtype in the dataset for the designated case type (e.g. UNIMODE_CASE) 
   

  class_counts          = np.zeros( n_classes, dtype=int )

  for dir_path, dirs, files in os.walk( args.data_dir ):                                                   # each iteration takes us to a new directory under data_dir

    for d in dirs:

      if not (d==args.data_dir):                                                                                # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data
  
        fqn = f"{dir_path}/{d}/{case_designation_flag}"
        if DEBUG>100:
          print ( f"\n\nCLASSI:         INFO:   fqn         {YELLOW}{fqn}{RESET}")

        count_this_case_flag=False

        try:
          fqn = f"{dir_path}/{d}/{case_designation_flag}"
          f = open( fqn, 'r' )
          count_this_case_flag=True
          if DEBUG>100:
            print ( f"\r{CLEAR_LINE}{DULL_WHITE}CLASSI:         INFO:  determine_class_counts() case  '{CYAN}{fqn}{RESET}{GREEN}' \r\033[130C is     a case flagged as '{CYAN}{case_designation_flag}{RESET}{GREEN}' - - including{RESET}{CLEAR_LINE}",  flush=True )
        except Exception:
          if DEBUG>100:
            print ( f"\r{CLEAR_LINE}{DULL_WHITE}CLASSI:         INFO:  determine_class_counts() case  '{CYAN}{fqn}{RESET}{RED} \r\033[130C is not a case flagged as '{CYAN}{case_designation_flag}{RESET}{RED}' - - skipping{RESET}{CLEAR_LINE}",  flush=True )
  
        try:                                                                                               # every tile has an associated label - the same label for every tile image in the directory
          label_file = f"{dir_path}/{d}/{args.class_numpy_file_name}"
          if DEBUG>100:
            print ( f"CLASSI:         INFO:   label_file  {ASPARAGUS}{label_file}{RESET}")
          label      = np.load( label_file )
        except Exception as e:
          print ( f"{RED}CLASSI:               FATAL: when processing: '{label_file}' (19263){RESET}", flush=True)        
          print ( f"{RED}CLASSI:                      reported error was: '{e}'{RESET}", flush=True)
          print ( f"{RED}CLASSI:                      halting now{RESET}", flush=True)
          sys.exit(0)

        if label[0]>args.highest_class_number:
          count_this_case_flag=False
          if DEBUG>2:
            print ( f"{ORANGE}CLASSI:         INFO: label is greater than '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{ORANGE}' - - not counting this example (label = {MIKADO}{label[0]}{RESET}{ORANGE}){RESET}"      )
          pass

        
        if ( count_this_case_flag==True ):
          class_counts[label[0]]+=1
          
          if DEBUG>100:
            print( f"CLASSI:         INFO:     class_counts   = {MIKADO}{class_counts}{RESET}", flush=True  )
            
  return class_counts


# ---------------------------------------------------------------------------------------------------------

def segment_cases( args, n_classes, class_names, n_tiles, pct_test ):


  # (1A) analyse dataset directory

  if args.use_unfiltered_data==True:
    rna_suffix = args.rna_file_suffix[1:]
  else:
    rna_suffix = args.rna_file_reduced_suffix
    
  cumulative_svs_file_count   = 0
  cumulative_tif_file_count   = 0
  cumulative_spcn_file_count  = 0
  cumulative_png_file_count   = 0
  cumulative_rna_file_count   = 0
  cumulative_other_file_count = 0
  dir_count                   = 0
  
  for dir_path, dirs, files in os.walk( args.data_dir ):                                                   # each iteration takes us to a new directory under data_dir

    if not (dir_path==args.data_dir):                                                                      # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data      
      
      dir_count += 1
      svs_file_count     = 0
      spcn_file_count    = 0
      tif_file_count     = 0
      rna_file_count     = 0
      png_file_count     = 0
      other_file_count   = 0
      

      for f in sorted( files ):
       
        if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))   ):
          svs_file_count              +=1
          cumulative_svs_file_count   +=1
        if (   ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))  ):
          tif_file_count             +=1
          cumulative_tif_file_count  +=1
        elif  ( f.endswith( 'spcn' ) ):
          spcn_file_count             +=1
          cumulative_spcn_file_count  +=1
        elif  ( f.endswith( 'png' ) ):
          png_file_count              +=1
          cumulative_png_file_count   +=1
        elif  ( f.endswith( rna_suffix ) ):
          rna_file_count              +=1
          cumulative_rna_file_count   +=1
        else:
          other_file_count            +=1
          cumulative_other_file_count +=1
        

  if DEBUG>0:
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():  summary of data files:{RESET}",                                          flush=True  )
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    directories count  =  {MIKADO}{dir_count:<6d}{RESET}",                   flush=True  )
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    svs   file  count  =  {MIKADO}{cumulative_svs_file_count:<6d}{RESET}",   flush=True  )
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    tif   file  count  =  {MIKADO}{cumulative_tif_file_count:<6d}{RESET}",   flush=True  )
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    spcn  file  count  =  {MIKADO}{cumulative_spcn_file_count:<6d}{RESET}",  flush=True  )
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    png   file  count  =  {MIKADO}{cumulative_png_file_count:<6d}{RESET}",   flush=True  )
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    rna   file  count  =  {MIKADO}{cumulative_rna_file_count:<6d}{RESET}{DULL_WHITE}  <<< note: same cases (sub-directories) may have more than one {MAGENTA}FPKM-UQ.txt{RESET}{DULL_WHITE} file. Nonetheless, only one per case will be used{RESET}",   flush=True  )
    print( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    other file  count  =  {MIKADO}{cumulative_other_file_count:<6d}{RESET}", flush=True  )
  

  # (1B) Locate and flag directories that contain BOTH an image and and rna-seq files

  if args.divide_cases=='True':

    if DEBUG>0:
      print ( f"{ORANGE}CLASSI:         INFO:    segment_cases():  option '{CYAN}DIVIDE_CASES  ( '-v'){RESET}{ORANGE}'  = {MIKADO}{args.divide_cases}{RESET}{ORANGE}, so will divide cases and set applicable flag files in the dataset directory ({MAGENTA}{args.data_dir}{RESET}{ORANGE}){RESET}",    flush=True )

    has_image_count            = 0
    has_rna_count              = 0
    matched_image_rna_count    = 0
  
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>888:  
        print( f"{DULL_WHITE}CLASSI:         INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) is skipped because it only contains sub-directories, not data
                
        dir_has_rna_data    = False
        dir_also_has_image  = False
  
        for f in sorted( files ):
          if  ( f.endswith( args.rna_file_suffix[1:]) ):
            dir_has_rna_data=True
            fqn = f"{dir_path}/HAS_RNA"
            has_rna_count += 1
            with open(fqn, 'w') as g:
              g.write( f"this directory contains rna data" )
            g.close  
            rna_file  = f
          if ( ( f.endswith( 'svs' ))  |  ( f.endswith( 'tif' ) )  |  ( f.endswith( 'tiff' ) )  ):
            dir_also_has_image=True
            fqn = f"{dir_path}/HAS_IMAGE"
            has_image_count += 1
            with open(fqn, 'w') as g:
              g.write( f"this directory contains image data" )
            g.close                           
        
        if dir_has_rna_data & dir_also_has_image:
          
          if DEBUG>555:
            print ( f"{WHITE}CLASSI:         INFO:   case {PINK}{args.data_dir}/{os.path.basename(dir_path)}{RESET} \r\033[100C has both matched and rna files (listed above) (count= {MIKADO}{matched_image_rna_count+1}{RESET})",  flush=True )
          fqn = f"{dir_path}/HAS_BOTH"
          with open(fqn, 'w') as g:
            g.write( f"this directory contains matched image and rna-seq data" )
          g.close  
          matched_image_rna_count+=1
  
    if DEBUG>0:
      print ( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    number of cases (directories) which contain BOTH matched and rna files = {MIKADO}{matched_image_rna_count}{RESET}",  flush=True )


  
  
  
  
    # (1C) Segment the cases as follows:
    #      (1Ca)  MULTIMODE____TEST ............................ MATCHED cases set aside for multimode testing only. The number of cases to be so flagged is given by config parameter "CASES_RESERVED_FOR_IMAGE_RNA"
    #      (1Cb)  UNIMODE_CASE____MATCHED ...................... MATCHED cases minus MULTIMODE____TEST cases; used for unimode training (generated embeddings are used for multimode training)
    #      (1Cc)  UNIMODE_CASE ................................. ALL cases minus multimode cases, and don't have to be matched. Constitute the largest possible set of cases for use in unimode image or rna training and testing (including as a prelude to multimode testing with the designated multimode test set where comparing unimode to multimode performance (which requires the use of the same cases for unimode and multimode) is not of interest
    #      (1Cd ) UNIMODE_CASE____IMAGE ........................ ALL cases minus multimode cases which contain an image -     used for unimode training ) constitute the largest possible (but umatched) set of cases for use in unimode image training (including as a prelude to multimode testing with the designated multimode test set, where comparing unimode to multimode performance (the latter requires the use of the SAME cases for unimode and multimode) is not of interest
    #      (1Ce ) UNIMODE_CASE____IMAGE_TEST ................... ALL cases minus multimode cases which contain an image - reserved for unimode testing  ) same criteria as UNIMODE_CASE____IMAGE, but reserved for testing
    #      (1Cf ) UNIMODE_CASE____RNA_FLAG ..................... ALL cases minus multimode cases which contain rna-seq  -     used for unimode training ) constitute the largest possible (but umatched) set of cases for use in unimode rna-seq training (including as a prelude to multimode testing with the designated multimode test set, where comparing unimode to multimode performance (the latter requires the use of the SAME cases for unimode and multimode) is not of interest
    #      (1Cg ) UNIMODE_CASE____RNA_TEST_FLAG ................ ALL cases minus multimode cases which contain rna-seq  - reserved for unimode testing  ) same criteria as UNIMODE_CASE____RNA_FLAG, but reserved for testing


    #        - yes it's confusing. sorry!


    # (1Ci) designate MULTIMODE____TEST cases.  Infinite loop with a break condition (necessary to give every case an equal chance of being randonly selected for inclusion in the MULTIMODE case set)
    
    directories_considered_count = 0
    multimode_case_test_count    = 0
    
    if DEBUG>0:
      if args.cases_reserved_for_image_rna>0:
        print ( f"{BOLD}{CHARTREUSE}CLASSI:         INFO:    segment_cases():    about to randomly designate {CYAN}CASES_RESERVED_FOR_IMAGE_RNA{RESET}{BOLD}{CHARTREUSE}  {RESET}(={MIKADO}{args.cases_reserved_for_image_rna}{RESET}){BOLD}{CHARTREUSE} cases flagged as '{ARYLIDE}HAS_BOTH{RESET}{BOLD}{CHARTREUSE}' to be exclusively reserved as {ARYLIDE}MULTIMODE____TEST{RESET}{BOLD}{CHARTREUSE} cases",  flush=True )
   
    while True:
      
      for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):                                 # select the multimode cases ...
    
        if DEBUG>55:  
          print( f"{DIM_WHITE}CLASSI:         INFO:     now considering case {ARYLIDE}{os.path.basename(dir_path)}{RESET}{DIM_WHITE} as a multimode case  " ) 
    
        
        if not (dir_path==args.data_dir):                                                                  # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data
  
          if DEBUG>55:
            print ( f"{PALE_GREEN}CLASSI:         INFO:   case   \r\033[60C{RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[120C has both image and rna files\r\033[140C (count= {matched_image_rna_count}{RESET}{PALE_GREEN})",  flush=True )
            
          try:
            fqn = f"{dir_path}/HAS_BOTH"        
            f = open( fqn, 'r' )
            if DEBUG>55:
              print ( f"{PALE_GREEN}CLASSI:         INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files (listed above)  \r\033[160C (count= {matched_image_rna_count}{RESET}{PALE_GREEN})",  flush=True )
              print ( f"{PALE_GREEN}CLASSI:         INFO:   multimode_case_test_count          = {AMETHYST}{multimode_case_test_count}{RESET}",          flush=True )
              print ( f"{PALE_GREEN}CLASSI:         INFO:   matched_image_rna_count            = {AMETHYST}{matched_image_rna_count}{RESET}",  flush=True )
              print ( f"{PALE_GREEN}CLASSI:         INFO:   cases_reserved_for_image_rna       = {AMETHYST}{args.cases_reserved_for_image_rna}{RESET}",        flush=True )
            selector = random.randint(0,500)                                                               # the high number has to be larger than the total number of matched cases to give every case a chance of being included 
            if ( selector==22 ) & ( multimode_case_test_count<args.cases_reserved_for_image_rna ):   # used 22 but it could be any number

              fqn = f"{dir_path}/MULTIMODE____TEST"         
              try:
                with open(fqn, 'r') as f:                                                                  # have to check that the case (directory) was not already flagged as a multimode cases, else it will do it again and think it was an additional case, therebody creating one (or more) fewer cases
                  pass
              except Exception:
                fqn = f"{dir_path}/MULTIMODE____TEST"         
                try:
                  with open(fqn, 'w') as f:
                    f.write( f"this case is designated as a multimode case" )
                    multimode_case_test_count+=1
                    f.close
                  if DEBUG>2:
                    print ( f"{PALE_GREEN}CLASSI:         INFO:    segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[122C has been randomly flagged as '{ASPARAGUS}MULTIMODE____TEST{RESET}{PALE_GREEN}'  \r\033[204C (count= {MIKADO}{multimode_case_test_count}{RESET}{PALE_GREEN})",  flush=True )
                except Exception:
                  print( f"{RED}CLASSI:       FATAL:  could not create '{CYAN}MULTIMODE____TEST{RESET}' file" )
                  time.sleep(10)
                  sys.exit(0)
  
          except Exception:
            if DEBUG>55:
              print ( f"{RED}CLASSI:       not a matched case" )
    
      directories_considered_count+=1
      if DEBUG>555:
        print ( f"c={c}" )      

      if multimode_case_test_count== args.cases_reserved_for_image_rna:
        if DEBUG>55:
          print ( f"{PALE_GREEN}CLASSI:         INFO:   multimode_case_test_count              = {AMETHYST}{multimode_case_test_count}{RESET}",          flush=True )
          print ( f"{PALE_GREEN}CLASSI:         INFO:   cases_reserved_for_image_rna           = {AMETHYST}{args.cases_reserved_for_image_rna}{RESET}",             flush=True )
        break


    # (1Cii) designate UNIMODE_CASE____MATCHED cases. Go through all MATCHED directories one time. Flag any MATCHED case other than those flagged as MULTIMODE____TEST case at 1Ca above with the UNIMODE_CASE____MATCHED
    
    unimode_case_matched_count    = 0

    for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):                                   # ... designate every matched case (HAS_BOTH) other than those flagged as MULTIMODE____TEST above to be a unimode case
  
      if DEBUG>1:  
        print( f"{DIM_WHITE}CLASSI:         INFO:   now considering case (directory) as a unimode case {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                    # the top level directory (dataset) is skipped because it only contains sub-directories, not data

        if DEBUG>55:
          print ( f"{PALE_GRshuffEEN}CLASSI:         INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files (listed above)  \r\033[160C (count= {matched_image_rna_count}{RESET}{PALE_GREEN})",  flush=True )
  
          
        try:
          fqn = f"{dir_path}/HAS_BOTH"
          f = open( fqn, 'r' )

          try:
            fqn = f"{dir_path}/MULTIMODE____TEST"                                                          # then we designated it to be a MULTIMODE case above, so ignore 
            f = open( fqn, 'r' )
          except Exception:                                                                                # these are the ones we want
            if DEBUG>555:
              print ( f"{PALE_GREEN}CLASSI:         INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files and has not already been designated as a mutimode case  \r\033[200C (count= {matched_image_rna_count}{RESET}{PALE_GREEN})",  flush=True )
              print ( f"{PALE_GREEN}CLASSI:         INFO:   unimode_case_matched_count            = {AMETHYST}{unimode_case_matched_count}{RESET}",            flush=True )
            if ( ( unimode_case_matched_count + multimode_case_test_count ) <= matched_image_rna_count ):  # if we don't yet have enough designated multimode cases (and hence designations in total)
              fqn = f"{dir_path}/UNIMODE_CASE____MATCHED"            
              with open(fqn, 'w') as f:
                f.write( f"this case is designated as a unimode case" )
              f.close
              unimode_case_matched_count+=1
              if DEBUG>44:
                print ( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{DULL_YELLOW} \r\033[122C has been randomly designated as a   unimode case  \r\033[204C (count= {MIKADO}{unimode_case_matched_count}{RESET}{DULL_WHITE})",  flush=True )


        except Exception:
          if DEBUG>555:
            print ( "not a multimode case" )
      
      
    # (1Ciii) designate the UNIMODE_CASE____MATCHED cases. Go through all directories one time. Flag other than MULTIMODE____TEST and  UNIMODE_CASE____MATCHED cases as UNIMODE_CASE
        
    unimode_case_unmatched_count=0
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                 # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>55:  
        print( f"{DIM_WHITE}CLASSI:           INFO:   now processing case (directory) {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                    # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data  

        for f in sorted( files ):          
                    
          try:
            fqn = f"{dir_path}/MULTIMODE____TEST"        
            f = open( fqn, 'r' )
            if DEBUG>555:
              print ( f"{RED}CLASSI:           INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{RED} \r\033[100C is a multimode case. Skipping",  flush=True )
            break
          except Exception:
            try:
              fqn = f"{dir_path}/UNIMODE_CASE"        
              f = open( fqn, 'r' )
              if DEBUG>555:
                print ( f"{RED}CLASSI:           INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{RED} \r\033[100C is in a directory containing the UNIMODE_CASE flag. Skipping",  flush=True )
              break
            except Exception:
              if DEBUG>44:
                print ( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[122C has been flagged with the  {ASPARAGUS}UNIMODE_CASE{RESET}  \r\033[204C (count= {MIKADO}{unimode_case_unmatched_count+1}{RESET})",  flush=True )
              fqn = f"{dir_path}/UNIMODE_CASE"            
              with open(fqn, 'w') as f:
                f.write( f"this case is not a designated multimode case" )
              f.close
              unimode_case_unmatched_count+=1                                                              # only segment_cases knows the value of unimode_case_unmatched_count, and we need in generate(), so we return it
                                                                  

    # (1Civ) Designate those IMAGE cases which are not also MULTIMODE cases. Go through directories one time. Flag UNIMODE_CASE which are ALSO image cases as UNIMODE_CASE____IMAGE
    
    if DEBUG>3:
      print ( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():  about to designate '{ARYLIDE}UNIMODE_CASE____IMAGE{RESET}{DULL_WHITE}' cases{RESET}",  flush=True )  
    
    directories_considered_count    = 0
    unimode_case_image_count        = 0
    class_counts                   = np.zeros( n_classes, dtype=int )
    
    n=0 
    
    while np.any( class_counts < 2 ):

      a = random.choice( range(  0,  1   ) )
      b = random.choice( range(  100,250 ) )
      c = random.choice( range(  100,250 ) )
      c = 120
      BB=f"\033[38;2;{a};{b};{c}m"
          
      if n>0:
        print ( f"\033[59;200H{CLEAR_LINE}",                                                                                                                             flush=True  )    
        print ( f"\033[60;200H{BOLD}{BB}  CLASSI:         INFO: some subtypes are not represented in the applicable subset.  Shuffling and trying again.{RESET}",        flush=True  )    
      
      for dir_path, dirs, files in os.walk( args.data_dir ):
    
        random.shuffle(dirs)
          
        if DEBUG>55:  
          print( f"{DIM_WHITE}CLASSI:           INFO:   now processing case (directory) {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
    
        if not (dir_path==args.data_dir): 
                      
          try:
            fqn = f"{dir_path}/HAS_IMAGE"        
            f = open( fqn, 'r' )
            if DEBUG>10:
              print ( f"{DULL_WHITE}CLASSI:           INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_GREEN}' \r\033[122C is an image case",  flush=True )
            try:
              fqn = f"{dir_path}/UNIMODE_CASE"        
              f = open( fqn, 'r' )
              if DEBUG>10:
                print ( f"{GREEN}CLASSI:           INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{GREEN} \r\033[122C is in a directory containing a UNIMODE IMAGE case",  flush=True )
              fqn = f"{dir_path}/UNIMODE_CASE____IMAGE"            
              with open(fqn, 'w') as f:
                f.write( f"this case is a UNIMODE_CASE____IMAGE case" )
              f.close
              if DEBUG>22:
                print ( f"{PALE_GREEN}CLASSI:           INFO:       segment_cases():  case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_GREEN}' \r\033[122C has been flagged with the UNIMODE_CASE____IMAGE  \r\033[204C (count= {MIKADO}{unimode_case_image_count+1}{RESET})",  flush=True )
              unimode_case_image_count+=1                                                                  # only segment_cases knows the value of unimode_case_image_count, and we need in generate(), so we return it
  
              
              try:                                                                                         # accumulate class counts. Every tile has an associated label - the same label for every tile image in the directory
                label_file = f"{dir_path}/{args.class_numpy_file_name}"
                if DEBUG>100:
                  print ( f"CLASSI:         INFO:   label_file  {ASPARAGUS}{label_file}{RESET}")
                label      = np.load( label_file )
                if label[0]>args.highest_class_number:
                  pass
                else:
                  class_counts[label[0]]+=1
                  
                if DEBUG>2:
                  np.set_printoptions(formatter={'int': lambda x: "{:>6d}".format(x)}) 
                  print( f"\033[61;200H{BOLD}{BB}  CLASSI:         INFO: class_counts                         = {CARRIBEAN_GREEN}{class_counts}{RESET}{CLEAR_LINE}", flush=True  )
  
              except Exception as e:
                print ( f"{RED}CLASSI:               FATAL: when processing: '{label_file}'  (21945){RESET}", flush=True)        
                print ( f"{RED}CLASSI:                      reported error was: '{e}'{RESET}", flush=True)
                print ( f"{RED}CLASSI:                      halting now{RESET}", flush=True)
                sys.exit(0)
  
            except Exception:
              if DEBUG>44:
                print ( f"{RED}CLASSI:           INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{RED}' \r\033[122C  is not a UNIMODE_CASE case - - skipping{RESET}",  flush=True )
          except Exception:
            if DEBUG>44:
              print ( f"{PALE_RED}CLASSI:           INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[122C is not an image case - - skipping{RESET}",  flush=True )                                                                    
      
      n+=1  


    # (1Cv) Designate those RNA cases which are not also MULTIMODE cases. Go through directories one time. Flag UNIMODE_CASE which are also rna cases as UNIMODE_CASE____RNA
    
    if DEBUG>3:
      print ( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():  about to designate '{ARYLIDE}UNIMODE_CASE____RNA{RESET}{DULL_WHITE}' cases{RESET}",  flush=True )  
    
    directories_considered_count    = 0
    unimode_case_rna_count          = 0
    
    for dir_path, dirs, files in os.walk( args.data_dir ):
  
      if DEBUG>55:  
        print( f"{DIM_WHITE}CLASSI:           INFO:   now processing case (directory) {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir): 
                    
        try:
          fqn = f"{dir_path}/HAS_RNA"        
          f = open( fqn, 'r' )
          if DEBUG>44:
            print ( f"{GREEN}CLASSI:           INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{GREEN}' \r\033[122C is an rna case",  flush=True )
          try:
            fqn = f"{dir_path}/UNIMODE_CASE"        
            f = open( fqn, 'r' )
            if DEBUG>2:
              print ( f"{GREEN}CLASSI:           INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{GREEN} \r\033[122C is in a directory containing the UNIMODE_CASE",  flush=True )
            fqn = f"{dir_path}/UNIMODE_CASE____RNA"            
            with open(fqn, 'w') as f:
              f.write( f"this case is a UNIMODE_CASE____RNA case" )
            f.close
            if DEBUG>22:
              print ( f"{PALE_GREEN}CLASSI:           INFO:       segment_cases():  case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_GREEN}' \r\033[122C has been flagged with the UNIMODE_CASE____RNA  \r\033[204C (count= {MIKADO}{unimode_case_rna_count+1}{RESET})",  flush=True )
            unimode_case_rna_count+=1                                                                      # only segment_cases knows the value of unimode_case_rna_count, and we need in generate(), so we return it
          except Exception:
            if DEBUG>44:
              print ( f"{RED}CLASSI:           INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{RED}' \r\033[122C  is not a UNIMODE_CASE case - - skipping{RESET}",  flush=True )
        except Exception:
          if DEBUG>44:
            print ( f"{PALE_RED}CLASSI:           INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[122C is not an rna case - - skipping{RESET}",  flush=True )                                                                    
        

    # (1Cvi) Designate 'UNIMODE_CASE____IMAGE_TEST' cases. Go through directories one time. Flag 'PCT_TEST' % of the UNIMODE_CASE IMAGE cases as UNIMODE_CASE____IMAGE_TEST
    #        These cases are used for unimode image testing. Necessary to strictly separated cases in this manner for image mode so that tiles from a single image do not end up in both the training and test sets   
    #        In image mode, tiles allocated to the training set cann't come from an image which is also contributing tiles to the test set. Ditto the reverse.
    #        This issue does not affect rna mode, where there is only one artefact per case. I.e. when input is rna, any rna sample can be allocated to either the training set or test set
    #
    #        Strategy: re-designate an appropriate number of the 'UNIMODE_CASE____IMAGE' to be 'UNIMODE_CASE____IMAGE_TEST' (delete the first flag)
    #
    #        TODO: IDEALLY HERE WE WOULD ALSO ENSURE THAT AT LEAST ONE CASE OF EACH SUBTYPE WAS DESIGNATED AS IMAGE_TEST. OTHERWISE SOME SUBTYPES CAN END UP WITH NO TEST EXAMPLES IF 
    #              THERE IS ONLY A VERY SMALL NUMBER OF EXAMPLES FOR THAT CASE IN TOTAL (BECAUSE THE TEST EXAMPLE QUOTE MIGHT BE FILLED WITHOUT COMING ACROSS AN EXAMPLE OF THAT SUBTYPE)
    
  

    cases_to_designate = int(pct_test * unimode_case_image_count)
        
    if DEBUG>0:
      print ( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    about to randomly re-designate int({CYAN}PCT_TEST{RESET}{DULL_WHITE} {MIKADO}{pct_test*100:4.2f}%{RESET}{DULL_WHITE} * {CYAN}unimode_case_image_count{RESET}{DULL_WHITE} {MIKADO}{unimode_case_image_count}{RESET}{DULL_WHITE}) = {MIKADO}{cases_to_designate} {ARYLIDE}UNIMODE_CASE____IMAGE{RESET}{DULL_WHITE} cases as reserved image test cases by placing the flag {ARYLIDE}UNIMODE_CASE____IMAGE_TEST{RESET}{DULL_WHITE}",  flush=True )
    
    unimode_case_image_test_count  = 0
    class_counts                   = np.zeros( n_classes, dtype=int )
    ratios                         = np.zeros( n_classes, dtype=int )


    startrow=16
    startcol=180
      
    UNIMODE_CASE____IMAGE_class_counts = determine_class_counts ( args, n_classes, class_names, n_tiles, 'UNIMODE_CASE____IMAGE' )   # available to be redesignated as ..._TEST cases
    nominally_required_per_class       =  UNIMODE_CASE____IMAGE_class_counts * pct_test
    required_per_class                 =  (UNIMODE_CASE____IMAGE_class_counts * pct_test).astype(int) + 1
    
    if DEBUG>0:
      np.set_printoptions(formatter={ 'int':   lambda x: f"{x:>6d}"}    ) 
      print( f"{SAVE_CURSOR}\033[{startrow+0};{startcol}H{BOLD}{BB}  CLASSI: segment_cases()         INFO: UNIMODE_CASE____IMAGE_class_counts         = {MIKADO}{UNIMODE_CASE____IMAGE_class_counts}{RESET}{CLEAR_LINE}", flush=True  )
      np.set_printoptions(formatter={ 'float': lambda x: f"{x:>6.2f}"} )  
      print( f"\033[{startrow+1};{startcol}H{BOLD}{BB}  CLASSI: segment_cases()         INFO: nominally_required_per_class for testing   = {MIKADO}{nominally_required_per_class}{RESET}{CLEAR_LINE}", flush=True  )
      np.set_printoptions(formatter={ 'int':   lambda x: f"{x:>6d}"}    )       
      print( f"\033[{startrow+2};{startcol}H{BOLD}{BB}  CLASSI: segment_cases()         INFO: required_per_class                             = {MIKADO}{required_per_class}{RESET}{CLEAR_LINE}", flush=True  )
 
    
    for the_class in range (0, n_classes ):

      cases_to_designate =  required_per_class[the_class]

      a  = random.choice( range(  100,250 ) )
      b  = random.choice( range(  0,  1   ) )
      c  = 120
      BB = f"\033[38;2;{a};{b};{c}m"
      
      if DEBUG>0:
        np.set_printoptions(formatter={'int': lambda x: "{:>6d}".format(x)}) 
        print( f"\033[{startrow+2+the_class};{startcol}H{BOLD}{BB}  CLASSI: segment_cases()         INFO: cases to designate for class {MIKADO}{the_class}{RESET}{BB}             = {MAGENTA}{cases_to_designate}{RESET}{CLEAR_LINE}", flush=True  )
      
      while True:

        for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):
          
          if DEBUG>100:  
            print( f"{DIM_WHITE}CLASSI:         INFO:   now considering case {ARYLIDE}{os.path.basename(dir_path)}{RESET}{DIM_WHITE} \r\033[130C as a candidate UNIMODE_CASE____IMAGE_TEST case  " ) 
              
          if not (dir_path==args.data_dir):                                                                # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data
            
            try:
              fqn = f"{dir_path}/UNIMODE_CASE____IMAGE"    
              f = open( fqn, 'r' )                
              if DEBUG>66:
                print ( f"{PALE_GREEN}CLASSI:           INFO:   case   \r\033[55C'{RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[130C is a     {CYAN}UNIMODE_CASE____IMAGE{RESET}{PALE_GREEN} case{RESET}",  flush=True )
              selector = random.randint(0,500)                                                           # the high number has to be larger than the total number of not a multimode cases to give every case a chance of being included 
              if ( selector>0 ):                                                                       # used 22 but it could be any number
                fqn = f"{dir_path}/UNIMODE_CASE____IMAGE_TEST"         
                try:                                                                                       # every tile has an associated label - the same label for every tile image in the directory
                  label_file = f"{dir_path}/{args.class_numpy_file_name}"
                  if DEBUG>100:
                    print ( f"CLASSI:         INFO:   label_file  {ASPARAGUS}{label_file}{RESET}")
                  label      = (np.load( label_file ))[0]
                  if DEBUG>100:
                    print ( f"CLASSI:         INFO:   label       {ASPARAGUS}{label}{RESET}")
                  if label>args.highest_class_number:
                    pass
                  elif label==the_class:
                    try:
                      with open(fqn, 'w') as f:
                        f.write( f"this case is designated as a UNIMODE_CASE____IMAGE_TEST case (for class {the_class})" )
                        class_counts[the_class]+=1
                        f.close
                        os.remove ( f"{dir_path}/UNIMODE_CASE____IMAGE" )
                      if DEBUG>9:
                        print ( f"{BLEU}CLASSI:           INFO:    segment_cases()():  case  {RESET}{CYAN}{dir_path}{RESET}{BLEU} \r\033[130C has been randomly (re-)designated as a UNIMODE_CASE____IMAGE_TEST case  \r\033[204C (class = {MIKADO}{the_class}{RESET}{BLEU}{RESET})",  flush=True )
                    except Exception:
                      print( f"{RED}CLASSI:       FATAL:  either could not create '{CYAN}UNIMODE_CASE____IMAGE_TEST{RESET}' file or delete the '{CYAN}UNIMODE_CASE____IMAGE{RESET}' " )  
                      time.sleep(10)
                      sys.exit(0)              
                  else:
                    pass
  
                except Exception as e:
                  print ( f"{RED}CLASSI:               FATAL: when processing: '{label_file}'    (05723){RESET}", flush=True)        
                  print ( f"{RED}CLASSI:                      reported error was: '{e}'{RESET}", flush=True)
                  print ( f"{RED}CLASSI:                      halting now{RESET}", flush=True)
                  sys.exit(0)

            except Exception:
              if DEBUG>66:
                print ( f"{RED}CLASSI:           INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[130C is not a {CYAN}UNIMODE_CASE____IMAGE{RESET}{RED} case - - skipping{RESET}",  flush=True )
        
          if class_counts[the_class] == cases_to_designate:
            break

        if class_counts[the_class] == cases_to_designate:
          break

    if DEBUG>0:
      np.set_printoptions(formatter={'int': lambda x: "{:>6d}".format(x)}) 
      print( f"\033[{startrow+2+len(class_counts)};{startcol}H{BOLD}{BB}  CLASSI: segment_cases()         INFO: class_counts for testing                   = {MAGENTA}{class_counts}{RESET}{CLEAR_LINE}", flush=True  )
      
            
    if DEBUG>0:
      ratios = np.divide( class_counts, UNIMODE_CASE____IMAGE_class_counts )
      np.set_printoptions(formatter={ 'float' : lambda x: f"{x:>6.2f}"} )   
      print( f"\033[{startrow+2+len(class_counts)+1};{startcol}H{BOLD}{GREEN}  CLASSI: segment_cases()         INFO: final ratios ({CYAN}pct_test={MIKADO}{pct_test}{RESET}{BOLD}{GREEN})                = {ratios}{RESET}{CLEAR_LINE}", flush=True  )

    unimode_case_image_test_count = np.sum ( class_counts )
    if DEBUG>0:
      print( f"\033[{startrow+2+len(class_counts)+2};{startcol}H{BOLD}{GREEN}  CLASSI: segment_cases()         INFO: unimode_case_image_test_count              = {MIKADO}{unimode_case_image_test_count}{RESET}{CLEAR_LINE}{RESTORE_CURSOR}", flush=True  )

    unimode_case_image_count = unimode_case_image_count - unimode_case_image_test_count
    
 
 
 
    # (1Cvii) Designate 'UNIMODE_CASE____RNA_TEST' cases. Go through directories one time. Re-flag 'PCT_TEST' % of the UNIMODE_CASE____RNA cases as UNIMODE_CASE____RNA_TEST and remove the UNIMODE_CASE____RNA flag
    #
    #        Strategy: re-designate an appropriate number of the 'UNIMODE_CASE____RNA' to be 'UNIMODE_CASE____RNA_TEST' (delete the first flag)
  

    cases_to_designate = int(pct_test * unimode_case_rna_count)
        
    if DEBUG>0:
      print ( f"{DULL_WHITE}CLASSI:         INFO:    segment_cases():    about to randomly re-designate int({CYAN}PCT_TEST{RESET}{DULL_WHITE} {MIKADO}{pct_test*100:4.2f}%{RESET}{DULL_WHITE} * {CYAN}unimode_case_rna_count{RESET}{DULL_WHITE}   {MIKADO}{unimode_case_rna_count}{RESET}{DULL_WHITE}) = {MIKADO}{cases_to_designate} {ARYLIDE}UNIMODE_CASE____RNA{RESET}{DULL_WHITE}   cases as reserved rna   test cases by placing the flag {ARYLIDE}UNIMODE_CASE____RNA_TEST{RESET}{DULL_WHITE}{CLEAR_LINE}",  flush=True )
    
    directories_considered_count   = 0
    unimode_case_rna_test_count    = 0
    
    while True:
      
      for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):
    
        if DEBUG>55:  
          print( f"{DIM_WHITE}CLASSI:         INFO:   now considering case {ARYLIDE}{os.path.basename(dir_path)}{RESET}{DIM_WHITE} \r\033[130C as a candidate UNIMODE_CASE____RNA_TEST case  " ) 
            
        if not (dir_path==args.data_dir):                                                                  # the top level directory (dataset) has to be skipped because it only contains sub-directories, not data
          try:
            fqn = f"{dir_path}/UNIMODE_CASE____RNA"    
            f = open( fqn, 'r' )                
            if DEBUG>66:
              print ( f"{PALE_GREEN}CLASSI:           INFO:   case   \r\033[55C'{RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[130C is a     {CYAN}UNIMODE_CASE____RNA{RESET}{PALE_GREEN} case{RESET}",  flush=True )
            selector = random.randint(0,500)                                                               # the high number has to be larger than the total number of not a multimode cases to give every case a chance of being included 
            if ( selector==22 ) & ( unimode_case_rna_test_count<cases_to_designate ):                      # used 22 but it could be any number
              fqn = f"{dir_path}/UNIMODE_CASE____RNA_TEST"         
              try:
                with open(fqn, 'w') as f:
                  f.write( f"this case is designated as a UNIMODE_CASE____RNA_TEST case" )
                  unimode_case_rna_test_count+=1
                  f.close
                  os.remove ( f"{dir_path}/UNIMODE_CASE____RNA" )
                if DEBUG>66:
                  print ( f"{BLEU}CLASSI:           INFO:    segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{BLEU} \r\033[130C has been randomly designated as a UNIMODE_CASE____RNA_TEST case  \r\033[204C (count= {MIKADO}{unimode_case_rna_test_count}{BLEU}{RESET})",  flush=True )
              except Exception:
                print( f"{RED}CLASSI:       FATAL:  either could not create '{CYAN}UNIMODE_CASE____RNA_TEST{RESET}' file or delete the '{CYAN}UNIMODE_CASE____RNA{RESET}' " )  
                time.sleep(10)
                sys.exit(0)              
          except Exception:
            if DEBUG>66:
              print ( f"{RED}CLASSI:           INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[130C is not a {CYAN}UNIMODE_CASE____RNA{RESET}{RED} case - - skipping{RESET}",  flush=True )
    
      directories_considered_count+=1
     
      if unimode_case_rna_test_count == cases_to_designate:
        if DEBUG>55:
          print ( f"{PALE_GREEN}CLASSI:         INFO:   unimode_case_rna_test_count  = {AMETHYST}{unimode_case_rna_test_count}{RESET}",          flush=True )
        break

    unimode_case_rna_count = unimode_case_rna_count - unimode_case_rna_test_count    
    

    if DEBUG>0:
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():    these flags have been placed:{RESET}{CLEAR_LINE}                                                   ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        HAS_IMAGE ................................. = {MIKADO}{has_image_count}{RESET}                 ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        HAS_RNA ................................... = {MIKADO}{has_rna_count}{RESET}                   ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        HAS_BOTH .................................. = {MIKADO}{matched_image_rna_count}{RESET}         ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        MULTIMODE____TEST . . . . . . . . . . . . . = {MIKADO}{multimode_case_test_count}{RESET}       ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        UNIMODE_CASE____MATCHED . . . . . . . . . . = {MIKADO}{unimode_case_matched_count}{RESET}      ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        UNIMODE_CASE  . . . . . . . . . . . . . . . = {MIKADO}{unimode_case_unmatched_count}{RESET}    ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        UNIMODE_CASE____IMAGE . . . . . . . . . . . = {MIKADO}{unimode_case_image_count}{RESET}        ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        UNIMODE_CASE____IMAGE_TEST  . . . . . . . . = {MIKADO}{unimode_case_image_test_count}{RESET}   ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        UNIMODE_CASE____RNA . . . . . . . . . . . . = {MIKADO}{unimode_case_rna_count}{RESET}          ",     flush=True )
        print ( f"{DULL_WHITE}{CLEAR_LINE}CLASSI:         INFO:    segment_cases():        UNIMODE_CASE____RNA_TEST  . . . . . . . . . = {MIKADO}{unimode_case_rna_test_count}{RESET}     ",     flush=True )

    return multimode_case_test_count, unimode_case_matched_count, unimode_case_unmatched_count, unimode_case_image_count, unimode_case_image_test_count, unimode_case_rna_count, unimode_case_rna_test_count


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
      print ( f"CLASSI:         INFO:      newline():             xmin                                    = {xmin}"                            )
      print ( f"CLASSI:         INFO:      newline():             xmax                                    = {xmax}"                            )
      print ( f"CLASSI:         INFO:      newline():             ymin                                    = {ymin}"                            )
      print ( f"CLASSI:         INFO:      newline():             ymax                                    = {ymax}"                            )

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

# ------------------------------------------------------------------------------
def analyse_probs( y1_hat, image_labels_values ):

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max( y1_hat, axis=1 )

    if DEBUG>99:
      y1_hat_numpy = (y1_hat.cpu().data).numpy()
      print ( "CLASSI:         INFO:      analyse_probs():               preds_tensor.shape           = {:}".format( preds_tensor.shape    ) ) 
      print ( "CLASSI:         INFO:      analyse_probs():               preds_tensor                 = \n{:}".format( preds_tensor      ) ) 
    
#    preds = np.squeeze( preds_tensor.cpu().numpy() )
    preds = preds_tensor.cpu().numpy()

    if DEBUG>9:
      print ( "CLASSI:         INFO:      analyse_probs():               type(preds)                  = {:}".format( type(preds)           ) )
      print ( "CLASSI:         INFO:      analyse_probs():               preds.shape                  = {:}".format( preds.shape           ) ) 
      print ( "CLASSI:         INFO:      analyse_probs():         FIRST  GROUP BELOW: preds"            ) 
      print ( "CLASSI:         INFO:      analyse_probs():         SECOND GROUP BELOW: y1_hat_numpy.T"   )
      np.set_printoptions(formatter={'int':   lambda x: "\033[1m{:^10d}\033[m".format(x)    }    )
      print ( preds[0:22] )
      #np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      #print (  np.transpose(y1_hat_numpy[0:22,:])  )

    p_full_softmax_matrix = functional.softmax( y1_hat, dim=1).cpu().numpy()

    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "CLASSI:         INFO:      analyse_probs():              type(p_full_softmax_matrix)     = {:}".format( type(p_full_softmax_matrix) )  )
      print ( "CLASSI:         INFO:      analyse_probs():               p_full_softmax_matrix          = \n{:}".format( np.transpose(p_full_softmax_matrix[0:22,:])   )  )

    # make a vector of the HIGHEST probability (for each example in the batch)    
    p_highest  = np.array(  [ functional.softmax( el, dim=0)[i].item() for i, el in zip(preds, y1_hat) ]   )


    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "CLASSI:         INFO:      analyse_probs():               p_highest.shape                = {:}".format( (np.array(p_highest)).shape )  )
      print ( "CLASSI:         INFO:      analyse_probs():               p_highest                      = \n{:}".format( np.array(p_highest) )  )
      
    # make a vector of the SECOND HIGHEST probability (for each example in the batch) (which is a bit trickier)
    p_2nd_highest = np.zeros((len(preds)))
    for i in range (0, len(p_2nd_highest)):
      p_2nd_highest[i] = max( [ el for el in p_full_softmax_matrix[i,:] if el != max(p_full_softmax_matrix[i,:]) ] )

    if DEBUG>99:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "CLASSI:         INFO:      analyse_probs():               p_2nd_highest              = \n{:}".format( p_2nd_highest   )  )  

    # make a vector of the probability the network gave for the true class (for each example in the batch)
    for i in range (0, len(image_labels_values)):
      p_true_class = np.choose( image_labels_values, p_full_softmax_matrix.T)
    
    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( f"CLASSI:         INFO:      analyse_probs():               p_true_class              = \n{p_true_class}"  )  
      
   
    return preds, p_full_softmax_matrix, p_highest, p_2nd_highest, p_true_class



# ------------------------------------------------------------------------------
def plot_scatter( args, writer, i, background_image, tile_size, image_labels, class_names, class_colours, preds, p_full_softmax_matrix, show_patch_images ):

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
    fig.legend(l, class_names, loc='upper right', fontsize=10, facecolor='white') 
  
  # (5) add patch level truth value and prediction 

  threshold_0=36     # total tiles, not width
  threshold_1=100
  threshold_2=400
  threshold_3=900
  threshold_4=30000
            
  t2=f"Cancer type:  {args.cancer_type_long}"
  t3=f"True subtype for the slide:"
  t4=f"{class_names[image_labels[idx]]}"
  t5=f"Predicted subtype for this patch:"
  t6=f"{class_names[np.argmax(np.sum(p_full_softmax_matrix, axis=0))]}"
  
  if total_tiles >=threshold_4:                    ## NOT OPTIMISED!  Need some more thresholds for values closer to theshold_3
    #          x     y
    ax.text(   0,  -1300, t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -1000, t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 4400, -1000, t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -700,  t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 4400, -700,  t6, size=10, ha="left",   color="black", style="italic" ) 
  elif threshold_4>total_tiles>=threshold_3:       ## OPTIMISED
    ax.text(   0,  -180,  t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -140,  t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -140,  t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -100,  t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -100,  t6, size=10, ha="left",   color="black", style="italic" )        
  elif threshold_3>total_tiles>=threshold_2:       ## OPTIMISED
    ax.text(   0,  -180,  t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -140,  t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -140,  t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -100,  t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -100,  t6, size=10, ha="left",   color="black", style="italic" )    
  elif threshold_2>total_tiles>=threshold_1:       ## OPTIMISED
    ax.text(   0,  -700,  t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -330,  t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -330,  t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -300,  t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -300,  t6, size=10, ha="left",   color="black", style="italic" )    
  elif threshold_1>total_tiles>=threshold_0:      ## OPTIMISED
    ax.text(   0,  -62,   t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -51,   t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 175,  -51,   t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -40,   t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 175,  -40,   t6, size=10, ha="left",   color="black", style="italic" )                   
  else: # (< threshold0)                          ## OPTIMISED
    ax.text(   0,  -32,   t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -25,   t3, size=10, ha="left",   color="black", style="normal" )
    ax.text(  90,  -25,   t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -18,   t5, size=10, ha="left",   color="black", style="normal" )
    ax.text(  90,  -18,   t6, size=10, ha="left",   color="black", style="italic" )    

  # (6) plot the points, organised to be at the centre of where the tiles would be on the background image, if it were tiled (the grid lines are on the tile borders)
  
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
      print ( f"CLASSI:         INFO:      major_ticks = {major_ticks}" )
    
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
          marker_size =int(80000/pixel_width)-2  # i.e. 13-1=12
        elif threshold_4<=nrows<threshold_5:   # seems ok
          marker_size =int(60000/pixel_width)
        else:
          marker_size = 1
        
        if DEBUG>8:
          print ( f"CLASSI:         INFO:      plot_scatter()  nrows       = {MIKADO}{nrows}{RESET}" )
          print ( f"CLASSI:         INFO:      plot_scatter()  marker_size = {MIKADO}{marker_size}{RESET}" )
          
        plt.scatter( x_npy, y_npy, c=class_colours[n], marker='x', s=marker_size, zorder=100 )             # 80000 is a good value for sqrt(14*14*64)=112x112
        
      except Exception as e:
        pass

      threshold_5=2500
      threshold_4=1600
      threshold_3=900
      threshold_2=400
      threshold_1=100
      threshold_0=36

      
      if   threshold_0<=total_tiles<threshold_1:
        linewidth=1
      elif threshold_1<=total_tiles<threshold_2:
        linewidth=0
      elif threshold_2<=total_tiles<threshold_3:
        linewidth=0
      else:
        linewidth=0
      
      if                 total_tiles >= threshold_5:
        plt.grid(True, which='major', alpha=1.0, color='none',   linestyle='-', linewidth=1 )
      elif threshold_4 < total_tiles <= threshold_5:
        plt.grid(True, which='major', alpha=1.0, color='silver', linestyle='-', linewidth=1 )
      elif threshold_3 < total_tiles <= threshold_4:
        plt.grid(True, which='major', alpha=1.0, color='silver', linestyle='-', linewidth=1 )
      elif threshold_2 < total_tiles <= threshold_3:
        plt.grid(True, which='major', alpha=1.0, color='silver', linestyle='-', linewidth=1 )
      elif threshold_1 < total_tiles <= threshold_2:
        plt.grid(True, which='major', alpha=1.0, color='silver', linestyle='-', linewidth=1 )
      else:
        plt.grid(True, which='major', alpha=1.0, color='grey',   linestyle='-', linewidth=1 )
        
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
  stats=f"Statistics: tile count: {total_tiles}; correctly predicted: {correct_predictions}/{total_tiles} ({100*pct_correct:2.1f}%)"
  plt.figtext( 0.15, 0.035, stats, size=14, color="grey", style="normal" )
  
  scattergram_name = [ "2 scattergram over tiles" if show_patch_images=='True' else "9 scattergram " ][0]
  plt.show
  writer.add_figure( scattergram_name, fig, i )
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
      print ( f"CLASSI:         INFO:        p_true_class.tolist() = {p_true_class.tolist()}" )
      print ( f"CLASSI:         INFO:        preds.tolist()        = {preds.tolist()}"        )
      print ( f"CLASSI:         INFO:        image_labels.tolist() = {image_labels.tolist()}"        )     
     
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

    if DEBUG>99:
      print ( f"CLASSI:         INFO:        plot_matrix():  (type: {MIKADO}{matrix_type}{RESET}) grid_p_full_softmax_matrix.shape  = {grid_p_full_softmax_matrix.shape}" ) 
      
    sd             = np.std( grid_p_full_softmax_matrix, axis=1 )    
    sd             = sd[np.newaxis,:]
    sd             = sd.T
    reshaped_to_2D = np.reshape(sd, (nrows,ncols))
    
    if DEBUG>9:
      print ( f"CLASSI:         INFO:        plot_matrix():  (type: {MIKADO}{matrix_type}{RESET}) reshaped_to_2D.shape  = {reshaped_to_2D.shape}" ) 
      print ( f"CLASSI:         INFO:        plot_matrix():  (type: {MIKADO}{matrix_type}{RESET}) reshaped_to_2D values = \n{reshaped_to_2D.T}" ) 
          
    cmap=cm.Greens
    tensorboard_label = "7 sd of class probs"

  else:
    print( f"\n{ORANGE}CLASSI:         WARNING: no such matrix_type {RESET}{MIKADO}{matrix_type}{RESET}{ORANGE}. Skipping.{RESET}", flush=True)

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
def plot_classes_preds(args, model, tile_size, batch_images, image_labels, batch_fnames, preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours):
    '''
    Generates matplotlib Figure using a trained network, along with a batch of images and labels, that shows the network's top prediction along with its probability, alongside the actual label, colouring this
    information based on whether the prediction was correct or not. Uses the "images_to_probs" function. 
    
    '''
    
    ##################################################################################################################################
    #
    #  (1) Training mode: the simple case because we are just displaying a set of random tiles which have passed through the network during training
    #
    if args.just_test=='False':
  
      # ~ number_to_plot = len(image_labels)    
      # ~ figure_width   = 10
      # ~ figure_height  = int(number_to_plot * .4)


      number_to_plot = 16    
      figure_width   = 18
      figure_height  = 20                                                                                # taller figure squashes the image grid (less gap). No idea why.
      
                
      # plot the images in the batch, along with predicted and true labels
      fig = plt.figure( figsize=( figure_width, figure_height ) )                                          # overall size ( width, height ) in inches
  
      if DEBUG>99:
        print ( "\nCLASSI:         INFO:      plot_classes_preds():             number_to_plot                          = {:}".format( number_to_plot    ) )
        print ( "CLASSI:         INFO:      plot_classes_preds():             figure width  (inches)                  = {:}".format( figure_width    ) )
        print ( "CLASSI:         INFO:      plot_classes_preds():             figure height (inches)                  = {:}".format( figure_height   ) )
  
      #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
      #plt.grid( False )
  
      ncols = int((   number_to_plot**.5 )           // 1 )
      nrows = int(( ( number_to_plot // ncols ) + 1 ) // 1 )
  
      if DEBUG>99:
        print ( "CLASSI:         INFO:      plot_classes_preds():             number_to_plot                          = {:}".format( number_to_plot  ) )
        print ( "CLASSI:         INFO:      plot_classes_preds():             nrows                                   = {:}".format( nrows           ) )
        print ( "CLASSI:         INFO:      plot_classes_preds():             ncols                                   = {:}".format( ncols           ) ) 
  
      for idx in np.arange( number_to_plot ):
  
          fq_link  = f"{args.data_dir}/{batch_fnames[idx]}.fqln"
          fq_name  = os.readlink     ( fq_link )
          dir_name = os.path.dirname ( fq_name )
          
          if DEBUG>2:
            print ( f"CLASSI:         INFO:      test:       file fq_link points to      = {MAGENTA}{fq_link}{RESET}"    )
            print ( f"CLASSI:         INFO:      test:       fq_link                     = {MAGENTA}{fq_name}{RESET}"                 )
            print ( f"CLASSI:         INFO:      test:       dir_name                    = {MAGENTA}{dir_name}{RESET}"                )
            
                  
          ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[])            # nrows, ncols, "index starts at 1 in the upper left corner and increases to the right", List of x-axis tick locations, List of y-axis tick locations
          ax.set_frame_on( False )
  
          img     = batch_images[idx]
          npimg_t = np.transpose(img, (1, 2, 0))
          plt.imshow(npimg_t)
  
          if DEBUG>99:
            print ( "CLASSI:         INFO:      plot_classes_preds():  idx={:}".format( idx ) )
            print ( "CLASSI:         INFO:      plot_classes_preds():  idx={:} probs[idx] = {:4.2e}, classes[preds[idx]] = {:<20s}, classes[labels[idx]] = {:<20s}".format( idx, probs[idx], classes[preds[idx]], classes[labels[idx]]  ) )
  
          ax.set_title( "p_1={:<.4f}\n p_2nd_highest={:<.4f}\n pred: {:}\ntruth: {:}".format( p_highest[idx], p_2nd_highest[idx], class_names[preds[idx]], class_names[image_labels[idx]] ),
                      loc        = 'center',
                      pad        = None,
                      size       = 8,
                      color      = ( "green" if preds[idx]==image_labels[idx] else "red") )
  
      fig.tight_layout( rect=[0, 0, 0, 0] )

      
      return fig


    ##################################################################################################################################
    #
    # (2) Test mode is much more complex, because we need to present an annotated 2D contiguous grid of tiles
    #
    
    if args.just_test=='True':
 
      non_specimen_tiles  = 0
      correct_predictions = 0  
  
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
        print ( f"CLASSI:         INFO:        plot_classes_preds():  {ORANGE if args.just_test=='True' else MIKADO} about to set up {MIKADO}{figure_width}x{figure_height} inch{RESET} figure and axes for {MIKADO}{nrows}x{ncols}={number_to_plot}{RESET} subplots. (Note: This takes a long time for larger values of nrows/ncols)", end="", flush=True )
            
      fig, axes = plt.subplots( nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=True, figsize=( figure_width, figure_height ) )        # This takes a long time to execute for larger values of nrows and ncols
    
      if DEBUG>0:
        print ( f"  ... done", flush=True )
      

      # (2b) add the legend 
      
      l=[]
      for n in range (0, len(class_colours)):
        l.append(mpatches.Patch(color=class_colours[n], linewidth=0))
        fig.legend(l, class_names, loc='upper right', fontsize=14, facecolor='lightgrey')      
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
      ax0.bar( x=[range(str(range(len(class_names))))], height=np.sum(p_full_softmax_matrix,axis=0),  width=int(number_to_plot/len(image_labels)), color=class_colours )
      # [c[0] for c in class_names]


      # (2d) process each tile; which entails allocating the tile to the correct spot in the subplot grid together plus annotated class information encoded as border color and centred 'x' of prediction was incorrect
      
      flag=0
      
      for r in range(nrows):
      
        for c in range(ncols):

          idx = (r*nrows)+c
          
          if args.just_test=='True':
            
            if DEBUG>0:
              if flag==0:
                  print ( f"CLASSI:         INFO:        plot_classes_preds():  {ORANGE if args.just_test=='True' else MIKADO} now processing sub-plot {RESET}", end="", flush=True )
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
              t4=f"{class_names[image_labels[idx]]}"
              t5=f"NN prediction from patch:"
              t6=f"{class_names[np.argmax(np.sum( p_full_softmax_matrix, axis=0)) ]}"
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
                print ( f"CLASSI:         INFO:      plot_classes_preds():             predicted_class                                   = {predicted_class}" )
            
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
        print ( "CLASSI:         INFO:      plot_classes_preds():  idx={:}".format( idx ) )
      if DEBUG>99:
        print ( "CLASSI:         INFO:      plot_classes_preds():  idx={:} p_highest[idx] = {:4.2f}, class_names[preds[idx]] = {:<20s}, class_names[image_labels[idx]] = {:<20s}".format( idx, p_highest[idx], class_names[preds[idx]], class_names[image_labels[idx]]  ) )
  
      if DEBUG>99:
        print ( f"CLASSI:         INFO:      plot_classes_preds():             idx                                     = {idx}"                            )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             p_highest[idx]                          = {p_highest[idx]:4.2f}"            )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             p_2nd_highest[idx]]                     = {p_2nd_highest[idx]:4.2f}"        )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             preds[idx]                              = {preds[idx]}"                     )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             class_names                             = {class_names}"                    )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             class_names                             = {class_names[1]}"                 )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             class_names                             = {class_names[2]}"                 )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             class_names[preds[idx]]                 = {class_names[preds[idx]]}"        )
        print ( f"CLASSI:         INFO:      plot_classes_preds():             class_names[image_labels[idx]]          = {class_names[image_labels[idx]]}" )
      
      return fig

# ------------------------------------------------------------------------------

def l1_penalty(model, l1_coef):
  
    """Compute L1 penalty. For implementation details, see:

    See: https://discuss.pytorch.org/t/simple-l2-regularization/139
    """
    reg_loss = 0
    for param in model.lnetimg.parameters_('y2'):
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

def save_model( log_dir, model ):
    """Save PyTorch model state dictionary
    """

    if args.input_mode == 'image':
      # ~ if args.pretrain=='True':
        # ~ try:
          # ~ fqn = f"{log_dir}/model_pretrained.pt"   # try and open it
          # ~ f = open( fqn, 'r' )
          # ~ if DEBUG>2:
            # ~ print( f"\r{COTTON_CANDY}CLASSI:         INFO:  pre-train option has been selected but a pre-trained model already exists. Saving state model dictionary as {fqn}{RESET}", end='', flush=True )
          # ~ f.close()
        # ~ except Exception as e:
          # ~ fqn = f"{log_dir}/model_pretrained.pt"
          # ~ print( f"{COTTON_CANDY}<< saving to: {fqn}{RESET} ", end='', flush=True )
      # ~ else:
      fqn = f"{log_dir}/model_image.pt"
        
    elif args.input_mode == 'rna':
      fqn = f"{log_dir}/model_rna.pt"

    elif args.input_mode == 'image_rna':
      fqn = f"{log_dir}/model_image_rna.pt"

    if DEBUG>44:
      print( f"\r\033[220C<<<{BOLD}{MIKADO}{fqn}{RESET}", end="", flush=True )
      
    model_state = model.state_dict()
    torch.save( model_state, fqn) 

# ------------------------------------------------------------------------------
    
def delete_selected( root, extension ):

  walker = os.walk( root, topdown=True )

  for root, dirs, files in walker:

    for f in files:
      fqf = root + '/' + f
      if DEBUG>99:
        print( f"CLASSI:         INFO:   examining file:   '\r\033[43C\033[36;1m{fqf}\033[m' \r\033[180C with extension '\033[36;1m{extension}\033[m'" )
      if ( f.endswith( extension ) ): 
        try:
          if DEBUG>99:
            print( f"CLASSI:         INFO:   will delete file  '\r\033[43C{MIKADO}{fqf}{RESET}'" )
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
def box_plot_by_subtype( args, class_names, n_genes, start_time, parameters, mags, probs, writer, total_runs_in_job, pct_test, run_level_classifications_matrix_acc ):
  
  np.set_printoptions(edgeitems=1000)
  np.set_printoptions(linewidth=1000)
  
  # recall that the we are going to plot statistics FOR EACH run in the the box plots, so we have to use the run_level_classifications_matrix accumulator rather than the already summarised job_level_classifications_matrix
  
  # (1) Check and maybe print some values. Not otherwise used.
  
  confusion_matrix                =  np.sum  ( run_level_classifications_matrix_acc, axis=0 )                                                          # sum across all examples to produce job level confusion matrix (2D array)
  if DEBUG>0:
    print( f'CLASSI:           INFO:    confusion_matrix (confusion matrix)       = \n{CARRIBEAN_GREEN}{confusion_matrix}{RESET}')
    print( f'CLASSI:           INFO:    total predictions (check sum)             =  {MIKADO}{np.sum(confusion_matrix)}{RESET}')

  total_predictions_by_subtype    = np.squeeze( ( np.expand_dims(np.sum  (  confusion_matrix, axis=0 ), axis=0 )  )  )                                 # sum down the columns to produces a row vector representing total subtypes  
  
  if DEBUG>0:
    np.set_printoptions(formatter={'int': lambda x: "{:>7d}".format(x)})    
    print( f'CLASSI:           INFO:    total_predictions_by_subtype              = \n{CARRIBEAN_GREEN}{total_predictions_by_subtype}{RESET}') 
  if DEBUG>2:
    print( f'CLASSI:           INFO:    total predictions (check sum)             =  {MIKADO}{np.sum(total_predictions_by_subtype)}{RESET}')   


  total_predictions_by_subtype[total_predictions_by_subtype == 0] = 1                                                                                  # to avoid divide by zero for any subtype which has so few examples that no predictions at all were made

  if DEBUG>2:    
    print( f'CLASSI:           INFO:    total_predictions_by_subtype mod to change subtypes with zero predictions overall so that they will have exactly one prediction = \n{CARRIBEAN_GREEN}{total_predictions_by_subtype}{RESET}') 


  correct_predictions_by_subtype  =  np.squeeze( np.array( [ confusion_matrix[i,i] for i in  range( 0 , len( confusion_matrix ))  ] )   )              # pick out diagonal elements (= number correct) to produce a row vector
  if DEBUG>0:
    print( f'CLASSI:           INFO:    correct_predictions_by_subtype            = \n{CARRIBEAN_GREEN}{correct_predictions_by_subtype}{RESET}')                                
  if DEBUG>2:
    print( f'CLASSI:           INFO:    total corects (check sum)                 =  {MIKADO}{np.sum(correct_predictions_by_subtype)}{RESET}')

  pct_correct_predictions_by_subtype  =  correct_predictions_by_subtype / total_predictions_by_subtype
  if DEBUG>0:
    np.set_printoptions(formatter={ 'float' : lambda x: f"   {CARRIBEAN_GREEN}{x:.1f}   "} )          
    print( f'CLASSI:           INFO:    pct_correct_predictions_by_subtype        = \n{CARRIBEAN_GREEN}{100*pct_correct_predictions_by_subtype}{RESET}')

  

  # (2) Extract two planes from 'run_level_classifications_matrix_acc' to derive 'pct_correct_predictions_plane' for the box plots (recalling that 'run_level_classifications_matrix_acc' is a 3D matrix with one plane for every run)
  
  all_predictions_plane           =   np.sum(  run_level_classifications_matrix_acc, axis=1 )[ 0:total_runs_in_job, : ]                                # sum elements (= numbers correct) from 3D volume down columns (axis 1) to produce a matrix
  total_predictions_made          =   np.sum(  all_predictions_plane )
  
  if DEBUG>2:
    np.set_printoptions(formatter={ 'int' : lambda x: f"{x:>5d}   "} )      
    
    print( f'CLASSI:           INFO:    all_predictions_plane (one row per run)   = \n{CARRIBEAN_GREEN}{all_predictions_plane}{RESET}')
    print( f'CLASSI:           INFO:    all_predictions_plane (check sum)         =  {MIKADO}{total_predictions_made}{RESET}')


  all_predictions_plane[all_predictions_plane == 0] = 1                                                                                                # to avoid divide by zero for any subtype which has so few examples that no predictions at all were made

  if DEBUG>2:    
    print( f'CLASSI:           INFO:    all_predictions_plane mod to change subtypes with zero predictions overall so that they will have exactly one prediction = \n{CARRIBEAN_GREEN}{all_predictions_plane}{RESET}') 


  expected_IFF_random_preds      =   100* total_predictions_by_subtype / total_predictions_made                                                        # what we'd expect if the classifications were entirely random

  if DEBUG>2:
    np.set_printoptions(formatter={ 'float' : lambda x: f"   {CARRIBEAN_GREEN}{x:.1f}   "} )    
    print( f"CLASSI:           INFO:    expected correct if random class'n        = \n{CARRIBEAN_GREEN}{expected_IFF_random_preds}{RESET}")


  correct_predictions_plane       =   np.transpose( np.array( [ run_level_classifications_matrix_acc[:,i,i] for i in  range( 0 , run_level_classifications_matrix_acc.shape[1] ) ]  )  ) [ 0:total_runs_in_job, : ]      # pick out diagonal elements (= numbers correct) from 3D volume  to produce a matrix
  if DEBUG>2:
    np.set_printoptions(formatter={ 'int' : lambda x: f"   {CARRIBEAN_GREEN}{x:>5d}   "} )          
    print( f'CLASSI:           INFO:    correct predictions (one row per run)     = \n{CARRIBEAN_GREEN}{correct_predictions_plane}{RESET}')
    print( f'CLASSI:           INFO:    total corects (check sum)                 = {MIKADO}{np.sum(correct_predictions_plane)}{RESET}')

  
  np.seterr( invalid='ignore', divide='ignore' )          
  pct_correct_predictions_plane   =   100 * np.divide( correct_predictions_plane, all_predictions_plane )
  if DEBUG>2:
    np.set_printoptions(formatter={ 'float' : lambda x: f"   {CARRIBEAN_GREEN}{x:5.1f}   "} )          
    print( f'CLASSI:           INFO:    pct_correct_predictions_plane (pre-NaN handling) = \n{CARRIBEAN_GREEN}{pct_correct_predictions_plane}{RESET}')


  num_rows                        =   pct_correct_predictions_plane.shape[0]
  pct_correct_predictions_plane   =   pct_correct_predictions_plane[~np.isnan(pct_correct_predictions_plane).any(axis=1), :]                           # delete any rows (runs) which contain a NaN because they will spoil the box plot
  num_rows_with_nan               =   pct_correct_predictions_plane.shape[0] - pct_correct_predictions_plane.shape[0]
  
  if DEBUG>2:
    np.set_printoptions(formatter={ 'float' : lambda x: f"   {MIKADO}{x:5.1f}   "} )          
    print( f'CLASSI:           INFO:    pct_correct_predictions_plane (one row per run) = \n{pct_correct_predictions_plane}{RESET}')
    print( f'CLASSI:           INFO:    {MAGENTA}number of rows with NaN = {MIKADO}{num_rows_with_nan}{RESET}')
  
  median_pct_correct_predictions_by_subtype  =  np.median ( pct_correct_predictions_plane, axis=0 )
  if DEBUG>2:
    np.set_printoptions(formatter={ 'float' : lambda x: f"   {CARRIBEAN_GREEN}{x:5.1f}   "} )          
    print( f'CLASSI:           INFO:    median_pct_correct_predictions_by_subtype = \n{CARRIBEAN_GREEN}{median_pct_correct_predictions_by_subtype}{RESET}')
    
  
  best_subtype_median      =  0 if np.around( np.max ( median_pct_correct_predictions_by_subtype ) ).astype(int) < 1 else np.around( np.max ( median_pct_correct_predictions_by_subtype ) ).astype(int)
  if DEBUG>2:
    print( f'CLASSI:           INFO:    best subtype median                       = {CARRIBEAN_GREEN}{best_subtype_median}{RESET}') 



  # (3) process and present box plots
  
  # "The box extends from the first quartile (Q1) to the third quartile (Q3) of the data, with a line at the median. 
  # The whiskers extend from the box by 1.5x the inter-quartile range (IQR). 
  # Flier points are those past the end of the whiskers. 
  # From https://en.wikipedia.org/wiki/Box_plot for reference."
  
  now     =  time.time()
  seconds = now - start_time 
  
  if DEBUG>99:
    print ( f"now     =  {now}"      )
    print ( f"seconds =  {seconds}"  )

  minutes =  ( seconds / 60          )  if  seconds >  60  else  0 
  seconds =  ( seconds - 60*minutes  )  if  minutes >=  1  else  seconds

  labels  = class_names[0:len(class_names)]
  labels = [elem[:15] for elem in labels]                                                                  # truncate the class names to be all the same length
  
  print(labels)

  # Titling

  now        = datetime.datetime.now()
  
  if len(labels) < 20:
    supertitle = f"Classification of {args.cancer_type_long} Subtypes\n{total_runs_in_job} experiment runs in this box plot.  Total run time: {round(minutes):02d}m {round(seconds):02d}s"
  else:
    supertitle = f"Classification of 14 Cancers into {len(class_names)} Subtypes\n{total_runs_in_job} experiment runs in this box plot.  Total run time: {round(minutes):02d}m:{round(seconds):02d}s"
  
  if args.input_mode=='image':
    title = f"{now:%d-%m-%y %H:%M}  {args.cases[0:25]} ({parameters['n_samples'][0]})  subtypes:{len(class_names)} NN:{parameters['nn_type_img'][0]}  optimizer:{parameters['nn_optimizer'][0]}  epochs:{args.n_epochs}\n  \
held-out:{int(100*parameters['pct_test'][0])}%  lr:{parameters['lr'][0]:<9.6f}  tiles:{parameters['n_tiles'][0]}  tile_size:{parameters['tile_size'][0]}  batch_size:{parameters['batch_size'][0]}  (mags:{mags} probs:{probs})"
  else:
    title = f"{now:%d-%m-%y %H:%M}  {args.cases[0:25]} ({parameters['n_samples'][0]}) / {args.rna_genes_tranche} (n_genes:{n_genes})   subtypes:{len(class_names)}  \
FPKM-UQ threshold cutoff: >{parameters['cutoff_percentile'][0]}%/<{parameters['low_expression_threshold'][0]} \nNeural Network:{parameters['nn_type_rna'][0]}  optimizer:{parameters['nn_optimizer'][0]}  epochs:{args.n_epochs}  \
batch size:{parameters['batch_size'][0]}   held-out:{int(100*parameters['pct_test'][0])}%  lr:{parameters['lr'][0]:<9.6f}  hidden layer:{parameters['hidden_layer_neurons'][0]}  xform:{parameters['gene_data_transform'][0]}  \
dropout:{parameters['dropout_1'][0]}  topology:{args.hidden_layer_encoder_topology}"

  # these colors are hard wired to the number and order of the pan-cancer subtypes (0008_global). Sorry!  
  pan_cancer_subtype_colors = [ "lightcoral",      "lightcoral",      "lightcoral", 
                                "lightsteelblue",  "lightsteelblue",  "lightsteelblue",  "lightsteelblue", "lightsteelblue",
                                "thistle",         "thistle", 
                                "paleturquoise",   "paleturquoise",   "paleturquoise",   "paleturquoise",                                 
                                "lightsalmon",     "lightsalmon", 
                                "powderblue",      "powderblue", 
                                "khaki",           "khaki", 
                                "plum",            "plum", 
                                "skyblue",         "skyblue",         "skyblue",   
                                "palegreen",       "palegreen",       "palegreen",
                                "wheat",           "wheat",
                                "rosybrown",       "rosybrown",
                                "lightpink",       "lightpink",       "lightpink",
                                "wheat",           "wheat"
                              ] 


  
  # Render portrait version of box plot

  figure_width  = 23
  figure_height = 16

  if len(labels) > 15:
    rotation  = 30
    font_big  = 18
    font_med  = 10
    font_sml  = 6
    font_tiny = 6
    base      = 0.75
    gap_1     = 3.00
    gap_2     = 3.00
    text_1="preds="
    text_2="right="
    text_3="med="    
    text_4="random"
    subtype_colors = pan_cancer_subtype_colors
  elif 5 < len(labels) <= 15:
    rotation  = 0
    font_big  = 20
    font_med  = 16
    font_sml  = 12
    font_tiny = 10
    base      = 0.75
    gap_1     = 2.75
    gap_2     = 2.5
    text_1    = "total predictions="
    text_2    = "total correct="
    text_3    = "median correct="    
    text_4    = "expected for random"
    c_m       = f"plt.cm.{eval('args.colour_map')}"                                                        # the 'eval' is so that the user input string will be treated as a variable
    subtype_colors = [ eval(c_m)(i) for i in range(len(labels))]                                           # makes an array of colours by calling the user defined colour map (which is a function, not a variable)  
  else:
    rotation  = 0
    font_big  = 20
    font_med  = 16
    font_sml  = 14
    font_tiny = 12
    base      = 0.75
    gap_1     = 2.75
    gap_2     = 2.5
    text_1    = "total predictions="
    text_2    = "total correct="
    text_3    = "median correct all runs="    
    text_4    = "expected for random"
    c_m       = f"plt.cm.{eval('args.colour_map')}"                                                        # the 'eval' is so that the user input string will be treated as a variable
    subtype_colors = [ eval(c_m)(i) for i in range(len(labels))]                                           # makes an array of colours by calling the user defined colour map (which is a function, not a variable)  
  


  fig, ax  = plt.subplots( figsize=( figure_width, figure_height ), constrained_layout=True )

  plt.xticks    ( fontsize=font_med, rotation=rotation, ha="right"                        )
  plt.yticks    ( fontsize=20                                                             )
  plt.ylabel    (  'subtypes correctly predicted (%)', weight='bold', fontsize=font_big   )
  plt.yticks    (  range(0, 100, 10)                                                      )
  fig.suptitle  (  supertitle,  color='black',  weight='bold',    fontsize=16             )
  ax.set_title  (  title,       color='black',                    fontsize=14             )  
  ax.set        (  ylim =(0, 100)                                                         )
  ax.xaxis.grid (  True, linestyle='dashed', color='lightgrey'                            )
  ax.yaxis.grid (  True, linestyle='dotted'                                               )

  
  alpha_lite=0.3
  alpha_hard=0.6
  line_props  = dict( color="black", alpha=alpha_lite, linewidth=2           )
  box_props   = dict( color="black",                   linestyle="dashdot"   )
  cap_props   = dict( color="black", alpha=alpha_lite                        )
  flier_props = dict( marker="o",    markersize=7                            )

  bp      = plt.boxplot( pct_correct_predictions_plane, labels=labels, vert=True, patch_artist=True, showfliers=True,  medianprops=dict(color="black", alpha=alpha_hard), boxprops=box_props, whiskerprops=line_props, capprops=cap_props, flierprops=flier_props )

  ax.annotate( f"Total predictions made {np.sum(all_predictions_plane):,}; of which correct: {np.sum(correct_predictions_plane):,} ({100*np.sum(correct_predictions_plane)/np.sum(all_predictions_plane):.1f}%)",
                   xy= (0.08,  0.2),  xycoords='figure fraction',  horizontalalignment='left', color='dimgray', fontsize=12  ) 
  ax.annotate( f"Subtypes for which accuracy >80% = {np.sum(median_pct_correct_predictions_by_subtype >= 80)}",
                   xy= (0.55,  0.2),  xycoords='figure fraction',  horizontalalignment='left',   color='dimgray', fontsize=12 )
  ax.annotate( f"Subtypes for which accuracy >90% = {np.sum(median_pct_correct_predictions_by_subtype >= 90)}",
                   xy= (0.77,  0.2),  xycoords='figure fraction',  horizontalalignment='left',   color='dimgray', fontsize=12 )

  if len(labels) > 15:
    ax.annotate( f"{text_3}", xy= (0.033,  0.1475),    xycoords='figure fraction',  horizontalalignment='left', color='blue',   fontsize=font_med  ) 
    ax.annotate( f"{text_2}", xy= (0.033,  0.1275),    xycoords='figure fraction',  horizontalalignment='left', color='green', fontsize=font_med  ) 
    ax.annotate( f"{text_1}", xy= (0.033,  0.1015),    xycoords='figure fraction',  horizontalalignment='left', color='red',  fontsize=font_med  ) 
  

  totals            = total_predictions_by_subtype
  corrects          = correct_predictions_by_subtype
  headline_correct  = np.around( np.sum(corrects)/np.sum(totals)*100,  0 )
  best_correct      = 0 if ( np.around( np.max(corrects/totals)*100 ).astype(int) ) < 1 else ( np.around( np.max(corrects/totals)*100 ).astype(int) )
  
  if (DEBUG>9):
    np.set_printoptions(formatter={ 'float' : lambda x: f"   {CARRIBEAN_GREEN}{x:6.3f}   "} ) 
    print ( f"CLASSI:           INFO:  headline_correct                           = {MIKADO}{headline_correct}{RESET}",  flush=True )
    print ( f"CLASSI:           INFO:  totals                                     = {MIKADO}{totals}{RESET}",            flush=True )
    print ( f"CLASSI:           INFO:  corrects                                   = {MIKADO}{corrects}{RESET}",          flush=True )
    print ( f"CLASSI:           INFO:  best_correct                               = {MIKADO}{best_correct:02d}{RESET}",  flush=True )
    
    
  for patch, color in zip( bp['boxes'], subtype_colors):
    patch.set_facecolor(color)
        
  for xtick in ax.get_xticks():                                                                            # get_xticks = get coordinates of xticks
    
    total    = totals   [xtick-1]
    correct  = corrects [xtick-1]
    percent  = 100*corrects[xtick-1]/totals[xtick-1]
    median   = median_pct_correct_predictions_by_subtype[xtick-1]
    random   = expected_IFF_random_preds[xtick-1]
    
    if len(labels) > 15:
      ax.text( x=xtick, y=base,                         s=f"{total:,}",                                horizontalalignment='center',  color='red',        fontsize=font_med   ) 
      ax.text( x=xtick, y=base+gap_1,                   s=f"{correct:,}",                              horizontalalignment='center',  color='green',      fontsize=font_med   )     
      ax.text( x=xtick, y=base+gap_1+gap_2,             s=f"{median:2.0f}%",                           horizontalalignment='center',  color='blue',       fontsize=font_med   )    
      ax.text( x=xtick, y=random-1.5,                   s=f"{text_4}",                                 horizontalalignment='center',  color='lightcoral', fontsize=font_tiny  )    
      plt.plot( [xtick-0.27, xtick+0.27], [random, random],           linewidth=1,     linestyle="--",                                color='lightcoral'                      )
    else:
      ax.text( x=xtick, y=base,                         s=f"{text_1}{total:,}",                        horizontalalignment='center',  color='dimgray',        fontsize=font_med   ) 
      ax.text( x=xtick, y=base+gap_1,                   s=f"{text_2}{correct:,}",                      horizontalalignment='center',  color='dimgray',      fontsize=font_med   )     
      ax.text( x=xtick, y=base+gap_1+gap_2,             s=f"{text_3}{median:2.0f}%",                   horizontalalignment='center',  color='dimgray',       fontsize=font_med   )    
      ax.text( x=xtick, y=random-1.5,                     s=f"{text_4}",                               horizontalalignment='center',  color='lightcoral', fontsize=font_tiny  )    
      plt.plot( [xtick-0.27, xtick+0.27], [random, random],           linewidth=1,     linestyle="--",                                color='lightcoral'                      )      
    

    if (DEBUG>99):
      print ( f"CLASSI:           INFO:  xtick                                    = {MIKADO}{xtick}{RESET}",  flush=True )
      print ( f"CLASSI:           INFO:  total                                    = {MIKADO}{total}{RESET}",  flush=True )

  if args.box_plot_show == "True":
    plt.show()
  
  writer.add_figure('Box Plot V', fig, 1)
  
  fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_AGG_{headline_correct.astype(int):02d}_BEST_{best_subtype_median:03d}__box_port"
  fqn = f"{fqn[0:255]}.png"
  fig.savefig(fqn)
    
  plt.close()
  
  


  # Render landscape version of box plot

  figure_width  = 30
  figure_height = 40


  if len(labels) < 8:
    rotation = 90
    font_big = 20
    font_med = 18
    font_sml = 16
    base     = 0.75
    gap_1    = 2.75
    gap_2    = 2.5
    text_1="predictions="
    text_2="correct="
    text_3="median correct all runs="    
    text_4="expected for random"
    c_m = f"plt.cm.{eval('args.colour_map')}"                                                              # the 'eval' is so that the user input string will be treated as a variable
    subtype_colors = [ eval(c_m)(i) for i in range(len(labels))]                                           # makes an array of colours by calling the user defined colour map (which is a function, not a variable)  
  else:
    rotation = 90
    font_big = 18
    font_med = 16
    font_sml = 14
    base     = 0.75
    gap_1    = 1.3
    gap_2    = 1.3
    text_1="preds="
    text_2="correct="
    text_3="median="    
    text_4="random"
    subtype_colors = pan_cancer_subtype_colors
     

        
  fig, ax       = plt.subplots( figsize=( figure_width, figure_height ), constrained_layout=False )

  plt.yticks    (  fontsize=font_sml                                         )
  plt.xticks    (  range( 0, 100, 10), fontsize=font_med                     )
  fig.suptitle  (  supertitle,  color='black',  weight='bold', fontsize=16   )
  ax.set_title  (  title,       color='black',                 fontsize=14   ) 
  ax.set        (  xlim =(0, 100)                                            )
  ax.xaxis.grid (  True, linestyle='dotted'                                  )  
  ax.yaxis.grid (  True, linestyle='dashed', color='lightgrey'               )  


  alpha_lite=0.3
  alpha_hard=0.6
  line_props  = dict( color="black", alpha=alpha_lite, linewidth=2           )
  box_props   = dict( color="black",                   linestyle="dashdot"   )
  cap_props   = dict( color="black", alpha=alpha_lite                        )
  flier_props = dict( marker="o",    markersize=7                            )

  bp      = plt.boxplot( pct_correct_predictions_plane, labels=labels, vert=False, patch_artist=True, showfliers=True,  medianprops=dict(color="black", alpha=alpha_hard), boxprops=box_props, whiskerprops=line_props, capprops=cap_props, flierprops=flier_props )

  ax.annotate( f"Total predictions made {np.sum(all_predictions_plane):,}; of which correct: {np.sum(correct_predictions_plane):,} ({100*np.sum(correct_predictions_plane)/np.sum(all_predictions_plane):.1f}%)",
                   xy= (0.01,  0.02),    xycoords='figure fraction',  horizontalalignment='left', color='dimgray', fontsize=15  ) 
  ax.annotate( f"Subtypes for which accuracy >80% = {np.sum(median_pct_correct_predictions_by_subtype >= 80)}",
                   xy= (0.50,  0.02),  xycoords='figure fraction',  horizontalalignment='left',   color='dimgray', fontsize=15 )
  ax.annotate( f"Subtypes for which accuracy >90% = {np.sum(median_pct_correct_predictions_by_subtype >= 90)}",
                   xy= (0.75,  0.02),  xycoords='figure fraction',  horizontalalignment='left',   color='dimgray', fontsize=15 )
  plt.xlabel (  'subtypes correctly predicted (%)', weight='bold', fontsize=font_big   )

  totals          = total_predictions_by_subtype
  corrects        = correct_predictions_by_subtype
    
    
  for patch, color in zip( bp['boxes'], subtype_colors):
    patch.set_facecolor(color)
        
  for ytick in ax.get_yticks():                                                                            # get_yticks = get coordinates of yticks

    total    = totals   [ytick-1]
    correct  = corrects [ytick-1]
    percent  = 100*corrects[ytick-1]/totals[ytick-1]
    median   = median_pct_correct_predictions_by_subtype[ytick-1]
    random   = expected_IFF_random_preds[ytick-1]
    
    ax.text( x=1,            y=ytick,       s=f"{text_1}{total:,}",                                        horizontalalignment='left',     color='#202020',     fontsize=10  ) 
    ax.text( x=10,           y=ytick,       s=f"{text_2}{correct:,}",                                      horizontalalignment='left',     color='#202020',     fontsize=10  )    
    ax.text( x=17,           y=ytick,       s=f"({percent:2.1f}%)",                                        horizontalalignment='left',     color='#202020',     fontsize=10  )    
    ax.text( x=22,           y=ytick,       s=f"{text_3}{median:2.1f}%",                                   horizontalalignment='left',     color='#202020',     fontsize=10  )  
    ax.text( x=random-0.7,   y=ytick,       s=f"{text_4}",                    rotation=rotation,           verticalalignment  ='center',   color='hotpink',     fontsize=7   )    
    plt.plot( [random, random],  [ytick-0.25, ytick+0.25],                    linewidth=1,  linestyle="-",                                   color='hotpink'                   )

   
    if (DEBUG>99):
      print ( f"CLASSI:           INFO:  ytick        = {MIKADO}{ytick}{RESET}",  flush=True )
      print ( f"CLASSI:           INFO:  total        = {MIKADO}{total}{RESET}",  flush=True )

 
  if args.box_plot_show == "True":
    plt.show()
  
  writer.add_figure('Box Plot H', fig, 1)
  
  
  fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_AGG_{headline_correct.astype(int):02d}_BEST_{best_subtype_median:03d}__box_land"
  fqn = f"{fqn[0:255]}.png"
  fig.savefig(fqn)
    
  plt.close()



    
  return

# --------------------------------------------------------------------------------------------  
def show_classifications_matrix( writer, total_runs_in_job, pct_test, epoch, pandas_matrix, class_names,  level ):

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

  index_names = class_names.copy()   
    
  #print ( "" )                                                                                            # peel off an all numbers pandas version to use for graphing etc
  pandas_version = pd.DataFrame( pandas_matrix, columns=class_names, index=index_names )
  #print(tabulate(pandas_version, headers='keys', tablefmt = 'psql'))     

  index_names.append( "subtype totals"  )
  index_names.append( "subtype correct" ) 
  index_names.append( "percent correct" )

  pandas_version_ext = pd.DataFrame( ext3_pandas_matrix, columns=class_names, index=index_names )     # this version has subtotals etc at the bottom so it's just for display  

  if DEBUG>4:
    print(tabulate( pandas_version_ext, headers='keys', tablefmt = 'fancy_grid' ) )   
  
  #display(pandas_version_ext)
 
 
  # (1) Save job level classification matrix as a csv file in logs directory

  if level=='job':

    now              = datetime.datetime.now()
    fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_conf_matrix_job"
    fqn = f"{fqn[0:255]}.csv"

    try:
      pandas_version.to_csv( fqn, sep='\t' )
      if DEBUG>0:
        print ( f"CLASSI:         INFO:     saving          job level confusion matrix to {MAGENTA}{fqn}{RESET}"  )
    except Exception as e:
      print ( f"{RED}CLASSI:         FATAL:     could not save file {MAGENTA}{fqn}{RESET}"  )
      print ( f"{RED}CLASSI:         FATAL:     error was: {e}{RESET}" )
      sys.exit(0)    
    
    fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}_{descriptor}_conf_matrix_job_ext"
    fqn = f"{fqn[0:255]}.csv"
    try:
      pandas_version_ext.to_csv( fqn, sep='\t' )
      if DEBUG>0:
        print ( f"CLASSI:         INFO:     saving extended job level confusion matrix to {MAGENTA}{fqn}{RESET}"  )
    except Exception as e:
      print ( f"{RED}CLASSI:         FATAL:     could not save file         = {MAGENTA}{fqn}{RESET}"  )
      print ( f"{RED}CLASSI:         FATAL:     error was: {e}{RESET}" )      
      sys.exit(0)
  
  return ( total_correct_by_subtype, total_examples_by_subtype )


# --------------------------------------------------------------------------------------------
def triang( df ):

  print( f"{BRIGHT_GREEN}CLASSI:         INFO: at top of triang(){RESET} ")  
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

  print( f"{MIKADO}CLASSI:         INFO: at top of color_vals(){RESET} ")   
  """
  Color dataframe using values
  """
  d = {'up' : 'orange',
       'dg' : 'black',
       'lt' : 'blue'}
  return [f'color : {i}' for i in triang(df_vals).loc[val.name, val.index].map(d).tolist()] 

# --------------------------------------------------------------------------------------------
def color_negative_red(val):  # not currently used

    #pd_percentage_correct_plane.style.applymap(color_negative_red) 

    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 1 else 'white'
    return 'color: %s' % color
    
# --------------------------------------------------------------------------------------------


if __name__ == '__main__':
  
  if DEBUG>1:
    print ( f"{BLEU}{sys.argv[1:]}{RESET}" );
  
  # using this to handle booleans in user parms, which argparse doesn't support
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

  p.add_argument('--repeat',                                                        type=int,    default=1                                       )
  p.add_argument('--skip_tiling',                                                   type=str,    default='False'                                 )                                
  p.add_argument('--skip_generation',                                               type=str,    default='False'                                 )                                
  p.add_argument('--pretrain',                                                      type=str,    default='False'                                 )                                
  p.add_argument('--log_dir',                                                       type=str,    default='logs'                                  )                
  p.add_argument('--base_dir',                                                      type=str,    default='/home/peter/git/pipeline'              )
  p.add_argument('--application_dir',                                               type=str,    default='/home/peter/git/pipeline/classi'       )
  p.add_argument('--data_dir',                                                      type=str,    default='/home/peter/git/pipeline/working_data' )     
  p.add_argument('--save_model_name',                                               type=str,    default='model.pt'                              )                             
  p.add_argument('--save_model_every',                                              type=int,    default=10                                      )                                     
  p.add_argument('--rna_file_name',                                                 type=str,    default='rna.npy'                               )                              
  p.add_argument('--rna_file_suffix',                                               type=str,    default='*FPKM-UQ.txt'                          )                        
  p.add_argument('--embedding_file_suffix_rna',                                     type=str                                                     )                        
  p.add_argument('--embedding_file_suffix_image',                                   type=str                                                     )                        
  p.add_argument('--embedding_file_suffix_image_rna',                               type=str                                                     )                        
  p.add_argument('--rna_file_reduced_suffix',                                       type=str,    default='_reduced'                              )                             
  p.add_argument('--use_unfiltered_data',                                           type=str2bool, nargs='?', const=True, default=True, help="If true, don't filter the genes, but rather use all of them"     )
  p.add_argument('--class_numpy_file_name',                                         type=str,    default='class.npy'                            )                            
  p.add_argument('--wall_time',                                                     type=int,    default=24                                     )
  p.add_argument('--seed',                                                          type=int,    default=0                                      )
  p.add_argument('--mode',                                                          type=str,    default='classify'                             )
  p.add_argument('--use_same_seed',                                                 type=str,    default='False'                                )
  p.add_argument('--nn_type_img',                                       nargs="+",  type=str,    default='VGG11'                                )
  p.add_argument('--nn_type_rna',                                       nargs="+",  type=str,    default='DENSE'                                )
  p.add_argument('--hidden_layer_encoder_topology', '--nargs-int-type', nargs='*',  type=int,                                                   )                             
  p.add_argument('--encoder_activation',                                            type=str,    default='none'                                 )                              
  p.add_argument('--nn_dense_dropout_1',                                nargs="+",  type=float,  default=0.0                                    )                                    
  p.add_argument('--nn_dense_dropout_2',                                nargs="+",  type=float,  default=0.0                                    )                                    
  p.add_argument('--dataset',                                                       type=str                                                    )
  p.add_argument('--cases',                                                         type=str,    default='UNIMODE_CASE'                         )
  p.add_argument('--divide_cases',                                                  type=str,    default='False'                                )
  p.add_argument('--cases_reserved_for_image_rna',                                  type=int                                                    )
  p.add_argument('--data_source',                                                   type=str                                                    )
  p.add_argument('--global_data',                                                   type=str                                                    )
  p.add_argument('--mapping_file_name',                                             type=str,    default='mapping_file'                         )
  p.add_argument('--target_genes_reference_file',                                   type=str                                                    )
  p.add_argument('--input_mode',                                                    type=str,    default='NONE'                                 )
  p.add_argument('--multimode',                                                     type=str,    default='NONE'                                 )
  p.add_argument('--n_samples',                                         nargs="+",  type=int,    default="101"                                  )                                    
  p.add_argument('--n_tiles',                                           nargs="+",  type=int,    default="50"                                   )       
  p.add_argument('--make_balanced',                                                 type=str,    default='True'                                 )
  p.add_argument('--highest_class_number',                                          type=int,    default="777"                                  )                                                             
  p.add_argument('--supergrid_size',                                                type=int,    default=1                                      )                                      
  p.add_argument('--patch_points_to_sample',                                        type=int,    default=1000                                   )                                   
  p.add_argument('--tile_size',                                         nargs="+",  type=int,    default=128                                    )                                    
  p.add_argument('--gene_data_norm',                                    nargs="+",  type=str,    default='NONE'                                 )                                 
  p.add_argument('--gene_data_transform',                               nargs="+",  type=str,    default='NONE'                                 )
  p.add_argument('--n_genes',                                                       type=int,    default=555                                    )                                   
  p.add_argument('--batch_size',                                        nargs="+",  type=int,   default=64                                      )                                     
  p.add_argument('--learning_rate',                                     nargs="+",  type=float, default=.007                                    )                                 
  p.add_argument('--tsne_learning_rate',                                nargs="+",  type=float, default=10.0                                    )                                 
  p.add_argument('--n_epochs',                                                      type=int,   default=17                                      )
  p.add_argument('--n_iterations',                                                  type=int,   default=251                                     )
  p.add_argument('--pct_test',                                          nargs="+",  type=float, default=0.2                                     )
  p.add_argument('--final_test_batch_size',                             nargs="?",  type=int,   default=1000                                    )                                   
  p.add_argument('--lr',                                                nargs="+",  type=float, default=0.0001                                  )
  p.add_argument('--latent_dim',                                                    type=int,   default=7                                       )
  p.add_argument('--l1_coef',                                                       type=float, default=0.1                                     )
  p.add_argument('--em_iters',                                                      type=int,   default=1                                       )
  p.add_argument('--clip',                                                          type=float, default=1                                       )
  p.add_argument('--max_consecutive_losses',                                        type=int,   default=7771                                    )
  p.add_argument('--optimizer',                                         nargs="+",  type=str,   default='ADAM'                                  )
  p.add_argument('--label_swap_pct',                                                type=float, default=0.0                                     )                                    
  p.add_argument('--make_grey_pct',                                                 type=float, default=0.0                                     ) 
  p.add_argument('--peer_noise_pct',                                                type=float, default=0.0                                     ) 
  p.add_argument('--regenerate',                                                    type=str,   default='True'                                  )
  p.add_argument('--just_profile',                                                  type=str,   default='False'                                 )                        
  p.add_argument('--just_test',                                                     type=str,   default='False'                                 )                        
  p.add_argument('--rand_tiles',                                                    type=str,   default='True'                                  )                         
  p.add_argument('--points_to_sample',                                              type=int,   default=100                                     )                            
  p.add_argument('--min_uniques',                                                   type=int,   default=0                                       )                              
  p.add_argument('--min_tile_sd',                                                   type=float, default=3                                       )                              
  p.add_argument('--greyness',                                                      type=int,   default=0                                       )                              
  p.add_argument('--stain_norm',                                        nargs="+",  type=str,   default='NONE'                                  )                         
  p.add_argument('--stain_norm_target',                                             type=str,   default='NONE'                                  )                         
  p.add_argument('--cancer_type',                                                   type=str,   default='NONE'                                  )                 
  p.add_argument('--cancer_type_long',                                              type=str,   default='NONE'                                  )                 
  p.add_argument('--class_colours',                                     nargs="*"                                                               )                 
  p.add_argument('--colour_map',                                                    type=str,   default='tab20'                                 )    
  p.add_argument('--target_tile_coords',                                nargs=2,    type=int,   default=[2000,2000]                             )                 
  p.add_argument('--zoom_out_prob',                                     nargs="*",  type=float,                                                 )                 
  p.add_argument('--zoom_out_mags',                                     nargs="*",  type=float,                                                   )                 

  p.add_argument('--a_d_use_cupy',                                                  type=str,   default='True'                                  )                    
  p.add_argument('--cutoff_percentile',                                 nargs="+",  type=float, default=100                                     )                    
  p.add_argument('--cov_uq_threshold',                                              type=float, default=0.0                                     )                    

  p.add_argument('--remove_unexpressed_genes',                                      type=str,   default='True'                                  )                               
  p.add_argument('--low_expression_threshold',                          nargs="+",  type=float, default=0.0                                     )                    


  p.add_argument('--figure_width',                                                  type=float, default=16                                      )                                  
  p.add_argument('--figure_height',                                                 type=float, default=16                                      )
  p.add_argument('--annotated_tiles',                                               type=str,   default='True'                                  )
  p.add_argument('--scattergram',                                                   type=str,   default='True'                                  )
  p.add_argument('--box_plot',                                                      type=str,   default='True'                                  )
  p.add_argument('--box_plot_show',                                                 type=str,   default='True'                                  )
  p.add_argument('--minimum_job_size',                                              type=float, default=5                                       )
  p.add_argument('--probs_matrix',                                                  type=str,   default='True'                                  )
  p.add_argument('--probs_matrix_interpolation',                                    type=str,   default='none'                                  )
  p.add_argument('--show_patch_images',                                             type=str,   default='True'                                  )    
  p.add_argument('--show_rows',                                                     type=int,   default=500                                     )                            
  p.add_argument('--show_cols',                                                     type=int,   default=100                                     ) 
  p.add_argument('--bar_chart_x_labels',                                            type=str,   default='rna_case_id'                           )
  p.add_argument('--bar_chart_show_all',                                            type=str,   default='True'                                  )
  p.add_argument('--bar_chart_sort_hi_lo',                                          type=str,   default='True'                                  )
  p.add_argument('-ddp', '--ddp',                                                   type=str,   default='False'                                 )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end     )
  p.add_argument('-n', '--nodes',                                                   type=int,   default=1,  metavar='N'                         )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end     )
  p.add_argument('-g', '--gpus',                                                    type=int,   default=1,  help='number of gpus per node'      )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end     )
  p.add_argument('-nr', '--nr',                                                     type=int,   default=0,  help='ranking within node'          )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end     )
  
  p.add_argument('--hidden_layer_neurons',                              nargs="+",  type=int,    default=2000                                   )     
  p.add_argument('--embedding_dimensions',                              nargs="+",  type=int,    default=1000                                   )    
  
  p.add_argument('--use_autoencoder_output',                                        type=str,   default='False'                                 ) # if "True", use file containing auto-encoder output (which must exist, in log_dir     ) as input rather than the usual input (e.g. rna-seq values     )
  p.add_argument('--ae_add_noise',                                                  type=str,   default='False'                                 )
  p.add_argument('--clustering',                                                    type=str,   default='NONE'                                  )
  p.add_argument('--n_clusters',                                                    type=int,                                                   )
  p.add_argument('--metric',                                                        type=str,   default="manhattan"                             )        
  p.add_argument('--epsilon',                                                       type=float, default="0.5"                                   )        
  p.add_argument('--perplexity',                                        nargs="+",  type=float, default=30.                                     )        
  p.add_argument('--momentum',                                                      type=float, default=0.8                                     )        
  p.add_argument('--min_cluster_size',                                              type=int,   default=3                                       )        
  p.add_argument('--render_clustering',                                             type=str,   default="False"                                 )        

  p.add_argument('--names_column',                                                  type=str, default="type_s"                                  )
  p.add_argument('--case_column',                                                   type=str, default="bcr_patient_uuid"                        )
  p.add_argument('--class_column',                                                  type=str, default="type_n"                                  )


  args, _ = p.parse_known_args()

  is_local = args.log_dir == 'experiments/example'

  args.n_workers  = 0 if is_local else 12
  args.pin_memory = torch.cuda.is_available()

  if DEBUG>2:
    print ( f"{GOLD}args.zoom_out_prob{RESET} =           ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------>    {YELLOW}{args.zoom_out_prob}{RESET}", flush=True)
    print ( f"{GOLD}args.zoom_out_mags{RESET} =           ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------>    {YELLOW}{args.zoom_out_mags}{RESET}", flush=True)
  
  main(args)
