"""============================================================================= 
Train LENET5
============================================================================="""

import sys
import math
import time
import torch
import cuda
import pplog
import argparse
import datetime
import matplotlib
import torchvision
import torch.utils.data

import numpy                 as np
import pandas                as pd
import seaborn               as sns
import matplotlib.pyplot     as plt
import matplotlib.lines      as mlines
import matplotlib.patches    as mpatches
import matplotlib.gridspec   as gridspec

from   pathlib                      import Path
from   random                       import randint
from   matplotlib.colors            import ListedColormap
from   matplotlib                   import cm
from   matplotlib.ticker            import (AutoMinorLocator, MultipleLocator)
from   sklearn                      import metrics 
from   pandas.plotting              import table
from   tabulate                     import tabulate
from   IPython.display              import display 

from   torch                        import optim
from   torch.nn.utils               import clip_grad_norm_
from   torch.nn                     import functional
from   torch.nn                     import DataParallel
from   itertools                    import product, permutations
from   PIL                          import Image
from   torch.utils.tensorboard      import SummaryWriter
from   torchvision                  import datasets, transforms

from   data                         import loader
from   data.dlbcl_image.config      import GTExV6Config
from   data.dlbcl_image.generate    import generate
from   models                       import LENETIMAGE
from   tiler_scheduler              import *
from   tiler_threader               import *
from   tiler_set_target             import *
from   tiler                        import *
from   otsne                        import otsne
from   sktsne                       import sktsne
from   _dbscan                      import _dbscan
from   h_dbscan                     import h_dbscan
# ~ from   plotly_play                  import plotly_play

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

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
AUREOLIN='\033[38;2;253;238;0m'
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

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)

global global_batch_count


run_level_total_correct             = []

global_batch_count    = 0
total_runs_in_job     = 0
final_test_batch_size = 0


# ------------------------------------------------------------------------------

#@profile
def main(args):


  """Main program: train -> test once per epoch
  """
  
  os.system("taskset -p 0xffffffff %d" % os.getpid())
  
  now = time.localtime(time.time())
  print(time.strftime( f"TRAINLENEJ:     INFO:  start time = %Y-%m-%d %H:%M:%S %Z", now ))
  start_time = time.time() 

  if DEBUG>2:
    print ( f"TRAINLENEJ:     INFO:     torch       version =    {MIKADO}{torch.__version__}{RESET}" )
    print ( f"TRAINLENEJ:     INFO:     torchvision version =    {MIKADO}{torchvision.__version__}{RESET}"  )
    print ( f"TRAINLENEJ:     INFO:     matplotlib  version =    {MIKADO}{matplotlib.__version__}{RESET}"   ) 
    print ( f"TRAINLENEJ:     INFO:     torchvision version =    {MIKADO}{torchvision.__version__}{RESET}"  )
    print ( f"TRAINLENEJ:     INFO:     seaborn     version =    {MIKADO}{sns.__version__}{RESET}"  )
    print ( f"TRAINLENEJ:     INFO:     pandas      version =    {MIKADO}{pd.__version__}{RESET}"  )  
  
  
  mode = 'TRAIN' if args.just_test!='True' else 'TEST'

  print( f"{GREY_BACKGROUND}TRAINLENEJ:     INFO:  common args:  \
{WHITE}mode={AUREOLIN}{mode}{WHITE}, \
input={AUREOLIN}{args.input_mode}{WHITE}, \
network={AUREOLIN}{args.nn_mode}{WHITE}, \
multimode={AUREOLIN}{args.multimode}{WHITE}, \
cases={AUREOLIN}{args.cases}{WHITE}, \
dataset={AUREOLIN}{args.dataset}{WHITE}, \
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
    print( f"{GREY_BACKGROUND}TRAINLENEJ:     INFO:  image args:   \
{WHITE}nn_type_img={AUREOLIN}{args.nn_type_img}{WHITE},\
use_tiler={AUREOLIN}{args.use_tiler}{WHITE},\
n_tiles={AUREOLIN}{args.n_tiles}{WHITE},\
h_class={AUREOLIN}{args.highest_class_number}{WHITE},\
tile_size={AUREOLIN}{args.tile_size}{WHITE},\
rand_tiles={AUREOLIN}{args.rand_tiles}{WHITE},\
greyness<{AUREOLIN}{args.greyness}{WHITE},\
sd<{AUREOLIN}{args.min_tile_sd}{WHITE},\
min_uniques>{AUREOLIN}{args.min_uniques}{WHITE},\
latent_dim={AUREOLIN}{args.latent_dim}{WHITE},\
label_swap={AUREOLIN}{args.label_swap_perunit}{WHITE},\
make_grey={AUREOLIN}{args.make_grey_perunit}{WHITE},\
stain_norm={AUREOLIN}{args.stain_norm,}{WHITE},\
annotated_tiles={AUREOLIN}{args.annotated_tiles}{WHITE},\
probs_matrix_interpolation={AUREOLIN}{args.probs_matrix_interpolation}{WHITE}\
                {RESET}"
, flush=True )

  elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
    print( f"{GREY_BACKGROUND}TRAINLENEJ:     INFO:  rna-seq args: \
nn_type_rna={CYAN}{args.nn_type_rna}{WHITE},\
hidden_layer_neurons={MIKADO}{args.hidden_layer_neurons}{WHITE}, \
gene_embed_dim={MIKADO}{args.gene_embed_dim}{WHITE}, \
nn_dense_dropout_1={MIKADO}{args.nn_dense_dropout_1}{WHITE}, \
nn_dense_dropout_2={MIKADO}{args.nn_dense_dropout_2}{WHITE}, \
n_genes={MIKADO}{args.n_genes}{WHITE}, \
gene_norm={YELLOW if not args.gene_data_norm[0]=='NONE'    else YELLOW if len(args.gene_data_norm)>1       else MIKADO}{args.gene_data_norm}{WHITE}, \
g_xform={YELLOW if not args.gene_data_transform[0]=='NONE' else YELLOW if len(args.gene_data_transform)>1  else MIKADO}{args.gene_data_transform}{WHITE} \
                                                                                  {RESET}"
, flush=True )

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
  use_tiler                     = args.use_tiler
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
  n_tiles                       = args.n_tiles
  n_epochs                      = args.n_epochs
  pct_test                      = args.pct_test
  batch_size                    = args.batch_size
  lr                            = args.learning_rate
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
  make_grey_perunit             = args.make_grey_perunit
  stain_norm                    = args.stain_norm
  stain_norm_target             = args.stain_norm_target
  annotated_tiles               = args.annotated_tiles
  figure_width                  = args.figure_width
  figure_height                 = args.figure_height  
  probs_matrix_interpolation    = args.probs_matrix_interpolation
  max_consecutive_losses        = args.max_consecutive_losses
  target_tile_coords            = args.target_tile_coords
  
  base_dir                      = args.base_dir
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
  bar_chart_x_labels            = args.bar_chart_x_labels
  bar_chart_sort_hi_lo          = args.bar_chart_sort_hi_lo
  remove_unexpressed_genes      = args.remove_unexpressed_genes
  remove_low_expression_genes   = args.remove_low_expression_genes
  low_expression_threshold      = args.low_expression_threshold
  encoder_activation            = args.encoder_activation
  hidden_layer_neurons          = args.hidden_layer_neurons
  gene_embed_dim                = args.gene_embed_dim
  
  use_autoencoder_output        = args.use_autoencoder_output  
  clustering                    = args.clustering  
  metric                        = args.metric  
  perplexity                    = args.momentum  
  momentum                      = args.perplexity  

  global last_stain_norm                                                                                   # Need to remember this across runs in a job
  global last_gene_norm                                                                                    # Need to remember this across runs in a job
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
  
  global file_name_prefix
  global class_colors

  multimode_case_count = unimode_case_count = not_a_multimode_case_count = not_a_multimode_case____image_count = not_a_multimode_case____image_test_count = 0


  if  0 in highest_class_number:
    print( f"{RED}TRAINLENEJ:     FATAL:  config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{RED}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{RED}') is not permitted to have the value {MIKADO}0{RESET}", flush=True)
    print( f"{RED}TRAINLENEJ:     FATAL: ... halting now{RESET}" )
    sys.exit(0)

  if  1 in highest_class_number:
    print( f"\n{CHARTREUSE}TRAINLENEJ:     WARNG:  config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{CHARTREUSE}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{CHARTREUSE}') contains the value {MIKADO}1{RESET}{CHARTREUSE}, which seems very odd", flush=True)
    print( f"{CHARTREUSE}TRAINLENEJ:     WARNG: ... continuing{RESET}" )
    time.sleep(4)

  if  2 in highest_class_number:
    print( f"\n{CHARTREUSE}TRAINLENEJ:     WARNG:  config setting '{CYAN}HIGHEST_CLASS_NUMBER{RESET}{CHARTREUSE}' (corresponding to python argument '{CYAN}--highest_class_number{RESET}{CHARTREUSE}') contains the value {MIKADO}2{RESET}{CHARTREUSE}, which is very low. Was this intentional?", flush=True)
    print( f"{CHARTREUSE}TRAINLENEJ:     WARNG: ... continuing{RESET}" )
    time.sleep(4)

  if sum(args.zoom_out_prob)!=1:
    print( f"\r{RESET}{ORANGE}TRAINLENEJ:     WARNG: the probabilities contained in configuration vectors '{CYAN}args.zoom_out_prob{RESET}{ORANGE}' do not add up to {MIKADO}1{RESET}{ORANGE} (FYI they add up to {MIKADO}{sum(args.zoom_out_prob)}{RESET}{ORANGE}) ... adjusting  first entry to make the total equal {MIKADO}1{RESET}", flush=True)
    
    first_entry = 1 - sum(args.zoom_out_prob[1:])
    args.zoom_out_prob[0] = first_entry

  if args.clustering == 'NONE':
    if  min(args.tile_size)<32:
      print( f"{RED}TRAINLENEJ:     FATAL:  config setting '{CYAN}TILE_SIZE{RESET}{RED}' (corresponding to python argument '{CYAN}--tile_size{RESET}{RED}') is not permitted to be less than {MIKADO}32{RESET}", flush=True)
      print( f"{RED}TRAINLENEJ:     FATAL: ... halting now{RESET}" )
      sys.exit(0)

  if  ( pretrain=='True' ) & ( input_mode=='image' ):
    print( f"{COTTON_CANDY}TRAINLENEJ:     INFO:  {CYAN}PRETRAIN{RESET}{COTTON_CANDY} option ({CYAN}-p True{RESET}{COTTON_CANDY}) (corresponding to python argument '{CYAN}--pretrain True{RESET}{COTTON_CANDY}') has been selected{RESET}", flush=True)

  if  ( pretrain=='True' ) & ( input_mode!='image' ):
    print( f"{RED}TRAINLENEJ:     FATAL: the {CYAN}PRETRAIN{RESET}{RED} option ({CYAN}-p True{RESET}{RED}) (corresponding to python argument '{CYAN}--pretrain True{RESET}{RED}') is only supported in image mode{RESET}", flush=True)
    print( f"{RED}TRAINLENEJ:     FATAL: ... halting now{RESET}" )
    sys.exit(0)

  if just_test=='False':
    if  not (  ( args.cases=='ALL_ELIGIBLE_CASES' ) | ( args.cases=='DESIGNATED_UNIMODE_CASE_FLAG' ) | ( args.cases=='DESIGNATED_MULTIMODE_CASE_FLAG' ) | ( args.cases=='NOT_A_MULTIMODE_CASE_FLAG' ) ):
      print( f"{RED}TRAINLENEJ:     FATAL: in training mode ('{CYAN}just_test=='False'{RESET}{RED})', user option  {CYAN}-c ('cases')  {RESET}{RED} = '{CYAN}{args.cases}{RESET}{RED}' is not supported{RESET}" )
      print( f"{RED}TRAINLENEJ:     FATAL: explanation:  in training mode the following options are supported: '{CYAN}ALL_ELGIBLE_CASES{RESET}{RED}', '{CYAN}DESIGNATED_UNIMODE_CASE_FLAG{RESET}{RED}', '{CYAN}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}{RED}', '{CYAN}NOT_A_MULTIMODE_CASE_FLAG{RESET}{RED}'" )
      print( f"{RED}TRAINLENEJ:     FATAL: ... halting now{RESET}" )
      sys.exit(0)
  else:
    if pretrain=='True':
      print( f"{RED}TRAINLENEJ:     FATAL: the {CYAN}PRETRAIN{RESET}{RED} option ({CYAN}-p True{RESET}{RED}) corresponding to python argument {CYAN}--pretrain True{RESET}{RED} is not supported in test mode (because it makes no sense){RESET}", flush=True)
      print( f"{RED}TRAINLENEJ:     FATAL: ... halting now{RESET}" )
      sys.exit(0)
    if ( input_mode=='image' ): 
      if  not ( ( args.cases=='ALL_ELIGIBLE_CASES' ) | ( args.cases=='NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG' ) | ( args.cases=='DESIGNATED_MULTIMODE_CASE_FLAG' )  ):
        print( f"{RED}TRAINLENEJ:     FATAL: in test mode ('{CYAN}just_test=='False'{RESET}{RED})', user option  {CYAN}-c ('cases')  {RESET}{RED} = '{CYAN}{args.cases}{RESET}{RED}' is not supported{RESET}" )
        print( f"{RED}TRAINLENEJ:     FATAL:   explanation:  in test mode ('{CYAN}just_test=='True'{RESET}{RED})' the following are supported: '{CYAN}ALL_ELIGIBLE_CASES{RESET}{RED}', '{CYAN}NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG{RESET}{RED}', '{CYAN}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}{RED}'" )
        print( f"{RED}TRAINLENEJ:     FATAL:   ... halting now{RESET}" )
        sys.exit(0)
      

  if  ( args.cases!='ALL_ELIGIBLE_CASES' ) & ( args.divide_cases == 'False' ):
    print( f"{RED}TRAINLENEJ:     CAUTION: user option {CYAN}-v ('divide_cases') {RESET}{RED} = {CYAN}False{RESET}{RED}, however option {CYAN}-c ('cases'){RESET}{RED} is NOT '{CYAN}ALL_ELIGIBLE_CASES{RESET}{RED}', so the requested subset of cases may or may not already exist{RESET}" )
    print( f"{RED}TRAINLENEJ:     CAUTION:   this will definitely cause problems unless the requested subset cases ({RESET}{RED}'{CYAN}{args.cases}{RESET}{RED}') already exist (in {RESET}{RED}'{CYAN}{args.data_dir}{RESET}{RED}') as a result of a previous run which had {CYAN}-v {'divide_cases'}{RESET}{RED} flag set" )
    print( f"{RED}TRAINLENEJ:     CAUTION:   ... NOT halting, but if the program crashes, you'll at least know the likely cause{RESET}" )
      
  c_m = f"plt.cm.{eval('colour_map')}"                                                                    # the 'eval' is so that the user input string will be treated as a variable
  class_colors = [ eval(c_m)(i) for i in range(len(args.class_names))]                                    # makes an array of colours by calling the user defined colour map (which is a function, not a variable)
  if DEBUG>555:
    print (f"TRAINLENEJ:     INFO:  class_colors = \n{MIKADO}{class_colors}{RESET}" )
    
  n_classes = len(args.class_names)
  run_level_classifications_matrix    =  np.zeros( (n_classes, n_classes), dtype=int )
  job_level_classifications_matrix    =  np.zeros( (n_classes, n_classes), dtype=int )
  # accumulator
  run_level_classifications_matrix_acc    =  np.zeros( ( 1000, n_classes,n_classes ), dtype=int )
  
  pplog.set_logfiles( log_dir )

  if ( input_mode=='image' ): 
    if 1 in batch_size:
      print ( f"{RED}TRAINLENEJ:     INFO: Sorry - parameter '{CYAN}BATCH_SIZE{RESET}{RED}' (currently '{MIKADO}{batch_size}{RESET}{RED}' cannot include a value <2 for images{RESET}" )
      print ( f"{RED}TRAINLENEJ:     INFO: halting now{RESET}" )      
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

  if ( args.cases=='DESIGNATED_MULTIMODE_CASE_FLAG' ):                                                                           
    if DEBUG>0:
      print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}args.cases{RESET}{ORANGE}' = {MAGENTA}{args.cases}{RESET}{ORANGE}! Therefore '{CYAN}N_SAMPLES{RESET}{ORANGE}' (currently {MIKADO}{n_samples[0]}{RESET}{ORANGE}) will be changed to the value of '{CYAN}CASES_RESERVED_FOR_IMAGE_RNA{RESET}{ORANGE} ({MIKADO}{args.cases_reserved_for_image_rna}{RESET}{ORANGE})" ) 
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
    print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set. No training will be performed{RESET}" )
    if n_epochs>1:
      print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set, so n_epochs (currently {MIKADO}{n_epochs}{RESET}{ORANGE}) has been set to {MIKADO}1{RESET}{ORANGE} for this run{RESET}" ) 
      n_epochs=1
    if ( multimode!='image_rna' ) & ( input_mode!='image_rna' ):
      print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set. Only one thread will be used for processing to ensure patch tiles will be processed in the correct sequence{RESET}" )
      if len(args.hidden_layer_neurons)>1:
        print( f"{RED}TRAINLENEJ:     INFO:  in test mode, ({CYAN}JUST_TEST=\"True\"{RESET}{RED}), only one value is allowed for the parameter '{CYAN}HIDDEN_LAYER_NEURONS{RESET}{RED}'. At the moment it has {MIKADO}{len(args.hidden_layer_neurons)}{RESET}{RED} values ... halting{RESET}" )
        sys.exit(0)        
      if input_mode=='image':
        if not tile_size_max**0.5 == int(tile_size_max**0.5):
          print( f"{RED}TRAINLENEJ:     INFO:  in test_mode, '{CYAN}TILE_SIZE{RESET}{ORANGE}' ({MIKADO}{tile_size}{RESET}{RED}) must be a perfect square (eg. 49, 64, 144, 256 ) ... halting [1586]{RESET}" )
          sys.exit(0)
      if len(batch_size)>1:
        print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'   flag is set but but '{CYAN}BATCH_SIZE{RESET}{ORANGE}' has {MIKADO}{len(batch_size)}{RESET}{ORANGE} values ({MIKADO}{batch_size}{RESET}{ORANGE}). Only the first value ({MIKADO}{batch_size[0]}{ORANGE}) will be used{RESET}" )
        del batch_size[1:]       
      if len(n_tiles)>1:
        print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'   flag is set but but '{CYAN}N_TILES{RESET}{ORANGE}'    has {MIKADO}{len(n_tiles)}{RESET}{ORANGE} values ({MIKADO}{n_tiles}{RESET}{ORANGE}). Only the first value ({MIKADO}{n_tiles[0]}{RESET}{ORANGE}) will be used{RESET}" )
        del n_tiles[1:] 
      n_tiles[0] = supergrid_size**2 * batch_size[0]
      print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set, therefore '{CYAN}N_TILES{RESET}{ORANGE}' has been set to '{CYAN}SUPERGRID_SIZE^2 * BATCH_SIZE{RESET}{ORANGE}' ({MIKADO}{supergrid_size} * {supergrid_size} * {batch_size} =  {n_tiles}{RESET} {ORANGE}) for this job{RESET}" )          
    else:
      print( f"{ORANGE}TRAINLENEJ:     INFO:   user argument  'MULTIMODE' = '{CHARTREUSE}{multimode}{RESET}{ORANGE}'. Embeddings will be generated.{RESET}"   )      
  else:
    if input_mode=='image':
      if not tile_size_max**0.5 == int(tile_size_max**0.5):
        print( f"{ORANGE}TRAINLENEJ:     WARNG: '{CYAN}TILE_SIZE{RESET}{CAMEL}' ({MIKADO}{tile_size_max}{RESET}{ORANGE}) isn't a perfect square, which is fine for training, but will mean you won't be able to use test mode on the model you train here{RESET}" )
      if supergrid_size>1:
        if DEBUG>99:
          print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'  flag is NOT set, so supergrid_size (currently {MIKADO}{supergrid_size}{RESET}{ORANGE}) will be ignored{RESET}" )
        args.supergrid_size=1

           
  if rand_tiles=='False':
    print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}RANDOM_TILES{RESET}{ORANGE}'  flag is not set. Tiles will be selected sequentially rather than at random. This is appropriate for test mode, but not training mode{RESET}" )     


  if ( input_mode=='image' ):
    
    # (1) make sure there are enough samples available to cover the user's requested "n_samples"
  
    image_file_count   = 0
  
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                      # each iteration takes us to a new directory under data_dir
  
      if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data      
        
        for f in files:
         
          if (   ( f.endswith( 'svs' ))  |  ( f.endswith( 'SVS' ))  | ( f.endswith( 'tif' ))  |  ( f.endswith( 'tiff' ))   ):
            image_file_count +=1
          
    if image_file_count<np.max(args.n_samples):
      print( f"{ORANGE}TRAINLENEJ:     WARNG:  there aren't enough samples. A file count reveals a total of {MIKADO}{image_file_count}{RESET}{ORANGE} SVS and TIF files in {MAGENTA}{args.data_dir}{RESET}{ORANGE}, whereas (the largest value in) user configuation parameter '{CYAN}N_SAMPLES[]{RESET}{ORANGE}' = {MIKADO}{np.max(args.n_samples)}{RESET})" ) 
      print( f"{ORANGE}TRAINLENEJ:     WARNG:  changing values of '{CYAN}N_SAMPLES{RESET}{ORANGE} larger than {RESET}{MIKADO}{image_file_count}{RESET}{ORANGE} to exactly {MIKADO}{image_file_count}{RESET}{ORANGE} and continuing" )
      args.n_samples = [  el if el<=image_file_count else image_file_count for el in args.n_samples   ]
      n_samples = args.n_samples
      
      
    else:
      print( f"TRAINLENEJ:     INFO:  {WHITE}a file count shows there is a total of {MIKADO}{image_file_count}{RESET} SVS and TIF files in {MAGENTA}{args.data_dir}{RESET}, which is sufficient to perform all requested runs (configured value of'{CYAN}N_SAMPLES{RESET}' = {MIKADO}{np.max(args.n_samples)}{RESET})" )

  if use_same_seed=='True':
    print( f"{ORANGE}TRAINLENEJ:     WARNG: '{CYAN}USE_SAME_SEED{RESET}{ORANGE}' flag is set. The same seed will be used for all runs in this job{RESET}" )
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
      print( f"{ORANGE}TRAINLENEJ:     WARN: there aren't enough samples. A file count reveals a total of {MIKADO}{rna_file_count}{RESET}{ORANGE} rna files in {MAGENTA}{args.data_dir}{RESET}{ORANGE}, whereas (the largest value in) user configuation parameter '{CYAN}N_SAMPLES{RESET}{ORANGE}' = {MIKADO}{np.max(args.n_samples)}{RESET})" ) 
      print( f"{ORANGE}TRAINLENEJ:     WARN: will change values in the config array '{CYAN}N_SAMPLES[]{RESET}{ORANGE}' which are greater than {RESET}{MIKADO}{rna_file_count}{RESET}{ORANGE} to exactly {MIKADO}{rna_file_count}{RESET}{ORANGE} and continue" )
      args.n_samples = [  el if el<=rna_file_count else rna_file_count for el in args.n_samples   ]
      n_samples      = args.n_samples

    else:
      if just_test!='True':
        print( f"TRAINLENEJ:     INFO:  {WHITE}a file count shows there is a total of {MIKADO}{rna_file_count}{RESET} rna files in {MAGENTA}{args.data_dir}{RESET}, which is sufficient to perform all requested runs (configured value of'{CYAN}N_SAMPLES{RESET}' = {MIKADO}{np.max(args.n_samples)}{RESET})" )
      else:
        print( f"TRAINLENEJ:     INFO:  {WHITE}a file count shows there is a total of {MIKADO}{rna_file_count}{RESET} rna files in {MAGENTA}{args.data_dir}{RESET}, which is sufficient to perform all requested runs (configured value of'{CYAN}N_TESTS{RESET}' = {MIKADO}{np.max(args.n_tests)}{RESET})" )


  if (DEBUG>0):
    print ( f"TRAINLENEJ:     INFO:  highest_class_number = {MIKADO}{highest_class_number}{RESET}",    flush=True)
    print ( f"TRAINLENEJ:     INFO:  n_samples            = {MIKADO}{n_samples}{RESET}",               flush=True)
    if ( input_mode=='image' ):
      print ( f"TRAINLENEJ:     INFO:  n_tiles              = {MIKADO}{n_tiles}{RESET}",                 flush=True)
      print ( f"TRAINLENEJ:     INFO:  tile_size            = {MIKADO}{tile_size}{RESET}",               flush=True)

    



  # (A)  SET UP JOB LOOP

  already_tiled=False
  already_generated=False
                          
  
  parameters = dict( 
                                 lr  =   lr,
                           pct_test  =   pct_test,
                          n_samples  =   n_samples,
                         batch_size  =   batch_size,
                            n_tiles  =   n_tiles,
               highest_class_number  =   highest_class_number,
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
  offset        = 14
  second_offset = 12

  total_runs_in_job = len(list(product(*param_values)))
    
  
  if DEBUG>0:
    print ( f"TRAINLENEJ:     INFO:  total_runs_in_job    =  {CARRIBEAN_GREEN}{total_runs_in_job}{RESET}"  )

  image_headings =\
f"\
\r\033[{start_column+0*offset}Clr\
\r\033[{start_column+1*offset}Cpct_test\
\r\033[{start_column+2*offset}Csamples\
\r\033[{start_column+3*offset}Cbatch_size\
\r\033[{start_column+4*offset}Ctiles/image\
\r\033[{start_column+5*offset}Chi_class_num\
\r\033[{start_column+6*offset}Ctile_size\
\r\033[{start_column+7*offset}Crand_tiles\
\r\033[{start_column+8*offset}Cnet_img\
\r\033[{start_column+9*offset}Coptimizer\
\r\033[{start_column+10*offset}Cstain_norm\
\r\033[{start_column+11*offset}Clabel_swap\
\r\033[{start_column+12*offset}Cgreyscale\
\r\033[{start_column+13*offset}Cjitter vector\
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
      for lr, pct_test, n_samples, batch_size, n_tiles, highest_class_number, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values):    
        print( f"{CARRIBEAN_GREEN}\
\r\033[2C\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{pct_test:<9.6f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{n_tiles:<5d}\
\r\033[{start_column+5*offset}C{highest_class_number:<2d}\
\r\033[{start_column+6*offset}C{tile_size:<3d}\
\r\033[{start_column+7*offset}C{rand_tiles:<5s}\
\r\033[{start_column+8*offset}C{nn_type_img:<10s}\
\r\033[{start_column+9*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+10*offset}C{stain_norm:<10s}\
\r\033[{start_column+11*offset}C{label_swap_perunit:<6.1f}\
\r\033[{start_column+12*offset}C{make_grey_perunit:<5.1f}\
\r\033[{start_column+13*offset}C{jitter:}\
{RESET}" )  

    elif input_mode=='rna':
      print(f"\n{UNDER}JOB:{RESET}")
      print(f"\033[2C\{rna_headings}{RESET}")
      
      for lr, pct_test, n_samples, batch_size, n_tiles, highest_class_number, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values):    

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

  if (just_test=='True') & (input_mode=='image') & (multimode!= 'image_rna'):   
    if not ( batch_size == int( math.sqrt(batch_size) + 0.5) ** 2 ):
      print( f"\033[31;1mTRAINLENEJ:     FATAL:  in test mode 'batch_size' (currently {batch_size}) must be a perfect square (4, 9, 16, 25 ...) to permit selection of a a 2D contiguous patch. Halting [2989].\033[m" )
      sys.exit(0)      

  
  # (B) RUN JOB LOOP

  run=0
  
  for lr, pct_test, n_samples, batch_size, n_tiles, highest_class_number, tile_size, rand_tiles, nn_type_img, nn_type_rna, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, nn_optimizer, stain_norm, gene_data_norm, gene_data_transform, label_swap_perunit, make_grey_perunit, jitter in product(*param_values): 
 
    if ( divide_cases == 'True' ):
      
      if just_test=='False':                                                                      
        multimode_case_count, unimode_case_count, not_a_multimode_case_count, not_a_multimode_case____image_count, not_a_multimode_case____image_test_count =     segment_cases( pct_test )  # boils down to setting flags in the directories of certain cases, esp. 'MULTIMODE_CASE_FLAG'
      else:
        print( f"{RED}TRAINLENEJ:     FATAL: user option  {CYAN}-v ('args.cases'){RESET}{RED} is not allowed in test mode ({CYAN}JUST_TEST=True{RESET}, {CYAN}--just_test 'True'{RESET}){RED}{RESET}" )
        print( f"{RED}TRAINLENEJ:     FATAL: explanation:  it will resegment the cases, meaning there is every chance cases you've trained on will end up in the test set{RESET}" )
        print( f"{RED}TRAINLENEJ:     FATAL: ... halting now{RESET}" )
        sys.exit(0)        
 
    use_unfiltered_data=""
    if use_unfiltered_data=='True':
      rna_genes_tranche="all_ENSG_genes_including_non_coding_genes"
    else:
      rna_genes_tranche=os.path.basename(target_genes_reference_file)    
    
    
    mags = ("_".join(str(z) for z in zoom_out_mags))
    prob = ("_".join(str(z) for z in zoom_out_prob))
    
    if input_mode=='image':
      file_name_prefix = f"_{args.cases[0:25]}_{args.dataset}_{nn_type_img}_runs_{total_runs_in_job}_e_{args.n_epochs:03d}_samps_{n_samples:03d}_tiles_{n_tiles:04d}_hi_clss_{highest_class_number:02d}_tlsz_{tile_size:03d}__mags_{mags}__probs_{prob}_bat_{batch_size:02d}_test_{int(100*pct_test):02d}_lr_{lr:01.5f}"
    elif input_mode=='rna':
      file_name_prefix = f"_{args.cases[0:25]}_{args.dataset}_{nn_type_rna}_runs_{total_runs_in_job}_e_{args.n_epochs:03d}_samps_{n_samples:03d}_hi_clss_{highest_class_number:02d}_bat_{batch_size:02d}_test_{int(100*pct_test):02d}_lr_{lr:01.5f}_hidd_{hidden_layer_neurons:04d}_dd_1_{int(100*nn_dense_dropout_1):04d}_tranche_{rna_genes_tranche}"
    else:
      file_name_prefix = f"_{args.cases[0:25]}_{args.dataset}_{nn_type_rna}_runs_{total_runs_in_job}_e_{args.n_epochs:03d}_samps_{n_samples:03d}_hi_clss_{highest_class_number:02d}_bat_{batch_size:02d}_test_{int(100*pct_test):02d}_lr_{lr:01.5f}_hidd_{hidden_layer_neurons:04d}_dd_1_{int(100*nn_dense_dropout_1):04d}_tranche_{rna_genes_tranche}"          

    # ~ if just_test=='True':
        # ~ print( f"{ORANGE}TRAINLENEJ:     INFO:  '{CYAN}JUST_TEST{RESET}{ORANGE}'     flag is set, so n_samples (currently {MIKADO}{n_samples}{RESET}{ORANGE}) has been set to {MIKADO}1{RESET}{ORANGE} for this run{RESET}" ) 
        # ~ n_samples = int(pct_test * n_samples )


    now              = datetime.datetime.now()    
    pplog.log_section(f"run = {now:%y-%m-%d %H:%M}   parameters = {file_name_prefix}")
    pplog.log_section(f"      zoom_out_mags = {zoom_out_mags}")
    pplog.log_section(f"      zoom_out_prob = {zoom_out_prob}")
    

    run+=1

    # accumulator
    if just_test!='True':
      aggregate_tile_probabilities_matrix =  np.zeros     ( ( n_samples, n_classes ),     dtype=float       )
      aggregate_tile_level_winners_matrix =  np.full_like ( aggregate_tile_probabilities_matrix, 0  )
      patches_true_classes                        =  np.zeros     ( ( n_samples            ),     dtype=int         )
      patches_case_id                             =  np.zeros     ( ( n_samples            ),     dtype=int         )    
      
      probabilities_matrix                        =  np.zeros     ( ( n_samples, n_classes ),     dtype=float       )              # same, but for rna        
      true_classes                                =  np.zeros     ( ( n_samples            ),     dtype=int         )              # same, but for rna 
      rna_case_id                                 =  np.zeros     ( ( n_samples            ),     dtype=int         )              # same, but for rna 
    else:
      aggregate_tile_probabilities_matrix =  np.zeros     ( ( args.n_tests, n_classes ),     dtype=float       )
      aggregate_tile_level_winners_matrix =  np.full_like ( aggregate_tile_probabilities_matrix, 0  )
      patches_true_classes                        =  np.zeros     ( ( args.n_tests            ),     dtype=int         )
      patches_case_id                             =  np.zeros     ( ( args.n_tests            ),     dtype=int         )    
      
      probabilities_matrix                        =  np.zeros     ( ( args.n_tests, n_classes ),     dtype=float       )              # same, but for rna        
      true_classes                                =  np.zeros     ( ( args.n_tests            ),     dtype=int         )              # same, but for rna 
      rna_case_id                                 =  np.zeros     ( ( args.n_tests            ),     dtype=int         )              # same, but for rna 
          

    if DEBUG>0:
      if input_mode=='image':
        print( f"\n\n{UNDER}RUN: { run} of {total_runs_in_job}{RESET}")
        print( f"\033[2C{image_headings}{RESET}") 
        print( f"{BITTER_SWEET}\
\r\033[2C\
\r\033[{start_column+0*offset}C{lr:<9.6f}\
\r\033[{start_column+1*offset}C{pct_test:<9.6f}\
\r\033[{start_column+2*offset}C{n_samples:<5d}\
\r\033[{start_column+3*offset}C{batch_size:<5d}\
\r\033[{start_column+4*offset}C{n_tiles:<5d}\
\r\033[{start_column+5*offset}C{highest_class_number:<2d}\
\r\033[{start_column+6*offset}C{tile_size:<3d}\
\r\033[{start_column+7*offset}C{rand_tiles:<5s}\
\r\033[{start_column+8*offset}C{nn_type_img:<10s}\
\r\033[{start_column+9*offset}C{nn_optimizer:<8s}\
\r\033[{start_column+10*offset}C{stain_norm:<10s}\
\r\033[{start_column+11*offset}C{label_swap_perunit:<6.1f}\
\r\033[{start_column+12*offset}C{make_grey_perunit:<5.1f}\
\r\033[{start_column+13*offset}C{jitter:}\
{RESET}" )  

      elif input_mode=='rna':
        print(f"\n\n{UNDER}RUN: {run} of {total_runs_in_job}{RESET}")
        print(f"\033[2C\{rna_headings}{RESET}")
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
  

      if DEBUG>0:
        print ("")
    
    
    final_test_batch_size =   int(n_samples * n_tiles * pct_test)
    
    if DEBUG>99:
      print( f"TRAINLENEJ:     INFO:          requested FINAL_TEST_BATCH_SIZE = {MIKADO}{int(args.final_test_batch_size)}{RESET}" )      
      print( f"TRAINLENEJ:     INFO:          N_SAMPLES (notional)            = {MIKADO}{n_samples}{RESET}" )
      print( f"TRAINLENEJ:     INFO:          N_TILES (per sample)            = {MIKADO}{n_tiles}{RESET}" )
      print( f"TRAINLENEJ:     INFO:          PCT_TEST                        = {MIKADO}{pct_test}{RESET}" )
      print( f"TRAINLENEJ:     INFO:          hence available test tiles      = {MIKADO}{int(final_test_batch_size)}{RESET}" )
    if args.final_test_batch_size > final_test_batch_size:
      print ( f"{ORANGE}TRAINLENEJ:     WARNING: there aren't enough test tiles to support a {CYAN}FINAL_TEST_BATCH_SIZE{RESET}{ORANGE} of {MIKADO}{args.final_test_batch_size}{RESET}{ORANGE} for this run{RESET}", flush=True )                
      print ( f"{ORANGE}TRAINLENEJ:              the number of test tiles available is {CYAN}N_SAMPLES{RESET} x {CYAN}N_TILES{RESET} x {CYAN}PCT_TEST{RESET}  = {MIKADO}{n_samples}{RESET} x {MIKADO}{n_tiles}{RESET} x {MIKADO}{pct_test}{RESET} = {MIKADO}{int(final_test_batch_size)}{RESET}{ORANGE}{RESET}", flush=True )                
      print ( f"{ORANGE}TRAINLENEJ:              {CYAN}FINAL_TEST_BATCH_SIZE{RESET}{ORANGE} has accordingly been set to {MIKADO}{int(final_test_batch_size)}{RESET} {ORANGE}for this run {RESET}", flush=True )
      args.final_test_batch_size = final_test_batch_size


    #(3) set up Tensorboard
    
    if DEBUG>1:    
      print( "TRAINLENEJ:     INFO: \033[1m3 about to set up Tensorboard\033[m" )
    
    writer = SummaryWriter(comment=f'_{randint(100, 999)}_{file_name_prefix}' )


    #print ( f"\033[36B",  flush=True )
    if DEBUG>1:    
      print( "TRAINLENEJ:     INFO:   \033[3mTensorboard has been set up\033[m" )


    # (1) Potentially schedule and run tiler threads
    
    if (input_mode=='image') & (multimode!='image_rna'):
      
      if skip_tiling=='False':
        
        if use_tiler=='internal':
          
          # need to re-tile if certain parameters have eiher INCREASED ('n_tiles' or 'n_samples') or simply CHANGED ( 'stain_norm' or 'tile_size') since the last run
          if ( ( already_tiled==True ) & ( ( stain_norm==last_stain_norm ) | (last_stain_norm=="NULL") ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) ):
            pass          # no need to re-tile                                                              
          else:           # must re-tile
            if DEBUG>0:
              print( f"TRAINLENEJ:     INFO: {BOLD}1 about to launch tiling processes{RESET}" )
            if DEBUG>1:
              print( f"TRAINLENEJ:     INFO:     stain normalization method = {CYAN}{stain_norm}{RESET}" )
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
                print( f"TRAINLENEJ:       INFO: {BOLD}about to set up stain normalization target{RESET}" )
              if stain_norm_target.endswith(".svs"):                                                       # ... then grab the user provided target
                norm_method = tiler_set_target( args, n_tiles, tile_size, stain_norm, stain_norm_target, writer )
              else:                                                                                        # ... and there MUST be a target
                print( f"TRAINLENEJ:     FATAL:    for {MIKADO}{stain_norm}{RESET} an SVS file must be provided from which the stain normalization target will be extracted" )
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
                    print( f"{SAVE_CURSOR}\r\033[{num_cpus}B{WHITE}TRAINLENEJ:     INFO: about to call tiler_threader with flag = {CYAN}{flag}{RESET}; count = {MIKADO}{count:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles = {MIKADO}{n_tiles}{RESET}{RESTORE_CURSOR}", flush=True )
                  slides_tiled_count = tiler_threader( args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method )

                if (  args.cases == 'DESIGNATED_MULTIMODE_CASE_FLAG' ):
                  
                  flag  = 'DESIGNATED_MULTIMODE_CASE_FLAG'
                  count = cases_reserved_for_image_rna
                  if DEBUG>0:
                    print( f"{SAVE_CURSOR}\r\033[{num_cpus}B{WHITE}TRAINLENEJ:     INFO: about to call tiler_threader with flag = {CYAN}{flag}{RESET}; count = {MIKADO}{count:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles = {MIKADO}{n_tiles}{RESET}{RESTORE_CURSOR}", flush=True )
                  slides_tiled_count = tiler_threader( args, flag, count, n_tiles, tile_size, batch_size, stain_norm, norm_method )



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
                  print( f"{SAVE_CURSOR}\r\033[{num_cpus+1}B{WHITE}TRAINLENEJ:     INFO: about to call tiler_threader with flag = {CYAN}{flag}{RESET}; slides_to_be_tiled = {MIKADO}{slides_to_be_tiled:3d}{RESET};   pct_test = {MIKADO}{pct_test:2.2f}{RESET};   n_samples_max = {MIKADO}{n_samples_max:3d}{RESET};   n_tiles_max = {MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )
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
                  print( f"{SAVE_CURSOR}\r{WHITE}TRAINLENEJ:     INFO: about to call {MAGENTA}tiler_threader{RESET}: flag={CYAN}{flag}{RESET}; train_count={MIKADO}{train_count:3d}{RESET}; %_test={MIKADO}{pct_test:2.2f}{RESET}; n_samples={MIKADO}{n_samples_max:3d}{RESET}; n_tiles={MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )
                slides_tiled_count = tiler_threader( args, flag, train_count, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job


                flag  = 'NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG'
                if DEBUG>0:
                  print( f"{SAVE_CURSOR}\r{WHITE}TRAINLENEJ:     INFO: about to call {MAGENTA}tiler_threader{RESET}: flag={CYAN}{flag}{RESET}; test_count={MIKADO}{test_count:3d}{RESET}; %_test={MIKADO}{pct_test:2.2f}{RESET}; n_samples={MIKADO}{n_samples_max:3d}{RESET}; n_tiles={MIKADO}{n_tiles_max}{RESET}{RESTORE_CURSOR}", flush=True )
                slides_tiled_count = tiler_threader( args, flag, test_count, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples & tiles that is required for any run within the job

                

            print ( f"{RESTORE_CURSOR}" )



          
            if just_profile=='True':                                                                       # then we are all done
              sys.exit(0)


    # (2) Regenerate Torch '.pt' file, if required

    if ( skip_generation=='False' ):
      
      if (input_mode=='image'):
        
        if ( ( already_tiled==True ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) & ( stain_norm==last_stain_norm ) ):    # all three have to be true, or else we must regenerate the .pt file
          pass  # PGD 201206 - TODO - This logic doesn't look correct
        else:
          if global_batch_count==0:
            if DEBUG>1:
              print( f"\r{RESET}TRAINLENEJ:     INFO: {BOLD}2  now generating torch '.pt' file from contents of dataset directories{RESET}" )
          else:
            print( f"\rTRAINLENEJ:     INFO: {BOLD}2  will regenerate torch '.pt' file from files, for the following reason(s):{RESET}" )            
            if n_tiles>n_tiles_last:
              print( f"                                    -- value of n_tiles   {MIKADO}({n_tiles})        \r\033[60Chas increased since last run{RESET}" )
            if n_samples>n_samples_last:
              print( f"                                    -- value of n_samples {MIKADO}({n_samples_last}) \r\033[60Chas increased since last run{RESET}")
            if not tile_size_last==tile_size:
              print( f"                                    -- value of tile_size {MIKADO}({tile_size})      \r\033[60Chas changed   since last run{RESET}")
         
        if DEBUG>5:
          print( f"TRAINLENEJ:     INFO: n_samples               = {MAGENTA}{n_samples}{RESET}"       )
          print( f"TRAINLENEJ:     INFO: args.n_samples          = {MAGENTA}{args.n_samples}{RESET}"  )
          print( f"TRAINLENEJ:     INFO: n_tiles                 = {MAGENTA}{n_tiles}{RESET}"         )
          print( f"TRAINLENEJ:     INFO: args.n_tiles            = {MAGENTA}{args.n_tiles}{RESET}"    )
          print( f"TRAINLENEJ:     INFO: batch_size              = {MAGENTA}{batch_size}{RESET}"      )
          print( f"TRAINLENEJ:     INFO: args.batch_size         = {MAGENTA}{args.batch_size}{RESET}" )
          print( f"TRAINLENEJ:     INFO: n_genes (from args)     = {MAGENTA}{n_genes}{RESET}"         )
          print( f"TRAINLENEJ:     INFO: gene_data_norm          = {MAGENTA}{gene_data_norm}{RESET}"  )            
                        
        n_genes = generate( args, n_samples, highest_class_number, multimode_case_count, unimode_case_count, not_a_multimode_case_count, not_a_multimode_case____image_count, not_a_multimode_case____image_test_count, pct_test, n_tiles, tile_size, gene_data_norm, gene_data_transform  ) 

        if DEBUG>5:
          print( f"TRAINLENEJ:     INFO: n_samples               = {BLEU}{n_samples}{RESET}"       )
          print( f"TRAINLENEJ:     INFO: args.n_samples          = {BLEU}{args.n_samples}{RESET}"  )
          print( f"TRAINLENEJ:     INFO: n_tiles                 = {BLEU}{n_tiles}{RESET}"         )
          print( f"TRAINLENEJ:     INFO: args.n_tiles            = {BLEU}{args.n_tiles}{RESET}"    )
          print( f"TRAINLENEJ:     INFO: batch_size              = {BLEU}{batch_size}{RESET}"      )
          print( f"TRAINLENEJ:     INFO: args.batch_size         = {BLEU}{args.batch_size}{RESET}" )
          print( f"TRAINLENEJ:     INFO: n_genes (from args)     = {BLEU}{n_genes}{RESET}"         )
          print( f"TRAINLENEJ:     INFO: gene_data_norm          = {BLEU}{gene_data_norm}{RESET}"  )            
          
        n_tiles_last   = n_tiles                                                                           # for the next run
        n_samples_last = n_samples                                                                         # for the next run
        tile_size_last = tile_size                                                                         # for the next run

        # The following is necessary because generate() is allowed to change the value of args.n_samples and args.batch_size, whereas n_samples and batch size are set in the 'product' loop above

        if ( args.cases!='ALL_ELIGIBLE_CASES' ):
          if n_samples != args.n_samples[0]:
            if DEBUG>0:
              print( f"{ORANGE}TRAINLENEJ:     INFO:   '{CYAN}N_SAMPLES{RESET}{ORANGE}' will be changed from {MIKADO}{n_samples}{RESET}{ORANGE} to {MIKADO}{args.n_samples[0]}{RESET}" ) 
            n_samples = args.n_samples[0]
          if batch_size != args.batch_size[0]:
            if DEBUG>0:
              print( f"{ORANGE}TRAINLENEJ:     INFO:   '{CYAN}BATCH_SIZE{RESET}{ORANGE}' will be changed from {MIKADO}{batch_size}{RESET}{ORANGE} to {MIKADO}{args.batch_size[0]}{RESET}" ) 
            batch_size = args.batch_size[0]

      
      elif ( input_mode=='rna' ) | ( input_mode=='image_rna' ) :
        
        must_generate=False
        if ( already_generated==False ):                                                                   # if we've never generated
          must_generate=True
        
        if not ( ( gene_data_norm==last_gene_norm ) & (last_gene_norm=="NULL") ):                          # if the type of normalization has changed since the last run, we have to regenerate
          must_generate=True
          
        if must_generate==True:
         
          n_genes = generate( args, n_samples, highest_class_number, multimode_case_count, unimode_case_count, not_a_multimode_case_count, not_a_multimode_case____image_count, not_a_multimode_case____image_test_count, pct_test, n_tiles, tile_size, gene_data_norm, gene_data_transform  )
          last_gene_norm=gene_data_norm
          already_generated=True 
                  
          # The following is necessary because generate() is allowed to change the value of args.n_samples and args.batch_size, whereas n_samples and batch size are set in the 'product' loop above

          if ( args.cases!='ALL_ELIGIBLE_CASES' ):
            if n_samples != args.n_samples[0]:
              if DEBUG>0:
                print( f"{ORANGE}TRAINLENEJ:     INFO: '{CYAN}n_samples{RESET}{ORANGE}' will be changed from {MIKADO}{n_samples} to {MIKADO}{args.n_samples[0]}{RESET}" ) 
              n_samples = args.n_samples[0]
            if batch_size != args.batch_size[0]:
              if DEBUG>0:
                print( f"{ORANGE}TRAINLENEJ:     INFO: '{CYAN}batch_size{RESET}{ORANGE}' will be changed from {MIKADO}{batch_size} to {MIKADO}{args.batch_size[0]}{RESET}" ) 
              batch_size = args.batch_size[0]

        else:
          if DEBUG>0:      
            print( f"\nTRAINLENEJ:     INFO: \033[1m3 gene_data_norm = {MIKADO}{gene_data_norm}{RESET} and last_gene_norm = {MIKADO}{last_gene_norm}{RESET} so no need to regenerate torch '.pt' file" )

      elif input_mode=='image_rna':
        print( f"{ORANGE}TRAINLENEJ:     INFO:   input mode = '{CHARTREUSE}{input_mode}{RESET}{ORANGE}'. concatentated image_rna embeddings will be generated.{RESET}"  )

      else:
        print( f"{RED}TRAINLENEJ:   FATAL:    input mode of type '{MIKADO}{input_mode}{RESET}{RED}' is not supported [200]{RESET}" )
        sys.exit(0)

      if DEBUG>5:
        print( f"TRAINLENEJ:     INFO: n_samples               = {MAGENTA}{n_samples}{RESET}"       )
        print( f"TRAINLENEJ:     INFO: args.n_samples          = {MAGENTA}{args.n_samples}{RESET}"  )
        print( f"TRAINLENEJ:     INFO: n_tiles                 = {MAGENTA}{n_tiles}{RESET}"         )
        print( f"TRAINLENEJ:     INFO: args.n_tiles            = {MAGENTA}{args.n_tiles}{RESET}"    )
        print( f"TRAINLENEJ:     INFO: batch_size              = {MAGENTA}{batch_size}{RESET}"      )
        print( f"TRAINLENEJ:     INFO: args.batch_size         = {MAGENTA}{args.batch_size}{RESET}" )
        print( f"TRAINLENEJ:     INFO: n_genes (from args)     = {MAGENTA}{n_genes}{RESET}"         )
        print( f"TRAINLENEJ:     INFO: gene_data_norm          = {MAGENTA}{gene_data_norm}{RESET}"  )            


     

    if clustering=='otsne':
      otsne ( args, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'TRAINLENEJ:       INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)

    elif clustering=='sktsne':
      sktsne(  args, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'TRAINLENEJ:       INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)

    elif clustering=='dbscan':
      _dbscan ( args, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'TRAINLENEJ:       INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)
      
    elif clustering=='h_dbscan':
      h_dbscan ( args, pct_test)
      writer.close()        
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time)       ,  0   )
      print( f'TRAINLENEJ:       INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      sys.exit(0)

    elif clustering!='NONE':
      print ( f"{RED}TRAINLENEJ:     FATAL:    there's no such clustering option as '{CYAN}{clustering}{RESET}'", flush=True)
      print ( f"{RED}TRAINLENEJ:     FATAL:    supported clustering algorithms are scikit-learn tsne ('{CYAN}sktsne{RESET}{RED}'), open tsne ('{CYAN}otsne{RESET}{RED}'), DBSCAN ('{CYAN}dbscan{RESET}{RED}'), HDBSCAN ('{CYAN}h_dbscan{RESET}{RED}'){RESET}", flush=True)
      print ( f"{RED}TRAINLENEJ:     FATAL:    halting now...{RESET}", flush=True)      
      sys.exit(0)

      

    # ~ elif clustering=='plotly_play':
      # ~ plotly_play ( args, pct_test)
      # ~ writer.close()        
      # ~ hours   = round( (time.time() - start_time) / 3600,  1   )
      # ~ minutes = round( (time.time() - start_time) /   60,  1   )
      # ~ seconds = round( (time.time() - start_time)       ,  0   )
      # ~ print( f'TRAINLENEJ:       INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
      # ~ sys.exit(0)
      
    # (4) Load experiment config.  (NOTE: Almost all configurable parameters are now provided via user arguments rather than this config file)

    
    if DEBUG>1:    
      print( f"TRAINLENEJ:     INFO: {BOLD}4 about to load experiment config{RESET}" )
    cfg = loader.get_config( nn_mode, lr, batch_size )                                                     #################################################################### change to just args at some point
#    GTExV6Config.INPUT_MODE         = input_mode                                                          # now using args
    GTExV6Config.MAKE_GREY          = make_grey_perunit                                                    # modify config class variable to take into account user preference
    GTExV6Config.JITTER             = jitter                                                               # modify config class variable to take into account user preference
#          if args.input_mode=='rna':  pplog.log_config(cfg) 

    # ~ pplog.log_section('Loading script arguments.')
    # ~ pplog.log_args(args)

    if DEBUG>1:      
      print( f"TRAINLENEJ:     INFO:   {ITALICS}experiment config has been loaded{RESET}" )
   


    #(5) Load network

    if DEBUG>1:                                                                                                       
      print( f"TRAINLENEJ:     INFO: {BOLD}5 about to load network {MIKADO}{nn_type_img}{RESET}{BOLD} and {MIKADO}{nn_type_rna}{RESET}" )  

    model = LENETIMAGE( args, cfg, input_mode, nn_type_img, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, args.latent_dim, args.em_iters  )

    if DEBUG>1: 
      print( f"TRAINLENEJ:     INFO:    {ITALICS}network loaded{RESET}" )


    # (6) maybe load existing models (two cases where this happens: (i) test mode and (ii) pretrain option selected )

    if pretrain=='True':                                                                                   # then load the last pretrained (as defined) model

      try:
        fqn = f"{log_dir}/model_pretrained.pt"
        model.load_state_dict(torch.load(fqn))
        print( f"{COTTON_CANDY}TRAINLENEJ:     INFO:  pre-trained model exists.  Will load and use the pre-trained mode rather than start with random weights{RESET}", flush=True)
      except Exception as e:
        print( f"{COTTON_CANDY}TRAINLENEJ:     INFO:  No pre-trained model exists.  Will commence training with random weights{RESET}", flush=True)
    
    elif just_test=='True':                                                                                  # then load the already trained model

      if args.input_mode == 'image':
        fqn = f"{log_dir}/model_image.pt"
      elif args.input_mode == 'rna':
        fqn = f"{log_dir}/model_rna.pt"
      elif args.input_mode == 'image_rna':
        fqn = f"{log_dir}/model_image_rna.pt"

      if DEBUG>0:
        print( f"{ORANGE}TRAINLENEJ:     INFO:  'just_test' flag is set.  About to load model state dictionary {MAGENTA}{fqn}{RESET}" )
        
      try:
        model.load_state_dict(torch.load(fqn))       
      except Exception as e:
        print ( f"{RED}TRAINLENEJ:     FATAL:  error when trying to load model {MAGENTA}'{fqn}'{RESET}", flush=True)    
        print ( f"{RED}TRAINLENEJ:     FATAL:    reported error was: '{e}'{RESET}", flush=True)
        print ( f"{RED}TRAINLENEJ:     FATAL:    explanation: this is a test run. ({CYAN}JUST_TEST==TRUE{RESET}{RED} (shell) or {CYAN}'just_test'=='True'{RESET}{RED} (python user argument). Perhaps you're using a different tile size ({CYAN}'TILE_SIZE'{RESET}{RED})than than the saved model uses{RESET}", flush=True)
        print ( f"{RED}TRAINLENEJ:     FATAL:    halting now...{RESET}", flush=True)      
        time.sleep(4)
        sys.exit(0)
                                            

    #(7) Send model to GPU(s)
    
    if DEBUG>1:    
      print( f"TRAINLENEJ:     INFO: {BOLD}6 about to send model to device{RESET}" )   
    model = model.to(device)
    if DEBUG>1:
      print( f"TRAINLENEJ:     INFO:     {ITALICS}model sent to device{RESET}" ) 
  
    #pplog.log_section('Model specs.')
    #pplog.log_model(model)
     
    
    if DEBUG>9:
      print( f"TRAINLENEJ:     INFO:   pytorch Model = {MIKADO}{model}{RESET}" )


    #(8) Fetch data loaders
    
    gpu        = 0
    world_size = 0
    rank       = 0
    

    if DEBUG>1: 
      print( f"TRAINLENEJ:     INFO: {BOLD}7 about to call dataset loader" )
    train_loader, test_loader, final_test_batch_size, final_test_loader = loader.get_data_loaders( args,
                                                         gpu,
                                                         cfg,
                                                         world_size,
                                                         rank,
                                                         batch_size,
                                                         args.n_workers,
                                                         args.pin_memory,                                                       
                                                         pct_test
                                                        )
    if DEBUG>1:
      print( "TRAINLENEJ:     INFO:   \033[3mdataset loaded\033[m" )
  
    #if just_test=='False':                                                                                # c.f. loader() Sequential'SequentialSampler' doesn't return indices
    #  pplog.save_test_indices(test_loader.sampler.indices)





    #(9) Select and configure optimizer

    if DEBUG>1:      
      print( f"TRAINLENEJ:     INFO: {BOLD}8 about to select and configure optimizer\033[m with learning rate = {MIKADO}{lr}{RESET}" )
    if nn_optimizer=='ADAM':
      optimizer = optim.Adam       ( model.parameters(),  lr=lr,  weight_decay=0,  betas=(0.9, 0.999),  eps=1e-08,               amsgrad=False                                    )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAMAX':
      optimizer = optim.Adamax     ( model.parameters(),  lr=lr,  weight_decay=0,  betas=(0.9, 0.999),  eps=1e-08                                                                 )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mAdamax optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADAGRAD':
      optimizer = optim.Adagrad    ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-10,               lr_decay=0, initial_accumulator_value=0          )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='SPARSEADAM':
      optimizer = optim.SparseAdam ( model.parameters(),  lr=lr,                   betas=(0.9, 0.999),  eps=1e-08                                                                 )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mSparseAdam optimizer selected and configured\033[m" )
    elif nn_optimizer=='ADADELTA':
      optimizer = optim.Adadelta   ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-06, rho=0.9                                                        )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mAdagrad optimizer selected and configured\033[m" )
    elif nn_optimizer=='ASGD':
      optimizer = optim.ASGD       ( model.parameters(),  lr=lr,  weight_decay=0,                                               alpha=0.75, lambd=0.0001, t0=1000000.0            )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mAveraged Stochastic Gradient Descent optimizer selected and configured\033[m" )
    elif   nn_optimizer=='RMSPROP':
      optimizer = optim.RMSprop    ( model.parameters(),  lr=lr,  weight_decay=0,                       eps=1e-08,  momentum=0,  alpha=0.99, centered=False                       )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mRMSProp optimizer selected and configured\033[m" )
    elif   nn_optimizer=='RPROP':
      optimizer = optim.Rprop      ( model.parameters(),  lr=lr,                                                                etas=(0.5, 1.2), step_sizes=(1e-06, 50)           )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mResilient backpropagation algorithm optimizer selected and configured\033[m" )
    elif nn_optimizer=='SGD':
      optimizer = optim.SGD        ( model.parameters(),  lr=lr,  weight_decay=0,                                   momentum=0.9, dampening=0, nesterov=True                      )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mStochastic Gradient Descent optimizer selected and configured\033[m" )
    elif nn_optimizer=='LBFGS':
      optimizer = optim.LBFGS      ( model.parameters(),  lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None  )
      if DEBUG>1:
        print( "TRAINLENEJ:     INFO:   \033[3mL-BFGS optimizer selected and configured\033[m" )
    else:
      print( "TRAINLENEJ:     FATAL:    Optimizer '{:}' not supported".format( nn_optimizer ) )
      sys.exit(0)
 
 
 
 
         
    # (10) Select Loss function
    
    if DEBUG>1:
      print( f"TRAINLENEJ:     INFO: {BOLD}9 about to select CrossEntropyLoss function{RESET}" )  
    loss_function = torch.nn.CrossEntropyLoss()
    
    if DEBUG>1:
      print( "TRAINLENEJ:     INFO:   \033[3mCross Entropy loss function selected\033[m" )  
    
    
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
                     
    print( f"TRAINLENEJ:     INFO: {BOLD}10 about to commence main loop, one iteration per epoch{RESET}" )

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

    test_total_loss_sum_ave_last           = 99999                                                         # used to determine whether total loss is increasing or decreasing
    test_lowest_total_loss_observed        = 99999
    test_lowest_total_loss_observed_epoch  = 0

    test_image_loss_sum_ave_last           = 99999
    test_lowest_image_loss_observed        = 99999    
    test_lowest_image_loss_observed_epoch  = 0     

    test_genes_loss_sum_ave_last           = 99999 
    test_lowest_genes_loss_observed        = 99999      
    test_lowest_genes_loss_observed_epoch  = 0 
  
    
    for epoch in range(1, n_epochs+1):
  
        if   args.input_mode=='image':
          print( f'\nTRAINLENEJ      INFO  in epoch {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}  mode:{MIKADO}{input_mode}{RESET} lr:{MIKADO}{lr}{RESET} samples:{MIKADO}{n_samples}{RESET} batch size:{MIKADO}{batch_size}{RESET} tile size:{MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide:{MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          print( f'\nTRAINLENEJ      INFO  in epoch {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}  mode:{MIKADO}{input_mode}{RESET} lr:{MIKADO}{lr}{RESET} samples:{MIKADO}{n_samples}{RESET} batch size:{MIKADO}{batch_size}{RESET} hidden layer neurons:{MIKADO}{hidden_layer_neurons}{RESET} embedded dimensions:{MIKADO}{batch_size if args.use_autoencoder_output==True  else "N/A" }{RESET}.  {DULL_WHITE}will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )
        else:
          print( f'\nTRAINLENEJ      INFO  in epoch {MIKADO}{epoch}{RESET} of {MIKADO}{n_epochs}{RESET}  mode:{MIKADO}{input_mode}{RESET} lr:{MIKADO}{lr}{RESET} samples:{MIKADO}{n_samples}{RESET} batch size:{MIKADO}{batch_size}{RESET} tile size:{MIKADO}{tile_size}x{tile_size}{RESET} tiles per slide:{MIKADO}{n_tiles}{RESET}.  {DULL_WHITE}will halt if test loss increases for {MIKADO}{max_consecutive_losses}{DULL_WHITE} consecutive epochs{RESET}' )

    
        if just_test=='True':                                                                              # skip trainiNG in 'test mode'
          pass
        
        # DO TRAINING
        else:
    
          train_loss_images_sum_ave, train_loss_genes_sum_ave, train_l1_loss_sum_ave, train_total_loss_sum_ave =\
                                                                                                       train ( args, epoch, train_loader, model, optimizer, loss_function, writer, train_loss_min, batch_size )
    
          if train_total_loss_sum_ave < train_lowest_total_loss_observed:
            train_lowest_total_loss_observed       = train_total_loss_sum_ave
            train_lowest_total_loss_observed_epoch = epoch
    
          if train_loss_images_sum_ave < train_lowest_image_loss_observed:
            train_lowest_image_loss_observed       = train_loss_images_sum_ave
            train_lowest_image_loss_observed_epoch = epoch

          if ( (train_total_loss_sum_ave < train_total_loss_sum_ave_last) | (epoch==1) ):
            consecutive_training_loss_increases = 0
            last_epoch_loss_increased = False
          else:
            last_epoch_loss_increased = True

          if DEBUG>0:
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
  




#        if (just_test=='True') & (multimode=='image_rna'):                                                 # skip testing in Test mode if multimode is True 
        if (just_test=='True') & (multimode=='image_rnaxxx'):                                                 # skip testing in Test mode if multimode is True 
          pass  
            
        # DO TESTING
        else:  
    
          show_all_test_examples=False
          test_loss_images_sum_ave, test_loss_genes_sum_ave, test_l1_loss_sum_ave, test_total_loss_sum_ave, correct_predictions, number_tested, max_correct_predictions, max_percent_correct, test_loss_min, embedding     =\
                        test ( cfg, args, epoch, test_loader,  model,  tile_size, loss_function, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                                             test_loss_min, show_all_test_examples, batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours)
  
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
  \r\033[166Cmins: total: {test_lowest_total_loss_observed*100/batch_size:6.2f}@{WHITE}e={test_lowest_total_loss_observed_epoch:<2d}{DULL_WHITE} |\
  \r\033[204Cimage:{CARRIBEAN_GREEN}{test_lowest_image_loss_observed*100/batch_size:>6.2f}@e={test_lowest_image_loss_observed_epoch:<2d}{DULL_WHITE} |\
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
  \r\033[166Cmins: total: {test_lowest_total_loss_observed*100/batch_size:6.2f}@{WHITE}e={test_lowest_total_loss_observed_epoch:<2d}{DULL_WHITE} |\
  \r\033[214Cgenes:{BITTER_SWEET}{test_lowest_genes_loss_observed*100/batch_size:>6.2f}@e={test_lowest_genes_loss_observed_epoch:<2d}{RESET}\
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
              print ( f"\r\033[232C\033[0K{BRIGHT_GREEN} < global low/saving{RESET}", end='' )
              print ( "\033[5B", end='' )
            
            if ( just_test=='False' ):
              save_model(args.log_dir, model) 
    
          if test_loss_genes_sum_ave < test_lowest_genes_loss_observed:
            test_lowest_genes_loss_observed       = test_loss_genes_sum_ave
            test_lowest_genes_loss_observed_epoch = epoch 
            if DEBUG>0:
              print ( "\033[5A", end='' )
              print ( f"\r\033[253C{BITTER_SWEET} < rna low {RESET}", end='' )
              print ( "\033[5B", end='' )
              
          if test_loss_images_sum_ave < test_lowest_image_loss_observed:
            test_lowest_image_loss_observed       = test_loss_images_sum_ave
            test_lowest_image_loss_observed_epoch = epoch
            if DEBUG>0:
              print ( "\033[5A", end='' )
              print ( f"\r\033[263C{CARRIBEAN_GREEN} < image low{RESET}", end='' )
              print ( "\033[5B", end='' )
  

          if args.input_mode=='rna':
            print ( "\033[8A", end='' )
          else:
            print ( "\033[8A", end='' )       
  
      #  ^^^  RUN FINISHES HERE ^^^



  
  
    # (C)  MAYBE CLASSIFY FINAL_TEST_BATCH_SIZE TEST SAMPLES USING THE BEST MODEL SAVED DURING THIS RUN
  
    if final_test_batch_size>0:
    
      if ( ( args.just_test!='True') &  (args.input_mode!='image_rna') )   |   ( (args.just_test=='True')  &  (args.input_mode=='image_rna') & (args.multimode=='image_rna')      ):
           
      
        if DEBUG>0:
          print ( "\033[8B" )        
          print ( f"TRAINLENEJ:     INFO:  test(): {BOLD}about to classify {MIKADO}{final_test_batch_size}{RESET}{BOLD} test samples through the best model this run produced"        )
        
        pplog.log ( f"\nTRAINLENEJ:     INFO:  test(): about to classify {final_test_batch_size} test samples through the best model this run produced"                                 )


        if args.input_mode == 'image':
          fqn = '%s/model_image.pt'     % log_dir
        elif args.input_mode == 'rna':
          fqn = '%s/model_rna.pt'       % log_dir
        elif args.input_mode == 'image_rna':
          fqn = '%s/model_image_rna.pt' % log_dir
    
          if DEBUG>0:
            print( f"TRAINLENEJ:     INFO:  about to load model state dictionary for best model (from {MIKADO}{fqn}{RESET})" )
  
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
        
        if DEBUG>0:
          print ( f"TRAINLENEJ:     INFO:      test():             final_test_batch_size               = {MIKADO}{final_test_batch_size}{RESET}" )
          
        # note that we pass 'final_test_loader' to test()
        test_loss_images_sum_ave, test_loss_genes_sum_ave, test_l1_loss_sum_ave, test_total_loss_sum_ave, correct_predictions, number_tested, max_correct_predictions, max_percent_correct, test_loss_min, embedding     =\
                          test ( cfg, args, epoch, final_test_loader,  model,  tile_size, loss_function, writer, max_correct_predictions, global_correct_prediction_count, global_number_tested, max_percent_correct, 
                                                                                                           test_loss_min, show_all_test_examples, final_test_batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours )    
    
      job_level_classifications_matrix               += run_level_classifications_matrix                     # accumulate for the job level stats. Has to be just after call to 'test()'    



    # (D)  MAYBE CREATE AND SAVE EMBEDDINGS FOR ALL TEST SAMPLES (IN TEST MODE, SO THE OPTIMUM MODEL HAS ALREADY BEEN LOADED AT STEP 5 ABOVE)
    
    if (just_test=='True') & (multimode=="image_rna"):

      if DEBUG>0:
        print( f"\r\033[7BTRAINLENEJ:     INFO:      test(): {BOLD}about to generate and save embeddings for all test samples{RESET}", flush=True )

      model.eval()                                                                                         # set model to evaluation mode

      embedding_count = 0
        
      for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in  enumerate( test_loader ):
          
        batch_images = batch_images.to(device)
        batch_genes  = batch_genes.to (device)
        image_labels = image_labels.to(device)

        if DEBUG>6:
          print( f"TRAINLENEJ:     INFO:      test(): for embeddings: embedding_count         = {MIKADO}{embedding_count+1}{RESET}",              flush=True )
          print( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch count             = {MIKADO}{i+1}{RESET}",                        flush=True )
          if args.input_mode=='image': 
            print( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_images size       = {BLEU}{batch_images.size()}{RESET}                                                     {MAGENTA}<<<<< Note: don't use dropout in test runs{RESET}", flush=True)
          if ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
            print( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_genes size        = {BLEU}{batch_genes.size()}{RESET}                                                     {MAGENTA}<<<<< Note: don't use dropout in test runs{RESET}", flush=True)
          print( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_fnames size       = {BLEU}{batch_fnames.size()}{RESET}",          flush=True)
        if DEBUG>888:
          print( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_fnames            = {PURPLE}{batch_fnames.cpu().numpy()}{RESET}", flush=True )

        gpu                = 0
        encoder_activation = 0
        if args.input_mode=='image':
          with torch.no_grad(): 
            y1_hat, y2_hat, embedding = model.forward( [ batch_images, 0            , batch_fnames] , gpu, encoder_activation  )          # y1_hat = image outputs
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          with torch.no_grad(): 
            y1_hat, y2_hat, embedding = model.forward( [ 0,            batch_genes  , batch_fnames], gpu, encoder_activation )            # y2_hat = rna outputs
                        

        if DEBUG>88:
          print( f"TRAINLENEJ:     INFO:      test(): for embeddings: embedding_count         = {MIKADO}{embedding_count+1}{RESET}",              flush=True )
          print( f"TRAINLENEJ:     INFO:      test(): for embeddings: returned embedding size = {ARYLIDE}{embedding.size()}{RESET}",          flush=True )
  
        batch_fnames_npy = batch_fnames.numpy()                                                            # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details

        if DEBUG>6:
          fq_link       = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"                                    # convert the saved integer to the matching file name
          save_path     =   os.path.dirname(os.readlink(fq_link))                 
          print( f"TRAINLENEJ:     INFO:      test(): (global count {MIKADO}{embedding_count:6d}{RESET}) saving {MIKADO}{batch_fnames_npy.shape[0]}{RESET} embeddings associated with case {MAGENTA}{save_path}{RESET}",                        flush=True )
          
                  
        if DEBUG>2:
          np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
          print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_fnames_npy.shape  = {batch_fnames_npy.shape}", flush=True )        
          print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_fnames_npy        = {batch_fnames_npy}",       flush=True )
  
        # save each embedding in its associated case directory using a randomly generated name
        if just_test=='True':                                                                               #  in test mode we are pushing inputs through the optimised model, which was saved during training mode

          for n in range( 0, batch_fnames_npy.shape[0] ):                                                    
  
            if args.input_mode=='image': 
              fq_link       = f"{args.data_dir}/{batch_fnames_npy[n]}.fqln"                                  # where to save the embedding (which case directory to save it to)
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_fnames_npy[{MIKADO}{n}{RESET}]   = {PINK}{batch_fnames_npy[n]}{RESET}",              flush=True )
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: fq_link                = {PINK}{fq_link}{RESET}",                          flush=True )
              save_path     =  os.path.dirname(os.readlink(fq_link))
              if DEBUG>8:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: save_path              = {PINK}{save_path}{RESET}",              flush=True )
              random_name   = f"_{randint(10000000, 99999999)}_image_rna_matched___image"
              save_fqn      = f"{save_path}/{random_name}"
              if DEBUG>8:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: save_fqn               = {PINK}{save_fqn}{RESET}",              flush=True )
              np.save( save_fqn, embedding.cpu().numpy()[n] )
  
            if ( args.input_mode=='rna' ):
              fq_link       = f"{args.data_dir}/{batch_fnames_npy[n]}.fqln"
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: batch_fnames_npy[{MIKADO}{n}{RESET}]   = {PINK}{batch_fnames_npy[n]}{RESET}",              flush=True )
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: fq_link                = {BLEU}{fq_link}{RESET}",                          flush=True )
              save_path     =   os.readlink(fq_link)                                                         # link is to the case directory for rna_seq (for tiles, it's to the patch file within the case directory)
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: save_path              = {BLEU}{save_path}{RESET}",              flush=True )
              random_name   = f"_image_rna_matched___rna"
              save_fqn      = f"{save_path}/{random_name}"
              if DEBUG>2:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: save_fqn               = {BLEU}{save_fqn}{RESET}",              flush=True )
              np.save( save_fqn, embedding.cpu().numpy()[n] )

            
        
          if DEBUG>88:
            np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
            print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: embedding [{MIKADO}{n},0:10{RESET}]     = {PINK}{embedding.cpu().numpy()[n,0:10]}{RESET}",  flush=True )
            print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: fq_link [{MIKADO}{n}{RESET}]            = {PINK}{fq_link}{RESET}",                          flush=True )
            print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: random name [{MIKADO}{n}{RESET}]        = {PINK}{ranndom_name}{RESET}",                     flush=True )
           #print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: points to                               = {PINK}{os.readlink(fq_link)}{RESET}",             flush=True )
            print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: save path                               = {BLEU}{save_path}{RESET}",                        flush=True )
            print ( f"TRAINLENEJ:     INFO:      test(): for embeddings: save fqn                                = {BLEU}{save_fqn}{RESET}",                         flush=True )
    
          embedding_count+=1


    if args.input_mode=='rna':
      print ( "\033[8A", end='' )
    else:
      print ( "\033[8A", end='' )  



    # (E)  ALWAYS DISPLAY & SAVE BAR CHARTS

    if (just_test=='True') & (multimode!="image_rna"):                                                     # don't currently produce bar-charts for embedded outputs ('image_rna')


      # case image:
        
      if input_mode=='image':
        
        pd.set_option('display.max_columns',  300)
        pd.set_option('display.max_colwidth', 300)      
        pd.set_option('display.width',       2000)
        
        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>3d}"})
          print ( f"\nTRAINLENEJ:     INFO:      patches_true_classes                                        = \n{AZURE}{patches_true_classes}{RESET}", flush=True )
          print ( f"\nTRAINLENEJ:     INFO:      patches_case_id                                             = \n{BLEU}{patches_case_id}{RESET}",     flush=True )        

        if args.cases=='DESIGNATED_MULTIMODE_CASE_FLAG':
          upper_bound_of_indices_to_plot_image = cases_reserved_for_image_rna
        else:  # correct for NOT_A_MULTIMODE_CASE_FLAG and NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG
          upper_bound_of_indices_to_plot_image = n_tests


        # case image- 1: PREDICTED - AGGREGATE probabilities
        
        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:      aggregate_tile_probabilities_matrix                 = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix}{RESET}", flush=True )

        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:      args.class_names                 = \n{CHARTREUSE}{class_names}{RESET}", flush=True )
          
  
        figure_width  = 20
        figure_height = 10
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
        ax.set_title ( args.cancer_type_long )
        plt.xticks( rotation=90 )
        plt.ylim  ( 0, n_tiles  )     
        #sns.set_theme(style="whitegrid")
        pd_aggregate_tile_probabilities_matrix                    = pd.DataFrame( aggregate_tile_probabilities_matrix )   [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix.columns            = args.class_names
        pd_aggregate_tile_probabilities_matrix[ 'agg_prob' ]      = np.sum(aggregate_tile_probabilities_matrix,   axis=1 )[0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ]  = pd_aggregate_tile_probabilities_matrix.max   (axis=1) [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix[ 'pred_class'   ]  = pd_aggregate_tile_probabilities_matrix.idxmax(axis=1) [0:upper_bound_of_indices_to_plot_image]  # grab class (which is the column index with the highest value in each row) and save as a new column vector at the end, to using for coloring 
        pd_aggregate_tile_probabilities_matrix[ 'true_class'   ]  = patches_true_classes                                  [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_probabilities_matrix[ 'n_classes'    ]  = len(class_names) 
        pd_aggregate_tile_probabilities_matrix[ 'case_id'      ]  = patches_case_id                                       [0:upper_bound_of_indices_to_plot_image]
        # ~ pd_aggregate_tile_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )
        #fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"
        

        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>3d}"})
          print ( f"\nTRAINLENEJ:     INFO:      upper_bound_of_indices_to_plot_image                              = {CHARTREUSE}{upper_bound_of_indices_to_plot_image}{RESET}",     flush=True      ) 
          print ( f"\nTRAINLENEJ:     INFO:      pd_aggregate_tile_probabilities_matrix[ 'case_id' ]         = \n{CHARTREUSE}{pd_aggregate_tile_probabilities_matrix[ 'case_id' ]}{RESET}",     flush=True      ) 
          print ( f"\nTRAINLENEJ:     INFO:      pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ]    = \n{CHARTREUSE}{pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ]}{RESET}",     flush=True )            
  
        if bar_chart_x_labels=='case_id':
          c_id = pd_aggregate_tile_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_aggregate_tile_probabilities_matrix.shape[0])]

        if DEBUG>1:
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:       (extended) pd_aggregate_tile_probabilities_matrix = \n{CHARTREUSE}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True )
          # ~ print ( f"\nTRAINLENEJ:     INFO:       (extended) aggregate_tile_probabilities_matrix    = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix}{RESET}", flush=True )
       
        if DEBUG>88:          
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:                                             aggregate_tile_probabilities_matrix = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix}{RESET}", flush=True )
        if DEBUG>88:          
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:          aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image]  = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image]}{RESET}", flush=True )
          print ( f"\nTRAINLENEJ:     INFO: np.argmax(aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image] = \n{CHARTREUSE}{np.argmax(aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image], axis=1)}{RESET}", flush=True )
          
        x_labels = [  str(el) for el in c_id ]
        cols     = [ class_colors[el] for el in np.argmax(aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_image], axis=1)  ]
                  
        if DEBUG>88:
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:                                                     cols = \n{CHARTREUSE}{cols}{RESET}", flush=True )
          print ( f"\nTRAINLENEJ:     INFO:                                                len(cols) = \n{CHARTREUSE}{len(cols)}{RESET}", flush=True )
          
        # ~ if DEBUG>0:
          # ~ print ( f"\nTRAINLENEJ:     INFO:      cols                = {MIKADO}{cols}{RESET}", flush=True )        
        
        p1 = plt.bar( x=x_labels, height=pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ], color=cols ) 
              
        # ~ ax = sns.barplot( x=c_id,  y=pd_aggregate_tile_probabilities_matrix[ 'max_agg_prob' ], hue=pd_aggregate_tile_probabilities_matrix['pred_class'], palette=class_colors, dodge=False )                  # in pandas, 'index' means row index
        ax.set_title   ("Score of Predicted Subtype (sum of tile-level probabilities)",  fontsize=16 )
        ax.set_xlabel  ("Case (Patch)",                                                  fontsize=14 )
        ax.set_ylabel  ("Aggregate Probabilities",                                       fontsize=14 )
        ax.tick_params (axis='x', labelsize=12,  labelcolor='black')
        # ~ ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
        # ~ plt.legend( args.class_names, loc=2, prop={'size': 14} )
        
        # ~ patch0 = mpatches.Patch(color=cols[1], label=args.class_names[0])
        # ~ patch1 = mpatches.Patch(color=cols[2], label=args.class_names[1])
        # ~ patch2 = mpatches.Patch(color=cols[3], label=args.class_names[2])
        # ~ patch3 = mpatches.Patch(color=cols[4], label=args.class_names[3])
        # ~ patch4 = mpatches.Patch(color=cols[5], label=args.class_names[4])
        # ~ patch5 = mpatches.Patch(color=cols[0], label=args.class_names[5])
        
        # ~ plt.legend( handles=[patch0, patch1, patch2, patch3, patch4, patch5 ], loc=2, prop={'size': 14} )
                
        correct_count = 0
        i=0
        for p in ax.patches:
          #ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',  fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
          if not np.isnan(p.get_height()):                                                                   # if it's a number, then it will be a height (y value)
            for index, row in pd_aggregate_tile_probabilities_matrix.iterrows():
              if DEBUG>888:
                print ( f"TRAINLENEJ:     INFO:      row['max_agg_prob']                       = {CHARTREUSE}{row['max_agg_prob']}{RESET}", flush=True )            
                print ( f"TRAINLENEJ:     INFO:      p.get_height()                            = {CHARTREUSE}{p.get_height()}{RESET}", flush=True )
                print ( f"TRAINLENEJ:     INFO:      patches_true_classes[{MIKADO}{i}{RESET}]  = {CHARTREUSE}{patches_true_classes[i]}{RESET}", flush=True ) 
              if row['max_agg_prob'] == p.get_height():                                                      # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>888:
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}args.class_names[row['true_class']]  = {RESET}{MIKADO}{args.class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}correct_count                        = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not args.class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=14, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  correct_count+=1
            i+=1 
  
        if DEBUG>1:
          print ( f"\nTRAINLENEJ:     INFO:      number correct (pd_aggregate_tile_probabilities_matrix) = {CHARTREUSE}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        if args.bar_chart_show_all=='True':
          writer.add_figure('images___aggregate_tile_level_probabs_matrix', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
        
        fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}_bar_chart_images___aggregated_tile_level_raw____probs.png"
        fig.savefig(fqn)
        
        
            

        # case image-2: PREDICTED - WINNER TAKE ALL probabilities
        
        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:      aggregate_tile_level_winners_matrix                = \n{AMETHYST}{aggregate_tile_level_winners_matrix}{RESET}", flush=True )
  
        figure_width  = 20
        figure_height = 10
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
        ax.set_title ( args.cancer_type_long )
        
        plt.xticks( rotation=90 )
        plt.ylim  ( 0, n_tiles  )     
        #sns.set_theme(style="whitegrid")
        pd_aggregate_tile_level_winners_matrix                      = pd.DataFrame( aggregate_tile_level_winners_matrix )    [0:upper_bound_of_indices_to_plot_image]
        pd_aggregate_tile_level_winners_matrix.columns              = args.class_names
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
          print ( f"\nTRAINLENEJ:     INFO:       (extended) pd_aggregate_tile_level_winners_matrix  = \n{BLEU}{pd_aggregate_tile_level_winners_matrix}{RESET}", flush=True )  
          

        if DEBUG>88:
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:                                             aggregate_tile_level_winners_matrix = \n{AMETHYST}{aggregate_tile_level_winners_matrix}{RESET}", flush=True )
          print ( f"\nTRAINLENEJ:     INFO:          aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image]  = \n{AMETHYST}{aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image]}{RESET}", flush=True )
          print ( f"\nTRAINLENEJ:     INFO: np.argmax(aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image] = \n{AMETHYST}{np.argmax(aggregate_tile_level_winners_matrix[0:upper_bound_of_indices_to_plot_image], axis=1)}{RESET}", flush=True )
          
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
        # ~ plt.legend( args.class_names,loc=2, prop={'size': 14} )
                
        correct_count=0
        i=0
        for p in ax.patches:
          #ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',  fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
          if not np.isnan(p.get_height()):
            for index, row in pd_aggregate_tile_level_winners_matrix.iterrows():
              if DEBUG>888:
                print ( f"TRAINLENEJ:     INFO:      row['max_tile_count']                     = {MIKADO}{row['max_tile_count']}{RESET}", flush=True )            
                print ( f"TRAINLENEJ:     INFO:      p.get_height()                            = {MIKADO}{p.get_height()}{RESET}", flush=True )
                print ( f"TRAINLENEJ:     INFO:      patches_true_classes[{MIKADO}{i}{RESET}]  = {MIKADO}{patches_true_classes[i]}{RESET}", flush=True ) 
              if row['max_tile_count'] == p.get_height():                                                    # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>888 :
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}args.class_names[row['true_class']]  = {RESET}{MIKADO}{args.class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}correct_count   max_tilmax                     = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not args.class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=14, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  correct_count+=1
            i+=1 
  
        if DEBUG>88:
          print ( f"\nTRAINLENEJ:     INFO:      number correct (pd_aggregate_tile_level_winners_matrix) = {MIKADO}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
        
        plt.tight_layout()
        
        if args.bar_chart_show_all=='True':        
          writer.add_figure('images___aggregate_tile_level_winners_matrix', fig, 0 )

        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}_bar_chart_images___aggregated_tile_level_winner_probs.png"
        fig.savefig(fqn)
        
        


        # Case image-3: probabilities assigned to TRUE classes 

        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        true_class_prob = aggregate_tile_probabilities_matrix[ range(0, patches_true_classes.shape[0]), patches_true_classes ]   # 'patches_true_classes' was established during test run
        pred_class_idx  = np.argmax ( aggregate_tile_probabilities_matrix, axis=1   )
        correct_count   = np.sum    (    patches_true_classes == pred_class_idx     )

        pd_aggregate_tile_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx [0:upper_bound_of_indices_to_plot_image]   # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c DESIGNATED_MULTIMODE_CASE_FLAG')
        pd_aggregate_tile_probabilities_matrix[ 'true_class_prob' ]  = true_class_prob[0:upper_bound_of_indices_to_plot_image]   # same
        # ~ pd_aggregate_tile_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )

        if DEBUG>0:
          print ( f"\nTRAINLENEJ:     INFO:      probabilities_matrix {CYAN}image{RESET} = \n{COTTON_CANDY}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True )    
                
                
        if bar_chart_x_labels=='case_id':                                                                  # user wants case ids as labels
          c_id = pd_aggregate_tile_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_aggregate_tile_probabilities_matrix.shape[0])]
          
        for i in range ( 0, aggregate_tile_probabilities_matrix.shape[0] ):
          agg_prob = pd_aggregate_tile_probabilities_matrix[ 'agg_prob'][i]
          arg_max  = np.argmax( aggregate_tile_probabilities_matrix[i,:] )
          if DEBUG>0:
            print ( f"TRAINLENEJ:     INFO:      i                                                                       = {COTTON_CANDY}{i}{RESET}", flush=True ) 
            print ( f"TRAINLENEJ:     INFO:      str(c_id[i])                                                            = {COTTON_CANDY}{str(c_id[i])}{RESET}", flush=True ) 
            print ( f"TRAINLENEJ:     INFO:      arg_max                                                                 = {COTTON_CANDY}{arg_max}{RESET}", flush=True ) 
            print ( f"TRAINLENEJ:     INFO:      class_names[ arg_max ]                                                  = {COTTON_CANDY}{class_names[ arg_max ]}{RESET}", flush=True ) 
            print ( f"TRAINLENEJ:     INFO:      height = [ aggregate_tile_probabilities_matrix[i,arg_max] / agg_prob ]  = {COTTON_CANDY}{[ aggregate_tile_probabilities_matrix[i,arg_max] / agg_prob ]}{RESET}", flush=True ) 
          plt.bar( x=[ str(c_id[i]) ],   height=[ aggregate_tile_probabilities_matrix[i,arg_max] / agg_prob ],  color=class_colors[ arg_max ], label=class_names[ arg_max ] )  # just plots the maximum value


        plt.title   ("Input Data = Slide Image Tiles;  Bar Height = Probability Assigned to **TRUE** Cancer Sub-type",            fontsize=16 )
        plt.xlabel  ("Case ID",                                                     fontsize=14 )
        plt.ylabel  ("Probability Assigned by Network",                             fontsize=14 )
        plt.ylim    (0.0, 1.0)
        plt.tick_params (axis='x', labelsize=8,   labelcolor='black')
        plt.tick_params (axis='y', labelsize=14,  labelcolor='black')
        plt.xticks  ( rotation=90 )

        plt.legend( args.class_names, loc=2, prop={'size': 14} )
            
        pct_correct = correct_count/ n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        writer.add_figure('images___probs_assigned_to_TRUE_classes', fig, 0 )
        
        # save version to logs directory
        now = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}_bar_chart_images___probs_assigned_to_TRUE_classes.png"
        
        fig.savefig(fqn)
          



        # Case image-4:  graph aggregate probabilities for ALL classses

        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        if DEBUG>55:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:       probabilities_matrix = \n{BLEU}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True ) 

        true_class_prob = aggregate_tile_probabilities_matrix[ range(0, patches_true_classes.shape[0]), patches_true_classes ]
        pred_class_idx  = np.argmax( aggregate_tile_probabilities_matrix, axis=1   )
        correct_count   = np.sum( patches_true_classes == pred_class_idx )

        if DEBUG>88:
          print ( f"\033[16B" )
          print ( f"\nTRAINLENEJ:     INFO:      patches_case_id                                = \n{ASPARAGUS}{patches_case_id}{RESET}",                              flush=True )  
          print ( f"\nTRAINLENEJ:     INFO:      pd_aggregate_tile_probabilities_matrix.shape   = {ASPARAGUS}{pd_aggregate_tile_probabilities_matrix.shape}{RESET}",  flush=True )                
          print ( f"\nTRAINLENEJ:     INFO:      true_class_prob                                = \n{ASPARAGUS}{true_class_prob}{RESET}",                               flush=True )
          print ( f"\nTRAINLENEJ:     INFO:      pred_class_idx                                 = \n{ASPARAGUS}{pred_class_idx}{RESET}",                               flush=True )
          print ( f"\nTRAINLENEJ:     INFO:      patches_true_classes                           = \n{ASPARAGUS}{patches_true_classes}{RESET}",                                 flush=True )
  
        plt.xticks( rotation=90 )
        pd_aggregate_tile_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx                                        [0:upper_bound_of_indices_to_plot_image]   # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c DESIGNATED_MULTIMODE_CASE_FLAG')
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
        plt.legend( args.class_names,loc=2, prop={'size': 14} )

  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
      
        writer.add_figure('images___probs_assigned_to_ALL__classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}_bar_chart_images___probs_assigned_to_ALL__classes.png"
        
        fig.savefig(fqn)



        fqn = f"{args.log_dir}/probabilities_dataframe_image.csv"
        try:
          pd_aggregate_tile_probabilities_matrix.to_csv ( fqn, sep='\t' )
          if DEBUG>88:
            print ( f"TRAINLENEJ:     INFO:     now saving  probabilities dataframe {ASPARAGUS}(image){RESET} to   {MAGENTA}{fqn}{RESET}"  )
        except Exception as e:
          print ( f"{ORANGE}TRAINLENEJ:     WARNING:     could not save file   = {ORANGE}{fqn}{RESET}"  )
          # ~ print ( f"{ORANGE}TRAINLENEJ:     WARNING:     error was: {e}{RESET}" )
          




      # Case rna: 
    
      elif input_mode=='rna':
        
        pd.set_option('display.max_columns',  300 )
        pd.set_option('display.max_rows',     600 )
        pd.set_option('display.max_colwidth', 300 )
        pd.set_option('display.width',        300 )
        pd.set_option("display.precision",      8 )
                          
        if DEBUG>88:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:      probabilities_matrix                 = \n{CAMEL}{probabilities_matrix}{RESET}", flush=True )
          print ( f"\nTRAINLENEJ:     INFO:      probabilities_matrix.shape           = {CAMEL}{probabilities_matrix.shape}{RESET}", flush=True )


        figure_width  = 20
        figure_height = 10

        if args.just_test!='True':        
          if args.cases!='ALL_ELIGIBLE_CASES':
            upper_bound_of_indices_to_plot_rna = n_samples
          elif args.cases!='DESIGNATED_MULTIMODE_CASE_FLAG':
            upper_bound_of_indices_to_plot_rna = cases_reserved_for_image_rna
          else:
            upper_bound_of_indices_to_plot_rna = n_samples
        else:
          upper_bound_of_indices_to_plot_rna = n_tests
          
          

        # Case rna-1:  bar chart showing probability of PREDICTED values
           
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        if DEBUG>55:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:       probabilities_matrix = \n{CAMEL}{probabilities_matrix}{RESET}", flush=True )

        true_class_prob = probabilities_matrix[ range(0, true_classes.shape[0]), true_classes ]
        pred_class_idx  = np.argmax ( probabilities_matrix, axis=1   )
        correct_count   = np.sum    ( true_classes == pred_class_idx )

        plt.xticks( rotation=90 )
        probabilities_matrix=probabilities_matrix[0:n_samples,:]                                  # possibly truncate rows because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c DESIGNATED_MULTIMODE_CASE_FLAG')
        pd_probabilities_matrix                       = pd.DataFrame( probabilities_matrix )
        pd_probabilities_matrix.columns               = args.class_names
        pd_probabilities_matrix[ 'agg_prob'        ]   = np.sum(probabilities_matrix,   axis=1 ) [0:upper_bound_of_indices_to_plot_rna]
        pd_probabilities_matrix[ 'max_agg_prob'    ]  = pd_probabilities_matrix.max   (axis=1)   [0:upper_bound_of_indices_to_plot_rna]
        pd_probabilities_matrix[ 'pred_class'      ]  = pd_probabilities_matrix.idxmax(axis=1)   [0:upper_bound_of_indices_to_plot_rna]    # grab class (which is the column index with the highest value in each row) and save as a new column vector at the end, to using for coloring 
        pd_probabilities_matrix[ 'true_class'      ]  = true_classes                             [0:upper_bound_of_indices_to_plot_rna]    # same
        pd_probabilities_matrix[ 'n_classes'       ]  = len(class_names) 
        pd_probabilities_matrix[ 'case_id'         ]  = rna_case_id                              [0:upper_bound_of_indices_to_plot_rna]    # same
        pd_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx                           [0:upper_bound_of_indices_to_plot_rna]    # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c DESIGNATED_MULTIMODE_CASE_FLAG')
        pd_probabilities_matrix[ 'true_class_prob' ]  = true_class_prob                          [0:upper_bound_of_indices_to_plot_rna]    # same
        # ~ pd_probabilities_matrix.sort_values( by='max_agg_prob', ascending=False, ignore_index=True, inplace=True )
 
        if DEBUG>0: ##################DON'T DELETE
          print ( "\033[20B" )
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:       (extended) pd_probabilities_matrix {CYAN}(rna){RESET} = \n{ARYLIDE}{pd_probabilities_matrix[0:upper_bound_of_indices_to_plot_rna]}{RESET}", flush=True ) 
  
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
        # ~ plt.legend( args.class_names,loc=2, prop={'size': 14} )        
        
        i=0
        for p in ax.patches:
          if not np.isnan(p.get_height()):                                                                   # if it's a number, then it will be a height (y value)
            for index, row in pd_probabilities_matrix.iterrows():
              if DEBUG>555:
                print ( f"TRAINLENEJ:     INFO:      row['max_agg_prob']                       = {CAMEL}{row['max_agg_prob']}{RESET}", flush=True )            
                print ( f"TRAINLENEJ:     INFO:      p.get_height()                            = {CAMEL}{p.get_height()}{RESET}", flush=True )
                print ( f"TRAINLENEJ:     INFO:      true_classes[{MIKADO}{i}{RESET}]  = {AMETHYST}{true_classes[i]}{RESET}", flush=True ) 
              if row['max_agg_prob'] == p.get_height():                                                      # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>555:
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}args.class_names[row['true_class']]  = {RESET}{MIKADO}{args.class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}correct_count                        = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not args.class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  pass
            i+=1 
  
  
        if DEBUG>0:
          print ( f"\nTRAINLENEJ:     INFO:      number correct (rna_seq_probabs_matrix) = {CHARTREUSE}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        # ~ writer.add_figure('rna_seq__probs_assigned_to_PREDICTED_classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}_bar_chart_rna_seq__probs_assigned_to_PREDICTED_classes.png"
        fig.savefig(fqn)
  
  
  
  
        # case rna-2:  bar chart showing probability assigned to TRUE classses
           
        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
        # ~ ax.set_title ( args.cancer_type_long )
              
        plt.xticks( rotation=90 )
        
        if bar_chart_x_labels=='case_id':
          c_id = pd_probabilities_matrix[ 'case_id' ]
        else:
          c_id = [i for i in range(pd_probabilities_matrix.shape[0])]

        if DEBUG>0:
          print ( f"TRAINLENEJ:     INFO:      probabilities_matrix {CYAN}(rna){RESET}  = \n{HOT_PINK}{probabilities_matrix}{RESET}", flush=True ) 

        for i in range ( 0, probabilities_matrix.shape[0] ):
          agg_prob = pd_probabilities_matrix[ 'agg_prob'][i]
          arg_max  = np.argmax( probabilities_matrix[i,:] )
          if DEBUG>0:
            print ( f"TRAINLENEJ:     INFO:      arg_max                   = {COTTON_CANDY}{arg_max}{RESET}", flush=True ) 
            print ( f"TRAINLENEJ:     INFO:      class_names[ arg_max ]    = {COTTON_CANDY}{class_names[ arg_max ]}{RESET}", flush=True ) 
          plt.bar( x=[ str(c_id[i]) ],   height=[ probabilities_matrix[i,arg_max] / agg_prob ],  color=class_colors[ arg_max ], label=class_names[ arg_max ] )


        # ~ ax = sns.barplot( x=c_id,  y=pd_probabilities_matrix[ 'true_class_prob' ], hue=pd_probabilities_matrix['pred_class'], palette=class_colors, dodge=False )                  # in pandas, 'index' means row index
        ax.set_title   ("Input Data = RNA-Seq UQ FPKM Values;  Bar Height = Probability Assigned to *TRUE* Cancer Sub-type",            fontsize=16 )
        ax.set_xlabel  ("Case ID",                                                     fontsize=14 )
        ax.set_ylabel  ("Probability Assigned by Network",                             fontsize=14 )
        ax.tick_params (axis='x', labelsize=8,   labelcolor='black')
        ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
        plt.ylim        (0.0, 1.0)
        # ~ plt.legend( args.class_names,loc=2, prop={'size': 14} )
        
        i=0
        for p in ax.patches:
          if not np.isnan(p.get_height()):                                                                   # if it's a number, then it will be a height (y value)
            for index, row in pd_probabilities_matrix.iterrows():
              if DEBUG>555:
                print ( f"TRAINLENEJ:     INFO:      row['max_agg_prob']                       = {COQUELICOT}{row['max_agg_prob']}{RESET}", flush=True )            
                print ( f"TRAINLENEJ:     INFO:      p.get_height()                            = {COQUELICOT}{p.get_height()}{RESET}", flush=True )
                print ( f"TRAINLENEJ:     INFO:      true_classes[{MIKADO}{i}{RESET}]  = {COQUELICOT}{true_classes[i]}{RESET}", flush=True ) 
              if row['max_agg_prob'] == p.get_height():                                                      # this logic is just used to map the bar back to the example (it's ugly, but couldn't come up with any other way)
                true_class = row['true_class']
                if DEBUG>555:
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< FOUND IT {RESET}",        flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}index                                = {RESET}{MIKADO}{index}{RESET}",                               flush=True ) 
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}true class                           = {RESET}{MIKADO}{true_class}{RESET}",                          flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}args.class_names[row['true_class']]  = {RESET}{MIKADO}{args.class_names[row['true_class']]}{RESET}", flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}pred class                           = {RESET}{MIKADO}{row['pred_class'][0]}{RESET}",                flush=True )
                    print ( f"TRAINLENEJ:     INFO:      {GREEN}correct_count                        = {RESET}{MIKADO}{correct_count}{RESET}",                       flush=True )                       
                if not args.class_names[row['true_class']] == row['pred_class'][0]:                          # this logic determines whether the prediction was correct or not
                  ax.annotate( f"{true_class}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=8, color=pkmn_type_colors[true_class], xytext=(0, 5), textcoords='offset points')
                else:
                  pass
            i+=1 
  
        if DEBUG>0:
          print ( f"\nTRAINLENEJ:     INFO:      number correct (rna_seq_probabs_matrix) = {COQUELICOT}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
                  
        writer.add_figure('rna_seq__probs_assigned_to_TRUE_classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}_bar_chart_rna_seq__probs_assigned_to_TRUE_classes.png"
        fig.savefig(fqn)
  
  
  
    
        # case rna-3:  bar chart showing probabilities assigned to ALL classses

        fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

        if DEBUG>55:
          np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
          print ( f"\nTRAINLENEJ:     INFO:       probabilities_matrix = \n{BLEU}{pd_probabilities_matrix}{RESET}", flush=True )
  
        plt.xticks( rotation=90 )
        pd_probabilities_matrix[ 'pred_class_idx'  ]  = pred_class_idx  [0:n_samples]                      # possibly truncate rows  because n_samples may have been changed in generate() if only a subset of the samples was specified (e.g. for option '-c DESIGNATED_MULTIMODE_CASE_FLAG')
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
        plt.legend( args.class_names,loc=2, prop={'size': 14} )

        if DEBUG>0:
          print ( f"\nTRAINLENEJ:     INFO:      number correct (pd_probabilities_matrix) = {COQUELICOT}{correct_count}{RESET}", flush=True )
  
        pct_correct = correct_count/n_samples
        stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
        plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
  
        plt.tight_layout()
      
        writer.add_figure('rna_seq__probs_assigned_to_ALL__classes', fig, 0 )
        
        # save version to logs directory
        now              = datetime.datetime.now()
              
        fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}_bar_chart_rna_seq__probs_assigned_to_ALL__classes.png"
        
        fig.savefig(fqn)
        
  

        fqn = f"{args.log_dir}/probabilities_dataframe_rna.csv"
        try:
          pd_probabilities_matrix.to_csv ( fqn, sep='\t' )
          if DEBUG>0:
            print ( f"TRAINLENEJ:     INFO:     now saving  probabilities dataframe {COQUELICOT}(rna){RESET}   to   {MAGENTA}{fqn}{RESET}"  )
        except Exception as e:
          print ( f"{ORANGE}TRAINLENEJ:     WARNING:     could not save file   = {ORANGE}{fqn}{RESET}"  )
          # ~ print ( f"{ORANGE}TRAINLENEJ:     WARNING:     error was: {e}{RESET}" )     
          
  
 
        
        
        
        
        # case multimode:

        if DEBUG>0:
          print ( f"TRAINLENEJ:     INFO:     now loading probabilities dataframe {CYAN}(image){RESET} from {MAGENTA}{fqn}{RESET} if it exists from an earlier run"  ) 
          
        image_dataframe_file_exists=False
        fqn = f"{args.log_dir}/probabilities_dataframe_image.csv"
        try:
          pd_aggregate_tile_probabilities_matrix = pd.read_csv( fqn, sep='\t'  )
          image_dataframe_file_exists=True
        except Exception as e:
          print ( f"{ORANGE}TRAINLENEJ:     INFO:     could not open file  {MAGENTA}{fqn}{RESET}{ORANGE} - it probably doesn't exist"  )
          print ( f"{ORANGE}TRAINLENEJ:     INFO:     explanation: if you want the bar chart which combines image and rna probabilities, you need to have performed both an image and an rna run. {RESET}" )                
          print ( f"{ORANGE}TRAINLENEJ:     INFO:     e.g. perform the following sequence of runs:{RESET}" )                 
          print ( f"{ORANGE}TRAINLENEJ:     INFO:          {CYAN}./do_all.sh     -d <cancer type code> -i image -c DESIGNATED_UNIMODE_CASE_FLAG -v true{RESET}" )                 
          print ( f"{ORANGE}TRAINLENEJ:     INFO:          {CYAN}./just_test.sh  -d <cancer type code> -i image -c DESIGNATED_UNIMODE_CASE_FLAG{RESET}" )                 
          print ( f"{ORANGE}TRAINLENEJ:     INFO:          {CYAN}./do_all.sh     -d <cancer type code> -i rna   -c DESIGNATED_UNIMODE_CASE_FLAG{RESET}" )                 
          print ( f"{ORANGE}TRAINLENEJ:     INFO:          {CYAN}./just_test.sh  -d <cancer type code> -i rna   -c DESIGNATED_UNIMODE_CASE_FLAG{RESET}" )   
          print ( f"{ORANGE}TRAINLENEJ:     INFO:     continuing...{RESET}" ) 

        if image_dataframe_file_exists:

          upper_bound_of_indices_to_plot_image = len(pd_aggregate_tile_probabilities_matrix.index)
          
          if DEBUG>0:
            print ( f"\nTRAINLENEJ:     INFO:      upper_bound_of_indices_to_plot_image = {COQUELICOT}{upper_bound_of_indices_to_plot_image}{RESET}", flush=True )
                      
          if upper_bound_of_indices_to_plot_image  !=   upper_bound_of_indices_to_plot_rna:
            print ( f"{ORANGE}TRAINLENEJ:     INFO:     for some reason the numbers of image examples and the number of rna examples to be plotted differ{RESET}"      ) 
            print ( f"{ORANGE}TRAINLENEJ:     INFO:        upper_bound_of_indices_to_plot_image = {MIKADO}{upper_bound_of_indices_to_plot_image}{RESET}"  ) 
            print ( f"{ORANGE}TRAINLENEJ:     INFO:        upper_bound_of_indices_to_plot_rna   = {MIKADO}{upper_bound_of_indices_to_plot_rna}{RESET}"  ) 
            print ( f"{ORANGE}TRAINLENEJ:     INFO:     possible explanation: one or both of the {CYAN}N_SAMPLES{RESET}{ORANGE} config settings is too small to have captured sufficient of the {CYAN}{args.cases}{RESET}{ORANGE} cases"      ) 
            print ( f"{ORANGE}TRAINLENEJ:     INFO:     skipping combined image+rna porbabilities plot that would otherwise have been generated{RESET}"      ) 
            print ( f"{ORANGE}TRAINLENEJ:     INFO:     continuing ...{RESET}"      ) 
            

          else:
            
            if DEBUG>0:
              np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
              print ( f"\nTRAINLENEJ:     INFO:     pd_aggregate_tile_probabilities_matrix {CYAN}(image){RESET} (from {MAGENTA}{fqn}{RESET}) = \n{COTTON_CANDY}{pd_aggregate_tile_probabilities_matrix[0:upper_bound_of_indices_to_plot_rna]}{RESET}", flush=True )   
              
            pd_aggregate_tile_probabilities_matrix[ 'true_class_prob' ] /= pd_aggregate_tile_probabilities_matrix[ 'agg_prob' ]   # image case only: normalize by dividing by number of tiles in the patch (which was saved as field 'agg_prob')
      
            if DEBUG>0:
              np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
              print ( f"\nTRAINLENEJ:     INFO:       pd_aggregate_tile_probabilities_matrix {CYAN}(image){RESET} normalized probabilities (from {MAGENTA}{fqn}{RESET}) = \n{COTTON_CANDY}{pd_aggregate_tile_probabilities_matrix}{RESET}", flush=True )  
              
            
            if DEBUG>0:
              print ( f"\nTRAINLENEJ:     INFO:     n me {CYAN}(rna){RESET} from {MAGENTA}{fqn}{RESET} if it exists from an earlier or the current run"  )  
         
            rna_dataframe_file_exists=False             
            fqn = f"{args.log_dir}/probabilities_dataframe_rna.csv"
            try:
              pd_probabilities_matrix = pd.read_csv(  fqn, sep='\t'  )
              rna_dataframe_file_exists=True
              if DEBUG>0:
                np.set_printoptions(formatter={'float': lambda x: f"{x:>7.2f}"})
                print ( f"\nTRAINLENEJ:     INFO:     pd_probabilities_matrix {CYAN}(rna){RESET} (from {MAGENTA}{fqn}{RESET}) = \n{ARYLIDE}{pd_probabilities_matrix}{RESET}", flush=True )  
            except Exception as e:
              print ( f"{ORANGE}TRAINLENEJ:     INFO:     could not open file  = {ORANGE}{fqn}{RESET}{ORANGE} - it probably doesn't exist"  )
              print ( f"{ORANGE}TRAINLENEJ:     INFO:     if you want the bar chart which combines image and rna probabilities, you need to have performed both an image and an rna run. {RESET}" )                
              print ( f"{ORANGE}TRAINLENEJ:     INFO:     e.g. perform the following sequence of runs:{RESET}" )                 
              print ( f"{ORANGE}TRAINLENEJ:     INFO:              {CYAN}./do_all.sh     -d <cancer type code> -i image -c DESIGNATED_UNIMODE_CASE_FLAG -v true{RESET}{ORANGE}'{RESET}" )                 
              print ( f"{ORANGE}TRAINLENEJ:     INFO:              {CYAN}./just_test.sh  -d <cancer type code> -i image -c DESIGNATED_UNIMODE_CASE_FLAG{RESET}" )                 
              print ( f"{ORANGE}TRAINLENEJ:     INFO:              {CYAN}./do_all.sh     -d <cancer type code> -i rna   -c DESIGNATED_UNIMODE_CASE_FLAG{RESET}" )                 
              print ( f"{ORANGE}TRAINLENEJ:     INFO:              {CYAN}./just_test.sh  -d <cancer type code> -i rna   -c DESIGNATED_UNIMODE_CASE_FLAG{RESET}" )   
              print ( f"{ORANGE}TRAINLENEJ:     INFO:     continuing...{RESET}" ) 
    
                        
      
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
                print ( f"\nTRAINLENEJ:     INFO:      upper_bound_of_indices_to_plot_rna                                   = {ARYLIDE}{upper_bound_of_indices_to_plot_rna}{RESET}", flush=True )
                print ( f"\nTRAINLENEJ:     INFO:      x_labels                                                             = \n{ARYLIDE}{x_labels}{RESET}", flush=True )
                print ( f"\nTRAINLENEJ:     INFO:      {CYAN}(rna){RESET} pd_probabilities_matrix                [ 'true_class_prob' ]   = \n{ARYLIDE}{set1}{RESET}", flush=True )
                print ( f"\nTRAINLENEJ:     INFO:      {CYAN}(img){RESET} pd_aggregate_tile_probabilities_matrix [ 'true_class_prob' ]   = \n{COTTON_CANDY}{set2}{RESET}", flush=True )
    
              
              p1 = plt.bar( x=x_labels, height=set1,               color=col0 )
              p2 = plt.bar( x=x_labels, height=set2, bottom=set1,  color=col1 )
             
              ax.set_title   ("Input Data = Imaga Tiles; RNA-Seq FPKM UQ;  Bar Height = Composite (Image + RNA-Seq) Probability Assigned to *TRUE* Cancer Sub-types",  fontsize=16 )
              ax.set_xlabel  ("Case ID",                                                     fontsize=14 )
              ax.set_ylabel  ("Probability Assigned by Network",                             fontsize=14 )
              ax.tick_params (axis='x', labelsize=8,   labelcolor='black')
              ax.tick_params (axis='y', labelsize=14,  labelcolor='black')
              # ~ plt.legend( args.class_names,loc=2, prop={'size': 14} )
              plt.xticks( rotation=90 )     
      
        
              if DEBUG>0:
                print ( f"\nTRAINLENEJ:     INFO:      number correct (image+rna) = {CHARTREUSE}{correct_count}{RESET}", flush=True )
        
              pct_correct = correct_count/n_samples
              stats=f"Statistics: sample count: {n_samples}; correctly predicted: {correct_count}/{n_samples} ({100*pct_correct:2.1f}%)"
              plt.figtext( 0.15, 0, stats, size=14, color="grey", style="normal" )
        
              plt.tight_layout()
                        
              writer.add_figure('z_multimode__probs_assigned_to_TRUE_classes', fig, 0 )         
            
  
  

   
    # (F)  MAYBE PROCESS AND DISPLAY RUN LEVEL CONFUSION MATRICES   
    
    if ( args.just_test!='True') | ( (args.just_test=='True')  &  (args.input_mode=='image_rna') & (args.multimode=='image_rna') ):
    
      #np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE if x==0 else WHITE if x<=5 else CARRIBEAN_GREEN} {x:>15d}"})  
      #print ( f"TRAINLENEJ:     INFO:  {ORANGE}run_level{RESET}_classifications_matrix (all test samples, using the best model that was saved during this run =\n" )
      #print ( f"         ", end='' ) 
      #print ( [ f"{name:.50s}" for name in class_names ] )    
      #print ( f"\n{run_level_classifications_matrix}{RESET}" )
  
  
      if DEBUG>4:
        print ( f"\n{run_level_classifications_matrix}" )
                 
      run_level_classifications_matrix_acc[run-1,:,:] = run_level_classifications_matrix[:,:]                # accumulate run_level_classifications_matrices
   
      if DEBUG>9:
        print ( f"\n{run_level_classifications_matrix_acc[run-1,:,:]}" )    
  
    
      print(  '\033[11B' )
      print( f"TRAINLENEJ:       INFO:    {BITTER_SWEET}Test predictions produced during training for this run{RESET}"         )
      print( f"TRAINLENEJ:       INFO:    {BITTER_SWEET}======================================================{RESET}"  )
      print( f"TRAINLENEJ:       INFO:                                                                                      "  )  
    
      total_correct, total_examples  = show_classifications_matrix( writer, total_runs_in_job, pct_test, epoch, run_level_classifications_matrix, level='run' )
  
  
      print( f"TRAINLENEJ:       INFO:    correct / examples  =  {BITTER_SWEET}{np.sum(total_correct, axis=0)} / {np.sum(run_level_classifications_matrix, axis=None)}{WHITE}  ({BITTER_SWEET}{100 * np.sum(total_correct, axis=0) / np.sum(run_level_classifications_matrix):3.1f}%){RESET}")
  
      for i in range( 0, len( run_level_classifications_matrix) ):                                           # reset for the next run   
        run_level_classifications_matrix[i] = 0  
    
  
      hours   = round( (time.time() - start_time) / 3600,  1   )
      minutes = round( (time.time() - start_time) /   60,  1   )
      seconds = round( (time.time() - start_time),     0       )
      #pplog.log_section('run complete in {:} mins'.format( minutes ) )
  
      print( f'TRAINLENEJ:       INFO:    elapsed time since job started: {MIKADO}{minutes}{RESET} mins ({MIKADO}{seconds:.1f}{RESET} secs)')
  
      print ( "\033[6A" )
            
    #  ^^^  JOB FINISHES HERE ^^^
  
  
  
  
  
    # (G)  MAYBE PROCESS AND DISPLAY JOB LEVEL CONFUSION MATRIX
    
    if (args.just_test!='True') & (total_runs_in_job>1) & (run==total_runs_in_job):
      
      print(  '\033[6B' )      
      print( f'TRAINLENEJ:       INFO:'                                                                                    )
      print( f"TRAINLENEJ:       INFO:    {CARRIBEAN_GREEN}Test predictions produced during training for this job{RESET}"     )
      print( f"TRAINLENEJ:       INFO:    {CARRIBEAN_GREEN}======================================================{RESET}"  )  
      print( f'TRAINLENEJ:       INFO:'                                                                                    )      
    
      total_correct, total_examples  = show_classifications_matrix( writer, total_runs_in_job, pct_test, epoch, job_level_classifications_matrix, level='job' )
    
      np.seterr( invalid='ignore', divide='ignore' )
      print( f"\n" )
      print( f'TRAINLENEJ:       INFO:    number of runs in this job                 = {MIKADO}{total_runs_in_job}{RESET}')
      print( f"TRAINLENEJ:       INFO:    total for ALL test examples over ALL runs  =  {CARRIBEAN_GREEN}{np.sum(total_correct, axis=0)} / {np.sum(job_level_classifications_matrix, axis=None)}  ({CARRIBEAN_GREEN}{100 * np.sum(total_correct, axis=0) / np.sum(job_level_classifications_matrix):3.1f}%){RESET}")
    
      np.set_printoptions(formatter={'int': lambda x: f"{CARRIBEAN_GREEN}{x:>6d}"})
      print( f'TRAINLENEJ:       INFO:    total correct per subtype over all runs:     {total_correct}{RESET}')
      np.set_printoptions(formatter={'float': lambda x: f"{CARRIBEAN_GREEN}{x:>6.1f}"})
      print( f'TRAINLENEJ:       INFO:     %    correct per subtype over all runs:     { 100 * np.divide( total_correct, total_examples) }{RESET}')
      np.seterr(divide='warn', invalid='warn')  
      
      if DEBUG>9:
        np.set_printoptions(formatter={'int': lambda x: f"{CARRIBEAN_GREEN}{x:>6d}    "})    
        print ( f"TRAINLENEJ:       INFO:    run_level_classifications_matrix_acc[0:total_runs_in_job,:,:]            = \n{run_level_classifications_matrix_acc[0:total_runs_in_job,:,:] }{RESET}" )
      if DEBUG>9:
        print ( f"TRAINLENEJ:       INFO:  run_level_classifications_matrix_acc                 = {MIKADO}{run_level_classifications_matrix_acc[ 0:total_runs_in_job, : ] }{RESET}"     )
  
    if ( args.box_plot=='True' ) & ( total_runs_in_job>=args.minimum_job_size ):
        box_plot_by_subtype( args, writer, total_runs_in_job, pct_test, run_level_classifications_matrix_acc )



  # (H)  CLOSE UP AND END
  writer.close()        

  hours   = round( (time.time() - start_time) / 3600,  1   )
  minutes = round( (time.time() - start_time) /   60,  1   )
  seconds = round( (time.time() - start_time)       ,  0   )
  #pplog.log_section('Job complete in {:} mins'.format( minutes ) )

  print( f'\033[18B')
  if ( args.just_test=='True') & ( args.input_mode=='rna' ):
    print( f'\033[12B')  
  
  print( f'TRAINLENEJ:       INFO: Job complete. The job ({MIKADO}{total_runs_in_job}{RESET} runs) took {MIKADO}{minutes}{RESET} minutes ({MIKADO}{seconds:.0f}{RESET} seconds) to complete')
            
  #pplog.log_section('Model saved.')
  



def train(args, epoch, train_loader, model, optimizer, loss_function, writer, train_loss_min, batch_size  ):

    """
    Train model and update parameters in batches of the whole training set
    """
    
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

        # from: https://stackoverflow.com/questions/44732217/why-do-we-need-to-explicitly-call-zero-grad
        # We explicitly need to call zero_grad() because, after loss.backward() (when gradients are computed), we need to use optimizer.step() to proceed gradient descent. More specifically, the gradients are not automatically zeroed because these two operations, loss.backward() and optimizer.step(), are separated, and optimizer.step() requires the just computed gradients.
        optimizer.zero_grad()

        batch_images = batch_images.to ( device )                                                          # send to GPU
        batch_genes  = batch_genes.to  ( device )                                                          # send to GPU
        image_labels = image_labels.to ( device )                                                          # send to GPU
        rna_labels   = rna_labels.to   ( device )                                                          # send to GPU

        if DEBUG>99:
          print ( f"TRAINLENEJ:     INFO:     train(): batch_images[0]                    = {MIKADO}\n{batch_images[0] }{RESET}", flush=True   )

        if DEBUG>99:
          print ( f"TRAINLENEJ:     INFO:     train(): type(batch_images)                 = {MIKADO}{type(batch_images)}{RESET}",  flush=True  )
          print ( f"TRAINLENEJ:     INFO:     train(): batch_images.size()                = {MIKADO}{batch_images.size()}{RESET}", flush=True  )


        if DEBUG>2:
          print( f"TRAINLENEJ:     INFO:      train(): about to call {MAGENTA}model.forward(){RESET}" )

        gpu                = 0                                                                             # to maintain compatability with NN_MODE=pre_compress
        encoder_activation = 0                                                                             # to maintain compatability with NN_MODE=pre_compress
        
        if args.input_mode=='image':
          y1_hat, y2_hat, embedding = model.forward( [ batch_images, 0          ,  batch_fnames] , gpu, encoder_activation  )          # perform a step. y1_hat = image outputs; y2_hat = rna outputs
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          if DEBUG>9:
            print ( f"TRAINLENEJ:     INFO:     train(): batch_genes.size()                = {batch_genes.size}" )
          y1_hat, y2_hat, embedding = model.forward( [0,             batch_genes,  batch_fnames],  gpu, encoder_activation )           # perform a step. y1_hat = image outputs; y2_hat = rna outputs


        if (args.input_mode=='image'):
          
          if DEBUG>2:
            np.set_printoptions(formatter={'float': lambda x:   "{:>6.2f}".format(x)})
            image_labels_numpy = (image_labels .cpu() .data) .numpy()
            y1_hat_numpy       = (y1_hat       .cpu() .data) .numpy()
            batch_fnames_npy   = (batch_fnames .cpu() .data) .numpy()
            random_pick        = random.randint( 0, y1_hat_numpy.shape[0]-1 )
            print ( f"TRAINLENEJ:     INFO:      test():        fq_link            [{random_pick:3d}]      (Truth)         = {MIKADO}{args.data_dir}/{batch_fnames_npy[random_pick]}.fqln{RESET}"     )            
            print ( f"TRAINLENEJ:     INFO:      test():        image_labels_numpy [{random_pick:3d}]      {GREEN}(Truth){RESET}         = {MIKADO}{image_labels_numpy[random_pick]}{RESET}"     )            
            print ( f"TRAINLENEJ:     INFO:      test():        y1_hat_numpy       [{random_pick:3d}]      {ORANGE}(Predictions){RESET}   = {MIKADO}{y1_hat_numpy[random_pick]}{RESET}"     )
            print ( f"TRAINLENEJ:     INFO:      test():        predicted class    [{random_pick:3d}]                      = {RED if image_labels_numpy[random_pick]!=np.argmax(y1_hat_numpy[random_pick]) else GREEN}{np.argmax(y1_hat_numpy[random_pick])}{RESET}"     )

            
          loss_images       = loss_function( y1_hat, image_labels )
          loss_images_value = loss_images.item()                                                           # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories
          
          if DEBUG>2:
            print ( f"TRAINLENEJ:     INFO:      test(): {MAGENTA}loss_images{RESET} (for this mini-batch)  = {PURPLE}{loss_images_value:6.3f}{RESET}" )
            # ~ time.sleep(.25)
        
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          if DEBUG>9:
            np.set_printoptions(formatter={'int': lambda x:   "{:>4d}".format(x)})
            rna_labels_numpy = (rna_labels.cpu().data).numpy()
            print ( "TRAINLENEJ:     INFO:      test():       rna_labels_numpy                = \n{:}".format( image_labels_numpy  ) )
          if DEBUG>9:
            np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
            y2_hat_numpy = (y2_hat.cpu().data).numpy()
            print ( "TRAINLENEJ:     INFO:      test():       y2_hat_numpy                      = \n{:}".format( y2_hat_numpy) )
          loss_genes        = loss_function( y2_hat, rna_labels )
          loss_genes_value  = loss_genes.item()                                                            # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories

        #l1_loss          = l1mapping_file_penalty(model, args.l1_coef)
        l1_loss           = 0

        if (args.input_mode=='image'):
          total_loss        = loss_images_value + l1_loss
        elif (args.input_mode=='rna') | (args.input_mode=='image_rna'):
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
          elif (args.input_mode=='rna') | (args.input_mode=='image_rna'):
            print ( f"\
\033[2K\r\033[27C{DULL_WHITE}train():\
\r\033[40Cn={i+1:>3d}{CLEAR_LINE}\
\r\033[73Closs_rna={loss_genes_value:5.2f}\
\r\033[96Cl1_loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{156+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
            print ( "\033[2A" )          


        if (args.input_mode=='image'):
          loss_images.backward()
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):          
          loss_genes.backward()

        # Perform gradient clipping *before* calling `optimizer.step()`.
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        
        if (args.input_mode=='image'):
          loss_images_sum      += loss_images_value
        if (args.input_mode=='rna') | (args.input_mode=='image_rna'):
          loss_genes_sum       += loss_genes_value
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss

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
                                                                                                        test_loss_min, show_all_test_examples, batch_size, nn_type_img, nn_type_rna, annotated_tiles, class_names, class_colours ):

    """Test model by pushing one or more held-out batches through the network
    """

    global class_colors 
    global file_name_prefix
    global global_batch_count
    global run_level_total_correct    
    global run_level_classifications_matrix
    global job_level_classifications_matrix

    
    model.eval()                                                                                           # set model to evaluation mode

    loss_images_sum     = 0
    loss_genes_sum      = 0
    l1_loss_sum         = 0
    total_loss_sum      = 0

      
    for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in  enumerate( test_loader ):
        
        batch_images = batch_images.to(device)
        batch_genes  = batch_genes .to(device)
        image_labels = image_labels.to(device)
        rna_labels   = rna_labels  .to(device)        

        gpu                = 0                                                                             # not used, but necessary to to maintain compatability with NN_MODE=pre_compress
        encoder_activation = 0                                                                             # not used, but necessary to to maintain compatability with NN_MODE=pre_compress

        if DEBUG>2:
          print( f"TRAINLENEJ:     INFO:      test(): about to call {COQUELICOT}model.forward(){RESET}" )

        if args.input_mode=='image':
          with torch.no_grad():                                                                            # don't need gradients for testing
            y1_hat, y2_hat, embedding = model.forward( [ batch_images, 0            , batch_fnames], gpu, encoder_activation  )          # perform a step. y1_hat = image outputs; y2_hat = rna outputs

          if DEBUG>2:
            np.set_printoptions(formatter={'float': lambda x:   "{:>6.2f}".format(x)})
            image_labels_numpy = (image_labels .cpu() .data) .numpy()
            y1_hat_numpy       = (y1_hat       .cpu() .data) .numpy()
            batch_fnames_npy   = (batch_fnames .cpu() .data) .numpy()
            random_pick        = random.randint( 0, y1_hat_numpy.shape[0]-1 )
            print ( f"TRAINLENEJ:     INFO:      test():        fq_link            [{random_pick:3d}]      (Truth)         = {MIKADO}{args.data_dir}/{batch_fnames_npy[random_pick]}.fqln{RESET}"     )            
            print ( f"TRAINLENEJ:     INFO:      test():        image_labels_numpy [{random_pick:3d}]      {GREEN}(Truth){RESET}         = {MIKADO}{image_labels_numpy[random_pick]}{RESET}"     )            
            print ( f"TRAINLENEJ:     INFO:      test():        y1_hat_numpy       [{random_pick:3d}]      {ORANGE}(Predictions){RESET}   = {MIKADO}{y1_hat_numpy[random_pick]}{RESET}"     )
            print ( f"TRAINLENEJ:     INFO:      test():        predicted class    [{random_pick:3d}]                      = {RED if image_labels_numpy[random_pick]!=np.argmax(y1_hat_numpy[random_pick]) else GREEN}{np.argmax(y1_hat_numpy[random_pick])}{RESET}"     )


        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          with torch.no_grad():                                                                            # don't need gradients for testing
            y1_hat, y2_hat, embedding = model.forward( [ 0,            batch_genes  , batch_fnames], gpu, encoder_activation )
          
        image_labels_values   =   image_labels.cpu().detach().numpy()
        rna_labels_values     =   rna_labels  .cpu().detach().numpy()
        batch_fnames_npy      =   batch_fnames.cpu().detach().numpy()        




        # move to a separate function ----------------------------------------------------------------------------------------------
        if   ( args.input_mode=='image' ):
          
          preds, p_full_softmax_matrix, p_highest, p_2nd_highest, p_true_class = analyse_probs( y1_hat, image_labels_values )          
          
        if ( args.input_mode=='image' ) & ( args.just_test=='True' ):
          
          if args.scattergram=='True':
            if DEBUG>2:
                print ( f"TRAINLENEJ:     INFO:      test():         global_batch_count {DIM_WHITE}(super-patch number){RESET} = {global_batch_count+1:5d}  {DIM_WHITE}({((global_batch_count+1)/(args.supergrid_size**2)):04.2f}){RESET}" )
                      
          if global_batch_count%(args.supergrid_size**2)==0:                                               # establish grid arrays on the FIRST batch of each grid
            grid_images                = batch_images.cpu().numpy()
            grid_labels                = image_labels.cpu().numpy()
            grid_preds                 = preds
            grid_p_highest             = p_highest
            grid_p_2nd_highest         = p_2nd_highest
            grid_p_true_class          = p_true_class
            grid_p_full_softmax_matrix = p_full_softmax_matrix 

            if DEBUG>88:
              print ( f"TRAINLENEJ:     INFO:      test():             batch_images.shape                      = {BLEU}{batch_images.shape}{RESET}"                  )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_images.shape                       = {BLEU}{grid_images.shape}{RESET}"                   )
              print ( f"TRAINLENEJ:     INFO:      test():             image_labels.shape                      = {BLEU}{image_labels.shape}{RESET}"                  )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_labels.shape                       = {BLEU}{grid_labels.shape}{RESET}"                   )
              print ( f"TRAINLENEJ:     INFO:      test():             preds.shape                             = {BLEU}{preds.shape}{RESET}"                         )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_preds.shape                        = {BLEU}{grid_preds.shape}{RESET}"                    )
              print ( f"TRAINLENEJ:     INFO:      test():             p_highest.shape                         = {BLEU}{p_highest.shape}{RESET}"                     )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_highest.shape                    = {BLEU}{grid_p_highest.shape}{RESET}"                )            
              print ( f"TRAINLENEJ:     INFO:      test():             p_2nd_highest.shape                     = {BLEU}{p_2nd_highest.shape}{RESET}"                 )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_2nd_highest.shape                = {BLEU}{grid_p_2nd_highest.shape}{RESET}"            )
              print ( f"TRAINLENEJ:     INFO:      test():             p_full_softmax_matrix.shape             = {BLEU}{p_full_softmax_matrix.shape}{RESET}"         )                                    
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_full_softmax_matrix.shape        = {BLEU}{grid_p_full_softmax_matrix.shape}{RESET}"    )
                      
          else:                                                                                            # ... accumulate for subsequent batches in the same grid 
            grid_images                = np.append( grid_images,                batch_images.cpu().numpy(), axis=0 )
            grid_labels                = np.append( grid_labels,                image_labels.cpu().numpy(), axis=0 )
            grid_preds                 = np.append( grid_preds,                 preds,                      axis=0 )
            grid_p_highest             = np.append( grid_p_highest,             p_highest,                  axis=0 )
            grid_p_2nd_highest         = np.append( grid_p_2nd_highest,         p_2nd_highest,              axis=0 )
            grid_p_true_class          = np.append( grid_p_true_class,          p_true_class,               axis=0 )
            grid_p_full_softmax_matrix = np.append( grid_p_full_softmax_matrix, p_full_softmax_matrix,      axis=0 )
  
            if DEBUG>88:
              print ( f"TRAINLENEJ:     INFO:      test():             grid_images.shape                       = {MIKADO}{grid_images.shape}{RESET}"                 )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_labels.shape                       = {MIKADO}{grid_labels.shape}{RESET}"                 )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_preds.shape                        = {MIKADO}{grid_preds.shape}{RESET}"                  )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_highest.shape                    = {MIKADO}{grid_p_highest.shape}{RESET}"              )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_2nd_highest.shape                = {MIKADO}{grid_p_2nd_highest.shape}{RESET}"          )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_true_class.shape                 = {MIKADO}{grid_p_true_class.shape}{RESET}"           )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_full_softmax_matrix.shape        = {MIKADO}{grid_p_full_softmax_matrix.shape}{RESET}"  )

            if DEBUG>88:
              np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_full_softmax_matrix              = \n{CHARTREUSE}{grid_p_full_softmax_matrix}{RESET}"  ) 

            if global_batch_count%(args.supergrid_size**2)==(args.supergrid_size**2)-1:                    # if it is the last batch in the grid (super-patch)
  
              index  = int(i/(args.supergrid_size**2))         # the entry we will update. (Because we aren't accumulating on every i'th batch, but rather on every  args.supergrid_size**2-1'th batch  (one time per grid))

              if DEBUG>5:
                np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():             index                           =  {MAGENTA}{index}{RESET}"  )

              patches_true_classes[index] =  image_labels.cpu().detach().numpy()[0]                        # all tiles in a patch belong to the same case, so we can chose any of them
              patches_case_id     [index] =  batch_fnames_npy[0]                                           # all tiles in a patch belong to the same case, so we can chose any of them

              if DEBUG>88:
                np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():             patches_case_id                 =  {MAGENTA}{patches_case_id}{RESET}{CLEAR_LINE}"  )
                print ( f"TRAINLENEJ:     INFO:      test():             patches_case_id[index]          =  {MAGENTA}{patches_case_id[index]}{RESET}{CLEAR_LINE}"  )
  
              grid_tile_probabs_totals_by_class = np.transpose(np.expand_dims( grid_p_full_softmax_matrix.sum( axis=0 ), axis=1 ))         # this is where we sum the totals across all tiles
              binary_matrix = np.zeros_like(grid_p_full_softmax_matrix)                                                                    # new matrix same shape as grid_p_full_softmax_matrix, with all values set to zero
              binary_matrix[ np.arange( len(grid_p_full_softmax_matrix)), grid_p_full_softmax_matrix.argmax(1) ] = 1                       # set the max value in each row to 1, all others zero
  
              if DEBUG>8:
                print ( f"TRAINLENEJ:     INFO:      test():         binary_matrix         = \n{CHARTREUSE}{binary_matrix}{RESET}"  )              
  
              grid_tile_winners_totals_by_class        = np.transpose(np.expand_dims( binary_matrix.sum( axis=0 ), axis=1 ))               # same, but 'winner take all' at the tile level
  
              if DEBUG>8:
                np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():         grid_tile_probabs_totals_by_class                     =    {CHARTREUSE}{grid_tile_probabs_totals_by_class}{RESET}"  )
                print ( f"TRAINLENEJ:     INFO:      test():         grid_tile_winners_totals_by_class                     =    {CHARTREUSE}{grid_tile_winners_totals_by_class}{RESET}"  )
                           
              aggregate_tile_probabilities_matrix[index] = grid_tile_probabs_totals_by_class
              aggregate_tile_level_winners_matrix[index] = grid_tile_winners_totals_by_class + random.uniform( 0.001, 0.01)   # necessary to make all the tile totals unique when we go looking for them later. ugly but necessary
  
              if DEBUG>8:
                np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():         aggregate_tile_probabilities_matrix                = \n{CHARTREUSE}{aggregate_tile_probabilities_matrix}{RESET}"  ) 

          if DEBUG>5:
            np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})
            print ( f"TRAINLENEJ:     INFO:      test():             global_batch_count              = {CHARTREUSE}{global_batch_count}{RESET}"  ) 
            print ( f"TRAINLENEJ:     INFO:      test():             args.supergrid_size**2          =  {CHARTREUSE}{args.supergrid_size**2}{RESET}"  ) 

          global_batch_count+=1
        
          if DEBUG>999:
              print ( f"TRAINLENEJ:     INFO:      test():             global_batch_count%(args.supergrid_size**2)                       = {global_batch_count%(args.supergrid_size**2)}"  )
          
          if global_batch_count%(args.supergrid_size**2)==0:
            if args.input_mode=='image':
              print("")
              
              if args.annotated_tiles=='True':
                
                fig=plot_classes_preds(args, model, tile_size, grid_images, grid_labels, 0,  grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_full_softmax_matrix, class_names, class_colours )
                writer.add_figure('1 annotated tiles', fig, epoch)
                plt.close(fig)

              batch_fnames_npy = batch_fnames.numpy()                                                      # batch_fnames was set up during dataset generation: it contains a link to the SVS file corresponding to the tile it was extracted from - refer to generate() for details
              
              if DEBUG>99:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():       batch_fnames_npy.shape      = {batch_fnames_npy.shape:}" )        
                print ( f"TRAINLENEJ:     INFO:      test():       batch_fnames_npy            = {batch_fnames_npy:}"       )
    
              fq_link = f"{args.data_dir}/{batch_fnames_npy[0]}.fqln"
              
              if DEBUG>28:
                np.set_printoptions(formatter={'int': lambda x: "{:>d}".format(x)})
                print ( f"TRAINLENEJ:     INFO:      test():       fq_link                     = {PINK}{fq_link:}{RESET}"                )
                print ( f"TRAINLENEJ:     INFO:      test():       file fq_link points to      = {PINK}{os.readlink(fq_link)}{RESET}"    )
              
              try:
                background_image = np.load(f"{fq_link}")
              except Exception as e:
                print ( f"{RED}TRAINLENEJ:     FATAL:  '{e}'{RESET}" )
                print ( f"{RED}TRAINLENEJ:     FATAL:     explanation: a required {MAGENTA}entire_patch.npy{RESET}{RED} file doesn't exist. (Probably none exist). These contain the background images used for the scattergram. {RESET}" )                
                print ( f"{RED}TRAINLENEJ:     FATAL:     if you used {CYAN}./just_test_dont_tile.sh{RESET}{RED} without first running {CYAN}./just_test.sh{RESET}{RED}' then tiling and patch generation will have been skipped ({CYAN}--skip_tiling = {MIKADO}'True'{RESET}{RED} in that script{RESET}{RED}){RESET}" )
                print ( f"{RED}TRAINLENEJ:     FATAL:     if so, run '{CYAN}./just_test.sh -d <cancer type code> -i <INPUT_MODE>{RESET}{RED}' at least one time so that these files will be generated{RESET}" )                 
                print ( f"{RED}TRAINLENEJ:     FATAL:     halting now ...{RESET}" )                 
                sys.exit(0)              

              
              if DEBUG>0:
                print ( f"TRAINLENEJ:     INFO:      test():        background_image.shape = {background_image.shape}" )
                
              if args.scattergram=='True':
                
                plot_scatter(args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_preds, p_full_softmax_matrix, show_patch_images='True')
                # ~ plot_scatter(args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_preds, p_full_softmax_matrix, show_patch_images='False')

              if (args.probs_matrix=='True') & (args.multimode!='image_rna'):
                
                # ~ # without interpolation
                # ~ matrix_types = [ 'margin_1st_2nd', 'confidence_RIGHTS', 'p_std_dev' ]
                # ~ for n, matrix_type in enumerate(matrix_types):
                  # ~ plot_matrix (matrix_type, args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_p_full_softmax_matrix, grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_true_class, 'none' )    # always display without probs_matrix_interpolation 
                # with  interpolation
                matrix_types = [ 'probs_true' ]
                for n, matrix_type in enumerate(matrix_types): 
                  plot_matrix (matrix_type, args, writer, (i+1)/(args.supergrid_size**2), background_image, tile_size, grid_labels, class_names, class_colours, grid_p_full_softmax_matrix, grid_preds, grid_p_highest, grid_p_2nd_highest, grid_p_true_class, args.probs_matrix_interpolation )
          # move to a separate function ----------------------------------------------------------------------------------------------
         










        # move to a separate function ----------------------------------------------------------------------------------------------
        if ( args.input_mode=='rna' ) & ( args.just_test=='True' ):
          
          preds, p_full_softmax_matrix, p_highest, p_2nd_highest, p_true_class = analyse_probs( y2_hat, rna_labels_values )
                      
          if DEBUG>5:
            print ( f"\n\nTRAINLENEJ:     INFO:      test():                                                batch = {BRIGHT_GREEN}{i+1}{RESET}"                        )
            print ( f"TRAINLENEJ:     INFO:      test():                                                count = {BLEU}{(i+1)*batch_size}{RESET}"                       ) 
            print ( f"TRAINLENEJ:     INFO:      test(): p_full_softmax_matrix.shape                          = {BLEU}{p_full_softmax_matrix.shape}{RESET}"            )                                    

          batch_index_lo = i*batch_size
          batch_index_hi = batch_index_lo + batch_size
          
          probabilities_matrix [batch_index_lo:batch_index_hi] = p_full_softmax_matrix # + random.uniform( 0.001, 0.01)                      # 'p_full_softmax_matrix' contains probs for an entire mini-batch; 'probabilities_matrix' has enough room for all cases
          true_classes         [batch_index_lo:batch_index_hi] = rna_labels_values
          rna_case_id          [batch_index_lo:batch_index_hi] = batch_fnames_npy                  [0:batch_size]

          if DEBUG>0:
            print ( f"TRAINLENEJ:     INFO:      test(): probabilities_matrix.shape                           = {BLEU}{probabilities_matrix.shape}{RESET}"  ) 
          if DEBUG>55:
            show_last=16
            np.set_printoptions(formatter={'float': lambda x: "{:>4.2f}".format(x)})       
            print ( f"TRAINLENEJ:     INFO:      test(): last {AMETHYST}{show_last}{RESET} entries in probabilities_matrix[{MIKADO}{batch_index_lo}{RESET}:{MIKADO}{batch_index_hi}{RESET}]     = \n{AMETHYST}{probabilities_matrix [args.n_samples[0]-show_last:args.n_samples[0]]}{RESET}"                       ) 
            np.set_printoptions(formatter={'int': lambda x: "{:^7d}".format(x)})   
            print ( f"TRAINLENEJ:     INFO:      test(): true_classes                       [{MIKADO}{batch_index_lo}{RESET}:{MIKADO}{batch_index_hi}{RESET}] =   {AMETHYST}{true_classes         [batch_index_lo          :batch_index_hi]}{RESET}"        )           
            print ( f"TRAINLENEJ:     INFO:      test(): rna_case_id                        [{MIKADO}{batch_index_lo}{RESET}:{MIKADO}{batch_index_hi}{RESET}] =   {AMETHYST}{rna_case_id          [batch_index_lo          :batch_index_hi]}{RESET}"        )   

         # move to a separate function ----------------------------------------------------------------------------------------------

        




        if (args.input_mode=='image'):
          loss_images       = loss_function(y1_hat, image_labels)
          loss_images_value = loss_images.item()                                                             # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories
 
          if DEBUG>2:
            print ( f"TRAINLENEJ:     INFO:      test(): {COQUELICOT}loss_images{RESET} (for this mini-batch)  = {PURPLE}{loss_images_value:6.3f}{RESET}" )
            time.sleep(.25)
             
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          loss_genes        = loss_function(y2_hat, rna_labels)
          loss_genes_value  = loss_genes.item()                                                              # use .item() to extract value from tensor: don't create multiple new tensors each of which will have gradient histories


        #l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0
        
        if (args.input_mode=='image'):
          total_loss        = loss_images_value + l1_loss
        elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
          total_loss        = loss_genes_value + l1_loss    
        


        if DEBUG>0:

          if ( args.input_mode=='image' ):
            print ( f"\
\033[2K\r\033[27Ctest():\
\r\033[40C{DULL_WHITE}n={i+1:>3d}{CLEAR_LINE}\
\r\033[49Closs_images={loss_images_value:5.2f}\
\r\033[96Cl1_ loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{150+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
            print ( "\033[2A" )
          elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
            print ( f"\
\033[2K\r\033[27Ctest():\
\r\033[40C{DULL_WHITE}n={i+1:>3d}{CLEAR_LINE}\
\r\033[73Closs_rna={loss_genes_value:5.2f}\
\r\033[96Cl1_loss={l1_loss:5.2f}\
\r\033[120CBATCH AVE LOSS      =\r\033[{150+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}{RESET}" )
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


    ### END OF "for i, ( batch_images, batch_genes, image_labels, rna_labels, batch_fnames ) in  enumerate( test_loader ):"
    ### end of one epoch
    
    
    
    

    if epoch % 1 == 0:                                                                                     # every ... epochs, do an analysis of the test results and display same
      
      if args.input_mode=='image':      
        y1_hat_values             = y1_hat.cpu().detach().numpy()
        y1_hat_values_max_indices = np.argmax( np.transpose(y1_hat_values), axis=0 )                       # indices of the highest values of y1_hat = highest probability class

      elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):      
        y2_hat_values             = y2_hat.cpu().detach().numpy()
        y2_hat_values_max_indices = np.argmax( np.transpose(y2_hat_values), axis=0 )                       # indices of the highest values of y2_hat = highest probability class
    
      
      image_labels_values       = image_labels.cpu().detach().numpy()
      rna_labels_values         =   rna_labels.cpu().detach().numpy()

      torch.cuda.empty_cache()
      
      if DEBUG>2:
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
      elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):         
        correct=np.sum( np.equal(y2_hat_values_max_indices, rna_labels_values))
            
      pct=100*correct/batch_size if batch_size>0 else 0
      global_pct = 100*(global_correct_prediction_count+correct) / (global_number_tested+batch_size) 
      if show_all_test_examples==False:
        print ( f"{CLEAR_LINE}                           test(): truth/prediction for first {MIKADO}{number_to_display}{RESET} examples from the most recent test batch \
  ( number correct this batch: {correct}/{batch_size} \
  = {MAGENTA if pct>=90 else PALE_GREEN if pct>=80 else ORANGE if pct>=70 else GOLD if pct>=60 else WHITE if pct>=50 else MAGENTA}{pct:>3.0f}%{RESET} )  \
  ( number correct overall: {global_correct_prediction_count+correct}/{global_number_tested+batch_size} \
  = {MAGENTA if global_pct>=90 else PALE_GREEN if global_pct>=80 else ORANGE if global_pct>=70 else GOLD if global_pct>=60 else WHITE if global_pct>=50 else MAGENTA}{global_pct:>3.0f}%{RESET} {DIM_WHITE}(number tested this run = epochs x test batches x batch size){RESET}" )
      else:
        run_level_total_correct.append( correct )
        print ( f"{CLEAR_LINE}                           test(): truth/prediction for {MIKADO}{number_to_display}{RESET} test examples \
  ( number correct  - all test examples - this run: {correct}/{batch_size} \
  = {MAGENTA if pct>=90 else PALE_GREEN if pct>=80 else ORANGE if pct>=70 else GOLD if pct>=60 else WHITE if pct>=50 else DIM_WHITE}{pct:>3.0f}%{RESET} )  \
  ( number correct  - all test examples - cumulative over all runs: {global_correct_prediction_count+correct}/{global_number_tested}  \
  = {MAGENTA if global_pct>=90 else PALE_GREEN if global_pct>=80 else ORANGE if global_pct>=70 else GOLD if global_pct>=60 else WHITE if global_pct>=50 else MAGENTA}{global_pct:>3.0f}%{RESET} )" )


      if args.input_mode=='image':   
        labs   = image_labels_values       [0:number_to_display] 
        preds  = y1_hat_values_max_indices [0:number_to_display]
        delta  = np.abs(preds - labs)
        np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>1d}{RESET}"})
        print (  f"truth = {labs}", flush=True   )
        print (  f"preds = {preds}", flush=True  )
        np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else DIM_WHITE}{x:>1d}{RESET}"})     
        print (  f"delta = {delta}", flush=True  )
      elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):   
        labs   = rna_labels_values         [0:number_to_display]
        preds  = y2_hat_values_max_indices [0:number_to_display]
        delta  = np.abs(preds - labs)
        np.set_printoptions(formatter={'int': lambda x: f"{DIM_WHITE}{x:>1d}{RESET}"})
        print (  f"truth = {labs}", flush=True   )
        print (  f"preds = {preds}", flush=True  )
        np.set_printoptions(formatter={'int': lambda x: f"{BRIGHT_GREEN if x==0 else DIM_WHITE}{x:>1d}{RESET}"})     
        print (  f"delta = {delta}", flush=True  )



      # ~ if ( args.just_test!='True') | ( (args.just_test=='True')  &  (args.input_mode=='image_rna') & (args.multimode=='image_rna') ):
       # grab test stats produced during training
      for i in range(0, len(preds) ):
        run_level_classifications_matrix[ labs[i], preds[i] ] +=1
      if DEBUG>8:
        print ( run_level_classifications_matrix, flush=True )
        #time.sleep(3)

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


      pplog.log(f"epoch = {epoch}" )
      pplog.log(f"test(): truth/prediction for first {number_to_display} examples from the most recent test batch ( number correct this batch: {correct}/{batch_size} = {pct:>3.0f}%  )  ( number correct overall: {global_correct_prediction_count+correct}/{global_number_tested+batch_size} = {global_pct:>3.0f}% (number tested this run = epochs x test batches x batch size)" )
      pplog.log(f"        truth = {labs}" )
      pplog.log(f"        preds = {preds}")
      pplog.log(f"        delta = {delta}") 
 

    if args.input_mode=='image':   
      y1_hat_values               = y1_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y1_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y1_hat_values_max_indices   = np.argmax( np.transpose(y1_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
    elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
      y2_hat_values               = y2_hat.cpu().detach().numpy()                                          # these are the raw outputs
      del y2_hat                                                                                           # immediately delete tensor to recover large amount of memory
      y2_hat_values_max_indices   = np.argmax( np.transpose(y2_hat_values), axis=0  )                      # these are the predicted classes corresponding to batch_images
      
    
    image_labels_values         = image_labels.cpu().detach().numpy()                                      # these are the true      classes corresponding to batch_images


    if args.input_mode=='image':
      correct_predictions              = np.sum( y1_hat_values_max_indices == image_labels_values )
    elif ( args.input_mode=='rna' ) | ( args.input_mode=='image_rna' ):
      correct_predictions              = np.sum( y2_hat_values_max_indices == rna_labels_values )


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
                                                                         
    writer.add_scalar( '1a_test_loss_ave',       total_loss_ave,          epoch )
    writer.add_scalar( '1b_test_loss_ave_min',   test_loss_min/(i+1),     epoch )    
    writer.add_scalar( '1c_num_correct',         correct_predictions,     epoch )
    writer.add_scalar( '1d_num_correct_max',     max_correct_predictions, epoch )
    writer.add_scalar( '1e_pct_correct',         pct_correct,             epoch ) 
    writer.add_scalar( '1f_max_percent_correct', max_percent_correct,     epoch ) 
  
    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      test():             batch_images.shape                       = {:}".format( batch_images.shape ) )
      print ( "TRAINLENEJ:     INFO:      test():             image_labels.shape                       = {:}".format( image_labels.shape ) )
      
#    if not args.just_test=='True':
#      if args.input_mode=='image':
#        writer.add_figure('Predictions v Truth', plot_classes_preds(args, model, tile_size, batch_images, image_labels, preds, p_highest, p_2nd_highest, p_full_softmax_matrix, class_names, class_colours), epoch)
        
    if args.just_test=='False':                                                                            # This call to plot_classes_preds() is for use by test() during training, and not for use in "just_test" mode (the latter needs support for supergrids)
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
      embedding=0
      
    return loss_images_sum_ave, loss_genes_sum_ave, l1_loss_sum_ave, total_loss_ave, correct_predictions, batch_size, max_correct_predictions, max_percent_correct, test_loss_min, embedding



# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
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
          print( f"TRAINLENET:       INFO:    \033[58Cdirectory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq file(s) and {MIKADO}{image_file_count:<2d}{RESET} image files(s) and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )
          time.sleep(0.5)       
        else:
          print( f"TRAINLENET:       INFO:    directory has {BLEU}{rna_file_count:<2d}{RESET} rna-seq files, {MIKADO}{image_file_count:<2d}{RESET} image files and {MIKADO}{png_file_count:<2d}{RESET} png data files{RESET}", flush=True  )

  if DEBUG>0:
    print( f"TRAINLENET:     INFO:    directories count  =  {MIKADO}{dir_count:<6d}{RESET}",                   flush=True  )
    print( f"TRAINLENET:     INFO:    image file  count  =  {MIKADO}{cumulative_image_file_count:<6d}{RESET}", flush=True  )
    print( f"TRAINLENET:     INFO:    tile  file  count  =  {MIKADO}{cumulative_png_file_count:<6d}{RESET}",   flush=True  )
    print( f"TRAINLENET:     INFO:    rna   file  count  =  {MIKADO}{cumulative_rna_file_count:<6d}{RESET}",   flush=True  )
    print( f"TRAINLENET:     INFO:    other file  count  =  {MIKADO}{cumulative_other_file_count:<6d}{RESET}", flush=True  )


  # (1B) Locate and flag directories that contain BOTH an image and and rna-seq files

  if args.divide_cases=='True':

    if DEBUG>0:
      print ( f"{ORANGE}TRAINLENET:     INFO:  divide_cases  ( {CYAN}-v{RESET}{ORANGE} option ) = {MIKADO}{args.divide_cases}{RESET}{ORANGE}, so will divide cases and set applicable flag files{RESET}",    flush=True )

    dirs_which_have_matched_image_rna_files    = 0
  
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>888:  
        print( f"{DIM_WHITE}TRAINLENET:     INFO:   now processing case (directory) {CYAN}{os.path.basename(dir_path)}{RESET}" )
  
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
            print ( f"{WHITE}TRAINLENET:     INFO:   case {PINK}{args.data_dir}/{os.path.basename(dir_path)}{RESET} \r\033[100C has both matched and rna files (listed above) (count= {MIKADO}{dirs_which_have_matched_image_rna_files+1}{RESET})",  flush=True )
          fqn = f"{dir_path}/HAS_MATCHED_IMAGE_RNA_FLAG"
          with open(fqn, 'w') as f:
            f.write( f"this directory contains matched image and rna-seq data" )
          f.close  
          dirs_which_have_matched_image_rna_files+=1
  
    if DEBUG>0:
      print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  number of cases (directories) which contain BOTH matched and rna files = {MIKADO}{dirs_which_have_matched_image_rna_files}{RESET}",  flush=True )


  
  
  
  
    # (1C) Segment the cases as follows:
    #      (1Ca)  DESIGNATED_MULTIMODE_CASE_FLAG ............... all MATCHED cases, used for multimode testing only. The amount of cases to be so flagged is given by config parameter CASES_RESERVED_FOR_IMAGE_RNA
    #      (1Cb)  DESIGNATED_UNIMODE_CASE_FLAG ................. all MATCHED cases minus designated multimode cases; used for unimode training (generated embeddings are used for multimode training)
    #      (1Cc)  NOT_A_MULTIMODE_CASE_FLAG .................... ALL cases minus multimode cases, and don't have to be matched. Constitute the largest possible set of cases for use in unimode image or rna training and testing (including as a prelude to multimode testing with the designated multimode test set where comparing unimode to multimode performance (which requires the use of the same cases for unimode and multimode) is not of interest
    #      (1Cd ) NOT_A_MULTIMODE_CASE____IMAGE_FLAG ........... ALL cases minus multimode cases which contain an image -     used for unimode training ) constitute the largest possible (but umatched) set of cases for use in unimode image training (including as a prelude to multimode testing with the designated multimode test set, where comparing unimode to multimode performance (the latter requires the use of the SAME cases for unimode and multimode) is not of interest
    #      (1Ce ) NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG ...... ALL cases minus multimode cases which contain an image - reserved for inimode testing  ) same criteria as NOT_A_MULTIMODE_CASE____IMAGE_FLAG, but reserved for testing


    #        - yes it's confusing. sorry!

    if DEBUG>0:
      print ( f"{WHITE}TRAINLENET:     INFO:      segment_cases():  about to segment cases by placing flags according to the following logic:         {CAMEL}DESIGNATED_UNIMODE_CASE_FLAG{RESET}{DULL_WHITE}   XOR {RESET}{ASPARAGUS} DESIGNATED_MULTIMODE_CASE_FLAG{RESET}",  flush=True )
      print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  config parameter '{CYAN}CASES_RESERVED_FOR_IMAGE_RNA{RESET}{DULL_WHITE}' = {MIKADO}{args.cases_reserved_for_image_rna}{RESET}{DULL_WHITE}, therefore {MIKADO}{args.cases_reserved_for_image_rna}{RESET}{DULL_WHITE} cases selected at random will be flagged with the    {ASPARAGUS}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}{DULL_WHITE} thereby exclusively setting them aside for multimode testing",  flush=True )


    # (1Ce) designate MULTIMODE cases.  Infinite loop with a break condition (necessary to give every case an equal chance of being randonly selected for inclusion in the MULTIMODE case set)
    
    directories_considered_count     = 0
    designated_multimode_case_count  = 0
    
    while True:
      
      for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):                                                      # select the multimode cases ...
    
        if DEBUG>55:  
          print( f"{DIM_WHITE}TRAINLENET:     INFO:   now considering case {ARYLIDE}{os.path.basename(dir_path)}{RESET}{DIM_WHITE} as a multimode case  " ) 
    
        
        if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
  
          if DEBUG>55:
            print ( f"{PALE_GREEN}TRAINLENET:     INFO:   case   \r\033[60C{RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[120C has both image and rna files\r\033[140C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
            
          try:
            fqn = f"{dir_path}/HAS_MATCHED_IMAGE_RNA_FLAG"        
            f = open( fqn, 'r' )
            if DEBUG>55:
              print ( f"{PALE_GREEN}TRAINLENET:     INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files (listed above)  \r\033[160C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
              print ( f"{PALE_GREEN}TRAINLENET:     INFO:   designated_multimode_case_count          = {AMETHYST}{designated_multimode_case_count}{RESET}",          flush=True )
              print ( f"{PALE_GREEN}TRAINLENET:     INFO:   dirs_which_have_matched_image_rna_files  = {AMETHYST}{dirs_which_have_matched_image_rna_files}{RESET}",  flush=True )
              print ( f"{PALE_GREEN}TRAINLENET:     INFO:   cases_reserved_for_image_rna             = {AMETHYST}{args.cases_reserved_for_image_rna}{RESET}",        flush=True )
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
                    print ( f"{PALE_GREEN}TRAINLENET:     INFO:      segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[122C has been randomly flagged as '{ASPARAGUS}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}{PALE_GREEN}'  \r\033[204C (count= {MIKADO}{designated_multimode_case_count}{RESET}{PALE_GREEN})",  flush=True )
                except Exception:
                  print( f"{RED}TRAINLENEJ:   FATAL:  could not create '{CYAN}DESIGNATED_MULTIMODE_CASE_FLAG{RESET}' file" )
                  time.sleep(10)
                  sys.exit(0)
  
          except Exception:
            if DEBUG>55:
              print ( f"{RED}TRAINLENEJ:   not a matched case" )
    
      directories_considered_count+=1
      if DEBUG>555:
        print ( f"c={c}" )      

      if designated_multimode_case_count== args.cases_reserved_for_image_rna:
        if DEBUG>55:
          print ( f"{PALE_GREEN}TRAINLENET:     INFO:   designated_multimode_case_count          = {AMETHYST}{designated_multimode_case_count}{RESET}",          flush=True )
          print ( f"{PALE_GREEN}TRAINLENET:     INFO:   cases_reserved_for_image_rna             = {AMETHYST}{args.cases_reserved_for_image_rna}{RESET}",             flush=True )
        break


    # (1Cb) designate UNIMODE cases. Go through all MATCHED directories one time. Flag any MATCHED case other than those flagged as DESIGNATED_MULTIMODE_CASE_FLAG case at 1Ci above with the DESIGNATED_UNIMODE_CASE_FLAG
    
    designated_unimode_case_count    = 0

    for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):                                   # ... designate every matched case (HAS_MATCHED_IMAGE_RNA_FLAG) other than those flagged as DESIGNATED_MULTIMODE_CASE_FLAG above to be a unimode case
  
      if DEBUG>1:  
        print( f"{DIM_WHITE}TRAINLENET:     INFO:   now considering case (directory) as a unimode case {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                    # the top level directory (dataset) is skipped because it only contains sub-directories, not data

        if DEBUG>55:
          print ( f"{PALE_GREEN}TRAINLENET:     INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files (listed above)  \r\033[160C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
  
          
        try:
          fqn = f"{dir_path}/HAS_MATCHED_IMAGE_RNA_FLAG"
          f = open( fqn, 'r' )

          try:
            fqn = f"{dir_path}/DESIGNATED_MULTIMODE_CASE_FLAG"                                             # then we designated it to be a MULTIMODE case above, so ignore 
            f = open( fqn, 'r' )
          except Exception:                                                                               # these are the ones we want
            if DEBUG>555:
              print ( f"{PALE_GREEN}TRAINLENET:     INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[100C has both matched and rna files and has not already been designated as a mutimode case  \r\033[200C (count= {dirs_which_have_matched_image_rna_files}{RESET}{PALE_GREEN})",  flush=True )
              print ( f"{PALE_GREEN}TRAINLENET:     INFO:   designated_unimode_case_count            = {AMETHYST}{designated_unimode_case_count}{RESET}",            flush=True )
            if ( ( designated_unimode_case_count + designated_multimode_case_count ) <= dirs_which_have_matched_image_rna_files ):                 # if we don't yet have enough designated multimode cases (and hence designations in total)
              fqn = f"{dir_path}/DESIGNATED_UNIMODE_CASE_FLAG"            
              with open(fqn, 'w') as f:
                f.write( f"this case is designated as a unimode case" )
              f.close
              designated_unimode_case_count+=1
              if DEBUG>44:
                print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{DULL_YELLOW} \r\033[122C has been randomly designated as a   unimode case  \r\033[204C (count= {MIKADO}{designated_unimode_case_count}{RESET}{DULL_WHITE})",  flush=True )


        except Exception:
          if DEBUG>555:
            print ( "not a multimode case" )
      
      
    # (1Cc) designate the 'NOT MULTIMODE' cases. Go through all directories one time. Flag ANY case (whether matched or not) other than those flagged as DESIGNATED_MULTIMODE_CASE_FLAG case at 1Ci above with the NOT_A_MULTIMODE_CASE_FLAG
    
    if DEBUG>0:
      print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  about to further segment cases by placing flags according to the following logic: {RESET}{ASPARAGUS}DESIGNATED_MULTIMODE_CASE_FLAG {RESET}{DULL_WHITE}XOR{RESET}{PALE_GREEN}  NOT_A_MULTIMODE_CASE_FLAG{RESET}",  flush=True )
    
    not_a_multimode_case_count=0
    for dir_path, dirs, files in os.walk( args.data_dir ):                                                      # each iteration takes us to a new directory under the dataset directory
  
      if DEBUG>55:  
        print( f"{DIM_WHITE}TRAINLENET:       INFO:   now processing case (directory) {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data  

        for f in sorted( files ):          
                    
          try:
            fqn = f"{dir_path}/DESIGNATED_MULTIMODE_CASE_FLAG"        
            f = open( fqn, 'r' )
            if DEBUG>555:
              print ( f"{RED}TRAINLENET:       INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{RED} \r\033[100C is a multimode case. Skipping",  flush=True )
            break
          except Exception:
            try:
              fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE_FLAG"        
              f = open( fqn, 'r' )
              if DEBUG>555:
                print ( f"{RED}TRAINLENET:       INFO:   case                                       {RESET}{AMETHYST}{dir_path}{RESET}{RED} \r\033[100C is in a directory containing the NOT_A_MULTIMODE_CASE_FLAG. Skipping",  flush=True )
              break
            except Exception:
              if DEBUG>44:
                print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{PALE_GREEN} \r\033[122C has been flagged with the  {ASPARAGUS}NOT_A_MULTIMODE_CASE_FLAG{RESET}  \r\033[204C (count= {MIKADO}{not_a_multimode_case_count+1}{RESET})",  flush=True )
              fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE_FLAG"            
              with open(fqn, 'w') as f:
                f.write( f"this case is not a designated multimode case" )
              f.close
              not_a_multimode_case_count+=1                                                                # only segment_cases knows the value of not_a_multimode_case_count, and we need in generate(), so we return it
                                                                  

    # (1Cd) Designate those IMAGE cases which are not also MULTIMODE cases. Go through directories one time. Flag NOT_A_MULTIMODE_CASE_FLAG which are also image cases as NOT_A_MULTIMODE_CASE____IMAGE_FLAG
    
    if DEBUG>3:
      print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  about to designate '{ARYLIDE}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{DULL_WHITE}' cases{RESET}",  flush=True )  
    
    directories_considered_count                    = 0
    designated_not_a_multimode_case____image_count  = 0
    
    for dir_path, dirs, files in os.walk( args.data_dir ):
  
      if DEBUG>55:  
        print( f"{DIM_WHITE}TRAINLENET:       INFO:   now processing case (directory) {ARYLIDE}{os.path.basename(dir_path)}{RESET}" )
  
      if not (dir_path==args.data_dir): 
                    
        try:
          fqn = f"{dir_path}/HAS_IMAGE_FLAG"        
          f = open( fqn, 'r' )
          if DEBUG>44:
            print ( f"{GREEN}TRAINLENET:       INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{GREEN}' \r\033[122C is an image case",  flush=True )
          try:
            fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE_FLAG"        
            f = open( fqn, 'r' )
            if DEBUG>2:
              print ( f"{GREEN}TRAINLENET:       INFO:   case                                       case \r\033[55C'{MAGENTA}{dir_path}{RESET}{GREEN} \r\033[122C is in a directory containing the NOT_A_MULTIMODE_CASE_FLAG",  flush=True )
            fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE____IMAGE_FLAG"            
            with open(fqn, 'w') as f:
              f.write( f"this case is a NOT_A_MULTIMODE_CASE____IMAGE_FLAG case" )
            f.close
            if DEBUG>22:
              print ( f"{PALE_GREEN}TRAINLENET:       INFO:       segment_cases():  case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_GREEN}' \r\033[122C has been flagged with the NOT_A_MULTIMODE_CASE____IMAGE_FLAG  \r\033[204C (count= {MIKADO}{designated_not_a_multimode_case____image_count+1}{RESET})",  flush=True )
            designated_not_a_multimode_case____image_count+=1                                                                # only segment_cases knows the value of not_a_multimode_case_count, and we need in generate(), so we return it
          except Exception:
            if DEBUG>44:
              print ( f"{RED}TRAINLENET:       INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{RED}' \r\033[122C  is not a NOT_A_MULTIMODE_CASE_FLAG case - - skipping{RESET}",  flush=True )
        except Exception:
          if DEBUG>44:
            print ( f"{PALE_RED}TRAINLENET:       INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[122C is not an image case - - skipping{RESET}",  flush=True )                                                                    
        

    # (1Ce) Designate 'NOT MULTIMODE IMAGE TEST' cases. Go through directories one time. Flag PCT_TEST % of the NOT_A_MULTIMODE_CASE_FLAG cases as NOT_A_MULTIMODE_CASE_IMAGE_TEST_FLAG
    #        These cases are used for unimode image testing. Necessary to strictly separated cases in this manner for image mode so that tiles from a single image do not end up in both the training and test sets   
    #        In image mode, tiles allocated to the training set cann't come from an image which is also contributing tiles to the test set. Ditto the reverse.
    #        This issue does not affect rna mode, where there is only one artefact per case. I.e. when input mode is rna, any rna sample can be allocated to either the training set or test set
    #
    #        Strategy: res-designate an appropriate number of the 'NOT_A_MULTIMODE_CASE____IMAGE_FLAG' to be 'NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG' (delete the first flag)
  

    cases_to_designate = int(pct_test * designated_not_a_multimode_case____image_count)
        
    if DEBUG>0:
      print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  about to randomly designate {ARYLIDE}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{DULL_WHITE} cases as reserved image test cases by placing the flag {ARYLIDE}NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG{RESET}{DULL_WHITE} in their case directories",  flush=True )

      print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  cases_to_designate = int({CYAN}PCT_TEST{RESET}{DULL_WHITE} {MIKADO}{pct_test*100:4.2f}%{RESET}{DULL_WHITE} * {CYAN}designated_not_a_multimode_case____image_count{RESET}{DULL_WHITE} {MIKADO}{designated_not_a_multimode_case____image_count}{RESET}{DULL_WHITE}) = {MIKADO}{cases_to_designate}",  flush=True )
    
    directories_considered_count                         = 0
    designated_not_a_multimode_case____image_test_count  = 0
    
    while True:
      
      for dir_path, dirs, files in os.walk( args.data_dir, topdown=True ):
    
        if DEBUG>55:  
          print( f"{DIM_WHITE}TRAINLENET:     INFO:   now considering case {ARYLIDE}{os.path.basename(dir_path)}{RESET}{DIM_WHITE} \r\033[130C as a candidate NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG case  " ) 
            
        if not (dir_path==args.data_dir):                                                                         # the top level directory (dataset) has be skipped because it only contains sub-directories, not data
          try:
            fqn = f"{dir_path}/NOT_A_MULTIMODE_CASE____IMAGE_FLAG"    
            f = open( fqn, 'r' )                
            if DEBUG>66:
              print ( f"{PALE_GREEN}TRAINLENET:       INFO:   case   \r\033[55C'{RESET}{AMETHYST}{dir_path}{RESET}{PALE_GREEN} \r\033[130C is a     {CYAN}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{PALE_GREEN} case{RESET}",  flush=True )
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
                  print ( f"{BLEU}TRAINLENET:       INFO:      segment_cases():  case  {RESET}{CYAN}{dir_path}{RESET}{BLEU} \r\033[130C has been randomly designated as a NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG case  \r\033[204C (count= {MIKADO}{designated_not_a_multimode_case____image_test_count}{BLEU}{RESET})",  flush=True )
              except Exception:
                print( f"{RED}TRAINLENET:   FATAL:  either could not create '{CYAN}NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG{RESET}' file or delete the '{CYAN}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}' " )  
                time.sleep(10)
                sys.exit(0)              
          except Exception:
            if DEBUG>66:
              print ( f"{RED}TRAINLENET:       INFO:   case \r\033[55C'{MAGENTA}{dir_path}{RESET}{PALE_RED} \r\033[130C is not a {CYAN}NOT_A_MULTIMODE_CASE____IMAGE_FLAG{RESET}{RED} case - - skipping{RESET}",  flush=True )
    
      directories_considered_count+=1
     
      if designated_not_a_multimode_case____image_test_count == cases_to_designate:
        if DEBUG>55:
          print ( f"{PALE_GREEN}TRAINLENET:     INFO:   designated_not_a_multimode_case____image_test_count  = {AMETHYST}{designated_not_a_multimode_case____image_test_count}{RESET}",          flush=True )
        break


    designated_not_a_multimode_case____image_count = designated_not_a_multimode_case____image_count - designated_not_a_multimode_case____image_test_count

    if DEBUG>0:
        print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  HAS_MATCHED_IMAGE_RNA_FLAG ................ flags placed = {MIKADO}{dirs_which_have_matched_image_rna_files}{RESET}",              flush=True )
        print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  DESIGNATED_MULTIMODE_CASE_FLAG ............ flags placed = {MIKADO}{designated_multimode_case_count}{RESET}",                      flush=True )
        print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  DESIGNATED_UNIMODE_CASE_FLAG .............. flags placed = {MIKADO}{designated_unimode_case_count}{RESET}",                        flush=True )
        print ( f"{DULL_WHITE}TRAINLENET      INFO:      segment_cases():  NOT_A_MULTIMODE_CASE_FLAG ................. flags placed = {MIKADO}{not_a_multimode_case_count}{RESET}",                           flush=True )
        print ( f"{DULL_WHITE}TRAINLENET      INFO:      segment_cases():  NOT_A_MULTIMODE_CASE____IMAGE_FLAG ........ flags placed = {MIKADO}{designated_not_a_multimode_case____image_count}{RESET}",       flush=True )
        print ( f"{DULL_WHITE}TRAINLENET:     INFO:      segment_cases():  NOT_A_MULTIMODE_CASE____IMAGE_TEST_FLAG ... flags placed = {MIKADO}{designated_not_a_multimode_case____image_test_count}{RESET}",  flush=True )


    
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
      print ( f"TRAINLENEJ:     INFO:      newline():             xmin                                    = {xmin}"                            )
      print ( f"TRAINLENEJ:     INFO:      newline():             xmax                                    = {xmax}"                            )
      print ( f"TRAINLENEJ:     INFO:      newline():             ymin                                    = {ymin}"                            )
      print ( f"TRAINLENEJ:     INFO:      newline():             ymax                                    = {ymax}"                            )

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

# ------------------------------------------------------------------------------
def analyse_probs( y1_hat, image_labels_values ):

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max( y1_hat, axis=1 )

    if DEBUG>99:
      y1_hat_numpy = (y1_hat.cpu().data).numpy()
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds_tensor.shape           = {:}".format( preds_tensor.shape    ) ) 
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds_tensor                 = \n{:}".format( preds_tensor      ) ) 
    
#    preds = np.squeeze( preds_tensor.cpu().numpy() )
    preds = preds_tensor.cpu().numpy()

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
    fig.legend(l, args.class_names, loc='upper right', fontsize=10, facecolor='white') 
  
  # (5) add patch level truth value and prediction 

  threshold_0=36     # total tiles, not width
  threshold_1=100
  threshold_2=400
  threshold_3=900
  threshold_4=30000
            
  t2=f"Cancer type:  {args.cancer_type_long}"
  t3=f"True subtype for the slide:"
  t4=f"{args.class_names[image_labels[idx]]}"
  t5=f"Predicted subtype for this patch:"
  t6=f"{args.class_names[np.argmax(np.sum(p_full_softmax_matrix, axis=0))]}"
  
  if total_tiles >=threshold_4:                    ## NOT OPTIMISED!  Need some more thresholds for values closer to theshold_3
    #          x     y
    ax.text(   0,  -1300, t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -1000, t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 4400, -1000, t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -700, t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 4400, -700, t6, size=10, ha="left",   color="black", style="italic" ) 
  elif threshold_4>total_tiles>=threshold_3:       ## OPTIMISED
    ax.text(   0,  -180, t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -140, t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -140, t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -100,  t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -100,  t6, size=10, ha="left",   color="black", style="italic" )        
  elif threshold_3>total_tiles>=threshold_2:       ## OPTIMISED
    ax.text(   0,  -180, t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -140, t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -140, t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -100,  t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -100,  t6, size=10, ha="left",   color="black", style="italic" )    
  elif threshold_2>total_tiles>=threshold_1:       ## OPTIMISED
    ax.text(   0,  -700, t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -330, t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -330, t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -300, t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 525,  -300, t6, size=10, ha="left",   color="black", style="italic" )    
  elif threshold_1>total_tiles>=threshold_0:      ## OPTIMISED
    ax.text(   0,  -62, t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -51, t3, size=10, ha="left",   color="black", style="normal" )
    ax.text( 175,  -51, t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -40, t5, size=10, ha="left",   color="black", style="normal" )
    ax.text( 175,  -40, t6, size=10, ha="left",   color="black", style="italic" )                   
  else: # (< threshold0)                          ## OPTIMISED
    ax.text(   0,  -32, t2, size=10, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
    ax.text(   0,  -25, t3, size=10, ha="left",   color="black", style="normal" )
    ax.text(  90,  -25, t4, size=10, ha="left",   color="black", style="italic" )
    ax.text(   0,  -18, t5, size=10, ha="left",  color="black", style="normal" )
    ax.text(  90,  -18, t6, size=10, ha="left",   color="black", style="italic" )    

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
          marker_size =int(80000/pixel_width)-2  # i.e. 13-1=12
        elif threshold_4<=nrows<threshold_5:   # seems ok
          marker_size =int(60000/pixel_width)
        else:
          marker_size = 1
        
        if DEBUG>8:
          print ( f"TRAINLENEJ:     INFO:      plot_scatter()  nrows       = {MIKADO}{nrows}{RESET}" )
          print ( f"TRAINLENEJ:     INFO:      plot_scatter()  marker_size = {MIKADO}{marker_size}{RESET}" )
          
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
  
  scattergram_name = [ "2 scattergram on tiles" if show_patch_images=='True' else "9 scattergram " ][0]
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

    if DEBUG>99:
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
  
      for idx in np.arange( number_to_plot ):
  
          fq_link  = f"{args.data_dir}/{batch_fnames[idx]}.fqln"
          fq_name  = os.readlink     ( fq_link )
          dir_name = os.path.dirname ( fq_name )
          
          if DEBUG>2:
            print ( f"TRAINLENEJ:     INFO:      test():       file fq_link points to      = {MAGENTA}{fq_link}{RESET}"    )
            print ( f"TRAINLENEJ:     INFO:      test():       fq_link                     = {MAGENTA}{fq_name}{RESET}"                 )
            print ( f"TRAINLENEJ:     INFO:      test():       dir_name                    = {MAGENTA}{dir_name}{RESET}"                )
            
                  
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
        print ( f"TRAINLENEJ:     INFO:        plot_classes_preds():  {ORANGE if args.just_test=='True' else MIKADO} about to set up {MIKADO}{figure_width}x{figure_height} inch{RESET} figure and axes for {MIKADO}{nrows}x{ncols}={number_to_plot}{RESET} subplots. (Note: This takes a long time for larger values of nrows/ncols)", end="", flush=True )
            
      fig, axes = plt.subplots( nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=True, figsize=( figure_width, figure_height ) )        # This takes a long time to execute for larger values of nrows and ncols
    
      if DEBUG>0:
        print ( f"  ... done", flush=True )
      

      # (2b) add the legend 
      
      l=[]
      for n in range (0, len(class_colours)):
        l.append(mpatches.Patch(color=class_colours[n], linewidth=0))
        fig.legend(l, args.class_names, loc='upper right', fontsize=14, facecolor='lightgrey')      
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
      ax0.bar( x=[range(str(range(len(args.class_names))))], height=np.sum(p_full_softmax_matrix,axis=0),  width=int(number_to_plot/len(image_labels)), color=class_colours )
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
              t4=f"{args.class_names[image_labels[idx]]}"
              t5=f"NN prediction from patch:"
              t6=f"{args.class_names[np.argmax(np.sum( p_full_softmax_matrix, axis=0)) ]}"
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

def save_model( log_dir, model ):
    """Save PyTorch model state dictionary
    """

    if args.input_mode == 'image':
      if args.pretrain=='True':
        try:
          fqn = f"{log_dir}/model_pretrained.pt"   # try and open it
          f = open( fqn, 'r' )
          if DEBUG>2:
            print( f"\r{COTTON_CANDY}TRAINLENEJ:     INFO:  pre-train option has been selected but a pre-trained model already exists. Saving state model dictionary as {fqn}{RESET}", end='', flush=True )
          f.close()
        except Exception as e:
          fqn = f"{log_dir}/model_pretrained.pt"
          print( f"{COTTON_CANDY}<< saving to: {fqn}{RESET} ", end='', flush=True )
      else:
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
def box_plot_by_subtype( args, writer, total_runs_in_job, pct_test, pandas_matrix ):
  
  # (1) Just some stats
  flattened              =  np.sum  ( pandas_matrix, axis=0 )                                                                          # sum across all examples to produce a 2D matrix
  
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    flattened.shape     = {CARRIBEAN_GREEN}{flattened.shape}{RESET}')
  total_examples_by_subtype     =  np.expand_dims(np.sum  (  flattened, axis=0 ), axis=0 )                                             # sum down the columns to produces a row vector
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    total_examples_by_subtype.shape  = {CARRIBEAN_GREEN}{total_examples_by_subtype.shape}{RESET}')
  if DEBUG>9:    
    print( f'TRAINLENEJ:       INFO:    total_examples_by_subtype        = {CARRIBEAN_GREEN}{total_examples_by_subtype}{RESET}') 
    
  if DEBUG>9:
    printloss_genes_value( f'TRAINLENEJ:       INFO:    flattened.shape     = {CARRIBEAN_GREEN}{flattened.shape}{RESET}')
  total_correct_by_subtype      =  np.array( [ flattened[i,i] for i in  range( 0 , len( flattened ))  ] )                              # pick out diagonal elements (= number correct) to produce a row vector
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    total_correct_by_subtype.shape   = {CARRIBEAN_GREEN}{total_correct_by_subtype.shape}{RESET}')
  if DEBUG>9:
    print( f'TRAINLENEJ:       INFO:    total_correct_by_subtype         = {CARRIBEAN_GREEN}{total_correct_by_subtype}{RESET}')                                
  
  
  # (2) process and present box plot

  total_values_plane            =   np.sum(  pandas_matrix, axis=1 )[ 0:total_runs_in_job, : ]                                         # sum elements (= numbers correct) from 3D volume down columns (axis 1)  to produce a matrix
  if DEBUG>8:
    print( f'\nTRAINLENEJ:       INFO:    total_values_plane.shape         = {CARRIBEAN_GREEN}{total_values_plane.shape}{RESET}')
  if DEBUG>8:
    np.set_printoptions(formatter={ 'int' : lambda x: f"   {CARRIBEAN_GREEN}{x:>6d}   "} )    
    print( f'TRAINLENEJ:       INFO:    total_values_plane               = \n{CARRIBEAN_GREEN}{total_values_plane}{RESET}')

  correct_values_plane          =   np.transpose( np.array( [ pandas_matrix[:,i,i] for i in  range( 0 , pandas_matrix.shape[1] ) ]  )  ) [ 0:total_runs_in_job, : ]      # pick out diagonal elements (= numbers correct) from 3D volume  to produce a matrix
  if DEBUG>8:
    print( f'TRAINLENEJ:       INFO:    correct_values_plane.shape       = {CARRIBEAN_GREEN}{correct_values_plane.shape}{RESET}')
  if DEBUG>8:
    np.set_printoptions(formatter={ 'int' : lambda x: f"   {CARRIBEAN_GREEN}{x:>6d}   "} )          
    print( f'TRAINLENEJ:       INFO:    correct_values_plane             = \n{CARRIBEAN_GREEN}{correct_values_plane}{RESET}')

  
  np.seterr( invalid='ignore', divide='ignore' )          
  percentage_correct_plane         =   100 * np.divide( correct_values_plane, total_values_plane )
  percentage_correct_plane_NO_NANS =   percentage_correct_plane[ ~np.isnan(percentage_correct_plane).any(axis=1) ]                     # remove rows with NaNs because the seaborn boxplot can't handle these
  if DEBUG>8:
    print( f'TRAINLENEJ:       INFO:    percentage_correct_plane.shape   = {CARRIBEAN_GREEN}{percentage_correct_plane.shape}{RESET}')
  if DEBUG>8:
    np.set_printoptions(formatter={'float': lambda x: f"   {CARRIBEAN_GREEN}{x:>6.2f}   "} )    
    print( f'TRAINLENEJ:       INFO:    percentage_correct_plane         = \n{CARRIBEAN_GREEN}{percentage_correct_plane}{RESET}')
    print( f'TRAINLENEJ:       INFO:    percentage_correct_plane_NO_NANS         = \n{CARRIBEAN_GREEN}{percentage_correct_plane_NO_NANS}{RESET}')
  np.seterr(divide='warn', invalid='warn') 
  
  
  npy_class_names = np.transpose(np.expand_dims( np.array(args.class_names), axis=0 ) )
  if DEBUG>0:
    print( f'TRAINLENEJ:       INFO:    npy_class_names.shape   = {CARRIBEAN_GREEN}{npy_class_names.shape}{RESET}')
    print( f'TRAINLENEJ:       INFO:    npy_class_names         = \n{CARRIBEAN_GREEN}{npy_class_names}{RESET}')
  
  pd_percentage_correct_plane =   pd.DataFrame( (percentage_correct_plane_NO_NANS), columns=npy_class_names )                 

  
  figure_width  = 41
  figure_height = 16 
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  ax.set_title ( args.cancer_type_long )
  plt.xticks(rotation=90)
  #sns.set_theme(style="whitegrid")
  ax = sns.boxplot( data=pd_percentage_correct_plane, orient='v', showfliers=False )
  #ax.set(ylim=(0, 100))
  #plt.show()
  writer.add_figure('Box Plot V', fig, 1)
  
  # save portrait version of box plot to logs directory
  now              = datetime.datetime.now()
  

  fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__box_plot_portrait.png"
  fig.savefig(fqn)
  
  figure_width  = 16
  figure_height = 4 
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  ax.set_title ( f"{args.cancer_type_long}_{args.mapping_file_name}_dataset")
  plt.xticks(rotation=0)
  #sns.set_theme(style="whitegrid")   
  ax = sns.boxplot( data=pd_percentage_correct_plane, orient='h', showfliers=False )
  ax.set(xlim=(0, 100))
  #plt.show()
  #writer.add_figure('Box Plot H', fig, 1)  # the landscape version doesn't work well in Tensorboard because it's short and wide
  
  # save landscape version of box  figure_width  = 4
  figure_height = 16 
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  ax.set_title ( args.cancer_type_long )
  plt.xticks(rotation=90)
  #sns.set_theme(style="whitegrid")
  ax = sns.boxplot( data=pd_percentage_correct_plane, orient='v', showfliers=False )
  #ax.set(ylim=(0, 100))
  #plt.show()
  writer.add_figure('Box Plot V', fig, 1)
  
  # save portrait version of box plot to logs directory
  now              = datetime.datetime.now()
  

  fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__box_plot_portrait.png"
  fig.savefig(fqn)
  
  figure_width  = 16
  figure_height = 4 
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  ax.set_title ( f"{args.cancer_type_long}_{args.mapping_file_name}_dataset")
  plt.xticks(rotation=0)
  #sns.set_theme(style="whitegrid")   
  ax = sns.boxplot( data=pd_percentage_correct_plane, orient='h', showfliers=False )
  ax.set(xlim=(0, 100))
  #plt.show()
  #writer.add_figure('Box Plot H', fig, 1)  # the landscape version doesn't work well in Tensorboard because it's short and wide
  
  # save landscape version of box plot to logs directory
  fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__box_plot_landscape.png"
  fig.savefig(fqn)
  
  plt.close('ALL_ELIGIBLE_CASES')
  fqn = f"{args.log_dir}/{now:%y%m%d%H}_{file_name_prefix}__box_plot_landscape.png"
  fig.savefig(fqn)
  
  plt.close('ALL_ELIGIBLE_CASES')
    
  return

# --------------------------------------------------------------------------------------------  
def show_classifications_matrix( writer, total_runs_in_job, pct_test, epoch, pandas_matrix, level ):

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

                                                                                               
  pandas_version_ext = pd.DataFrame( ext3_pandas_matrix, columns=args.class_names, index=index_names )     # this version has subtotals etc at the bottom so it's just for display
  print(tabulate( pandas_version_ext, headers='keys', tablefmt = 'fancy_grid' ) )   
  
  #display(pandas_version_ext)mapping_file
 
 
  # (1) Save job level classification matrix as a csv file in logs directory

  if level=='job':

    now              = datetime.datetime.now()
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

  mapping_file
  
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
    p = argparse.ArgumentParser()

    p.add_argument('--skip_tiling',                                                   type=str,   default='False'                            )                                
    p.add_argument('--skip_generation',                                               type=str,   default='False'                            )                                
    p.add_argument('--pretrain',                                                      type=str,   default='False'                            )                                
    p.add_argument('--log_dir',                                                       type=str,   default='data/dlbcl_image/logs'            )                
    p.add_argument('--base_dir',                                                      type=str,   default='/home/peter/git/pipeline'         )
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
    p.add_argument('--n_samples',                                         nargs="+",  type=int,    default="101"                             )                                    
    p.add_argument('--n_tiles',                                           nargs="+",  type=int,    default="50"                              )       
    p.add_argument('--n_tests',                                                       type=int,    default="16"                              )       
    p.add_argument('--highest_class_number',                              nargs="+",  type=int,    default="989"                             )                                                             
    p.add_argument('--supergrid_size',                                                type=int,    default=1                                 )                                      
    p.add_argument('--patch_points_to_sample',                                        type=int,    default=1000                              )                                   
    p.add_argument('--tile_size',                                         nargs="+",  type=int,    default=128                               )                                    
    p.add_argument('--gene_data_norm',                                    nargs="+",  type=str,    default='NONE'                            )                                 
    p.add_argument('--gene_data_transform',                               nargs="+",  type=str,    default='NONE'                            )
    p.add_argument('--n_genes',                                                       type=int,    default=506                               )                                   
    p.add_argument('--remove_unexpressed_genes',                                      type=str,    default='True'                            )                               
    p.add_argument('--remove_low_expression_genes',                                   type=str,   default='True'                             )                                
    p.add_argument('--low_expression_threshold',                                      type=float, default=0                                  )                                
    p.add_argument('--batch_size',                                        nargs="+",  type=int,   default=64                                 )                                     
    p.add_argument('--learning_rate',                                     nargs="+",  type=float, default=.00082                             )                                 
    p.add_argument('--n_epochs',                                                      type=int,   default=10                                 )
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
    p.add_argument('--use_tiler',                                                     type=str,   default='external'                         )                 
    p.add_argument('--cancer_type',                                                   type=str,   default='NONE'                             )                 
    p.add_argument('--cancer_type_long',                                              type=str,   default='NONE'                             )                 
    p.add_argument('--class_names',                                       nargs="*",  type=str,   default='NONE'                             )                 
    p.add_argument('--long_class_names',                                  nargs="+",  type=str,   default='NONE'                             ) 
    p.add_argument('--class_colours',                                     nargs="*"                                                          )                 
    p.add_argument('--colour_map',                                                    type=str,   default='tab10'                            )    
    p.add_argument('--target_tile_coords',                                nargs=2,    type=int,   default=[2000,2000]                        )                 
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
    p.add_argument('-ddp', '--ddp',                                                   type=str,   default='False'                            )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    p.add_argument('-n', '--nodes',                                                   type=int,   default=1,  metavar='N'                    )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    p.add_argument('-g', '--gpus',                                                    type=int,   default=1,  help='number of gpus per node' )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    p.add_argument('-nr', '--nr',                                                     type=int,   default=0,  help='ranking within node'     )  # only supported for 'NN_MODE=pre_compress' ATM (the auto-encoder front-end)
    
    p.add_argument('--hidden_layer_neurons',                              nargs="+",  type=int,    default=2000                              )     
    p.add_argument('--gene_embed_dim',                                    nargs="+",  type=int,    default=1000                              )    
    
    p.add_argument('--use_autoencoder_output',                                        type=str,   default='True'                             ) # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values)
    p.add_argument('--clustering',                                                    type=str,   default='NONE'                             )
    p.add_argument('--metric',                                                        type=str,   default="manhattan"                        )        
    p.add_argument('--perplexity',                                                    type=int,   default=30                                 )        
    p.add_argument('--momentum',                                                      type=float, default=0.8                                )        
        
    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 12
    args.pin_memory = torch.cuda.is_available()

    if DEBUG>99:
      print ( f"{GOLD}args.multimode{RESET} =           ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------>    {YELLOW}{args.multimode}{RESET}")
    
    main(args)
