"""============================================================================= 
Train LENET5
============================================================================="""

import sys
import math
import time
import cuda
import pprint
import argparse
import numpy as np
import torch
from tiler_scheduler import *
from tiler_threader import *
from tiler_set_target import *
from tiler import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import cm
#from matplotlib import figure
#from pytorch_memlab import profile

from   data                            import loader
from   data.dlbcl_image.config         import GTExV6Config
from   data.dlbcl_image.generate_image import generate_image
from   models                          import LENETIMAGE
from   torch                           import optim
from   torch.nn.utils                  import clip_grad_norm_
from   torch.nn                        import functional as F
from   torch.nn                        import DataParallel
from   itertools                       import product, permutations
from   PIL                             import Image

import torchvision
import torch.utils.data
from   torch.utils.tensorboard import SummaryWriter
from   torchvision    import datasets, transforms

DEBUG=1
last_stain_norm='NULL'
last_gene_norm='NULL'

np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=300)

#torch.backends.cudnn.benchmark   = False                                                                      #for CUDA memory optimizations
torch.backends.cudnn.enabled     = True                                                                     #for CUDA memory optimizations


# ------------------------------------------------------------------------------

DIM_WHITE='\033[37;2m'
CYAN='\033[36;1m'
RED='\033[31;1m'
PALE_RED='\033[31m'
ORANGE='\033[38;5;136m'
PALE_ORANGE='\033[38;5;172m'
GREEN='\033[32;1m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

device = cuda.device()

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)

global global_batch_count

global_batch_count=0

# ------------------------------------------------------------------------------

#@profile
def main(args):

  """Main program: train -> test once per epoch
  """
  
  os.system("taskset -p 0xffffffff %d" % os.getpid())
  
  global last_stain_norm                                                                                   # Need to remember this across runs
  global last_gene_norm                                                                                    # Need to remember this across runs
  
  print ( "\ntorch       version =      {:}".format (  torch.__version__       )  )
  print ( "torchvision version =      {:}".format (  torchvision.__version__ )  ) 
  
  now = time.localtime(time.time())
  print(time.strftime("TRAINLENEJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
  start_time = time.time()

  pprint.set_logfiles( args.log_dir )
  
  print( "TRAINLENEJ:     INFO: passed in arguments (may yet be over-ridden) are:\
 dataset=\033[36;1m{:}\033[m,\
 mode=\033[36;1m{:}\033[m,\
 use_tiler=\033[36;1m{:}\033[m,\
 nn=\033[36;1m{:}\033[m,\
 nn_optimizer=\033[36;1m{:}\033[m,\
 batch_size=\033[36;1m{:}\033[m,\
 learning_rate=\033[36;1m{:}\033[m,\
 epochs=\033[36;1m{:}\033[m,\
 samples=\033[36;1m{:}\033[m,\
 n_genes=\033[36;1m{:}\033[m,\
 gene_norm=\033[36;1m{:}\033[m,\
 n_tiles=\033[36;1m{:}\033[m,\
 rand_tiles=\033[36;1m{:}\033[m,\
 greyness<\033[36;1m{:}\033[m,\
 sd<\033[36;1m{:}\033[m,\
 min_uniques>\033[36;1m{:}\033[m,\
 latent_dim=\033[36;1m{:}\033[m,\
 label_swap=\033[36;1m{:}\033[m,\
 make_grey=\033[36;1m{:}\033[m,\
 stain_norm=\033[36;1m{:}\033[m,\
 tensorboard_images=\033[36;1m{:}\033[m,\
 max_consec_losses=\033[36;1m{:}\033[m"\
.format( args.dataset, args.input_mode, args.use_tiler, args.nn_type, args.optimizer, args.batch_size, args.learning_rate, args.n_epochs, args.n_samples, args.n_genes,  args.gene_data_norm, args.n_tiles, args.rand_tiles, args.greyness, \
args.min_tile_sd, args.min_uniques, args.latent_dim, args.label_swap_perunit, args.make_grey_perunit, args.stain_norm, args.tensorboard_images, args.max_consecutive_losses  ), flush=True )
  skip_preprocessing     = args.skip_preprocessing
  skip_generation        = args.skip_generation
  dataset                = args.dataset
  class_names            = args.class_names
  cancer_type            = args.cancer_type
  cancer_type_long       = args.cancer_type_long    
  long_class_names       = args.long_class_names  
  class_colours          = args.class_colours
  input_mode             = args.input_mode
  use_tiler              = args.use_tiler
  nn_type                = args.nn_type
  nn_optimizer           = args.optimizer
  n_samples              = args.n_samples
  n_tiles                = args.n_tiles
  batch_size             = args.batch_size
  lr                     = args.learning_rate
  rand_tiles             = args.rand_tiles
  n_genes                = args.n_genes
  gene_data_norm         = args.gene_data_norm  
  n_epochs               = args.n_epochs
  greyness               = args.greyness
  min_tile_sd            = args.min_tile_sd
  min_uniques            = args.min_uniques
  label_swap_perunit     = args.label_swap_perunit
  make_grey_perunit      = args.make_grey_perunit
  stain_norm             = args.stain_norm
  stain_norm_target      = args.stain_norm_target
  tensorboard_images     = args.tensorboard_images
  max_consecutive_losses = args.max_consecutive_losses
  target_tile_coords     = args.target_tile_coords
  
  base_dir               = args.base_dir
  data_dir               = args.data_dir
  log_dir                = args.log_dir
  tile_size              = args.tile_size
  rna_file_name          = args.rna_file_name
  class_numpy_file_name  = args.class_numpy_file_name
  regenerate             = args.regenerate
  just_profile           = args.just_profile
  just_test              = args.just_test
  save_model_name        = args.save_model_name
  save_model_every       = args.save_model_every
  supergrid_size         = args.supergrid_size
  
  if supergrid_size<1:
    print( f"{RED}TRAINLENEJ:     FATAL:  paramater 'supergrid_size' (current value {supergrid_size}) must be an integer greater than zero ... halting now{RESET}" )
    sys.exit(0)
  
  n_samples_max=np.max(n_samples)
  n_tiles_max=np.max  (n_tiles)
  n_tiles_last=0                                                                                           # used to trigger regeneration of tiles if a run requires more tiles that the preceeding run 
  n_samples_last=0
  tile_size_last=0                                                                                         # used to trigger regeneration of tiles if a run requires more tiles that the preceeding run 
  n_classes=len(class_names)
  
  
  if just_test=='True':
    print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'  flag is set. No training will be performed{RESET}" )
#    print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'just_test'  flag is set -- n_epochs (currently {n_epochs}) will be changed to 1 for this job{RESET}" )
#    n_epochs=1
       
  if rand_tiles=='False':
    print( f"{ORANGE}TRAINLENEJ:     INFO:  CAUTION! 'rand_tiles' flag is not set. Tiles will be selected sequentially rather than at random{RESET}" )     

  if (DEBUG>0):
    print ( f"TRAINLENEJ:     INFO:  n_classes   = {CYAN}{n_classes}{RESET}",                 flush=True)
    print ( f"TRAINLENEJ:     INFO:  class_names = {CYAN}{class_names}{RESET}",               flush=True)


  
  # (A)  SET UP JOB LOOP

  already_tiled=False
                          
  parameters = dict( 
                                 lr  =   lr,
                          n_samples  =   n_samples,
                         batch_size  =   batch_size,
                            n_tiles  =   n_tiles,
                          tile_size  =   tile_size,
                         rand_tiles  =  [ rand_tiles ],
                            nn_type  =   nn_type,
                        nn_optimizer =  nn_optimizer,
                          stain_norm =  stain_norm,
                      gene_data_norm =  gene_data_norm,                          
                  label_swap_perunit = [   0.0   ],
                   make_grey_perunit = [   0.0   ],
                              jitter = [  [ 0.0, 0.0, 0.0, 0.0 ] ]  )

  param_values = [v for v in parameters.values()]

  if DEBUG>0:
    print("\033[0Clr\r\033[14Cn_samples\r\033[26Cbatch_size\r\033[38Cn_tiles\r\033[51Ctile_size\r\033[61Crand_tiles\r\033[71Cnn_type\r\033[81Coptimizer\r\033[91Cstain_norm\r\033[103Cgene_norm\
\r\033[113Clabel_swap\r\033[124Cgreyscale\r\033[134Cjitter vector\033[m")
    for       lr,      n_samples,        batch_size,                 n_tiles,         tile_size,        rand_tiles,         nn_type,       nn_optimizer,         stain_norm,          gene_data_norm,\
          label_swap_perunit, make_grey_perunit,   jitter in product(*param_values):
      print( f"\033[0C{CYAN}{lr:9.6f} \r\033[14C{n_samples:<5d} \r\033[26C{batch_size:<5d} \r\033[38C{n_tiles:<5d} \r\033[51C{tile_size:<3d} \r\033[61C{rand_tiles:<5s} \r\033[71C{nn_type:<8s} \r\033[81C{nn_optimizer:<8s}\
\r\033[91C{stain_norm:<10s} \r\033[103C{gene_data_norm:<10s} \r\033[113C{label_swap_perunit:<6.1f} \r\033[124C{make_grey_perunit:<5.1f}  \r\033[134C{jitter:}{RESET}" )      

  # ~ for lr, batch_size  in product(*param_values): 
      # ~ comment = f' batch_size={batch_size} lr={lr}'

  if just_test=='True':
    #if not (n_tiles%batch_size==0):
    #  print( f"\033[31;1mTRAINLENEJ:     FATAL:  in test mode 'tiles per image' must be an integral multiple of 'batch size'. Halting.\033[m" )
    #  sys.exit(0)
    if not ( batch_size == int( math.sqrt(batch_size) + 0.5) ** 2 ):
      print( f"\033[31;1mTRAINLENEJ:     FATAL:  in test mode 'batch_size' (currently {batch_size}) must be a perfect square (4, 19, 16, 25 ...) to permit selection of a a 2D contiguous patch. Halting.\033[m" )
      sys.exit(0)
    if not ( n_tiles == batch_size*supergrid_size**2 ):
      print( f"{RED}TRAINLENEJ:     FATAL:  n_tiles={n_tiles}; however for supergrid_size={supergrid_size}{RESET}{RED}, the number of tiles must be a {supergrid_size**2}x the batch_size ({batch_size}), (i.e. n_tiles={batch_size*supergrid_size**2}){RESET}" )
      sys.exit(0)      

  run=0


  # (B) RUN JOB LOOP
  for lr, n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type, nn_optimizer, stain_norm, gene_data_norm, label_swap_perunit, make_grey_perunit, jitter in product(*param_values): 

    if DEBUG>0:
      print("TRAINLENEJ:     INFO: job level parameters:  \nlr\r\033[10Cn_samples\r\033[26Cbatch_size\r\033[38Cn_tiles\r\033[51Ctile_size\r\033[61Crand_tiles\r\033[71Cnn_type\r\033[81Coptimizer\r\033[91Cstain_norm\
\r\033[103Cgene_norm\r\033[113Clabel_swap\r\033[124Cgreyscale\r\033[134Cjitter vector\033[36;1m\n{:}\033[m".format( param_values ) )
    
    run+=1

    if DEBUG>0:
      print( "\n\033[1;4mRUN  {:}\033[m          learning rate=\033[36;1;4m{:}\033[m  n_samples=\033[36;1;4m{:}\033[m  batch size=\033[36;1;4m{:}\033[m    n_tiles=\033[36;1;4m{:}\033[m   tile_size=\033[36;1;4m{:}\033[m \
rand_tiles=\033[36;1;4m{:}\033[m  nn_type=\033[36;1;4m{:}\033[m nn_optimizer=\033[36;1;4m{:}\033[m stain_norm=\033[36;1;4m{:}\033[m gene_data_norm=\033[36;1;4m{:}\033[m label swaps=\033[36;1;4m{:}\033[m\
make grey=\033[36;1;4m{:}\033[m, jitter=\033[36;1;4m{:}\033[m"\
.format( run, lr,  n_samples, batch_size, n_tiles, tile_size, rand_tiles, nn_type, nn_optimizer, stain_norm, gene_data_norm, label_swap_perunit, make_grey_perunit, jitter) )

    #(1) set up Tensorboard
    
    print( "TRAINLENEJ:       INFO: \033[1m1 about to set up Tensorboard\033[m" )
    
    if input_mode=='image':
      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type}; opt={nn_optimizer}; n_samps={n_samples}; n_t={n_tiles}; t_sz={tile_size}; rnd={rand_tiles}; tot_tiles={n_tiles * n_samples}; n_epochs={n_epochs}; bat={batch_size}; stain={stain_norm};  uniques>{min_uniques}; grey>{greyness}; sd<{min_tile_sd}; lr={lr}; lbl_swp={label_swap_perunit*100}%; greyscale={make_grey_perunit*100}% jit={jitter}%' )
    elif input_mode=='rna':
      writer = SummaryWriter(comment=f' {dataset}; mode={input_mode}; NN={nn_type}; opt={nn_optimizer}; n_samps={n_samples}; n_genes={n_genes}; gene_norm={gene_data_norm}; n_epochs={n_epochs}; batch={batch_size}; lr={lr}')
    else:
      print( "TRAINLENEJ:     FATAL:    input of type '{:}' is not supported".format( nn_type ) )
      sys.exit(0)

    print( "TRAINLENEJ:       INFO:   \033[3mTensorboard has been set up\033[m" ) 
    
    
    # (2) potentially schedule and run tiler threads
    
    if input_mode=='image':
      if skip_preprocessing=='False':
        if use_tiler=='internal':
          # need to re-tile if certain parameters have eiher INCREASED ('n_tiles' or 'n_samples') or simply CHANGED ( 'stain_norm' or 'tile_size') since the last run
          if ( ( already_tiled==True ) & ( ( stain_norm==last_stain_norm ) | (last_stain_norm=="NULL") ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) ):
            pass          # no need to re-tile                                                              
          else:           # must re-tile
            if DEBUG>0:
              print( f"TRAINLENEJ:       INFO: {BOLD}2 about to launch tiling processes{RESET}" )
              print( f"TRAINLENEJ:       INFO:   about to delete all existing tiles from {CYAN}{data_dir}{RESET}")
              print( f"TRAINLENEJ:       INFO:   stain normalization method = {CYAN}{stain_norm}{RESET}" )
            delete_selected( data_dir, "png" )
            last_stain_norm=stain_norm
            already_tiled=True
  
            if DEBUG>999:
              print( f"TRAINLENEJ:       INFO:   n_samples_max                   = {CYAN}{n_samples_max}{RESET}")
              print( f"TRAINLENEJ:       INFO:   n_tiles_max                     = {CYAN}{n_tiles_max}{RESET}")
    
            if stain_norm=="NONE":                                                                         # we are NOT going to stain normalize ...
              norm_method='NONE'
            else:                                                                                          # we are going to stain normalize ...
              if DEBUG>0:
                print( f"TRAINLENEJ:       INFO: {BOLD}2 about to set up stain normalization target{RESET}" )
              if stain_norm_target.endswith(".svs"):                                                       # ... then grab the user provided target
                norm_method = tiler_set_target( args, stain_norm, stain_norm_target, writer )
              else:                                                                                        # ... and there MUST be a target
                print( f"TRAINLENEJ:     FATAL:    for {CYAN}{stain_norm}{RESET} an SVS file must be provided from which the stain normalization target will be extracted" )
                sys.exit(0)
    
            if DEBUG>99:
              print( f"TRAINLENEJ:       INFO: about to call tile threader with n_samples_max={CYAN}{n_samples_max}{RESET}; n_tiles_max={CYAN}{n_tiles_max}{RESET}  " )         
            result = tiler_threader( args, n_samples_max, n_tiles_max, tile_size, batch_size, stain_norm, norm_method )               # we tile the largest number of samples that is required for any run within the job
            
            if just_profile=='True':
              sys.exit(0)


    # (3) Regenerate Torch '.pt' file, if required

    if skip_preprocessing=='False':
      
      if input_mode=='image':
        
        if ( ( already_tiled==True ) & (n_tiles<=n_tiles_last ) & ( n_samples<=n_samples_last ) & ( tile_size_last==tile_size ) & ( stain_norm==last_stain_norm ) ):    # all three have to be true, or else we must regenerate the .pt file
          pass
        else:
          if global_batch_count==0:
            print( f"TRAINLENEJ:     INFO: \033[1m3  will generate torch '.pt' file from files{RESET}" )
          else:
            print( f"TRAINLENEJ:     INFO: \033[1m3  will regenerate torch '.pt' file from files, for the following reason(s):{RESET}" )            
            if n_tiles>n_tiles_last:
              print( f"                                    -- value of n_tiles   {CYAN}({n_tiles})        \r\033[60Chas increased since last run{RESET}" )
            if n_samples>n_samples_last:
              print( f"                                    -- value of n_samples {CYAN}({n_samples_last}) \r\033[60Chas increased since last run{RESET}")
            if not tile_size_last==tile_size:
              print( f"                                    -- value of tile_size {CYAN}({tile_size})      \r\033[60Chas changed   since last run{RESET}")
                        
          generate_image( args, n_samples, n_tiles, tile_size, n_genes, "NULL" )


        n_tiles_last   = n_tiles                                                                           # for the next run
        n_samples_last = n_samples                                                                         # for the next run
        tile_size_last = tile_size                                                                         # for the next run

      
      elif input_mode=='rna':
        if ( not ( gene_data_norm==last_gene_norm ) & (last_gene_norm=="NULL") ):
          if DEBUG>0:      
            print( f"\nTRAINLENEJ:     INFO: \033[1m3 about to regenerate torch '.pt' file from gene data normalization = {CYAN}{gene_data_norm}{RESET}" )
          generate_image( args, n_samples, n_tiles, n_genes, gene_data_norm )
          last_gene_norm=gene_data_norm
        else:
          if DEBUG>0:      
            print( f"\nTRAINLENEJ:     INFO: \033[1m3 gene_data_norm = {CYAN}{gene_data_norm}{RESET} and last_gene_norm = {CYAN}{last_gene_norm}{RESET} so no need to regenerate torch '.pt' file" )
      
      else:
        print( f"\033[nTRAINLENEJ:      : FATAL:        no such gene data normalization mode as: {gene_data_norm} ... halting now[188]\033[m" ) 
        sys.exit(0)



    # (4) Load experiment config.  Actually most configurable parameters are now provided via user args

    print( f"TRAINLENEJ:     INFO: {BOLD}4 about to load experiment config{RESET}" )
#    pprint.log_section('Loading config.')
    cfg = loader.get_config( args.nn_mode, lr, batch_size )                                                #################################################################### change to just args at some point
    GTExV6Config.INPUT_MODE         = input_mode                                                           # modify config class variable to take into account user preference
    GTExV6Config.MAKE_GREY          = make_grey_perunit                                                    # modify config class variable to take into account user preference
    GTExV6Config.JITTER             = jitter                                                               # modify config class variable to take into account user preference
#    pprint.log_config(cfg) 
#    pprint.log_section('Loading script arguments.')
#    pprint.log_args(args)
  
    print( f"TRAINLENEJ:     INFO:   {ITALICS}experiment config loaded{RESET}" )
   
    
    #(5) Load model
                                                                                                     
    print( f"TRAINLENEJ:     INFO: {BOLD}5 about to load model {nn_type}{RESET} with parameters: args.latent_dim={CYAN}{args.latent_dim}{RESET}, args.em_iters={CYAN}{args.em_iters}{RESET}" ) 
    model = LENETIMAGE(cfg, nn_type, n_classes, tile_size, args.latent_dim, args.em_iters )                                    

# LENETIMAGE  (model, cfg,  nn_type,  tile_size,  args.latent_dim,  args.em_iters   )
# def __init__(self,  cfg,  nn_type,  tile_size,       latent_dim,       em_iters=1 ):
# def __init__(model, cfg,  nn_type,  tile_size,       latent_dim,       em_iters=1 ):    

# self.cfg        = cfg                         so LHS model.cfg           = cfg  (cfg was as passed in as a parameter)
# self.image_net  = cfg.get_image_net(nn_type)  so LHS model.get_image_net = RHS cfg.get_image_net( nn_type, tile_size ) = RHS get_image_net(cfg, nn_type, tile_size ) = RHS vgg19_bn(cfg, tile_size): model.get_image_net = vgg19_bn(cfg, tile_size)
# self.genes_net  = cfg.get_genes_net()         so LHS model.get_genes_net = RHS cfg.get_genes_net                       = RHS get_genes_net(cfg)         = RHS AELinear
# self.latent_dim = latent_dim                  so LHS model.latent_dim    = latent_dim
 
    print( f"TRAINLENEJ:     INFO:    {ITALICS}model loaded{RESET}" )

    if just_test=='True':                                                                                  # then load already trained model from HDD
      if DEBUG>0:
        print( f"TRAINLENEJ:     INFO:   about to load {CYAN}{save_model_name}{RESET} from {CYAN}{log_dir}{RESET}" )
      fpath = '%s/model.pt' % log_dir
      try:
        model.load_state_dict(torch.load(fpath))       
      except Exception as e:
        print( "\033[31;1mTRAINLENEJ:     INFO:  CAUTION! 'There is no trained model. Predictions will be meaningless\033[m" )        
        time.sleep(2)
        pass

      #if torch.cuda.device_count()==2:                                                                    # for Dreedle, which has two bridged Titan RTXs
      # model = DataParallel(model, device_ids=[0, 1])
###    traced_model = torch.jit.trace(model.eval(), torch.rand(10), model.eval())                                                     


   
    #(6)
    
    print( f"TRAINLENEJ:     INFO: {BOLD}6 about to send model to device{RESET}" )   
    model = model.to(device)
    print( f"TRAINLENEJ:     INFO:     {ITALICS}model sent to device{RESET}" ) 
  
    pprint.log_section('Model specs.')
    pprint.log_model(model)
     
    
    if DEBUG>9:
      print( f"TRAINLENEJ:     INFO:   pytorch Model = {CYAN}{model}{RESET}" )
    
    GTExV6Config.LABEL_SWAP_PERUNIT = label_swap_perunit
    
    #(7)
    
    print( "TRAINLENEJ:     INFO: \033[1m7 about to call dataset loader\033[m with parameters: cfg=\033[36;1m{:}\033[m, batch_size=\033[36;1m{:}\033[m, args.n_worker=\033[36;1m{:}\033[m, args.pin_memory=\033[36;1m{:}\033[m, args.pct_test=\033[36;1m{:}\033[m".format( cfg, batch_size, args.n_workers, args.pin_memory, args.pct_test) )
    train_loader, test_loader = loader.get_data_loaders(args,
                                                        cfg,
                                                        batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.pct_test
                                                        )
                                                        
    print( "TRAINLENEJ:     INFO:   \033[3mdataset loaded\033[m" )
  
    if just_test=='False':                                                                                # c.f. loader() Sequential'SequentialSampler' doesn't return indices
      pprint.save_test_indices(test_loader.sampler.indices)
  
    #(8)
      
    print( "TRAINLENEJ:     INFO: \033[1m8 about to select and configure optimizer\033[m with learning rate = \033[35;1m{:}\033[m".format( lr ) )
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
 
         
    #(9)
    
    print( "TRAINLENEJ:     INFO: \033[1m9 about to select CrossEntropyLoss function\033[m" )  
    loss_function = torch.nn.CrossEntropyLoss()
      
    print( "TRAINLENEJ:     INFO:   \033[3mCross Entropy loss function selected\033[m" )  
    
    number_correct_max   = 0
    pct_correct_max      = 0
    test_loss_min        = 999999
    train_loss_min       = 999999
    
    
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
   
    #(10)
                     
    print( "TRAINLENEJ:     INFO: \033[1m10 about to commence training loop, one iteration per epoch\033[m" )
  
    train_total_loss_ave_last              = 99999
    test_total_loss_ave_last               = 99999
    consecutive_training_loss_increases    = 0
    consecutive_test_loss_increases        = 0
    last_epoch_loss_increased              = True
    train_lowest_total_loss_observed       = 99999
    train_lowest_total_loss_observed_epoch = 0
    test_lowest_total_loss_observed        = 99999
    test_lowest_total_loss_observed_epoch  = 0
    
    train_lowest_image_loss_observed       = 99999
    train_lowest_image_loss_observed_epoch = 0
    test_lowest_image_loss_observed        = 99999
    test_lowest_image_loss_observed_epoch  = 0   
    
    for epoch in range(1, n_epochs + 1):
  
        print( f'TRAINLENEJ:     INFO:   epoch: \033[35;1m{epoch}\033[m of \033[35;1m{n_epochs}\033[m, batch size: \033[35;1m{batch_size}\033[m.  \033[38;2;140;140;140mwill halt if test loss increases for \033[35;1m{max_consecutive_losses}\033[m \033[38;2;140;140;140mconsecutive epochs\033[m' )
    
        if just_test=='True':        
          pass     
        
        else:
          
          if DEBUG>1:
            print('TRAINLENEJ:     INFO:   6.1 running training step ')
    
          train_loss1_sum_ave, train_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave = train (      args, epoch, train_loader, model, optimizer, loss_function, writer, train_loss_min, batch_size )
    
          if train_total_loss_ave < train_lowest_total_loss_observed:
            train_lowest_total_loss_observed       = train_total_loss_ave
            train_lowest_total_loss_observed_epoch = epoch
    
          if train_loss1_sum_ave < train_lowest_image_loss_observed:
            train_lowest_image_loss_observed       = train_loss1_sum_ave
            train_lowest_image_loss_observed_epoch = epoch

          if DEBUG>0:
            if ( (train_total_loss_ave < train_total_loss_ave_last) | (epoch==1) ):
              consecutive_training_loss_increases = 0
              last_epoch_loss_increased = False
              print ( f"\r\033[1C\033[2K\033[38;2;140;140;140m                          train():\r\033[49Closs_images={train_loss1_sum_ave:.5f}   \r\033[73Closs_unused=   \r\033[96Cl1_loss={train_l1_loss_sum_ave:.4f}   BATCH AVG =\r\033[124C\033[38;2;0;127;0m{train_total_loss_ave:9.4f}   \033[38;2;140;140;140mlowest total loss=\r\033[154C{train_lowest_total_loss_observed:.4f} at epoch {train_lowest_total_loss_observed_epoch:2d}    lowest image loss=\r\033[195C{train_lowest_image_loss_observed:.4f} at epoch {train_lowest_image_loss_observed_epoch:2d}\033[m", end=''  )
            else:
              last_epoch_loss_increased = True
              print ( f"\r\033[1C\033[2K\033[38;2;140;140;140m                          train():\r\033[49Closs_images={train_loss1_sum_ave:.5f}   \r\033[73Closs_unused=   \r\033[96Cl1_loss={train_l1_loss_sum_ave:.4f}   BATCH AVG =\r\033[124C\033[38;2;127;83;0m{train_total_loss_ave:9.4f}   \033[38;2;140;140;140mlowest total loss=\r\033[154C{train_lowest_total_loss_observed:.4f} at epoch {train_lowest_total_loss_observed_epoch:2d}    lowest image loss=\r\033[195C{train_lowest_image_loss_observed:.4f} at epoch {train_lowest_image_loss_observed_epoch:2d}\033[m", end='' )
              if last_epoch_loss_increased == True:
                consecutive_training_loss_increases +=1
                if consecutive_training_loss_increases == 1:
                  print ( "\033[38;2;127;82;0m <<< training loss increased\033[m", end='' )
                else:
                  print ( "\033[38;2;127;82;0m <<< {0:2d} consecutive training loss increases (s) !!!\033[m".format( consecutive_training_loss_increases ), end='' )
                print ( '')
    
            if (last_epoch_loss_increased == False):
              print ('')
    
          train_total_loss_ave_last = train_total_loss_ave
  
  
        if DEBUG>1:
          print('TRAINLENEJ:     INFO:   6.2 running test step ')
  
        test_loss1_sum_ave, test_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave, number_correct_max, pct_correct_max, test_loss_min     =\
                                                                               test ( cfg, args, epoch, test_loader,  model,  tile_size, loss_function, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, nn_type, tensorboard_images, class_names, class_colours)

  
        if DEBUG>0:
          if ( (test_total_loss_ave < (test_total_loss_ave_last)) | (epoch==1) ):
            consecutive_test_loss_increases = 0
            last_epoch_loss_increased = False
            print ( f"\r\033[1C\033[K                           test():\r\033[49Closs_images={test_loss1_sum_ave:.5f}   \r\033[73Closs_unused=   \r\033[96Cl1_loss={test_l1_loss_sum_ave:.4f}   BATCH AVG =\r\033[124C\033[38;2;0;255;0m{test_total_loss_ave:9.4f}\033[m   lowest TEST loss =\r\033[153C{test_lowest_total_loss_observed:.4f} at epoch {test_lowest_total_loss_observed_epoch:2d}\033[m    \033[38;2;140;140;140mlowest image loss=\r\033[195C{test_lowest_image_loss_observed:.4f} at epoch {test_lowest_image_loss_observed_epoch:2d}\033[m", end = '' )
          else:
            last_epoch_loss_increased = True
            print ( f"\r\033[1C\033[K                           test():\r\033[49Closs_images={test_loss1_sum_ave:.5f}   \r\033[73Closs_unused=   \r\033[96Cl1_loss={test_l1_loss_sum_ave:.4f}   BATCH AVG =\r\033[124C\033[38;2;255;0;0m{test_total_loss_ave:9.4f}\033[m   lowest TEST loss =\r\033[153C{test_lowest_total_loss_observed:.4f} at epoch {test_lowest_total_loss_observed_epoch:2d}\033[m    \033[38;2;140;140;140mlowest image loss=\r\033[195C{test_lowest_image_loss_observed:.4f} at epoch {test_lowest_image_loss_observed_epoch:2d}\033[m", end = '')
            if last_epoch_loss_increased == True:
              consecutive_test_loss_increases +=1
              if consecutive_test_loss_increases == 1:
                print ( "\033[38;2;255;0;0m <<< test loss increased\033[m", end='' )
              else:
                print ( "\033[38;2;255;0;0m <<< {0:2d} consecutive test loss increases !!!\033[m".format( consecutive_test_loss_increases ), end='')
              print ( '')

              if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before, so that the most recent model for which the loss improved will be saved
                  now = time.localtime(time.time())
                  print(time.strftime("TRAINLENEJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
                  sys.exit(0)
          
          if (last_epoch_loss_increased == False):
            print ('')
  
        test_total_loss_ave_last = test_total_loss_ave
        
        if test_total_loss_ave < test_lowest_total_loss_observed:
          test_lowest_total_loss_observed       = test_total_loss_ave
          test_lowest_total_loss_observed_epoch = epoch
          if DEBUG>0:
            print( f"TRAINLENEJ:     INFO:   {GREEN}{ITALICS}new low test loss ... saving model to {log_dir}{RESET}\033[m" )
          save_model(args.log_dir, model)
  
        if test_loss1_sum_ave < test_lowest_image_loss_observed:
          test_lowest_image_loss_observed       = test_loss1_sum_ave
          test_lowest_image_loss_observed_epoch = epoch
        
  
  #            if DEBUG>0:
  #              print( "TRAINLENEJ:     INFO:   saving samples to \033[35;1m{:}\033[m".format( args.log_dir ) )
  #            save_samples(args.log_dir, model, test_loader, cfg, epoch)
   
  #      if epoch%save_model_every == 0:
  #          if DEBUG>0:
  #            print( f"TRAINLENEJ:     INFO:   about to save model to \033[35;1m{log_dir}\033[m" )
  #          save_model(args.log_dir, model)
  #          if DEBUG>0:
  #            print( f"TRAINLENEJ:     INFO:   \033[3mmodel saved \033[m" )
            
    print( "TRAINLENEJ:     INFO: \033[33;1mtraining complete\033[m" )
  
    hours   = round((time.time() - start_time) / 3600, 1  )
    minutes = round((time.time() - start_time) / 60,   1  )
    #pprint.log_section('Job complete in {:} mins'.format( minutes ) )
  
    print('TRAINLENEJ:     INFO: run completed in {:} mins'.format( minutes ) )
    
    writer.close()                                                                                         # PGD 200206
    
   # if DEBUG>0:
   #   print( f"TRAINLENEJ:     INFO:   about to save model to \033[35;1m{log_dir}\033[m" )
   # save_model(args.log_dir, model)
   # if DEBUG>0:
   #   print( f"TRAINLENEJ:     INFO:   \033[3mmodel saved \033[m" )
              
    #pprint.log_section('Model saved.')
# ------------------------------------------------------------------------------






def train(args, epoch, train_loader, model, optimizer, loss_function, writer, train_loss_min, batch_size  ):
    """
    Train LENET5 model and update parameters in batches of the whole training set
    """
    
    if DEBUG>1:
      print( "TRAINLENEJ:     INFO:     at top of train() and parameter train_loader() = \033[35;1m{:}\033[m".format( train_loader ) )
    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:     at top of train() with parameters \033[35;1margs: \033[m{:}, \033[35;1mtrain_loader: \033[m{:}, \033[35;1mmodel: \033[m{:}, \033[35;1moptimizer: \033[m{:}".format(args, train_loader, model, optimizer ) )

    if DEBUG>1:
      print( "TRAINLENEJ:     INFO:     train(): about to call \033[33;1mmodel.train()\033[m" )

    model.train()                                                                                          # set model to training mode

    if DEBUG>1:
      print( "TRAINLENEJ:     INFO:     train(): done\033[m" )

    loss1_sum        = 0
    loss2_sum        = 0
    l1_loss_sum      = 0
    total_loss_sum   = 0


    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:     train(): about to enumerate over dataset" )
    
    for i, ( batch_images, batch_labels ) in enumerate( train_loader ):                                    # fetch a batch each of images and labels
        
        if DEBUG>99:
          print( f"TRAINLENEJ:     INFO:     train(): len(batch_images) = \033[33;1m{len(batch_images)}\033[m" )
          print( f"TRAINLENEJ:     INFO:     train(): len(batch_labels) = \033[33;1m{len(batch_labels)}\033[m" )
        if DEBUG>999:
          print( f"{ batch_labels.cpu().detach().numpy()},  ", flush=True, end="" )          
                  
        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     train(): about to call \033[33;1moptimizer.zero_grad()\033[m" )

        optimizer.zero_grad()

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     train(): done" )

        batch_images = batch_images.to ( device )                                                          # send to GPU
        batch_labels = batch_labels.to ( device )                                                          # send to GPU

        if DEBUG>9:
          print ( "TRAINLENEJ:     INFO:     train():       type(batch_images)                 = {:}".format( type(batch_images)       ) )
          print ( "TRAINLENEJ:     INFO:     train():       batch_images.size()                = {:}".format( batch_images.size()       ) )

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:      train(): about to call \033[33;1mmodel.forward()\033[m" )

        y1_hat  = model.forward( batch_images )                                                          # perform a step: VGG11.forward( batch_images )
          
        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:      train(): done" )
             
      
        if DEBUG>9:
          print ( "TRAINLENEJ:     INFO:      train():       2 type(y1_hat)                      = {:}".format( type(y1_hat)       ) )
          print ( "TRAINLENEJ:     INFO:      train():       2 y1_hat.shape                      = {:}".format( y1_hat.shape       ) )
          print ( "TRAINLENEJ:     INFO:      train():       2 type(batch_labels)                = {:}".format( type(batch_labels)  ) )
          print ( "TRAINLENEJ:     INFO:      train():       2 batch_labels.shape                = {:}".format( batch_labels.shape  ) )
        if DEBUG>9:
          y1_hat_numpy = (y1_hat.cpu().data).numpy()
          print ( "TRAINLENEJ:     INFO:      train():       y1_hat                            = \n{:}".format( y1_hat_numpy) )
          print ( "TRAINLENEJ:     INFO:      train():       batch_labels                      = \n{:}".format( batch_labels  ) )

        loss_images = loss_function(y1_hat, batch_labels)
        loss_images_value = loss_images.item()                                                             # use .item() to extract just the value: don't create multiple new tensors each of which will have gradient histories
        #l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0
        total_loss        = loss_images_value + l1_loss

        if DEBUG>0:
          print ( f"\033[2K                          train():     \033[38;2;140;140;140m\r\033[40Cn={i+1:>3d}    \r\033[49Closs_images={loss_images_value:.5f}   \r\033[73Closs_unused=   \r\033[96Cl1_loss={l1_loss:.4f}   BATCH AVE =\r\033[{124+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{PALE_GREEN if total_loss<1 else PALE_ORANGE if 1<=total_loss<2 else PALE_RED}{total_loss:9.4f}\033[m" )
          print ( "\033[2A" )
          
        loss_images.backward()
        # Perform gradient clipping *before* calling `optimizer.step()`.
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss1_sum      += loss_images_value
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss     

        del y1_hat
        del loss_images
        del batch_labels
        torch.cuda.empty_cache()

        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      train():       type(loss1_sum)                      = {:}".format( type(loss1_sum)       ) )
          
    loss1_sum_ave    = loss1_sum      / (i+1)
    loss2_sum_ave    = loss2_sum      / (i+1)
    l1_loss_sum_ave  = l1_loss_sum    / (i+1)
    total_loss_ave   = total_loss_sum / (i+1)

    if total_loss_sum < train_loss_min:
      train_loss_min = total_loss_sum

    writer.add_scalar( 'loss_train', total_loss_sum, epoch )
    writer.add_scalar( 'loss_train_min',      train_loss_min, epoch )

    if DEBUG>99:
      print ( "TRAINLENEJ:     INFO:      train():       type(loss1_sum_ave)                      = {:}".format( type(loss1_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      train():       type(loss2_sum_ave)                      = {:}".format( type(loss2_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      train():       type(l1_loss_sum_ave)                    = {:}".format( type(l1_loss_sum_ave)   ) )
      print ( "TRAINLENEJ:     INFO:      train():       type(total_loss_ave)                     = {:}".format( type(total_loss_ave)    ) )

    return loss1_sum_ave, loss2_sum_ave, l1_loss_sum_ave, total_loss_ave

# ------------------------------------------------------------------------------








def test( cfg, args, epoch, test_loader, model, tile_size, loss_function, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, nn_type, tensorboard_images, class_names, class_colours ):

    """Test model by pusing a held out batch through the network
    """

    global global_batch_count

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to test model by computing the average loss on a held-out dataset. No parameter updates" )

    model.eval()                                                                                           # set model to evaluation mod

    loss1_sum      = 0
    loss2_sum      = 0
    l1_loss_sum    = 0
    total_loss_sum = 0

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to enumerate  " )
      
    for i, (batch_images, batch_labels) in  enumerate( test_loader ):
        
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        if DEBUG>9:
          print ( "TRAINLENEJ:     INFO:      test():       type(batch_images)                      = {:}".format( type(batch_images)       ) )
          print ( "TRAINLENEJ:     INFO:      test():       batch_images.shape                      = {:}".format( batch_images.shape       ) )

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     test(): about to call \033[33;1mmodel.forward()\033[m" )

        with torch.no_grad():                                                                             # PGD 200129 - Don't need gradients for testing, so this should save some GPU memory (tested: it does)
          y1_hat = model.forward( batch_images )                                                          
    
        preds, p_max, p_2, sm = analyse_probs( y1_hat )
        
    
        if args.just_test=='True':

          if DEBUG>0:
              print ( f"TRAINLENEJ:     INFO:      test():             global_batch_count {DIM_WHITE}(super-patch number){RESET} = {global_batch_count+1:5d}  {DIM_WHITE}({((global_batch_count+1)/(args.supergrid_size**2)):04.2f}){RESET}", end="" )
                      
          if global_batch_count%(args.supergrid_size**2)==0:
            grid_images = batch_images.cpu().numpy()
            grid_labels = batch_labels.cpu().numpy()
            grid_preds  = preds
            grid_p_max  = p_max
            grid_p_2    = p_2
            grid_sm     = sm 

            if DEBUG>99:
              print ( f"TRAINLENEJ:     INFO:      test():             batch_images.shape                      = {batch_images.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_images.shape                       = {grid_images.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             batch_labels.shape                      = {batch_labels.shape}" )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_labels.shape                       = {grid_labels.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             preds.shape                             = {preds.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_preds.shape                        = {grid_preds.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             p_max.shape                             = {p_max.shape}" )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_max.shape                        = {grid_p_max.shape}" )            
              print ( f"TRAINLENEJ:     INFO:      test():             p_2.shape                               = {p_2.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_2.shape                          = {grid_p_2.shape}" )
              print ( f"TRAINLENEJ:     INFO:      test():             sm.shape                                = {sm.shape}" )                                    
              print ( f"TRAINLENEJ:     INFO:      test():             grid_sm.shape                           = {grid_sm.shape}" )
                      
          else:
            grid_images = np.append( grid_images, batch_images.cpu().numpy(), axis=0 )
            grid_labels = np.append( grid_labels, batch_labels.cpu().numpy(), axis=0 )
            grid_preds  = np.append( grid_preds,  preds,                      axis=0 )
            grid_p_max  = np.append( grid_p_max,  p_max,                      axis=0 )
            grid_p_2    = np.append( grid_p_2,    p_2,                        axis=0 )
            grid_sm     = np.append( grid_sm,     sm,                         axis=0 )
  
            if DEBUG>99:
              print ( f"TRAINLENEJ:     INFO:      test():             grid_images.shape                       = {grid_images.shape}"  )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_labels.shape                       = {grid_labels.shape}"  )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_preds.shape                        = {grid_preds.shape}"   )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_max.shape                        = {grid_p_max.shape}"   )            
              print ( f"TRAINLENEJ:     INFO:      test():             grid_p_2.shape                          = {grid_p_2.shape}"     )
              print ( f"TRAINLENEJ:     INFO:      test():             grid_sm.shape                           = {grid_sm.shape}"      )  

          global_batch_count+=1
        
          if DEBUG>999:
              print ( f"TRAINLENEJ:     INFO:      test():             global_batch_count%(args.supergrid_size**2)                       = {global_batch_count%(args.supergrid_size**2)}"  )
              
          if global_batch_count%(args.supergrid_size**2)==0:
            if GTExV6Config.INPUT_MODE=='image':
              print("")
              fig=plot_classes_preds(args, model, tile_size, grid_images, grid_labels, grid_preds, grid_p_max, grid_p_2, grid_sm, class_names, class_colours )
              writer.add_figure('Predictions v Truth', fig, epoch)
              plt.close(fig)

        if DEBUG>9:
          y1_hat_numpy = (y1_hat.cpu().data).numpy()
          print ( "TRAINLENEJ:     INFO:      test():        type(y1_hat)                      = {:}".format( type(y1_hat_numpy)       ) )
          print ( "TRAINLENEJ:     INFO:      test():        y1_hat.shape                      = {:}".format( y1_hat_numpy.shape       ) )
          print ( "TRAINLENEJ:     INFO:      test():        batch_labels.shape                = {:}".format( batch_labels.shape  ) )
        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      test():        y1_hat                            = \n{:}".format( y1_hat_numpy) )
          print ( "TRAINLENEJ:     INFO:      test():        batch_labels                      = \n{:}".format( batch_labels  ) )
        
        loss_images       = loss_function( y1_hat, batch_labels )
        loss_images_value = loss_images.item()                                                             # use .item() to extract just the value: don't create multiple new tensors each of which will have gradient histories

        if DEBUG>9:
          print ( "\033[2K                           test():      loss_images, loss_images_values ={:}, {:}".format( loss_images_value,  loss_images_value))


        #l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0
        total_loss        = loss_images_value + l1_loss

        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      test():       type(loss)                      = {:}".format( type(loss)       ) )

        if DEBUG>0:
          if (not args.just_test=='True'):
            print ( f"\033[2K                           test():     \033[38;2;140;140;140m\r\033[40C{ 'p' if args.just_test=='True' else 'n'}={i+1:>3d}    \r\033[49Closs_images={loss_images_value:.5f}   \r\033[73Closs_unused=   \r\033[96Cl1_loss={l1_loss:.4f}   BATCH AVE =\r\033[{124+6*int((total_loss*5)//1) if total_loss<1 else 156+6*int((total_loss*1)//1) if total_loss<12 else 250}C{GREEN if total_loss<1 else ORANGE if 1<=total_loss<2 else RED}{total_loss:9.4f}\033[m" )
            print ( "\033[2A" )
          else:
            print ( f"\033[38;2;140;140;140m\r\033[131CLOSS=\r\033[{136+7*int((total_loss*5)//1) if total_loss<1 else 178+7*int((total_loss*1)//1) if total_loss<12 else 250}C{GREEN if total_loss<1 else ORANGE if 1<=total_loss<2 else RED}{total_loss:9.4f}\033[m" )
            print ( "\033[1A" )

        loss1_sum      += loss_images_value                                                                # use .item() to extract just the value: don't create a new tensor
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss  

        del loss_images
        torch.cuda.empty_cache()

    if epoch % 1 == 0:
      y1_hat_values             = y1_hat.cpu().detach().numpy()
      y1_hat_values_max_indices = np.argmax( np.transpose(y1_hat_values), axis=0 )
      batch_labels_values       = batch_labels.cpu().detach().numpy()

      torch.cuda.empty_cache()    
      
      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      test():        y1_hat.shape                      = {:}".format( y1_hat.shape                     ) )
        print ( "TRAINLENEJ:     INFO:      test():        y1_hat_values_max_indices.shape   = {:}".format( y1_hat_values_max_indices.shape  ) )
        print ( "TRAINLENEJ:     INFO:      test():        batch_labels_values.shape         = {:}".format( batch_labels_values.shape        ) )
      
      number_to_display=batch_size
      print ( "" )
      correct=np.sum( np.equal(y1_hat_values_max_indices, batch_labels_values))
      print ( f"TRAINLENEJ:     INFO:     test(): truth/prediction for first {number_to_display} examples from the last test batch (number correct = \u001b[4m{correct}/{batch_size} = {100*correct/batch_size}%)\033[m" )
      np.set_printoptions(formatter={'int': lambda x: "{:>2d}".format(x)})
      print (  batch_labels_values[0:number_to_display]          ) 
      print (  y1_hat_values_max_indices[0:number_to_display]    )

      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      test():       y1_hat.shape                     = {:}".format( y1_hat_values.shape          ) )
        np.set_printoptions(formatter={'float': lambda x: "{0:10.2e}".format(x)})
        print (  "{:}".format( (np.transpose(y1_hat_values))[:,:number_to_display] )  )
        np.set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})

      if DEBUG>9:
        number_to_display=16  
        print ( "TRAINLENEJ:     INFO:      test():       FIRST  GROUP BELOW: y1_hat"                                                                      ) 
        print ( "TRAINLENEJ:     INFO:      test():       SECOND GROUP BELOW: y1_hat_values_max_indices (prediction)"                                      )
        print ( "TRAINLENEJ:     INFO:      test():       THIRD  GROUP BELOW: batch_labels_values (truth)"                                                 )
        np.set_printoptions(formatter={'float': '{: >6.2f}'.format}        )
        print ( f"{(np.transpose(y1_hat_values)) [:,:number_to_display] }" )
        np.set_printoptions(formatter={'int': '{: >6d}'.format}            )
        print ( " {:}".format( y1_hat_values_max_indices    [:number_to_display]        ) )
        print ( " {:}".format( batch_labels_values          [:number_to_display]        ) )
 
 
    y1_hat_values               = y1_hat.cpu().detach().numpy()                                            # these are the raw outputs
    del y1_hat                                                                                             # immediately delete tensor to recover large amount of memory
    y1_hat_values_max_indices   = np.argmax( np.transpose(y1_hat_values), axis=0  )                        # these are the predicted classes corresponding to batch_images
    batch_labels_values         = batch_labels.cpu().detach().numpy()                                      # these are the true      classes corresponding to batch_images
    number_correct              = np.sum( y1_hat_values_max_indices == batch_labels_values )
    pct_correct                 = number_correct / batch_size * 100

    loss1_sum_ave    = loss1_sum       / (i+1)
    loss2_sum_ave    = loss2_sum       / (i+1)
    l1_loss_sum_ave  = l1_loss_sum     / (i+1)
    total_loss_ave   = total_loss_sum  / (i+1)

    if total_loss_sum    <  test_loss_min:
       test_loss_min     =  total_loss_sum

    if number_correct    >  number_correct_max:
      number_correct_max =  number_correct

    if pct_correct       >  pct_correct_max:
      pct_correct_max    =  pct_correct
    
    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      test():             total_loss_sum                           = {:}".format( total_loss_sum           ) )
      print ( "TRAINLENEJ:     INFO:      test():             test_loss_min                            = {:}".format( test_loss_min            ) )
      print ( "TRAINLENEJ:     INFO:      test():             number_correct                           = {:}".format( number_correct           ) )
      print ( "TRAINLENEJ:     INFO:      test():             number_correct_max                       = {:}".format( number_correct_max       ) )
      print ( "TRAINLENEJ:     INFO:      test():             pct_correct                              = {:}".format( pct_correct              ) )
      print ( "TRAINLENEJ:     INFO:      test():             pct_correct_max                          = {:}".format( pct_correct_max          ) )
    
    writer.add_scalar( 'loss_test',        total_loss_sum,     epoch )
    writer.add_scalar( 'loss_test_min',    test_loss_min,      epoch )    
    writer.add_scalar( 'num_correct',      number_correct,     epoch )
    writer.add_scalar( 'num_correct_max',  number_correct_max, epoch )
    writer.add_scalar( 'pct_correct',      pct_correct,        epoch ) 
    writer.add_scalar( 'pct_correct_max',  pct_correct_max,    epoch ) 

    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      test():             batch_images.shape                       = {:}".format( batch_images.shape ) )
      print ( "TRAINLENEJ:     INFO:      test():             batch_labels.shape                       = {:}".format( batch_labels.shape ) )
      
#    if not args.just_test=='True':
#      if GTExV6Config.INPUT_MODE=='image':
#        writer.add_figure('Predictions v Truth', plot_classes_preds(args, model, tile_size, batch_images, batch_labels, preds, p_max, p_2, sm, class_names, class_colours), epoch)
        
    if args.just_test=='False':
      if GTExV6Config.INPUT_MODE=='image':
        fig=plot_classes_preds(args, model, tile_size, batch_images.cpu().numpy(), batch_labels.cpu().numpy(), preds, p_max, p_2, sm, class_names, class_colours)
        writer.add_figure('Predictions v Truth', fig, epoch)
        plt.close(fig)

    del batch_images
    del batch_labels
    
#    if args.just_test=='True':
#      if GTExV6Config.INPUT_MODE=='image':
#        it=list(permutations( range(0, batch_size)  ) )
#        writer.add_figure('Predictions v Truth', plot_classes_preds(args, model, tile_size, batch_images, batch_labels, preds, p_max, p_2, sm, class_names, class_colours), epoch)

    if DEBUG>99:
      print ( "TRAINLENEJ:     INFO:      test():       type(loss1_sum_ave)                      = {:}".format( type(loss1_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(loss2_sum_ave)                      = {:}".format( type(loss2_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(l1_loss_sum_ave)                    = {:}".format( type(l1_loss_sum_ave)   ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(total_loss_ave)                     = {:}".format( type(total_loss_ave)    ) )

    return loss1_sum_ave, loss2_sum_ave, l1_loss_sum_ave, total_loss_ave, number_correct_max, pct_correct_max, test_loss_min



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
def analyse_probs( y1_hat ):

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(y1_hat, axis=1)

    if DEBUG>9:
      y1_hat_numpy = (y1_hat.cpu().data).numpy()
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds_tensor.shape           = {:}".format( preds_tensor.shape    ) ) 
      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds_tensor                 = \n{:}".format( preds_tensor      ) ) 
    
    preds = np.squeeze(preds_tensor.cpu().numpy())

    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               type(preds)                  = {:}".format( type(preds)           ) )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               preds.shape                  = {:}".format( preds.shape           ) ) 
      print ( "TRAINLENEJ:     INFO:      analyse_probs():         FIRST  GROUP BELOW: preds"            ) 
      print ( "TRAINLENEJ:     INFO:      analyse_probs():         SECOND GROUP BELOW: y1_hat_numpy.T"   )
      np.set_printoptions(formatter={'int':   lambda x: "\033[1m{:^10d}\033[m".format(x)    }    )
      print ( preds[0:22] )
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print (  np.transpose(y1_hat_numpy[0:22,:])  )

    p_max  = np.array([F.softmax(el, dim=0)[i].item() for i, el in zip(preds, y1_hat)] )    # regarding the -1 dimension, see https://stackoverflow.com/questions/59704538/what-is-a-dimensional-range-of-1-0-in-pytorch

    # extract the SECOND HIGHEST probability for each example (which is a bit trickier)
    sm = F.softmax( y1_hat, dim=1).cpu().numpy()
    p_2 = np.zeros((len(preds)))
    for i in range (0, len(p_2)):
      p_2[i] = max( [ el for el in sm[i,:] if el != max(sm[i,:]) ] )     
      
    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               sm                         = \n{:}".format( np.transpose(sm[0:22,:])   )  )
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
    

    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():              type(sm)                   = {:}".format( type(sm) )  )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               sm                         = \n{:}".format( np.transpose(sm[0:22,:])   )  )
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      #print ( "TRAINLENEJ:     INFO:      analyse_probs():               p_2              = \n{:}".format( p_2   )  )                       
    
    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               p_max.shape                = {:}".format( (np.array(p_max)).shape )  )
      print ( "TRAINLENEJ:     INFO:      analyse_probs():               p_max                      = \n{:}".format( np.array(p_max[0:22]) )  )
   
    return preds, p_max, p_2, sm


# ------------------------------------------------------------------------------
def plot_classes_preds(args, model, tile_size, batch_images, batch_labels, preds, p_max, p_2, sm, class_names, class_colours):
    '''
    Generates matplotlib Figure using a trained network, along with a batch of images and labels, that shows the network's top prediction along with its probability, alongside the actual label, colouring this
    information based on whether the prediction was correct or not. Uses the "images_to_probs" function. 
    
    From: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    
    '''
    
    if args.just_test=='True':
      
      non_specimen_tiles=0
      number_correct=0      
      
      # plot the images in the batch, along with predicted and true labels
      figure_width   = 14
      figure_height  = 14
      fig = plt.figure( figsize=( figure_width, figure_height )  )                                         # overall size ( width, height ) in inches

      number_to_plot = batch_labels.shape[0]   
      ncols = int(number_to_plot**.5)
      nrows = ncols

      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             tiles to plot                           = {:}".format( number_to_plot  ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             nrows                                   = {:}".format( nrows           ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             ncols                                   = {:}".format( ncols           ) )      

      patch=[]
      for n in range (0, len(class_colours)):
        patch.append(mpatches.Patch(color=class_colours[n], linewidth=0))
        fig.legend(patch, args.long_class_names, loc='upper right', fontsize=14, facecolor='lightgrey')      
      #fig.tight_layout( pad=0 )

    else:
  
      number_to_plot = len(batch_labels)    
      figure_width   = 30
      figure_height  = int(number_to_plot * .4)
          
      # plot the images in the batch, along with predicted and true labels
      fig = plt.figure( figsize=( figure_width, figure_height ) )                                         # overall size ( width, height ) in inches
      fig.tight_layout( rect=[0, 0, 1, 1] )
  
      if DEBUG>99:
        print ( "\nTRAINLENEJ:     INFO:      plot_classes_preds():             number_to_plot                          = {:}".format( number_to_plot    ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             figure width  (inches)                  = {:}".format( figure_width    ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             figure height (inches)                  = {:}".format( figure_height   ) )

      #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
      #plt.grid( False )
  
      ncols = int((   number_to_plot**.5 )           // 1  )
      nrows = int(( ( number_to_plot // ncols ) + 1 ) // 1 )
  
      if DEBUG>99:
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             number_to_plot                          = {:}".format( number_to_plot  ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             nrows                                   = {:}".format( nrows           ) )
        print ( "TRAINLENEJ:     INFO:      plot_classes_preds():             ncols                                   = {:}".format( ncols           ) ) 

     
#     for idx in np.arange( number_to_plot ):

    if args.just_test=='True':

      break_1=6    # rows
      break_2=18   # rows
      break_3=25   # rows

      
      fig, axes = plt.subplots( nrows=nrows, ncols=ncols, figsize=( figure_width, figure_height ) )
      
      patch=[]
      for n in range (0, len(class_colours)):
        patch.append(mpatches.Patch(color=class_colours[n], linewidth=0))
        fig.legend(patch, args.long_class_names, loc='upper right', fontsize=14, facecolor='lightgrey')      
      #fig.tight_layout( pad=0 )      
      
      gs = axes[1, -1].get_gridspec()
      # remove underlying axes from the region we want to use for the bar chart
      if nrows<=break_1:                                            
          axes[nrows-1, ncols-1].remove()                                                                                 # delete this cell (the one in the bottom right hand corner)
      elif break_1<nrows<=break_2:
        for i, j in product(range(nrows-2, nrows), range(ncols-2, ncols )):                                               # delete all these cells (cartesian product)
          axes[i,j].remove()
      elif break_2<nrows<=break_3:
        for i, j in product(range(nrows-3, nrows), range(ncols-3, ncols )):                                               # delete all these cells (cartesian product)
          axes[i,j].remove()      
      else:
        pass
      #axbig = fig7.add_subplot(gs[8:, -1])
      #axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5), xycoords='axes fraction', va='center')
    
      if nrows<=break_1:      
          ax0 = fig.add_subplot( gs[nrows-1:, ncols-1:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))   # where to place top LH corner of the bar chart
      elif break_1<nrows<=break_2:
           ax0 = fig.add_subplot( gs[nrows-2:, ncols-2:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart
      elif break_2<nrows<=break_3:
           ax0 = fig.add_subplot( gs[nrows-3:, ncols-3:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart
      else:
           ax0 = fig.add_subplot( gs[nrows-4:, ncols-4:], yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))  # where to place top LH corner of the bar chart

                      
#      ax0 = fig.add_subplot( nrows, ncols, nrows*ncols, yticks=np.arange(0, number_to_plot, int(number_to_plot**0.5)))
#      pos1 = ax0.get_position()
#      pos2 = [pos1.x0 + 2, pos1.y0 + 2,  pos1.width * 0.5, pos1.height * 0.5]
#      ax0.set_position(pos2)
      ax0.grid( color='silver', linestyle='--', linewidth=1, axis='y', alpha=0 )
      ax0.set_xlabel("sum of tile probs Vs. class", size=11)
      ax0.yaxis.set_ticks_position("right")
      ax0.tick_params(labelsize=10) 
      ax0.set_ylim(0,number_to_plot) 
      ax0.set_facecolor("xkcd:mint" if batch_labels[0]==np.argmax(np.sum(sm,axis=0)) else "xkcd:faded pink" )      
      ax0.bar( x=['1', '2', '3', '4', '5', '6', '7'], height=np.sum(sm,axis=0),  width=int(number_to_plot/len(batch_labels)), color=class_colours )

  # [c[0] for c in class_names]
        
    for idx in range( number_to_plot ): # reserving last subplot for the bar chart
        
        if args.just_test=='True':
          
          if DEBUG>0:
            if idx%10==0:
              print ( f"{DIM_WHITE}..{idx}", end="" ) 

          #ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )            # nrows, ncols, "index starts at 1 in the upper left corner and increases to the right", List of x-axis tick locations, List of y-axis tick locations
          
          if nrows<=break_1:    
            if ( idx in excludes( number_to_plot, 1)  ):
              pass
            else:
              ax=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
          elif break_1<nrows<=break_2:
            if ( idx in excludes( number_to_plot, plot_box_side_length=2)  ):
              pass
            else:
              ax=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
          elif break_2<nrows<=break_3:
            if ( idx in excludes( number_to_plot, plot_box_side_length=3)  ):
              pass
            else:
              ax=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
          else:
            if ( idx in excludes( number_to_plot, plot_box_side_length=4)  ):
              pass
            else:
              ax=plt.subplot( nrows, ncols, idx+1, xticks=[], yticks=[], frame_on=True, autoscale_on=True  )
         
          threshold_0=36
          threshold_1=100
          threshold_2=400
          threshold_3=900
                 
          if idx==0:
            t1=f"{int(number_to_plot**.5)//1}x{int(number_to_plot**.5)//1}"
            ax.text( -120,  20, t1, size=12, ha="left", color="goldenrod", style="normal" )
            t2=f"Cancer type: {args.cancer_type_long}"
            t3=f"Truth label for this WSI:"
            t4=f"{args.long_class_names[batch_labels[idx]]}"
            t5=f"NN prediction from patch:"
            t6=f"{args.long_class_names[np.argmax(np.sum(sm,axis=0))]}"
            if len(batch_labels)>=threshold_3:
              ax.text( -550, -400, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
              ax.text( -550, -300, t3, size=14, ha="left",   color="black", style="normal" )
              ax.text(  550, -300, t4, size=14, ha="left",   color="black", style="italic" )
              ax.text( -550, -200, t5, size=14, ha="left",   color="black", style="normal" )
              ax.text(  550, -200, t6, size=14, ha="left",   color="black", style="italic" )
            elif threshold_3>len(batch_labels)>=threshold_2: #OK
              ax.text( -380, -300, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
              ax.text( -380, -200, t3, size=14, ha="left",   color="black", style="normal" )
              ax.text(  400, -200, t4, size=14, ha="left",   color="black", style="italic" )
              ax.text( -380, -120, t5, size=14, ha="left",   color="black", style="normal" )
              ax.text(  400, -120, t6, size=14, ha="left",   color="black", style="italic" )
            elif threshold_2>len(batch_labels)>=threshold_1: #OK
              ax.text( -200, -180, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
              ax.text( -200, -120, t3, size=14, ha="left",   color="black", style="normal" )
              ax.text(  375, -120, t4, size=14, ha="left",   color="black", style="italic" )
              ax.text( -200, -80, t5, size=14, ha="left",   color="black", style="normal" )
              ax.text(  375, -80, t6, size=14, ha="left",   color="black", style="italic" )
            elif threshold_1>len(batch_labels)>=threshold_0: #OK
              ax.text( -100, -75, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
              ax.text( -100, -50, t3, size=14, ha="left",   color="black", style="normal" )
              ax.text(  230, -50, t4, size=14, ha="left",   color="black", style="italic" )
              ax.text( -100, -30, t5, size=14, ha="left",   color="black", style="normal" )
              ax.text(  230, -30, t6, size=14, ha="left",   color="black", style="italic" )               
            else: # (< threshold0) #OK
              ax.text( -60,  -60, t2, size=16, ha="left",   color="black", style="normal", fontname="DejaVu Sans", weight='bold' )            
              ax.text( -60,  -35, t3, size=14, ha="left",   color="black", style="normal" )
              ax.text(  95, -35, t4, size=14, ha="left",   color="black", style="italic" )
              ax.text( -60,  -20, t5, size=14, ha="left",   color="black", style="normal" )
              ax.text(  95, -20, t6, size=14, ha="left",   color="black", style="italic" )                           
            
            if DEBUG>99:
              predicted_class=np.argmax(np.sum(sm,axis=0))
              print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             predicted_class                                   = {predicted_class}" )
              
          
          tile_rgb_npy=batch_images[idx]
          tile_rgb_npy_T = np.transpose(tile_rgb_npy, (1, 2, 0))         
          tile_255 = tile_rgb_npy_T * 255
          tile_uint8 = np.uint8( tile_255 )
          tile_norm_PIL = Image.fromarray( tile_uint8 )
          tile = tile_norm_PIL.convert("RGB")

          IsBadTile = check_badness( args, tile )
          
          if IsBadTile:                                                                                   # because such tiles were never looked at during training
            non_specimen_tiles+=1
            pass
          else:
            if len(batch_labels)>=threshold_3:
              font_size=8
              left_offset=int(0.6*tile_size)
              top_offset =int(0.9*tile_size)            
              p=int(10*(p_max[idx]-.01)//1)
              p_txt=p
            elif len(batch_labels)>=threshold_2:
              font_size=10
              left_offset=int(0.45*tile_size)
              top_offset =int(0.90*tile_size)            
              p=np.around(p_max[idx]-.01,decimals=1)
              p_txt=p
            elif len(batch_labels)>=threshold_1:
              font_size=14
              left_offset=int(0.6*tile_size)
              top_offset =int(0.92*tile_size)            
              p=np.around(p_max[idx]-.01,decimals=1)
              p_txt=p
            else: 
              p=np.around(p_max[idx],2)
              p_txt = f"p={p}"   
              font_size=16
              left_offset=4
              top_offset =int(0.95*tile_size)
              
            if p_max[idx]>=0.75:
              c="orange"
            elif p_max[idx]>0.50:
              c="orange"
            else:
              c="orange"

            if len(batch_labels)>=threshold_3:
              c="red"
                                          
            ax.text( left_offset, top_offset, p_txt, size=font_size, color=c, style="normal", weight="bold" )

            if (preds[idx]==batch_labels[idx].item()):
              number_correct+=1
            else:
              c=class_colours[preds[idx]]
              if len(batch_labels)>=threshold_3:
                font_size=13
                left_offset=int(0.3*tile_size)
                top_offset =int(0.6*tile_size)  
              elif len(batch_labels)>=threshold_2:
                left_offset=int(0.4*tile_size)
                top_offset =int(0.6*tile_size)  
                font_size=16
              elif len(batch_labels)>=threshold_1:
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
                
              ax.text( left_offset, top_offset, text, size=font_size, color=c, style="normal", weight="bold" )
                      
        else:
          ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[] )                                              # nrows, ncols, "index starts at 1 in the upper left corner and increases to the right", List of x-axis tick locations, List of y-axis tick locations

        if args.just_test=='True':
          total_tiles     =  len(batch_labels)
          specimen_tiles  =  total_tiles - non_specimen_tiles
          if specimen_tiles>0:
            pct_correct     =   (number_correct/specimen_tiles)
          else:
            pct_correct     =   0
    
          if idx==total_tiles-2:
            ax2 = fig.gca()
            stats=f"Statistics: tile count: {total_tiles}; background tiles: {non_specimen_tiles}; specimen tiles: {specimen_tiles}; correctly predicted: {number_correct}/{specimen_tiles} ({pct_correct*100}%)"
            plt.figtext( 0.15, 0.055, stats, size=14, color="black", style="normal" )
          
        img=batch_images[idx]
        npimg_t = np.transpose(img, (1, 2, 0))
        if args.just_test=='False':
          plt.imshow(npimg_t)
        else:
          plt.imshow(npimg_t, aspect='auto')
          plt.subplots_adjust(wspace=0, hspace=0)        

        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:}".format( idx ) )
        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:} p_max[idx] = {:4.2f}, class_names[preds[idx]] = {:<20s}, class_names[batch_labels[idx]] = {:<20s}".format( idx, p_max[idx], class_names[preds[idx]], class_names[batch_labels[idx]]  ) )

        if DEBUG>99:
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             idx                                     = {idx}"                            )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             p_max[idx]                              = {p_max[idx]:4.2f}"                )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             p_2[idx]]                               = {p_2[idx]:4.2f}"                  )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             preds[idx]                              = {preds[idx]}"                     )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names                             = {class_names}"                    )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names                             = {class_names[1]}"                 )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names                             = {class_names[2]}"                 )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names[preds[idx]]                 = {class_names[preds[idx]]}"        )
          print ( f"TRAINLENEJ:     INFO:      plot_classes_preds():             class_names[batch_labels[idx]]          = {class_names[batch_labels[idx]]}" )

        if args.just_test=='False':
          ax.set_title( "p_1={:<.4f}\n p_2={:<.4f}\n pred: {:}\ntruth: {:}".format( p_max[idx], p_2[idx], class_names[preds[idx]], class_names[batch_labels[idx]] ),
                      loc        = 'center',
                      pad        = None,
                      size       = 8,
                      color      = ( "green" if preds[idx]==batch_labels[idx].item() else "red") )
        else:
          if not IsBadTile:
            if preds[idx]==batch_labels[idx].item():
              ax.patch.set_edgecolor(class_colours[preds[idx]])
              if len(batch_labels)>threshold_3:
                ax.patch.set_linewidth('1')
              if len(batch_labels)>threshold_2:
                ax.patch.set_linewidth('2')
              elif len(batch_labels)>threshold_1:
                ax.patch.set_linewidth('3')
              else:
                ax.patch.set_linewidth('4')
            else:
              ax.patch.set_edgecolor('magenta')
              ax.patch.set_linestyle(':')
              if len(batch_labels)>threshold_3:
                ax.patch.set_linewidth('1')              
              if len(batch_labels)>threshold_2:
                ax.patch.set_linewidth('2')
              elif len(batch_labels)>threshold_1:
                ax.patch.set_linewidth('3')
              else:
                ax.patch.set_linewidth('6')

    print ( f"{RESET}")
    
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
    """Save PyTorch model's state dictionary
    """
    fpath = '%s/model.pt' % log_dir
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
            print( f"TRAINLENEJ:     INFO:   will delete file  '\r\033[43C{CYAN}{fqf}{RESET}'" )
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

    p.add_argument('--skip_preprocessing',            type=str,   default='False')                                # USED BY main() to enable user to skip tile generation
    p.add_argument('--skip_generation',               type=str,   default='False')                                # USED BY main() to enable user to skip torch database generation
    p.add_argument('--log_dir',                       type=str,   default='data/dlbcl_image/logs')                # used to store logs and to periodically save the model
    p.add_argument('--base_dir',                      type=str,   default='/home/peter/git/pipeline')             # NOT CURRENTLY USED
    p.add_argument('--data_dir',                      type=str,   default='/home/peter/git/pipeline/dataset')     # USED BY generate()
    p.add_argument('--save_model_name',               type=str,   default='model.pt')                             # USED BY main()
    p.add_argument('--save_model_every',              type=int,   default=10)                                     # USED BY main()    
    p.add_argument('--rna_file_name',                 type=str,   default='rna.npy')                              # USED BY generate()
    p.add_argument('--rna_file_suffix',               type=str,   default='*FPKM-UQ.txt' )                        # USED BY generate()
    p.add_argument('--rna_file_reduced_suffix',       type=str,   default='_reduced')                             # USED BY generate()
    p.add_argument('--class_numpy_file_name',         type=str,   default='class.npy')                            # USED BY generate()
    p.add_argument('--wall_time',                     type=int,   default=24)
    p.add_argument('--seed',                          type=int,   default=0)
    p.add_argument('--nn_mode',                       type=str,   default='dlbcl_image')
    p.add_argument('--nn_type',            nargs="+", type=str,   default='VGG11')
    p.add_argument('--dataset',                       type=str,   default='SARC')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--input_mode',                    type=str,   default='NONE')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--n_samples',          nargs="+", type=int,   default=101)                                    # USED BY generate()      
    p.add_argument('--n_tiles',            nargs="+", type=int,   default=100)                                    # USED BY generate() and all ...tiler() functions 
    p.add_argument('--supergrid_size',                type=int,   default=1)                                      # USED BY main()
    p.add_argument('--tile_size',          nargs="+", type=int,   default=128)                                    # USED BY many
    p.add_argument('--gene_data_norm',     nargs="+", type=str,   default='NONE')                                 # USED BY tiler()
    p.add_argument('--n_genes',                       type=int,   default=60482)                                  # USED BY generate()      
    p.add_argument('--batch_size',         nargs="+", type=int,   default=256)                                    # USED BY tiler() 
    p.add_argument('--learning_rate',      nargs="+", type=float, default=.00082)                                 # USED BY main()                               
    p.add_argument('--n_epochs',                      type=int,   default=10)
    p.add_argument('--pct_test',                      type=float, default=0.2)
    p.add_argument('--lr',                            type=float, default=0.0001)
    p.add_argument('--latent_dim',                    type=int,   default=7)
    p.add_argument('--l1_coef',                       type=float, default=0.1)
    p.add_argument('--em_iters',                      type=int,   default=1)
    p.add_argument('--clip',                          type=float, default=1)
    p.add_argument('--max_consecutive_losses',        type=int,   default=7771)
    p.add_argument('--optimizer',          nargs="+", type=str,   default='ADAM')
    p.add_argument('--label_swap_perunit',            type=int,   default=0)                                    
    p.add_argument('--make_grey_perunit',             type=float, default=0.0)                                    
    p.add_argument('--tensorboard_images',            type=str,   default='True')
    p.add_argument('--regenerate',                    type=str,   default='True')
    p.add_argument('--just_profile',                  type=str,   default='False')                                # USED BY tiler()    
    p.add_argument('--just_test',                     type=str,   default='False')                                # USED BY tiler()    
    p.add_argument('--rand_tiles',                    type=str,   default='True')                                 # USED BY tiler()      
    p.add_argument('--points_to_sample',              type=int,   default=100)                                    # USED BY tiler()
    p.add_argument('--min_uniques',                   type=int,   default=0)                                      # USED BY tiler()
    p.add_argument('--min_tile_sd',                   type=float, default=3)                                      # USED BY tiler()
    p.add_argument('--greyness',                      type=int,   default=0)                                      # USED BY tiler()
    p.add_argument('--stain_norm',         nargs="+", type=str,   default='NONE')                                 # USED BY tiler()
    p.add_argument('--stain_norm_target',             type=str,   default='NONE')                                 # USED BY tiler_set_target()
    p.add_argument('--use_tiler',                     type=str,   default='external'  )                           # USED BY main()
    p.add_argument('--cancer_type',                   type=str,   default='NONE'      )                           # USED BY main()
    p.add_argument('--cancer_type_long',              type=str,   default='NONE'      )                           # USED BY main()
    p.add_argument('--class_names',        nargs="+"                                  )                           # USED BY main()
    p.add_argument('--long_class_names',   nargs="+"                                  )                           # USED BY main()
    p.add_argument('--class_colours',      nargs="*"                                  )    
    p.add_argument('--target_tile_coords', nargs=2,   type=int, default=[2000,2000]       )                       # USED BY tiler_set_target()
        
    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 12
    args.pin_memory = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    main(args)
