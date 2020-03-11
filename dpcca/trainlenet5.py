"""============================================================================= 
Train LENET5
============================================================================="""

import sys
import time
import cuda
import pprint
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
#from matplotlib import figure

from   data                    import loader
from   data.dlbcl_image.config import GTExV6Config
from   data.dlbcl_image.generate_image import generate_image
from   models                  import LENETIMAGE
from   torch                   import optim
from   torch.nn.utils          import clip_grad_norm_
from   torch.nn                import functional as F
from   torch.nn                import DataParallel
from   itertools               import product

import torchvision
import torch.utils.data
from   torch.utils.tensorboard import SummaryWriter
from   torchvision    import datasets, transforms

DEBUG=1

np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=300)

# ------------------------------------------------------------------------------
    
LOG_EVERY        = 1
SAVE_MODEL_EVERY = 5

device = cuda.device()

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)


# constant for classes used in tensorboard images tab for the SARC dataset

classes = ('dediff. liposarcoma', 'leiomyosarcoma', 'myxofibrosarcoma', 'pleomorphic MFH', 'synovial', 'undiff. pleomorphic', 'MPNST', 'desmoid', 'giant cell MFH' )
# ------------------------------------------------------------------------------

def main(args):

  """Main program: train -> test once per epoch while saving samples as needed.
  """
  
  print( "TRAINLENEJ:     INFO: passed in arguments (may yet be over-ridden) are:\
 dataset=\033[36;1m{:}\033[m,\
 mode=\033[36;1m{:}\033[m,\
 nn=\033[36;1m{:}\033[m,\
 optimizer=\033[36;1m{:}\033[m,\
 batch_size=\033[36;1m{:}\033[m,\
 epochs=\033[36;1m{:}\033[m,\
 samples=\033[36;1m{:}\033[m,\
 genes=\033[36;1m{:}\033[m,\
 tiles_per_image=\033[36;1m{:}\033[m,\
 whiteness=\033[36;1m{:}\033[m,\
 greyness=\033[36;1m{:}\033[m,\
 latent_dim=\033[36;1m{:}\033[m,\
 label_swap=\033[36;1m{:}\033[m,\
 make_grey=\033[36;1m{:}\033[m,\
 colour_norm=\033[36;1m{:}\033[m,\
 tensorboard_images=\033[36;1m{:}\033[m,\
 max_consec_losses=\033[36;1m{:}\033[m"\
.format( args.dataset, args.input_mode, args.nn_type, args.optimizer, args.batch_size, args.n_epochs, args.n_samples, args.n_genes, args.n_tiles, args.whiteness, args.greyness, args.latent_dim, args.label_swap_perunit, args.make_grey_perunit, args.colour_norm, args.tensorboard_images, args.max_consecutive_losses  ), flush=True )
  dataset            = args.dataset
  input_mode         = args.input_mode
  nn_optimizer       = args.optimizer
  n_samples          = args.n_samples
  n_tiles            = args.n_tiles
  n_genes            = args.n_genes
  n_epochs           = args.n_epochs
  whiteness          = args.whiteness
  greyness           = args.greyness
  label_swap_perunit = args.label_swap_perunit
  make_grey_perunit  = args.make_grey_perunit
  colour_norm        = args.colour_norm
  tensorboard_images = args.tensorboard_images

  base_dir              = args.base_dir
  data_dir              = args.data_dir
  tile_size             = args.tile_size
  rna_file_name         = args.rna_file_name
  class_numpy_file_name = args.class_numpy_file_name
  regenerate            = args.regenerate
    
  print ( "torch       version =      {:}".format (  torch.__version__       )  )
  print ( "torchvision version =      {:}".format (  torchvision.__version__ )  ) 
  
  now = time.localtime(time.time())
  print(time.strftime("TRAINLENEJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
  start_time = time.time()

  pprint.set_logfiles( args.log_dir )

  # (A)  SET UP JOB LOOP

  #parameters = dict( lr=[.01, .001],  batch_size=[100, 1000],  shuffle=[True, False])
  parameters = dict(             lr =  [ .00082 ],
                          n_samples =  [  8, 16, 32, 64 ],
                         batch_size =  [   64  ],
                            nn_type =  [ 'VGG11' ],
                        nn_optimizer = [ 'ADAM'  ],
                  label_swap_perunit = [   0.0   ],
                   make_grey_perunit = [   0.0   ],
                              jitter = [  [ 0.0, 0.0, 0.0, 0.0 ] ]  )

  param_values = [v for v in parameters.values()]


  if DEBUG>0:
    print('TRAINLENEJ:     INFO: job level parameters  (learning rate,   n_samples, batch_size, nn_type, optimizer, label_swap_perunit, make_grey_perunit, jitter  ) = \033[36;1m{:}\033[m'.format( param_values ) )
  if DEBUG>9:
    print('TRAINLENEJ:     INFO: batch parameter - cartesian product ( learning rate x n_samples x batch_size x nn_type x optimizer x label_swap_perunit x make_grey_perunit x jitter ) =\033[35;1m')
    for lr,  n_samples, batch_size, nn_type, nn_optimizer, label_swap_perunit, make_grey_perunit, jitter in product(*param_values):  
      print( lr, n_samples, batch_size,  nn_type, nn_optimizer, label_swap_perunit, make_grey_perunit, jitter )

  # ~ for lr, batch_size  in product(*param_values): 
      # ~ comment = f' batch_size={batch_size} lr={lr}'

  run=0


  # (B) RUN JOB LOOP
  for lr, n_samples, batch_size, nn_type, nn_optimizer, label_swap_perunit, make_grey_perunit, jitter in product(*param_values): 
    
    run+=1

    if DEBUG>0:
      print( "\n\033[1;4mRUN  {:}\033[m          learning rate=\033[36;1m{:}\033[m  n_samples=\033[36;1m{:}\033[m  batch size=\033[36;1m{:}\033[m  nn_type=\033[36;1m{:}\033[m nn_optimizer=\033[36;1m{:}\033[m label swaps=\033[36;1m{:}\033[m make grey=\033[36;1m{:}\033[m, jitter=\033[36;1m{:}\033[m".format( run, lr,  n_samples, batch_size, nn_type, nn_optimizer, label_swap_perunit, make_grey_perunit, jitter) )
 
    # (0)

    if regenerate=='True':
      print( "TRAINLENEJ:     INFO: \033[1m0 about to regenerate '.pt' file from dataset\033[m\n" )
      generate_image( args, n_samples )

    # (1)

    print( "TRAINLENEJ:     INFO: \033[1m1 about to load experiment config\033[m" )
  
#    pprint.log_section('Loading config.')
    cfg = loader.get_config( args.nn_mode, lr, batch_size )                                                # PGD 200302 - The arguments aren't currently used
    GTExV6Config.INPUT_MODE         = input_mode                                                           # modify config class variable to take into account user preference
    GTExV6Config.MAKE_GREY          = make_grey_perunit                                                    # modify config class variable to take into account user preference
    GTExV6Config.JITTER             = jitter                                                               # modify config class variable to take into account user preference
#    pprint.log_config(cfg)
#    pprint.log_section('Loading script arguments.')
#    pprint.log_args(args)
  
    print( "TRAINLENEJ:     INFO:   \033[3mexperiment config loaded\033[m" )
   
    
    #(2)
    print( "TRAINLENEJ:     INFO: \033[1m2 about to load LENET5 model\033[m with parameters: args.latent_dim=\033[36;1m{:}\033[m, args.em_iters=\033[36;1m{:}\033[m".format( args.latent_dim, args.em_iters) ) 
    model = LENETIMAGE(cfg, nn_type, args.latent_dim, args.em_iters )            # yields model.image_net() = e.g model.VGG11() (because model.get_image_net() in config returns the e.g. VGG11 class)
 
    #if torch.cuda.device_count()==2:                                             # for Dreedle, which has two bridged Titan RTXs
     # model = DataParallel(model, device_ids=[0, 1])

###    traced_model = torch.jit.trace(model.eval(), torch.rand(10), model.eval())                                                     
    print( "TRAINLENEJ:     INFO:    \033[3mmodel loaded\033[m" )  
   
    #(3)
    print( "TRAINLENEJ:     INFO: \033[1m3 about send model to device\033[m" )   
    model = model.to(device)
    print( "TRAINLENEJ:     INFO:   \033[3mmodel sent to device\033[m" ) 
  
    pprint.log_section('Model specs.')
    pprint.log_model(model)
     
    
    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:   pytorch Model = {:}".format(model))
    
    GTExV6Config.LABEL_SWAP_PERUNIT = label_swap_perunit
    #(4)
    print( "TRAINLENEJ:     INFO: \033[1m4 about to call dataset loader\033[m with parameters: cfg=\033[36;1m{:}\033[m, batch_size=\033[36;1m{:}\033[m, args.n_worker=\033[36;1m{:}\033[m, args.pin_memory=\033[36;1m{:}\033[m, args.cv_pct=\033[36;1m{:}\033[m".format( cfg, batch_size, args.n_workers, args.pin_memory, args.cv_pct) )
    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.cv_pct
                                                        )
                                                        
    print( "TRAINLENEJ:     INFO:   \033[3mdataset loaded\033[m" )
  
    pprint.save_test_indices(test_loader.sampler.indices)
  
    #(5)  
    print( "TRAINLENEJ:     INFO: \033[1m5 about to select and configure optimizer\033[m with learning rate = \033[35;1m{:}\033[m".format( lr ) )
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
 
         
    #(6)
    print( "TRAINLENEJ:     INFO: \033[1m6 about to select CrossEntropyLoss function\033[m" )  
    loss_function = torch.nn.CrossEntropyLoss()
      
    print( "TRAINLENEJ:     INFO:   \033[3mCross Entropy loss function selected\033[m" )  
    
    #(7)
    print( "TRAINLENEJ:     INFO: \033[1m7 about to set up Tensorboard\033[m" )
    if input_mode=='image':
      writer = SummaryWriter(comment=f' dataset={dataset}; mode={input_mode}; NN={nn_type}; opt={nn_optimizer}; n_samples={n_samples}; tiles_per_image={n_tiles}; total_tiles={n_tiles * n_samples}; n_epochs={n_epochs}; batch={batch_size}; color_norm={colour_norm};  white<{whiteness}; grey>{greyness};  lr={lr}; lbl_swp={label_swap_perunit*100}%; greyscale={make_grey_perunit*100}% jit={jitter}%' )
    elif input_mode=='rna':
      writer = SummaryWriter(comment=f' dataset={dataset}; mode={input_mode}; NN={nn_type}; opt={nn_optimizer}; n_samples={n_samples}; n_genes={n_genes}; n_epochs={n_epochs}; batch={batch_size}; lr={lr}')
    else:
      print( "TRAINLENEJ:     FATAL:    input of type '{:}' is not supported".format( nn_type ) )
      sys.exit(0)

    number_correct_max   = 0
    pct_correct_max      = 0
    test_loss_min        = 999999
    train_loss_min       = 999999
    print( "TRAINLENEJ:     INFO:   \033[3mTensorboard has been set up\033[m" ) 
    
    
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
   
    #(8)
                     
    print( "TRAINLENEJ:     INFO: \033[1m8 about to commence training loop, one iteration per epoch\033[m" )
  
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
  
        print('TRAINLENEJ:     INFO:   epoch: \033[35;1m{:}\033[m, batch size: \033[35;1m{:}\033[m.  \033[38;2;140;140;140mWill save best model and halt when test loss increases for \033[35;1m{:}\033[m \033[38;2;140;140;140mconsecutive epochs'.format( epoch, batch_size, args.max_consecutive_losses ) )
    
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
            print ( "\r\033[1C\033[2K\033[38;2;140;140;140m                          train():\r\033[47Closs_images=\r\033[59C{0:.4f}   loss_unused=\r\033[85C{1:.4f}   l1_loss=\r\033[102C{2:.4f}   BATCH AVG =\r\033[122C\033[38;2;0;127;0m{3:9.4f}   \033[38;2;140;140;140mlowest total loss=\r\033[153C{4:.4f} at epoch {5:2d}    lowest image loss=\r\033[195C{6:.4f} at epoch {7:2d}\033[m".format(        train_loss1_sum_ave, train_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave, train_lowest_total_loss_observed, train_lowest_total_loss_observed_epoch, train_lowest_image_loss_observed, train_lowest_image_loss_observed_epoch ), end=''  )
          else:
            last_epoch_loss_increased = True
            print ( "\r\033[1C\033[2K\033[38;2;140;140;140m                          train():\r\033[47Closs_images=\r\033[59C{0:.4f}   loss_unused=\r\033[85C{1:.4f}   l1_loss=\r\033[102C{2:.4f}   BATCH AVG =\r\033[122C\033[38;2;127;83;0m{3:9.4f}\033[m   \033[38;2;140;140;140mlowest total loss=\r\033[153C{4:.4f} at epoch {5:2d}    lowest image loss=\r\033[195C{6:.4f} at epoch {7:2d}\033[m".format( train_loss1_sum_ave, train_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave, train_lowest_total_loss_observed, train_lowest_total_loss_observed_epoch, train_lowest_image_loss_observed, train_lowest_image_loss_observed_epoch ), end='' )
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
                                                                               test ( cfg, args, epoch, test_loader,  model,  loss_function, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, nn_type, tensorboard_images )
  
        if test_total_loss_ave < test_lowest_total_loss_observed:
          test_lowest_total_loss_observed       = test_total_loss_ave
          test_lowest_total_loss_observed_epoch = epoch
  
        if test_loss1_sum_ave < test_lowest_image_loss_observed:
          test_lowest_image_loss_observed       = test_loss1_sum_ave
          test_lowest_image_loss_observed_epoch = epoch
  
        if DEBUG>0:
          if ( (test_total_loss_ave < (test_total_loss_ave_last)) | (epoch==1) ):
            consecutive_test_loss_increases = 0
            last_epoch_loss_increased = False
            print ( "\r\033[1C\033[2K                           test():\r\033[47Closs_images=\r\033[59C{0:.4f}   loss_unused=\r\033[85C{1:.4f}   l1_loss=\r\033[102C{2:.4f}\033[m   BATCH AVG =\r\033[122C\033[38;2;0;255;0m{3:9.4f}\033[m   lowest TEST loss =\r\033[153C{4:.4f} at epoch {5:2d}\033[m    \033[38;2;140;140;140mlowest image loss=\r\033[195C{6:.4f} at epoch {7:2d}\033[m".format(       test_loss1_sum_ave, test_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave, test_lowest_total_loss_observed, test_lowest_total_loss_observed_epoch, test_lowest_image_loss_observed, test_lowest_image_loss_observed_epoch ), end = '' )
          else:
            last_epoch_loss_increased = True
            print ( "\r\033[1C\033[2K                           test():\r\033[47Closs_images=\r\033[59C{0:.4f}   loss_unused=\r\033[85C{1:.4f}\033[m   l1_loss=\r\033[102C{2:.4f}\033[m   BATCH AVG =\r\033[122C\033[38;2;255;0;0m{3:9.4f}\033[m   lowest TEST loss =\r\033[153C{4:.4f} at epoch {5:2d}\033[m    \033[38;2;140;140;140mlowest image loss=\r\033[195C{6:.4f} at epoch {7:2d}\033[m".format( test_loss1_sum_ave, test_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave, test_lowest_total_loss_observed, test_lowest_total_loss_observed_epoch, test_lowest_image_loss_observed, test_lowest_image_loss_observed_epoch), end = '')
            if last_epoch_loss_increased == True:
              consecutive_test_loss_increases +=1
              if consecutive_test_loss_increases == 1:
                print ( "\033[38;2;255;0;0m <<< test loss increased\033[m", end='' )
              else:
                print ( "\033[38;2;255;0;0m <<< {0:2d} consecutive test loss increases !!!\033[m".format( consecutive_test_loss_increases ), end='')
              print ( '')

              if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before SAVE_MODEL_EVERY so that the most recent model for which the loss improved will be saved
                  now = time.localtime(time.time())
                  print(time.strftime("TRAINLENEJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
                  sys.exit(0)
          
          if (last_epoch_loss_increased == False):
            print ('')
  
        test_total_loss_ave_last = test_total_loss_ave
  
  #        if epoch % LOG_EVERY == 0:
  #            if DEBUG>0:
  #              print( "TRAINLENEJ:     INFO:   saving samples to \033[35;1m{:}\033[m".format( args.log_dir ) )
  #            save_samples(args.log_dir, model, test_loader, cfg, epoch)
        if epoch % (args.max_consecutive_losses + 1) == 0:
            if DEBUG>0:
              print( "TRAINLENEJ:     INFO:   saving model   to \033[35;1m{:}\033[m".format( args.log_dir ) )
            save_model(args.log_dir, model)
            
    print( "TRAINLENEJ:     INFO: training complete \033[33;1mdone\033[m" )
  
    hours   = round((time.time() - start_time) / 3600, 1  )
    minutes = round((time.time() - start_time) / 60,   1  )
    pprint.log_section('Job complete in {:} mins'.format( minutes ) )
  
    print('TRAINLENEJ:     INFO: run completed in {:} mins'.format( minutes ) )
    
    writer.close()                                                                                         # PGD 200206
    
    save_model(args.log_dir, model)
    pprint.log_section('Model saved.')
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
          print ( "\033[2K                          train():     \033[38;2;140;140;140mn=\r\033[41C{0:2d}    loss_images=\r\033[59C{1:.4f}   l1_loss=\r\033[102C{2:.4f}   BATCH AVE =\r\033[122C{2:.4f}\033[m".format( i+1, loss_images_value, l1_loss, total_loss ))
          print ( "\033[2A" )
          
        loss_images.backward()
        # Perform gradient clipping *before* calling `optimizer.step()`.
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss1_sum      += loss_images_value
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss     

        del loss_images
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






def test( cfg, args, epoch, test_loader, model, loss_function, writer, number_correct_max, pct_correct_max, test_loss_min, batch_size, nn_type, tensorboard_images ):

    """Test model by pusing a held out batch through the network
    """

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to test model by computing the average loss on a held-out dataset. No parameter updates" )

    model.eval()                                                                                           # set model to evaluation mod

    loss1_sum      = 0
    loss2_sum      = 0
    l1_loss_sum    = 0
    total_loss_sum = 0

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to enumerate  " )
      
    for i, (batch_images, batch_labels) in enumerate(test_loader):

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        if DEBUG>9:
          print ( "TRAINLENEJ:     INFO:      test():       type(batch_images)                      = {:}".format( type(batch_images)       ) )
          print ( "TRAINLENEJ:     INFO:      test():       batch_images.shape                      = {:}".format( batch_images.shape       ) )

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     test(): about to call \033[33;1mmodel.forward()\033[m" )

        with torch.no_grad():                                                                             # PGD 200129 - Don't need gradients for testing, so this should save some GPU memory (tested: it does)
          y1_hat = model.forward( batch_images )                                                          

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
          print ( "\033[2K                           test():      \033[38;2;140;140;140ms=\r\033[41C{0:2d}    loss_images=\r\033[59C{1:.4f}\033[m  l1_loss=\r\033[102C{2:.4f}\033[m   BATCH AVE =\r\033[122C\033[38;2;255;255;0m{3:.4f}\033[m".format( i+1, loss_images_value, l1_loss, total_loss ))
          print ( "\033[2A" )
          
        loss1_sum      += loss_images_value                                                                # use .item() to extract just the value: don't create a new tensor
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss  

        del loss_images
        torch.cuda.empty_cache()

    if epoch % 5 == 0:
      y1_hat_values             = y1_hat.cpu().detach().numpy()
      y1_hat_values_max_indices = np.argmax( np.transpose(y1_hat_values), axis=0 )
      batch_labels_values       = batch_labels.cpu().detach().numpy()
      
      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      test():        y1_hat.shape                      = {:}".format( y1_hat.shape                     ) )
        print ( "TRAINLENEJ:     INFO:      test():        y1_hat_values_max_indices.shape   = {:}".format( y1_hat_values_max_indices.shape  ) )
        print ( "TRAINLENEJ:     INFO:      test():        batch_labels_values.shape         = {:}".format( batch_labels_values.shape        ) )
      
      number_to_display=44
      print ( "" )
      print ( "TRAINLENEJ:     INFO:     test(): truth/prediction for first few examples from the last test batch (number correct = \u001b[4m{:}\033[m/{:})".format(np.sum( np.equal(y1_hat_values_max_indices, batch_labels_values)), batch_labels_values.shape[0] )   )
      np.set_printoptions(formatter={'int': lambda x: "{:4d}".format(x)})
      print (  batch_labels_values[0:number_to_display]          ) 
      print (  y1_hat_values_max_indices[0:number_to_display]    )

      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      test():       y1_hat.shape                     = {:}".format( y1_hat_values.shape          ) )
        np.set_printoptions(formatter={'float': lambda x: "{0:10.2e}".format(x)})
        print (  "{:}".format( (np.transpose(y1_hat_values))[:,:number_to_display] )  )
        np.set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})

      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      test():       FIRST  GROUP BELOW: y1_hat"                                                                      ) 
        print ( "TRAINLENEJ:     INFO:      test():       SECOND GROUP BELOW: batch_labels_values"                                                         )
        print ( "TRAINLENEJ:     INFO:      test():       THIRD  GROUP BELOW: y1_hat_values_max_indices"                                                   )
        np.set_printoptions(formatter={'float': '{: >9.2f}'.format}        )
        print ( "{:}".format( (np.transpose(y1_hat_values)) [:5,:25]     ) )
        np.set_printoptions(formatter={'int': '{: >9d}'.format}            )
        print ( " {:}".format( batch_labels_values          [:25]        ) )
        print ( " {:}".format( y1_hat_values_max_indices    [:25]        ) )
 
 
    y1_hat_values               = y1_hat.cpu().detach().numpy()
    y1_hat_values_max_indices   = np.argmax( np.transpose(y1_hat_values), axis=0  )
    batch_labels_values         = batch_labels.cpu().detach().numpy()
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
      
    if GTExV6Config.INPUT_MODE=='image':
        writer.add_figure('Predictions v Truth', plot_classes_preds(model, batch_images, batch_labels),  epoch)

    if DEBUG>99:
      print ( "TRAINLENEJ:     INFO:      test():       type(loss1_sum_ave)                      = {:}".format( type(loss1_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(loss2_sum_ave)                      = {:}".format( type(loss2_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(l1_loss_sum_ave)                    = {:}".format( type(l1_loss_sum_ave)   ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(total_loss_ave)                     = {:}".format( type(total_loss_ave)    ) )

    return loss1_sum_ave, loss2_sum_ave, l1_loss_sum_ave, total_loss_ave, number_correct_max, pct_correct_max, test_loss_min



# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def imshow(img):

    '''
    helper function to show an image
    
    From: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    '''

    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# ------------------------------------------------------------------------------
def images_to_probs(model, images):
    '''
    Generates predictions and corresponding probabilities from a trained network and a list of images
    
    From: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    '''

    with torch.no_grad():
      y1_hat = model.forward( images )

    y1_hat_numpy = (y1_hat.cpu().data).numpy()

    if DEBUG>9:
      y1_hat_numpy = (y1_hat.cpu().data).numpy()
      print ( "TRAINLENEJ:     INFO:      images_to_probs():       type(y1_hat)                      = {:}".format( type(y1_hat)          ) )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():       type(y1_hat_numpy)                = {:}".format( type(y1_hat_numpy)    ) )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():       y1_hat.shape                      = {:}".format( y1_hat_numpy.shape    ) )
      if DEBUG>99:
        print ( "TRAINLENEJ:     INFO:      images_to_probs():       y1_hat                            = \n{:}".format( y1_hat_numpy) )

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(y1_hat, axis=1)

    if DEBUG>9:
      y1_hat_numpy = (y1_hat.cpu().data).numpy()
      print ( "TRAINLENEJ:     INFO:      images_to_probs():             preds_tensor.shape           = {:}".format( preds_tensor.shape    ) ) 
      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      images_to_probs():             preds_tensor                 = \n{:}".format( preds_tensor      ) ) 
    
    preds = np.squeeze(preds_tensor.cpu().numpy())

    if DEBUG>9:
      print ( "TRAINLENEJ:     INFO:      images_to_probs():             type(preds)                  = {:}".format( type(preds)           ) )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():             preds.shape                  = {:}".format( preds.shape           ) ) 
      if DEBUG>9:
        print ( "TRAINLENEJ:     INFO:      images_to_probs():       FIRST  GROUP BELOW: preds"            ) 
        print ( "TRAINLENEJ:     INFO:      images_to_probs():       SECOND GROUP BELOW: y1_hat_numpy.T"   )
        np.set_printoptions(formatter={'int':   lambda x: "\033[1m{:^10d}\033[m".format(x)    }    )
        print ( preds[0:22] )
        np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
        print (  np.transpose(y1_hat_numpy[0:22,:])  )

  
    p_max  = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, y1_hat)]      # regarding the -1 dimension, see https://stackoverflow.com/questions/59704538/what-is-a-dimensional-range-of-1-0-in-pytorch

    # extract the SECOND HIGHEST probability for each example (which is a bit trickier)
    sm = F.softmax( y1_hat, dim=1).cpu().numpy()
    p_2 = np.zeros((len(preds)))
    for i in range (0, len(p_2)):
      p_2[i] = max( [ el for el in sm[i,:] if el != max(sm[i,:]) ] )     
      
                             
    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():            type(sm)                   = {:}".format( type(sm) )  )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():             sm                         = \n{:}".format( np.transpose(sm[0:22,:])   )  )
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():             p2              = \n{:}".format( p2   )  )                       
    
    if DEBUG>9:
      np.set_printoptions(formatter={'float': lambda x: "{0:10.4f}".format(x) }    )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():             p_max.shape                = {:}".format( (np.array(p_max)).shape )  )
      print ( "TRAINLENEJ:     INFO:      images_to_probs():             p_max                      = \n{:}".format( np.array(p_max[0:22]) )  )
   
    return preds, p_max, p_2


# ------------------------------------------------------------------------------
def plot_classes_preds(model, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images and labels from a batch, that shows the network's top prediction along with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not. Uses the "images_to_probs" function. 
    
    From: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    '''
    
    preds, p_max, p_2 = images_to_probs( model, images )

    number_to_plot = len(labels)    
    figure_width   = 15
    figure_height  = int(number_to_plot * .4)
        
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure( figsize=( figure_width, figure_height ) )                                         # overall size ( width, height ) in inches

    if DEBUG>9:
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

    for idx in np.arange( number_to_plot-1 ):

        ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[])            # nrows, ncols, "index starts at 1 in the upper left corner and increases to the right", List of x-axis tick locations, List of y-axis tick locations
        ax.set_frame_on( False )

        imshow( images[idx] )

        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:}".format( idx ) )
          print ( "TRAINLENEJ:     INFO:      plot_classes_preds():  idx={:} p_max[idx] = {:4.2f}, classes[preds[idx]] = {:<20s}, classes[labels[idx]] = {:<20s}".format( idx, p_max[idx], classes[preds[idx]], classes[labels[idx]]  ) )

        ax.set_title( "p_1={:<.4f}\n p_2={:<.4f}\n pred: {:}\ntruth: {:}".format( p_max[idx], p_2[idx], classes[preds[idx]], classes[labels[idx]] ),
                      loc        = 'center',
                      pad        = None,
                      size       = 8,
                      color      = ( "green" if preds[idx]==labels[idx].item() else "red") )

    fig.tight_layout( rect=[0, 0.03, 1, 0.95] )


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
    """Save PyTorch model's state dictionary for provenance.
    """
    fpath = '%s/model.pt' % log_dir
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--log_dir',                type=str,   default='data/dlbcl_image/logs')                # used to store logs and to periodically save the model
    p.add_argument('--base_dir',               type=str,   default='/home/peter/git/pipeline')             # NOT CURRENTLY USED
    p.add_argument('--data_dir',               type=str,   default='/home/peter/git/pipeline/dataset')     # USED BY GENERATE()
    p.add_argument('--rna_file_name',          type=str,   default='rna.npy')                              # USED BY GENERATE()
    p.add_argument('--class_numpy_file_name',  type=str,   default='class.npy')                            # USED BY GENERATE()
    p.add_argument('--wall_time',              type=int,   default=24)
    p.add_argument('--seed',                   type=int,   default=0)
    p.add_argument('--nn_mode',                type=str,   default='dlbcl_image')
    p.add_argument('--nn_type',                type=str,   default='VGG11')
    p.add_argument('--dataset',                type=str,   default='SARC')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--input_mode',             type=str,   default='NONE')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--n_samples',              type=int,   default=105)                                    # USED BY GENERATE() and for Tensorboard
    p.add_argument('--n_tiles',                type=int,   default=100)                                    # USED BY GENERATE() and for Tensorboard
    p.add_argument('--tile_size',              type=int,   default=128)                                    # USED BY GENERATE() and for Tensorboard                                    
    p.add_argument('--n_genes',                type=int,   default=60482)                                  # USED BY GENERATE() and for Tensorboard
    p.add_argument('--batch_size',             type=int,   default=256)                                         
    p.add_argument('--n_epochs',               type=int,   default=10)
    p.add_argument('--cv_pct',                 type=float, default=0.1)
    p.add_argument('--lr',                     type=float, default=0.0001)
    p.add_argument('--latent_dim',             type=int,   default=7)
    p.add_argument('--l1_coef',                type=float, default=0.1)
    p.add_argument('--em_iters',               type=int,   default=1)
    p.add_argument('--clip',                   type=float, default=1)
    p.add_argument('--max_consecutive_losses', type=int,   default=7771)
    p.add_argument('--optimizer',              type=str,   default='ADAM')
    p.add_argument('--greyness',               type=int,   default=0)                                      # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--whiteness',              type=float, default=0.1)                                    # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--label_swap_perunit',     type=int,   default=0)                                    
    p.add_argument('--make_grey_perunit',      type=int,   default=0)                                    
    p.add_argument('--colour_norm',            type=str,   default='NONE')                                 # taken in as an argument so that it can be used as a label in Tensorboard
    p.add_argument('--tensorboard_images',     type=str,   default='True')
    p.add_argument('--regenerate',             type=str,   default='True')
 
    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 4
    args.pin_memory = torch.cuda.is_available()

    # For easy debugging locally.
    if is_local:
        LOG_EVERY        = 1
        SAVE_MODEL_EVERY = 20

    torch.manual_seed(args.seed)
    main(args)
