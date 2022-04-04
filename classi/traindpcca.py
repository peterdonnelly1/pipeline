"""=============================================================================
Train deep probabilistic CCA (DPCCJ).
============================================================================="""

import sys
import time
import cuda
import pprint
import argparse
import numpy as np

from modes              import loader
from  models            import DPCCA

# ~ from PIL                import ImageTk
from PIL                import Image
from PIL                import ImageFont
from PIL                import ImageDraw

import torch
from   torch          import optim
import torch.utils.data
from   torch.nn.utils import clip_grad_norm_
from   torch.nn       import functional as F

import torchvision
from   torch.utils.tensorboard import SummaryWriter
from   torchvision import datasets, transforms

from constants  import *

DEBUG=1

np.set_printoptions(edgeitems=8)
np.set_printoptions(linewidth=200)

# ------------------------------------------------------------------------------

LOG_EVERY        = 1

device = cuda.device()

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)

# ------------------------------------------------------------------------------

def main(args):
    """Main program: train -> test once per epoch while saving samples as needed.
    """

    lr=args.lr
    batch_size=args.batch_size


    now = time.localtime(time.time())
    print(time.strftime("TRAINDPCCJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
    print("TRAINDPCCJ:     Torch       version = {:}".format( torch.__version__ )        )
    print("TRAINDPCCJ:     Torchvision version = {:}".format( torchvision.__version__ )  )

    start_time = time.time()
    pprint.set_logfiles( args.log_dir, descriptor )

    torch.cuda.empty_cache()                # PGD 200128 - for CUDA memory optimizations
    torch.backends.cudnn.benchmark = True   # PGD 200128 - for CUDA memory optimizations
    torch.backends.cudnn.enabled   = True   # PGD 200128 - for CUDA memory optimizations
    
    pprint.log_section('Loading config.')

    print( "TRAINDPCCJ:     INFO: passed in arguments are:\
 mode=\033[35;1m{:}\033[m,\
 batch_size=\033[35;1m{:}\033[m,\
 n_epochs=\033[35;1m{:}\033[m,\
 latent_dim=\033[35;1m{:}\033[m,\
 max_consecutive_losses=\033[35;1m{:}\033[m"\
#.format( args.dataset, args.batch_size, args.n_epochs, args.latent_dim, args.max_consecutive_losses), flush=True )
.format( args.mode, args.batch_size, args.n_epochs, args.latent_dim, args.max_consecutive_losses), flush=True )
    # (1)
    print( "TRAINDPCCJ:     INFO: \033[1m1 about to load experiment configuration parameter\033[m" )
    #cfg = loader.get_config(args.dataset)
    cfg = loader.get_config( args.mode, lr, batch_size )    
    print( "TRAINDPCCJ:     INFO:   cfg = \033[35;1m{:}\033[m".format( cfg ) )                                                         

    pprint.log_config(cfg)
    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    print( "TRAINDPCCJ:     INFO:   experiment config loaded\033[m" )
    
    #(2)
    print( "TRAINDPCCJ:     INFO: \033[1m2 about to load DPCCJ model\033[m with parameters: args.latent_dim=\033[35;1m{:}\033[m, args.em_iters=\033[35;1m{:}\033[m".format( args.latent_dim, args.em_iters) )                                                         
    model = DPCCA( cfg, args.latent_dim, args.em_iters )                                                   # model is a DPCCA object (nn.Module)
    print( "TRAINDPCCJ:     INFO:   model loaded\033[m" ) 
 
 
    #(3) 
    print( "TRAINDPCCJ:     INFO: \033[1m3 about send model to device\033[m" )   
    model = model.to(device)
    print( "TRAINDPCCJ:     INFO:   model sent to device\033[m" ) 

    pprint.log_section('Model specs.')
    pprint.log_model(model)  
    
    if DEBUG>9:
      print( "TRAINDPCCJ:     INFO: Pytorch Model = {:}".format(model))
    
    #(4)
    print( "TRAINDPCCJ:     INFO: \033[1m4 about to load the dataset\033[m, with parameters: cfg=\033[35;1m{:}\033[m, args.batch_size=\033[35;1m{:}\033[m, args.n_worker=\033[35;1m{:}\033[m, args.pin_memory=\033[35;1m{:}\033[m, args.pct_test=\033[35;1m{:}\033[m".format( cfg, args.batch_size, args.n_workers, args.pin_memory, args.pct_test) )
    train_loader, test_loader = loader.get_data_loaders(args,
                                                        cfg,
                                                        args.batch_size,
                                                        n_samples,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.pct_test)
                                                        
    print( "TRAINDPCCJ:     INFO:   dataset loaded\033[m" )

    pprint.save_test_indices(test_loader.sampler.indices)

    #(5)
    print( "TRAINDPCCJ:     INFO: \033[1m5 about to select and configure Adam optimizer\033[m, with learning rate = \033[35;1m{:}\033[m".format( args.lr ) )  
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print( "TRAINDPCCJ:     INFO:   Adam optimizer selected and configured\033[m" )
    
    pprint.log_section('Training model.\n\n'\
                       'Epoch\t\tTrain x1 err\tTrain x2 err\tTrain l1\t'\
                       '\tTest x1 err\tTest x2 err\tTest l1')
 
    #(6)
    torch.cuda.empty_cache()      # PGD
    print( "TRAINDPCCJ:     INFO: \033[1m6 about to commence training loop, one iteration per epoch\033[m" )

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
    
    for epoch in range(1, args.n_epochs + 1):

        print('TRAINDPCCJ:     INFO:   epoch: \033[35;1m{:}\033[m, batch size: \033[35;1m{:}\033[m.  Will save best model and halt when test set total loss increases for \033[35;1m{:}\033[m consecutive epochs'.format( epoch, args.batch_size, args.max_consecutive_losses ) )
    
        if DEBUG>1:
          print('TRAINDPCCJ:     INFO:   7.1 running training step ')
  
        # PUSH ONE MINI-BATCH THROUGH TRAINING
        torch.cuda.empty_cache()      # PGD
        train_ae_loss1_sum_ave, train_ae_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave = train (      args,        train_loader, model, optimizer )

        if train_total_loss_ave < train_lowest_total_loss_observed:
          train_lowest_total_loss_observed       = train_total_loss_ave
          train_lowest_total_loss_observed_epoch = epoch

        if train_ae_loss1_sum_ave < train_lowest_image_loss_observed:
          train_lowest_image_loss_observed       = train_ae_loss1_sum_ave
          train_lowest_image_loss_observed_epoch = epoch

        if DEBUG>0:
          if ( (train_total_loss_ave < train_total_loss_ave_last) | (epoch==1) ):
            consecutive_training_loss_increases = 0
            last_epoch_loss_increased = False
            print ( "TRAINDPCCJ:     INFO:     train():\r\033[47Cae_loss_images=\r\033[62C\033[38;2;140;140;140m{0:.2f}\033[m   ae_loss_genes=\r\033[85C\033[38;2;140;140;140m{1:.2f}\033[m   \r\033[100Cl1_loss=\033[38;2;140;140;140m{2:.2f}\033[m   \r\033[120CTOTAL LOSS=\033[38;2;0;255;0m{3:9.2f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.2f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.2f} at epoch {7:2d}\033[m".format( train_ae_loss1_sum_ave, train_ae_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave, train_lowest_total_loss_observed, train_lowest_total_loss_observed_epoch, train_lowest_image_loss_observed, train_lowest_image_loss_observed_epoch ) )
          else:
            last_epoch_loss_increased = True
            print ( "TRAINDPCCJ:     INFO:     train():\r\033[47Cae_loss_images=\r\033[62C\033[38;2;140;140;140m{0:.2f}\033[m   ae_loss_genes=\r\033[85C\033[38;2;140;140;140m{1:.2f}\033[m   \r\033[100Cl1_loss=\033[38;2;140;140;140m{2:.2f}\033[m   \r\033[120CTOTAL LOSS=\033[38;2;255;165;0m{3:9.2f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.2f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.2f} at epoch {7:2d}\033[m".format( train_ae_loss1_sum_ave, train_ae_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave, train_lowest_total_loss_observed, train_lowest_total_loss_observed_epoch, train_lowest_image_loss_observed, train_lowest_image_loss_observed_epoch ) )
            if last_epoch_loss_increased == True:
              consecutive_training_loss_increases +=1
              if consecutive_training_loss_increases == 1:
                print ( "TRAINDPCCJ:     \033[38;2;255;165;0mNOTE:\033[m     train():              \033[38;2;255;165;0mtotal training loss increased\033[m" )
              else:
                print ( "TRAINDPCCJ:     \033[38;2;255;165;0mNOTE:\033[m     train():             \033[38;2;255;165;0m{0:2d} consecutive training loss increases (s) !!!\033[m".format( consecutive_training_loss_increases ) )

        train_total_loss_ave_last = train_total_loss_ave



        if DEBUG>1:
          print('TRAINDPCCJ:     INFO:   7.2 running test step ')
  
        # TEST THE SPECIFIED NUMBER OF EXAMPLES
        torch.cuda.empty_cache()      # PGD  
        test_ae_loss1_sum_ave, test_ae_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave     = test  ( cfg, args, epoch, test_loader,  model            )

        if test_total_loss_ave < test_lowest_total_loss_observed:
          test_lowest_total_loss_observed       = test_total_loss_ave
          test_lowest_total_loss_observed_epoch = epoch

        if test_ae_loss1_sum_ave < test_lowest_image_loss_observed:
          test_lowest_image_loss_observed       = test_ae_loss1_sum_ave
          test_lowest_image_loss_observed_epoch = epoch

        if DEBUG>0:
          if ( (test_total_loss_ave < test_total_loss_ave_last) | (epoch==1) ):
            consecutive_test_loss_increases = 0
            last_epoch_loss_increased = False
            print ( "TRAINDPCCJ:     INFO:      test():\r\033[47Cae_loss_images=\r\033[62C\033[38;2;140;140;140m{0:.2f}\033[m   ae_loss_genes=\r\033[85C\033[38;2;140;140;140m{1:.2f}\033[m   \r\033[100Cl1_loss=\033[38;2;140;140;140m{2:.2f}\033[m   \r\033[120CTOTAL LOSS=\033[38;2;0;255;0m{3:9.2f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.2f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.2f} at epoch {7:2d}\033[m".format( test_ae_loss1_sum_ave, test_ae_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave, test_lowest_total_loss_observed, test_lowest_total_loss_observed_epoch, test_lowest_image_loss_observed, test_lowest_image_loss_observed_epoch ) )
          else:
            last_epoch_loss_increased = True
            print ( "TRAINDPCCJ:     INFO:      test():\r\033[47Cae_loss_images=\r\033[62C\033[38;2;140;140;140m{0:.2f}\033[m   ae_loss_genes=\r\033[85C\033[38;2;140;140;140m{1:.2f}\033[m   \r\033[100Cll_loss=\033[38;2;140;140;140m{2:.2f}\033[m   \r\033[120CTOTAL LOSS=\033[38;2;255;0;0m{3:9.2f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.2f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.2f} at epoch {7:2d}\033[m".format( test_ae_loss1_sum_ave, test_ae_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave, test_lowest_total_loss_observed, test_lowest_total_loss_observed_epoch, test_lowest_image_loss_observed, test_lowest_image_loss_observed_epoch))
            if last_epoch_loss_increased == True:
              consecutive_test_loss_increases +=1
              if consecutive_test_loss_increases == 1:
                print ( "TRAINDPCCJ:     \033[38;2;255;0;0mNOTE:\033[m      test():              \033[38;2;255;0;0mtotal test loss increased\033[m" )
              else:
                print ( "TRAINDPCCJ:     \033[38;2;255;0;0mNOTE:\033[m      test():             \033[38;2;255;0;0m{0:2d} consecutive test loss increase(s) !!!\033[m".format( consecutive_test_loss_increases ) )
              if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before SAVE_MODEL_EVERY so that the most recent model for which the loss improved will be saved
                  now = time.localtime(time.time())
                  print(time.strftime("TRAINDPCCJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
                  sys.exit(0)
  
        test_total_loss_ave_last = test_total_loss_ave
        
        

        if epoch % LOG_EVERY == 0:
            if DEBUG>0:
              print( "TRAINDPCCJ:     INFO:   saving samples to \033[35;1m{:}\033[m".format( args.log_dir ) )
            save_samples(args.log_dir, model, test_loader, cfg, epoch)
        if epoch % (args.max_consecutive_losses + 1) == 0:
            if DEBUG>0:
              print( "TRAINDPCCJ:     INFO:   saving model   to \033[35;1m{:}\033[m".format( args.log_dir ) )
            save_model(args.log_dir, model)

  
    print( "TRAINDPCCJ:     INFO: training complete \033[33;1mdone\033[m" )

    hours   = round((time.time() - start_time) / 3600, 1  )
    minutes = round((time.time() - start_time) / 60,   1  )
    pprint.log_section('Job complete in {:} mins'.format( minutes ) )

    print('TRAINDPCCJ:     INFO: run completed in {:} mins'.format( minutes ) )
    
    save_model(args.log_dir, model)
    pprint.log_section('Model saved.')

# ------------------------------------------------------------------------------

def train(args, train_loader, model, optimizer):
    """
    Train PCCA model and update parameters in batches of the whole train set

    """
    
    if DEBUG>1:
      print( "TRAINDPCCJ:     INFO:     at top of train() and parameter train_loader() = \033[35;1m{:}\033[m".format( train_loader ) )
    if DEBUG>9:
      print( "TRAINDPCCJ:     INFO:     at top of train() with parameters \033[35;1margs: \033[m{:}, \033[35;1mtrain_loader: \033[m{:}, \033[35;1mmodel: \033[m{:}, \033[35;1moptimizer: \033[m{:}".format(args, train_loader, model, optimizer ) )

    if DEBUG>1:
      print( "TRAINDPCCJ:     INFO:     train(): about to call \033[33;1mmodel.train()\033[m" )

    model.train()

    if DEBUG>1:
      print( "TRAINDPCCJ:     INFO:     train(): done\033[m" )

    ae_loss1_sum        = 0
    ae_loss2_sum        = 0
    l1_loss_sum         = 0
    total_loss_sum      = 0

    if DEBUG>1:
      print( "TRAINDPCCJ:     INFO:     train(): about to push a mini-batch (matched image batch and genes batch) of \033[35;1m{:}\033[m examples through the model".format(args.batch_size ) )
    
    for i, ( batch_images, batch_genes ) in enumerate( train_loader ):

        #if DEBUG>0:
          #batch_images_clone = batch_images.clone()
          #batch_genes_clone = batch_genes.clone()
          #print( "TRAINDPCCJ:     INFO:       train():     current enumeration and i=\033[35;1m{:}\033[m".format( i ) )
          #print( "TRAINDPCCJ:     INFO:       train(): for current enumeration values shapes are: batch_images: \033[35;1m{:}\033[m, batch_genes: \033[35;1m{:}\033[m".format( batch_images_clone.numpy().shape, batch_genes_clone.numpy().shape ) )

        if DEBUG>9:
          if ( i%10==0 ):
            print( "TRAINDPCCJ:     INFO:     train():     current enumeration values: i=\033[35;1m{:}\033[m, \nbatch_images=\n\033[35;1m{:}\033[m, \nbatch_genes=\n\033[35;1m{:}\033[m".format( i+1, batch_images_clone.numpy(), batch_genes_clone.numpy() ) )
            time.sleep(5)
                  
        if DEBUG>1:
          print( "TRAINDPCCJ:     INFO:     train(): about to call \033[33;1moptimizer.zero_grad()\033[m" )

        optimizer.zero_grad()

        if DEBUG>1:
          print( "TRAINDPCCJ:     INFO:     train(): done" )

        batch_images = batch_images.to(device)
        batch_genes  = batch_genes.to (device)

        if DEBUG>1:
          print( "TRAINDPCCJ:     INFO:     train(): about to call \033[33;1mmodel.forward()\033[m" )
        
        batch_imagesr, batch_genesr = model.forward( [batch_images, batch_genes] )
        
        if DEBUG>1:
          print ( "TRAINDPCCJ:     INFO:      train(): batch_imagesr.shape = {:}".format( batch_imagesr.shape ) )
          print ( "TRAINDPCCJ:     INFO:      train(): batch_genesr.shape  = {:}".format( batch_genesr.shape ) )

        if DEBUG>1:
          print( "TRAINDPCCJ:     INFO:     train(): done" )
        
        ae_loss_images = F.mse_loss(batch_imagesr, batch_images)
        ae_loss_genes  = F.mse_loss(batch_genesr,  batch_genes)
        l1_loss        = l1_penalty(model, args.l1_coef)
        loss           = ae_loss_images + ae_loss_genes  + l1_loss
        
        if DEBUG>1:
          print ( "TRAINDPCCJ:     INFO:      train():       batch_genesr.shape                      = {:}".format( batch_genesr.shape  ) )
          print ( "TRAINDPCCJ:     INFO:      train():       batch_genesr[0:10]                      = {:}".format( batch_genesr[0:10]  ) )
          print ( "TRAINDPCCJ:     INFO:      train():       batch_genes.shape                       = {:}".format( batch_genes.shape   ) )
          print ( "TRAINDPCCJ:     INFO:      train():       batch_genes[0:10]                       = {:}".format( batch_genes [0:10]  ) )	        

        if DEBUG>0:
          print ( f"TRAINDPCCJ:     INFO:     train():     n=\r\033[41C\033[38;2;140;140;140m{0:2d}\033[m    ae_loss_images=\r\033[62C\033[38;140;140;140m{BLUE}{1:.2f}{RESET}\033[m   ae_loss_genes=\r\033[83C\033[38;2;140;140;140m{ORANGE}{2:12.2f}{RESET}\033[m   \r\033[100Cll_loss=\033[38;2;140;140;140m{3:.2f}\033[m   \r\033[120CTOTAL LOSS=\033[38;2;255;165;0m{4:9.2f}\033[m".format( i, ae_loss_images, ae_loss_genes , l1_loss, loss ))
          print ( "\033[2A" )
          
        loss.backward()
        # Perform gradient clipping *before* calling `optimizer.step()`.
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        ae_loss1_sum   += ae_loss_images.item()
        ae_loss2_sum   += ae_loss_genes .item()
        l1_loss_sum    += l1_loss.item()
        total_loss_sum  = ae_loss1_sum + ae_loss2_sum + l1_loss_sum  

        del ae_loss_images            # PGD
        del ae_loss_genes             # PGD
        del l1_loss                   # PGD
        del loss                      # PGD
        torch.cuda.empty_cache()      # PGD


    ae_loss1_sum_ave    = ae_loss1_sum    / (i+1)
    ae_loss2_sum_ave    = ae_loss2_sum    / (i+1)
    l1_loss_sum_ave     = l1_loss_sum     / (i+1)
    total_loss_ave      = total_loss_sum  / (i+1)

    return ae_loss1_sum_ave, ae_loss2_sum_ave, l1_loss_sum_ave, total_loss_ave

# ------------------------------------------------------------------------------

def test(cfg, args, epoch, test_loader, model            ):
    """Test model by computing the average loss on a held-out dataset. No parameter updates.
    """


    if DEBUG>1:
      print( "TRAINDPCCJ:     INFO:      test(): about to test model by computing the average loss on a held-out dataset. No parameter updates" )

    model.eval()

    ae_loss1_sum = 0
    ae_loss2_sum = 0
    l1_loss_sum  = 0

    ae_loss1_sum        = 0
    ae_loss2_sum        = 0
    l1_loss_sum         = 0
    total_loss_sum      = 0

    if DEBUG>1:
      print( "TRAINDPCCJ:     INFO:      test(): about to enumerate  " )
      
    for i, (batch_images, batch_genes) in enumerate(test_loader):

        batch_images = batch_images.to(device)
        batch_genes = batch_genes.to(device)

        if DEBUG>1:
          print( "TRAINDPCCJ:     INFO:     test(): about to call \033[33;1mmodel.forward()\033[m" )
          
        '''
        batch_images_reconstructed, batch_genes_reconstructed = model.forward([batch_images, batch_genes])  # model is DPCCJ

        ae_loss_images = F.mse_loss(batch_images_reconstructed, batch_images)
        ae_loss_genes  = F.mse_loss(batch_genes_reconstructed, batch_genes)
        l1_loss  = l1_penalty(model, args.l1_coef)
        loss     = ae_loss_images + ae_loss_genes  + l1_loss
        '''
        
        with torch.no_grad():                                                                              # PGD 200128 - Don't need gradients for testing, so this should save some GPU memory (tested: it does)
          y1, y2 = model.forward([batch_images, batch_genes])  # model is DPCCJ
    
        ae_loss_images = F.mse_loss(y1, batch_images)
        ae_loss_genes  = F.mse_loss(y2, batch_genes)
        l1_loss  = l1_penalty(model, args.l1_coef)
        loss     = ae_loss_images + ae_loss_genes  + l1_loss        
        
        if DEBUG>0:
          print ( "TRAINDPCCJ:     INFO:     test():      s=\r\033[41C\033[38;2;140;140;140m{0:2d}\033[m    ae_loss_images=\r\033[62C\033[38;2;140;140;140m{1:.2f}\033[m   ae_loss_genes=\r\033[85C\033[38;2;140;140;140m{2:.2f}\033[m   \r\033[100Cll_loss=\033[38;2;140;140;140m{3:.2f}\033[m   \r\033[120CTTOTAL LOSS=\033[38;2;255;255;0m{4:9.2f}\033[m".format( i+1, ae_loss_images, ae_loss_genes , l1_loss, loss ))
          print ( "\033[2A" )

        ae_loss1_sum   += ae_loss_images.item()
        ae_loss2_sum   += ae_loss_genes .item()
        l1_loss_sum    += l1_loss.item()
        total_loss_sum  = ae_loss1_sum + ae_loss2_sum + l1_loss_sum 

        del ae_loss_images            # PGD
        del ae_loss_genes             # PGD
        del l1_loss                   # PGD
        del loss                      # PGD
        torch.cuda.empty_cache()

        #if i == 0 and epoch % LOG_EVERY == 0:
        if epoch % LOG_EVERY==0:
          if DEBUG>99:
            print( "TRAINDPCCJ:     INFO:      test(): about to save comparisons  " )
          cfg.save_comparison( args.log_dir, batch_images, y1, epoch, is_x1=True)
          cfg.save_comparison( args.log_dir, batch_genes,  y2, epoch, is_x1=False)
            
    ae_loss1_sum_ave    = ae_loss1_sum    / (i+1)
    ae_loss2_sum_ave    = ae_loss2_sum    / (i+1)
    l1_loss_sum_ave     = l1_loss_sum     / (i+1)
    total_loss_ave      = total_loss_sum  / (i+1)

    return ae_loss1_sum_ave, ae_loss2_sum_ave, l1_loss_sum_ave, total_loss_ave

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

def save_samples(log_dir, model, test_loader, cfg, epoch):
    
    """Save samples from test set
    """

    with torch.no_grad():
        
        n  = len( test_loader.sampler.indices )                                                            # test_loader sampler = SubsetRandomSampler(test_inds)
        
        images_batch = torch.Tensor(n, cfg.N_CHANNELS, cfg.IMG_SIZE, cfg.IMG_SIZE)                         # make empty tensor for a batch of images
        genes_batch  = torch.Tensor(n, cfg.N_GENES)                                                        # make empty tensor for a batch of genes

        labels   = []

        if DEBUG>8:
          print( f"CLASSI:         INFO:   number if test indices (tiles) to sample and save = {CYAN}{n}{RESET}" )          
          print( f"CLASSI:         INFO:   directory to save them to                         = {CYAN}{log_dir}{RESET}" )                     
          print( f"CLASSI:         INFO:   cfg.N_CHANNELS                                    = {CYAN}{cfg.N_CHANNELS}{RESET}" )
          print( f"CLASSI:         INFO:   cfg.IMG_SIZE                                      = {CYAN}{cfg.IMG_SIZE}{RESET}" )
          print( f"CLASSI:         INFO:   cfg.N_GENES                                       = {CYAN}{cfg.N_GENES}{RESET}" )

        if DEBUG>8:
          print( f"CLASSI:         INFO:   indices of {MAGENTA}{n}{RESET} random test tiles = {MAGENTA}MAGENTA below {RESET}", flush=True )     

        for i in range(n):

            j = test_loader.sampler.indices[i]           
            
            x1, x2   = test_loader.dataset[j]                                                              # load the image and gene info

            if DEBUG>8:
              if i==0:
                np.set_printoptions(edgeitems=200)
                np.set_printoptions(linewidth=260)
                np.set_printoptions(formatter={'int': lambda x: "{:>10.2f}".format(x)})
                print( f"CLASSI:         INFO:   type(x1)  = {YELLOW}{type(x1)}{RESET}", flush=True )
                print( f"CLASSI:         INFO:   np.shape (x1) = {YELLOW}{np.shape(x1.cpu().numpy())}{RESET}", flush=True )                
                #print( f"CLASSI:         INFO:   tile (x1) = {YELLOW}{x1.cpu().numpy()}{RESET}", flush=True )
                
                
            if DEBUG>9:
              if i==0:
                print( f"CLASSI:         INFO:   gene (x2) = {YELLOW}{x2.cpu().numpy()}{RESET}", flush=True)   


            label    = test_loader.dataset.labels[j]
            images_batch[i] = x1*255
            genes_batch [i] = x2
            labels.append(label)
            
            if DEBUG>8:
              if i==0:
                print( f"CLASSI:         INFO:   tile (x1) = {BLUE}{images_batch[i].cpu().numpy()}{RESET}", flush=True )            

            if DEBUG>8:
              print( f"{MAGENTA}{j}, {RESET}", end="", flush=True)                                                      # 


        print("")
        
        if DEBUG>9:
          print( f"CLASSI:         INFO:   labels = {CYAN}{labels}{RESET}", flush=True )          
        
        images_batch = images_batch.to(device)
        genes_batch  = genes_batch .to(device)

        cfg.save_samples( log_dir, model, epoch, images_batch, genes_batch, labels )


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

    p.add_argument('--just_test',              type=str,   default='False')      
    p.add_argument('--log_dir',                type=str,   default='logs')
    p.add_argument('--wall_time',              type=int,   default=24)
    p.add_argument('--seed',                   type=int,   default=0)
    p.add_argument('--mode',                   type=str,   default='classify')    
    p.add_argument('--batch_size',             type=int,   default=128)
    p.add_argument('--n_epochs',               type=int,   default=100)
    p.add_argument('--pct_test',               type=float, default=0.1)
    p.add_argument('--lr',                     type=float, default=0.001)
    p.add_argument('--latent_dim',             type=int,   default=2)
    p.add_argument('--l1_coef',                type=float, default=0.1)
    p.add_argument('--em_iters',               type=int,   default=1)
    p.add_argument('--clip',                   type=float, default=1)
    p.add_argument('--max_consecutive_losses', type=int,   default=10)
    p.add_argument('--normalize_rna_values',   type=str,   default="yes")

    args, _ = p.parse_known_args()

    is_local = args.log_dir == 'experiments/example'

    args.n_workers  = 0 if is_local else 4
    args.pin_memory = torch.cuda.is_available()

    # For easy debugging locally.
    if is_local:
        LOG_EVERY        = 1

    torch.manual_seed(args.seed)
    main(args)
