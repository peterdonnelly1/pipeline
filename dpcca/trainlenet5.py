"""============================================================================= 
Train LENET5
============================================================================="""

import sys
import time
import cuda
import pprint
import argparse
import numpy as np
from   data           import loader
from   models         import LENETIMAGE

import torch
from   torch          import optim
import torch.utils.data
from   torch.nn.utils import clip_grad_norm_
from   torch.nn       import functional as F

DEBUG=1

np.set_printoptions(edgeitems=8)
np.set_printoptions(linewidth=200)

# ------------------------------------------------------------------------------
    
LOG_EVERY        = 1
SAVE_MODEL_EVERY = 5

device = cuda.device()

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)

# ------------------------------------------------------------------------------

def main(args):
    """Main program: train -> test once per epoch while saving samples as needed.
    """
    
    now = time.localtime(time.time())
    print(time.strftime("TRAINLENEJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))

    start_time = time.time()
    pprint.set_logfiles(args.directory)

    pprint.log_section('Loading config.')

    print( "TRAINLENEJ:     INFO: passed in arguments are:\
 dataset=\033[35;1m{:}\033[m,\
 batch_size=\033[35;1m{:}\033[m,\
 n_epochs=\033[35;1m{:}\033[m,\
 latent_dim=\033[35;1m{:}\033[m,\
 max_consecutive_losses=\033[35;1m{:}\033[m"\
.format( args.dataset, args.batch_size, args.n_epochs, args.latent_dim, args.max_consecutive_losses), flush=True )


    # (1)
    print( "TRAINLENEJ:     INFO: \033[1m1 about to load experiment config\033[m" )
      
    cfg = loader.get_config(args.dataset)
    pprint.log_config(cfg)
    pprint.log_section('Loading script arguments.')
    pprint.log_args(args)

    print( "TRAINLENEJ:     INFO:   experiment config loaded\033[m" )
    
    #(2)
    print( "TRAINLENEJ:     INFO: \033[1m2 about to load LENET5 model\033[m with parameters: args.latent_dim=\033[35;1m{:}\033[m, args.em_iters=\033[35;1m{:}\033[m".format( args.latent_dim, args.em_iters) )                                                         
    model = LENETIMAGE(cfg, args.latent_dim, args.em_iters)
    print( "TRAINLENEJ:     INFO:   model loaded\033[m" )  
 
    #(3)
    print( "TRAINLENEJ:     INFO: \033[1m3 about send model to device\033[m" )   
    model = model.to(device)
    print( "TRAINLENEJ:     INFO:   model sent to device\033[m" ) 

    pprint.log_section('Model specs.')
    pprint.log_model(model)  
    
    if DEBUG>9:
      print( "TRAINLENEJ:     INFO: Pytorch Model = {:}".format(model))
    
    #(4)
    print( "TRAINLENEJ:     INFO: \033[1m4 about to call dataset loader\033[m with parameters: cfg=\033[35;1m{:}\033[m, args.batch_size=\033[35;1m{:}\033[m, args.n_worker=\033[35;1m{:}\033[m, args.pin_memory=\033[35;1m{:}\033[m, args.cv_pct=\033[35;1m{:}\033[m".format( cfg, args.batch_size, args.n_workers, args.pin_memory, args.cv_pct) )
    train_loader, test_loader = loader.get_data_loaders(cfg,
                                                        args.batch_size,
                                                        args.n_workers,
                                                        args.pin_memory,
                                                        args.cv_pct)
                                                        
    print( "TRAINLENEJ:     INFO:   dataset loaded\033[m" )

    pprint.save_test_indices(test_loader.sampler.indices)

    #(5)
    print( "TRAINLENEJ:     INFO: \033[1m5 about to select and configure Adam model\033[m with learning rate = \033[35;1m{:}\033[m".format( args.lr ) )  
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print( "TRAINLENEJ:     INFO:   Adam optimizer selected and configures\033[m" )
    
    
    #(6)
    print( "TRAINLENEJ:     INFO: \033[1m6 about to select Torch CrossEntropyLoss function\033[m" )  
    loss_function = torch.nn.CrossEntropyLoss()   ###NEW
    print( "TRAINLENEJ:     INFO:   Torch CrossEntropyLoss function selected" )  
    

    pprint.log_section('Training model.\n\n'\
                       'Epoch\t\tTrain x1 err\tTrain x2 err\tTrain l1\t'\
                       '\tTest x1 err\tTest x2 err\tTest l1')
 
    #(7)                      
    print( "TRAINLENEJ:     INFO: \033[1m7 about to commence training loop, one iteration per epoch\033[m" )

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

        print('TRAINLENEJ:     INFO:   epoch: \033[35;1m{:}\033[m, batch size: \033[35;1m{:}\033[m.  Will save best model and halt when test set total loss increases for \033[35;1m{:}\033[m consecutive epochs'.format( epoch, args.batch_size, args.max_consecutive_losses ) )
    
        if DEBUG>1:
          print('TRAINLENEJ:     INFO:   6.1 running training step ')
  
        train_loss1_sum_ave, train_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave = train (      args,        train_loader, model, optimizer, loss_function )

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
            print ( "TRAINLENEJ:     INFO:     train():\r\033[47Closs_images=\r\033[59C\033[38;2;140;140;140m{0:.4f}\033[m   loss_unused=\r\033[85C\033[38;2;140;140;140m{1:.4f}\033[m   l1_loss=\r\033[102C\033[38;2;140;140;140m{2:.4f}\033[m   TOTAL LOSS=\r\033[122C\033[38;2;0;255;0m{3:9.4f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.4f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.4f} at epoch {7:2d}\033[m".format( train_loss1_sum_ave, train_loss2_sum_ave, train_l1_loss_sum_ave, train_total_loss_ave,    train_lowest_total_loss_observed, train_lowest_total_loss_observed_epoch, train_lowest_image_loss_observed, train_lowest_image_loss_observed_epoch ) )
          else:
            last_epoch_loss_increased = True
            print ( "TRAINLENEJ:     INFO:     train():\r\033[47Closs_images=\r\033[59C\033[38;2;140;140;140m{0:.4f}\033[m   loss_unused=\r\033[85C\033[38;2;140;140;140m{1:.4f}\033[m   l1_loss=\r\033[102C\033[38;2;140;140;140m{2:.4f}\033[m   TOTAL LOSS=\r\033[122C\033[38;2;255;165;0m{3:9.4f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.4f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.4f} at epoch {7:2d}\033[m".format( train_loss1_sum_ave, train_loss2_sum_ave, train_l1_loss_sum_ave, train_l1_loss_sum_ave, train_lowest_total_loss_observed, train_lowest_total_loss_observed_epoch, train_lowest_image_loss_observed, train_lowest_image_loss_observed_epoch ) )
            if last_epoch_loss_increased == True:
              consecutive_training_loss_increases +=1
              if consecutive_training_loss_increases == 1:
                print ( "TRAINLENEJ:     NOTE:     train():              \033[38;2;255;165;0mtotal training loss increased\033[m" )
              else:
                print ( "TRAINLENEJ:     NOTE:     train():             \033[38;2;255;165;0m{0:2d} consecutive training loss increases (s) !!!\033[m".format( consecutive_training_loss_increases ) )

        train_total_loss_ave_last = train_total_loss_ave

        if DEBUG>1:
          print('TRAINLENEJ:     INFO:   6.2 running test step ')
  
        test_loss1_sum_ave, test_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave     = test  ( cfg, args, epoch, test_loader,  model,            loss_function )

        if test_total_loss_ave < test_lowest_total_loss_observed:
          test_lowest_total_loss_observed       = test_total_loss_ave
          test_lowest_total_loss_observed_epoch = epoch

        if test_loss1_sum_ave < test_lowest_image_loss_observed:
          test_lowest_image_loss_observed       = test_loss1_sum_ave
          test_lowest_image_loss_observed_epoch = epoch

        if DEBUG>0:
          if ( (test_total_loss_ave < test_total_loss_ave_last) | (epoch==1) ):
            consecutive_test_loss_increases = 0
            last_epoch_loss_increased = False
            print ( "TRAINLENEJ:     INFO:      test():\r\033[47Closs_images=\r\033[59C\033[38;2;140;140;140m{0:.4f}\033[m   loss_unused=\r\033[85C\033[38;2;140;140;140m{1:.4f}\033[m   l1_loss=\r\033[102C\033[38;2;140;140;140m{2:.4f}\033[m   TOTAL LOSS=\r\033[122C\033[38;2;0;255;0m{3:9.4f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.4f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.4f} at epoch {7:2d}\033[m".format( test_loss1_sum_ave, test_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave, test_lowest_total_loss_observed, test_lowest_total_loss_observed_epoch, test_lowest_image_loss_observed, test_lowest_image_loss_observed_epoch ) )
          else:
            last_epoch_loss_increased = True
            print ( "TRAINLENEJ:     INFO:      test():\r\033[47Closs_images=\r\033[59C\033[38;2;140;140;140m{0:.4f}\033[m   loss_unused=\r\033[85C\033[38;2;140;140;140m{1:.4f}\033[m   l1_loss=\r\033[102C\033[38;2;140;140;140m{2:.4f}\033[m   TOTAL LOSS=\r\033[122C\033[38;2;255;0;0m{3:9.4f}\033[m   lowest total loss=\r\033[153C\033[38;2;140;140;140m{4:.4f} at epoch {5:2d}\033[m    lowest image loss=\r\033[195C\033[38;2;140;140;140m{6:.4f} at epoch {7:2d}\033[m".format( test_loss1_sum_ave, test_loss2_sum_ave, test_l1_loss_sum_ave, test_total_loss_ave, test_lowest_total_loss_observed, test_lowest_total_loss_observed_epoch, test_lowest_image_loss_observed, test_lowest_image_loss_observed_epoch))
            if last_epoch_loss_increased == True:
              consecutive_test_loss_increases +=1
              if consecutive_test_loss_increases == 1:
                print ( "TRAINLENEJ:     NOTE:      test():              \033[38;2;255;0;0mtotal test loss increased\033[m" )
              else:
                print ( "TRAINLENEJ:     NOTE:      test():             \033[38;2;255;0;0m{0:2d} consecutive test loss increase(s) !!!\033[m".format( consecutive_test_loss_increases ) )
              if consecutive_test_loss_increases>args.max_consecutive_losses:  # Stop one before SAVE_MODEL_EVERY so that the most recent model for which the loss improved will be saved
                  now = time.localtime(time.time())
                  print(time.strftime("TRAINLENEJ:     INFO: %Y-%m-%d %H:%M:%S %Z", now))
                  sys.exit(0)
  
        test_total_loss_ave_last = test_total_loss_ave

        if DEBUG>1:
          print('TRAINLENEJ:     INFO:   train_msgs (average loss ove eopoch): {:}, test_msgs (average loss ove eopoch): {:}'.format(train_msgs, test_msgs))

#        if epoch % LOG_EVERY == 0:
#            if DEBUG>0:
#              print( "TRAINLENEJ:     INFO:   saving samples to \033[35;1m{:}\033[m".format( args.directory ) )
#            save_samples(args.directory, model, test_loader, cfg, epoch)
        if epoch % (args.max_consecutive_losses + 1) == 0:
            if DEBUG>0:
              print( "TRAINLENEJ:     INFO:   saving model   to \033[35;1m{:}\033[m".format( args.directory ) )
            save_model(args.directory, model)
            
    print( "TRAINLENEJ:     INFO: training complete \033[33;1mdone\033[m" )

    hours   = round((time.time() - start_time) / 3600, 1  )
    minutes = round((time.time() - start_time) / 60,   1  )
    pprint.log_section('Job complete in {:} mins'.format( minutes ) )

    print('TRAINLENEJ:     INFO: run completed in {:} mins'.format( minutes ) )
    
    save_model(args.directory, model)
    pprint.log_section('Model saved.')
# ------------------------------------------------------------------------------

def train(args, train_loader, model, optimizer, loss_function):
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
    
    for i, (batch_images, batch_tissues) in enumerate(train_loader):                                         # fetch a batch of each

                  
        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     train(): about to call \033[33;1moptimizer.zero_grad()\033[m" )

        optimizer.zero_grad()

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     train(): done" )

        batch_images  = batch_images.to(device)
        batch_tissues = batch_tissues.to (device)

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     train(): about to call \033[33;1mmodel.forward()\033[m" )

        y1_hat  = model.forward( batch_images )

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     train(): done" )
             
      
        if DEBUG>9:
          y1_hat_numpy = (y1_hat.cpu().data).numpy()
          print ( "TRAINLENEJ:     INFO:      train():       type(y1_hat)                      = {:}".format( type(y1_hat_numpy)       ) )
          print ( "TRAINLENEJ:     INFO:      train():       y1_hat.shape                      = {:}".format( y1_hat_numpy.shape       ) )
          print ( "TRAINLENEJ:     INFO:      train():       y1_hat                            = \n{:}".format( y1_hat_numpy[0:2,0:2] ) )
        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      train():       batch_tissues.shape                  = {:}".format( batch_tissues.shape  ) )
          print ( "TRAINLENEJ:     INFO:      train():       batch_tissues[0:10]                  = {:}".format( batch_tissues[0:64]  ) )

        loss_images = loss_function(torch.transpose(y1_hat, 1, 0), batch_tissues.view(256,1).squeeze() )  
        loss_images_value = loss_images.item()                                                             # use .item() to extract just the value: don't create multiple new tensors each of which will have gradient histories
        #l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0
        total_loss        = loss_images_value + l1_loss

        if DEBUG>0:
          print ( "TRAINLENEJ:     INFO:     train():     n=\r\033[41C\033[38;2;140;140;140m{0:2d}\033[m    loss_images=\r\033[59C\033[38;2;180;180;0m{1:.4f}\033[m   l1_loss=\r\033[102C\033[38;2;180;180;0m{2:.4f}\033[m   TOTAL LOSS=\r\033[122C\033[38;2;255;255;0m{2:.4f}\033[m".format( i, loss_images_value, l1_loss, total_loss ))
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

    if DEBUG>99:
      print ( "TRAINLENEJ:     INFO:      train():       type(loss1_sum_ave)                      = {:}".format( type(loss1_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      train():       type(loss2_sum_ave)                      = {:}".format( type(loss2_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      train():       type(l1_loss_sum_ave)                    = {:}".format( type(l1_loss_sum_ave)   ) )
      print ( "TRAINLENEJ:     INFO:      train():       type(total_loss_ave)                     = {:}".format( type(total_loss_ave)    ) )

    return loss1_sum_ave, loss2_sum_ave, l1_loss_sum_ave, total_loss_ave

# ------------------------------------------------------------------------------

def test( cfg, args, epoch, test_loader, model, loss_function ):
    """Test model by computing the average loss on a held-out dataset. No parameter updates.
    """

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to test model by computing the average loss on a held-out dataset. No parameter updates" )

    model.eval()                                                                                           # set model to evaluation mode

    loss1_sum      = 0
    loss2_sum      = 0
    l1_loss_sum    = 0

    loss1_sum      = 0
    loss2_sum      = 0
    l1_loss_sum    = 0
    total_loss_sum = 0

    if DEBUG>9:
      print( "TRAINLENEJ:     INFO:      test(): about to enumerate  " )
      
    for i, (batch_images, batch_tissues) in enumerate(test_loader):

        batch_images =   batch_images.to(device)
        batch_tissues  = batch_tissues.to(device)

        if DEBUG>9:
          print( "TRAINLENEJ:     INFO:     test(): about to call \033[33;1mmodel.forward()\033[m" )
          
        y1_hat = model.forward( batch_images )                                                             # model is now LENET5

        if DEBUG>9:
          y1_hat_values = (y1_hat.cpu().data).numpy()
          print ( "TRAINLENEJ:     INFO:      test():       type(y1_hat)                      = {:}".format( type(y1_hat_values)       ) )
          print ( "TRAINLENEJ:     INFO:      test():       y1_hat.shape                      = {:}".format( y1_hat_values.shape       ) )
          print ( "TRAINLENEJ:     INFO:      test():       y1_hat                            = \n{:}".format( y1_hat_values[0:2,0:2] ) )
        
        loss_images       = loss_function(torch.transpose(y1_hat, 1, 0), batch_tissues.view(256,1).squeeze() ) 
        loss_images_value = loss_images.item()                                                             # use .item() to extract just the value: don't create multiple new tensors each of which will have gradient histories
        #l1_loss          = l1_penalty(model, args.l1_coef)
        l1_loss           = 0
        total_loss        = loss_images_value + l1_loss

        if DEBUG>99:
          print ( "TRAINLENEJ:     INFO:      test():       type(loss)                      = {:}".format( type(loss)       ) )

        if DEBUG>0:
          print ( "TRAINLENEJ:     INFO:     test():      s=\r\033[41C\033[38;2;140;140;140m{0:2d}\033[m    loss_images=\r\033[59C\033[38;2;255;255;0m{1:.4f}\033[m  l1_loss=\r\033[102C\033[38;2;255;255;0m{2:.4f}\033[m   TOTAL LOSS=\r\033[122C\033[38;2;255;255;0m{3:.4f}\033[m".format( i, loss_images_value, l1_loss, total_loss ))
          print ( "\033[2A" )

        loss1_sum      += loss_images_value                                                                # use .item() to extract just the value: don't create a new tensor
        l1_loss_sum    += l1_loss
        total_loss_sum += total_loss  

        del loss_images
        torch.cuda.empty_cache()

    if epoch % 10 == 0:
      y1_hat_values                = y1_hat.cpu().detach().numpy()
      y1_hat_values_max_indices    = np.argmax( y1_hat_values, axis=0  )
      y1_hat_values_at_max_indices = np.amax  ( y1_hat_values, axis=0  )
      batch_tissues_values         = batch_tissues.cpu().detach().numpy()  
      if DEBUG>99:
        print ( "TRAINLENEJ:     INFO:      test():       y1_hat.shape                     = {:}".format( y1_hat_values.shape               ) )
      print ( "" )
      print ( "TRAINLENEJ:     INFO:     test(): truth/prediction/values for first few examples from the last test batch (number correct = \u001b[4m{:}\033[m/{:})".format(np.sum( np.equal(y1_hat_values_max_indices, batch_tissues_values)), batch_tissues_values.shape[0] )   )
      np.set_printoptions(formatter={'int': lambda x: "{0:5d}".format(x)})
      print (  batch_tissues_values[0:44]  ) 
      print (  y1_hat_values_max_indices[0:44]    )
      np.set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})
      print (  y1_hat_values_at_max_indices[0:44]  ) 

          
    loss1_sum_ave    = loss1_sum       / (i+1)
    loss2_sum_ave    = loss2_sum       / (i+1)
    l1_loss_sum_ave  = l1_loss_sum     / (i+1)
    total_loss_ave   = total_loss_sum  / (i+1)

    if DEBUG>99:
      print ( "TRAINLENEJ:     INFO:      test():       type(loss1_sum_ave)                      = {:}".format( type(loss1_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(loss2_sum_ave)                      = {:}".format( type(loss2_sum_ave)     ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(l1_loss_sum_ave)                    = {:}".format( type(l1_loss_sum_ave)   ) )
      print ( "TRAINLENEJ:     INFO:      test():       type(total_loss_ave)                     = {:}".format( type(total_loss_ave)    ) )

    return loss1_sum_ave, loss2_sum_ave, l1_loss_sum_ave, total_loss_ave

# ------------------------------------------------------------------------------

def l1_penalty(model, l1_coef):
    """Compute L1 penalty. For implementation details, see:

    https://discuss.pytorch.org/t/simple-l2-regularization/139
    """
    reg_loss = 0
    for param in model.lnetimg.parameters_('y2_hat'):
        reg_loss += torch.norm(param, 1)
    return l1_coef * reg_loss

# ------------------------------------------------------------------------------
# NOT USED
def save_samples(directory, model, test_loader, cfg, epoch):
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

        cfg.save_samples(directory, model, epoch, x1_batch, x2_batch, labels)

# ------------------------------------------------------------------------------

def save_model(directory, model):
    """Save PyTorch model's state dictionary for provenance.
    """
    fpath = '%s/model.pt' % directory
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    #p.add_argument('--directory',  type=str,   default='experiments/example')
    p.add_argument('--directory',              type=str,   default='data/dlbcl_image/logs') # WATCH!!
    p.add_argument('--wall_time',              type=int,   default=24)
    p.add_argument('--seed',                   type=int,   default=0)
    p.add_argument('--dataset',                type=str,   default='dlbcl_image') ## WATCH!!!
    p.add_argument('--batch_size',             type=int,   default=128)
    p.add_argument('--n_epochs',               type=int,   default=30)
    p.add_argument('--cv_pct',                 type=float, default=0.1)
    p.add_argument('--lr',                     type=float, default=0.0008)
    p.add_argument('--latent_dim',             type=int,   default=2)
    p.add_argument('--l1_coef',                type=float, default=0.1)
    p.add_argument('--em_iters',               type=int,   default=1)
    p.add_argument('--clip',                   type=float, default=1)
    p.add_argument('--max_consecutive_losses', type=int,   default=5)

    args, _ = p.parse_known_args()

    is_local = args.directory == 'experiments/example'

    args.n_workers  = 0 if is_local else 4
    args.pin_memory = torch.cuda.is_available()

    # For easy debugging locally.
    if is_local:
        LOG_EVERY        = 1
        SAVE_MODEL_EVERY = 20

    torch.manual_seed(args.seed)
    main(args)
