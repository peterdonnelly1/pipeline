"""============================================================================
Configuration for the DLBC data set with LENET  
============================================================================"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from   torchvision.utils import save_image

from   models import LENET5, AELINEAR, AEDENSE, VGG, VGGNN, INCEPT3, DENSE, DEEPDENSE, CONV1D, DCGANAE128, AEDCECCAE_3, AEDCECCAE_5, AEVGG16
from   models.vggnn import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, make_layers, configs
#from   models.incept3 import incept3
from modes.classify.dataset import classifyDataset
from modes.config import Config

from constants  import *

DEBUG=1

# ------------------------------------------------------------------------------

class classifyConfig(Config):

    # Class variables: only parameters that will not change across an entire job (job = many runs of the model)

    # ~ N_CHANNELS     = 3
    # ~ IMG_EMBED_DIM  = 7                                                                                     # Has to be the same as the number of classes
    # ~ N_GENES        = 12                                                                                    # Not used
    # ~ GENE_EMBED_DIM = 1000                                                                                  # PGD THIS WAS ORIGINALLY 1000
    LABEL_SWAP_PERUNIT   = 0.0                                                                             # 1.0 =change 100% of labels to a random class                                                            - use for validation
    MAKE_GREY            = 0.0                                                                             # 1.0 =change 100% of RGB images to 3-channel Greyscale etc                                               - use for validation
    JITTER               = [0.0 ,0.0, 0.0 ,0.1]                                                            # torchvision.transforms.ColorJitter(brightness=[0,1], contrast=[0,1], saturation=[0,1], hue=[-0.5,+0.5]) - use for validation

    # Instance variables: parameters that may change from run to run (such as learning rate or batch_size) 

    def __init__(self, lr,  batch_size ):
   
      if DEBUG>1:
        print( f"CONFIG:         INFO:     at {CYAN} __init__():{RESET}   current learning rate / batch_size  = {MIKADO}{lr}, {batch_size}{RESET} respectively" )

# ------------------------------------------------------------------------------

    def get_image_net( self, args, input_mode, nn_type_img, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2, tile_size ):

      if DEBUG>0:
        print( f"CONFIG:         INFO:     at {CYAN}get_image_net(){RESET}:   nn_type_img  = {CYAN}{nn_type_img}{RESET}" )

      if   nn_type_img=='LENET5':
        return LENET5         ( self, args, n_classes, tile_size )
      elif nn_type_img=='VGG':
        return VGG( self )
      elif nn_type_img=='VGG11':
        return vgg11_bn       ( self, args, n_classes, tile_size )
      elif nn_type_img=='VGG13':
        return vgg13_bn       ( self, args, n_classes, tile_size )       
      elif nn_type_img=='VGG16':
        return vgg16_bn       ( self, args, n_classes, tile_size )
      elif nn_type_img=='VGG19':
        return vgg19_bn       ( self, args, n_classes, tile_size )
      elif nn_type_img=='INCEPT3':
        return INCEPT3        ( self, args, n_classes, tile_size )
      elif nn_type_img=='AE3LAYERCONV2D':
        return AE3LAYERCONV2D ( self, args, n_classes, tile_size )
      elif nn_type_img=='AEDCECCAE_3':
        return AEDCECCAE_3    ( self, args, n_classes, tile_size )
      elif nn_type_img=='AEDCECCAE_5':
        return AEDCECCAE_5    ( self, args, n_classes, tile_size )
      elif nn_type_img=='AEVGG16':
        return AEVGG16        ( self, args, n_classes, tile_size )
      else: 
        print( f"{BOLD}{RED}CONFIG:              FATAL:  Sorry, there is no neural network model available for {CYAN}image{RESET}{RED} processing named: '{CYAN}{nn_type_img}{RESET}{RED}' ... halting now.{RESET}" )        
        sys.exit(0)

# ------------------------------------------------------------------------------

    def get_genes_net( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2  ):
      
      if DEBUG>2:
        print( "CONFIG:         INFO:     at \033[35;1m get_genes_net()\033[m:   nn_type_rna  = \033[36;1m{:}\033[m".format( nn_type_rna ) )

      if DEBUG>9:
        print( "CONFIG:         INFO:     at \033[35;1m get_genes_net()\033[m:   nn_type_rna  = \033[36;1m{:}\033[m".format( nn_type_rna ) )

      if nn_type_rna=='DENSE':
        return DENSE           ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      if nn_type_rna=='DEEPDENSE':
        return DEEPDENSE       ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='CONV1D':
        return CONV1D          ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='DENSEPOSITIVE':
        return DENSEPOSITIVE   ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='DCGANAE128':
        return DCGANAE128      (self)
      elif nn_type_rna=='AELINEAR':
        return AELINEAR        ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='AEDENSE':
        return AEDENSE         ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='AEDENSEPOSITIVE':
        return AEDENSEPOSITIVE ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='AEDEEPDENSE':
        return AEDEEPDENSE     ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='TTVAE':
        return TTVAE           ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2 )  
      else:
        print( f"{BOLD}{RED}CONFIG:              FATAL:  Sorry, there is no neural network model available for {CYAN}rna{RESET}{RED} processing named: '{CYAN}{nn_type_rna}{RESET}{RED}' ... halting now.{RESET}" )        
        exit(0)
# ------------------------------------------------------------------------------

    def get_dataset( self, args, which_dataset, gpu ):

      return classifyDataset( self, which_dataset, args )

# ------------------------------------------------------------------------------

    def save_samples(self, directory, model, desc, x1, x2, labels):

      if DEBUG>9:
        print( "CONFIG:         INFO:       at top of save_samples() and parameter directory = \033[35;1m{:}\033[m".format( directory ) )
        
        n_samples = 100
        nc        = self.N_CHANNELS
        w         = self.IMG_SIZE

        # Visualize images.
        # -----------------
        x1r, x2r = model.sample([x1, x2], n_samples)
        x1r = x1r.view(n_samples, nc, w, w)
        fname = '%s/sample_images_%s.png' % (directory, desc)
        save_image(x1r.cpu(), fname)

# ------------------------------------------------------------------------------

    def save_image_samples(self, directory, model, desc, x1):
        n_samples = 64
        nc = self.N_CHANNELS
        w = self.IMG_SIZE

        x1_recon, _ = model.sample(None, n_samples)
        x1_recon = x1_recon.view(n_samples, nc, w, w)
        fname = '%s/sample_%s.png' % (directory, desc)
        save_image(x1_recon.cpu(), fname)

# ------------------------------------------------------------------------------

    def save_comparison(self, directory, x, x_recon, desc, is_x1=None):
        """Save image samples from learned image likelihood.
        """
        if is_x1:
            self.save_image_comparison(directory, x, x_recon, desc)
        else:
            self.save_genes_comparison(directory, x, x_recon, desc)

# ------------------------------------------------------------------------------

    def save_image_comparison(self, directory, x, x_recon, desc):

        nc = x_recon.size()[1]                                                                             # number of channels
        w  = x_recon.size()[2]                                                                             # width
        
        x1_fpath = '%s/%s_images_recon.png' % (directory, desc)
        N = min(x.size(0), 24)                                                                             # PGD 200614 - Number of images pairs to save for display
        recon = x_recon.view(-1, nc, w, w)[:N]
        x = x.view(-1, nc, w, w)[:N]
        comparison = torch.cat([x, recon])
        save_image(comparison.cpu(), x1_fpath, nrow=N)

# ------------------------------------------------------------------------------

    def save_genes_comparison(self, directory, x, xr, desc):
        n, _ = x.shape
        x    = x.detach().cpu().numpy()
        xr   = xr.detach().cpu().numpy()

        x_cov  = np.cov(x)
        xr_cov = np.cov(xr)

        comparison = np.hstack([x_cov, xr_cov])
        plt.imshow(comparison)

        fpath = '%s/%s_genes_recon.png' % (directory, str(desc))
        plt.savefig(fpath)
        plt.close('all')
        plt.clf()
