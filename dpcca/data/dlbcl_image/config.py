"""============================================================================
Configuration for the DLBC data set with LENET  
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from   torchvision.utils import save_image

from   models import LENET5, AELinear, VGG, VGGNN, INCEPT3, DENSE, CONV1D, DCGANAE128
from   models.vggnn import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, make_layers, configs
#from   models.incept3 import incept3
from   data.dlbcl_image.dataset import GTExV6Dataset
from   data.config import Config

WHITE='\033[37;1m'
DIM_WHITE='\033[37;2m'
CYAN='\033[36;1m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
BLUE='\033[38;2;0;0;255m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
ORANGE='\033[38;2;255;127;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GREEN='\033[32;1m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

DEBUG=1

# ------------------------------------------------------------------------------

class GTExV6Config(Config):

    # Class variables: only parameters that will not change across an entire job (job = many runs of the model)

    ROOT_DIR       = 'data/dlbcl_image'
    
#    INPUT_MODE     = 'rna'                                                                               # valid values are 'image', 'rna', 'image_rna'
    
#    IMG_SIZE      =  128
#   IMG_SIZE      =  399         # PGD 200219 - USE THIS SIZE FOR INCEPTION V3
    N_CHANNELS    =  3
    IMG_EMBED_DIM  = 7           # Has to be the same as the number of classes

#   IMG_SIZE       = 28          # FOR MNIST ONLY
#   N_CHANNELS     = 1           # FOR MNIST ONLY
#   IMG_EMBED_DIM  = 10          # FOR MNIST ONLY

#    N_PIXELS       = N_CHANNELS * IMG_SIZE * IMG_SIZE
    N_GENES        = 506
    GENE_EMBED_DIM = 1000         # PGD THIS WAS ORIGINALLY 1000

    LABEL_SWAP_PERUNIT   = 0.0                                                                             # 1.0 =change 100% of labels to a random class                                                            - use for validation
    MAKE_GREY            = 0.0                                                                             # 1.0 =change 100% of RGB images to 3-channel Greyscale etc                                               - use for validation
    JITTER               = [0.0 ,0.0, 0.0 ,0.1]                                                            # torchvision.transforms.ColorJitter(brightness=[0,1], contrast=[0,1], saturation=[0,1], hue=[-0.5,+0.5]) - use for validation

    # Instance variables: parameters that may change from run to run (such as learning rate or batch_size) 

    def __init__(self, lr,  batch_size ):
   
      if DEBUG>1:
        print( "CONFIG:         INFO:     at \033[35;1m __init__()\033[m:   current learning rate / batch_size  = \033[36;1m{:}, {:}\033[m respectively".format( lr,  batch_size ) )

# ------------------------------------------------------------------------------

    def get_image_net(self, nn_type, n_classes, n_genes, tile_size ):


      if DEBUG>0:
        print( "CONFIG:         INFO:     at \033[35;1m get_image_net()\033[m:   nn_type  = \033[36;1m{:}\033[m".format( nn_type ) )

      if   nn_type=='LENET5':
        return LENET5(self)
      elif nn_type=='VGG':
        return VGG(self)
      elif nn_type=='VGG11':
        return vgg11_bn(self, n_classes, tile_size)
      elif nn_type=='VGG13':
        return vgg13_bn(self, n_classes, tile_size)       
      elif nn_type=='VGG16':
        return vgg16_bn(self, n_classes, tile_size)
      elif nn_type=='VGG19':
        return vgg19_bn(self, n_classes, tile_size)
      elif nn_type=='INCEPT3':
        return INCEPT3(self,  n_classes, tile_size) 
      elif nn_type=='DENSE':
        return DENSE(self, n_classes, n_genes)
      elif nn_type=='CONV1D':
        return CONV1D(self)
      elif nn_type=='DCGANAE128':
        return DCGANAE128(self)        
      else: 
        print( f"\033[31;1mCONFIG:         FATAL:  Sorry, there is no neural network model called: '{nn_type}' ... halting now.\033[m" )        
        exit(0)

# ------------------------------------------------------------------------------

    def get_genes_net(self, n_classes, n_genes ):
#        return AELinear(self)
        return DENSE( self, n_classes, n_genes )

# ------------------------------------------------------------------------------

    def get_dataset(self, args):
      print ( "CONFIG:         INFO:   at \033[35;1mget_dataset\033[m")
      return GTExV6Dataset(self, args)

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
        nc = self.N_CHANNELS
        w  = self.IMG_SIZE

        x1_fpath = '%s/%s_images_recon.png' % (directory, desc)
        N = min(x.size(0), 24)                                             # PGD 200614 - Number of images pairs to save for display
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
