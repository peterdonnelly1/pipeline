"""============================================================================
Configuration for the DLBC data set with LENET  
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from   torchvision.utils import save_image

from   models import LENET5, AELinear
from   data.dlbcl_image.dataset import GTExV6Dataset     # NEW
from   data.config import Config

DEBUG=1

# ------------------------------------------------------------------------------

class GTExV6Config(Config):

    # Class variables: only parameters that will not change across an entire job (job = many runs of the model)

    ROOT_DIR       = 'data/dlbcl_image'
#   N_SAMPLES      = 5900   # this is the number of examples << DOESN'T SEEM TO BE USED
    IMG_SIZE       = 128
    N_CHANNELS     = 3
    N_PIXELS       = 3 * IMG_SIZE * IMG_SIZE
    IMG_EMBED_DIM  = 3           # Has to be the same as the number of classes. For both 'eye' and 'dlbc' we have 3 classes: 0, 1 and 2
    N_GENES        = 60482
    GENE_EMBED_DIM = 1000         # PGD THIS WAS ORIGINALLY 1000

    # Instance variables: parameters that may change from run to run (such as learning rate or batch_size) 

    def __init__(self, lr,  batch_size ):
   
      if DEBUG>0:
        print( "GTEXV6CONFIG:   INFO:   __init__():   current learning rate / batch_size  = \033[35;1m{:}, {:}\033[m respectively".format( lr,  batch_size ) )

# ------------------------------------------------------------------------------

    def get_image_net(self):
        #return DCGANAE128(self)
        return LENET5(self)
        
# ------------------------------------------------------------------------------

    def get_genes_net(self):
        return AELinear(self)

# ------------------------------------------------------------------------------

    def get_dataset(self, **kwargs):
      print ( "CONFIG:         INFO:   at \033[33;1mget_dataset\033[m")
      return GTExV6Dataset(self)

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
        N = min(x.size(0), 8)
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
