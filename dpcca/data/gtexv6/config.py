"""============================================================================
Configuration for the GTEx V6 data set.
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from   torchvision.utils import save_image

from   models import DCGANAE128, AELinear
from   data.gtexv6.dataset import GTExV6Dataset
from   data.config import Config

from   data.gtexv6.dataset import GTExV6Dataset                                                            # use for any dataset in dpcca mode

DEBUG=1

# ------------------------------------------------------------------------------

class GTExV6Config(Config):

#    ROOT_DIR       = 'data/gtexv6'
    ROOT_DIR       = '/home/peter/git/pipeline/dpcca/data/dlbcl_image'    # PGD 200609 ##############################################################################################################################################
#   N_SAMPLES      = 4800   # this is the number of examples << DOESN'T SEEM TO BE USED
    IMG_SIZE       = 128
    N_CHANNELS     = 3
    N_PIXELS       = 3 * IMG_SIZE * IMG_SIZE
    #N_GENES        = 60482
    N_GENES        = 506                                                                                   # PGD 200609 ##############################################################################################################################################
    IMG_EMBED_DIM  = 1000
    GENE_EMBED_DIM = 1000                                                                                  # PGD THIS WAS ORIGINALLY 1000
    
# ------------------------------------------------------------------------------

    def get_image_net(self):
        return DCGANAE128(self)

# ------------------------------------------------------------------------------

    def get_genes_net(self):
        return AELinear(self)

# -------------------------------------------------------------------------------

    def get_dataset(self, args):
      print ( "CONFIG:  	INFO:   at \033[33;1mget_dataset\033[m")
      return GTExV6Dataset(self)

# ------------------------------------------------------------------------------

    def save_samples(self, directory, model, desc,  x1,           x2,          labels ):
   #cfg.save_samples(      directory, model, epoch, images_batch, genes_batch, labels )


      if DEBUG>0:
        print( "CONFIG:         INFO:       at top of save_samples() and parameter directory = \033[35;1m{:}\033[m".format( directory ) )
        
        n_samples = len(x1)
        nc        = self.N_CHANNELS
        w         = self.IMG_SIZE

        # Visualize images.
        # -----------------
        
        x1r, x2r = model.sample([x1, x2], n_samples)                                                       #  See: dpcca.py.sample - "Sample from fitted PCCJ-VAE model" - returns reconstructed versions of x1 and x2
        x1r = x1r.view(n_samples, nc, w, w)
        fname = '%s/sample_images_%s.png' % (directory, desc)
        save_image(x1r.cpu(), fname)


# ------------------------------------------------------------------------------
    # NOT USED
    def save_image_samples(self, directory, model, desc, x1):
        n_samples = 64
        nc = self.N_CHANNELS
        w = self.IMG_SIZE

        x1_recon, _ = model.sample(None, n_samples)
        x1_recon = x1_recon.view(n_samples, nc, w, w)
        fname = '%s/sample_%s.png' % (directory, desc)
        save_image(x1_recon.cpu(), fname)

# ------------------------------------------------------------------------------

    def save_comparison(self,      directory,  x,             x_recon,  desc,   is_x1=None):
#  cfg.save_comparison(       args.directory,  batch_images,  y1,       epoch,  is_x1=True)
#  cfg.save_comparison(       args.directory,  batch_genes,   y2,       epoch,  is_x1=False)

        """Save image samples from learned image likelihood.
        """

        if DEBUG>99:
          print( "CONFIG:  	INFO:        save_comparison(): about to save comparisons  " )

        if is_x1:
            self.save_image_comparison(directory, x, x_recon, desc)
        else:
            self.save_genes_comparison(directory, x, x_recon, desc)

# ------------------------------------------------------------------------------

    def save_image_comparison(self, directory, x, x_recon, desc):
      
      
        nc = self.N_CHANNELS
        w  = self.IMG_SIZE

        if DEBUG>2:
          print( "CONFIG:  	INFO:          save_image_comparison(): about to save image comparisons  " )

        x1_fpath = '%s/%s_images_recon.png' % (directory, desc)
        N = min(x.size(0), 8)                                                                              # PGD 200131 - it's the number of image pairs to show in the panel of images. Saved in applicable logs subfolder
        #N = min(x.size(0), 32)
        
        recon = x_recon.view(-1, nc, w, w)[:N]
        
        x = x.view(-1, nc, w, w)[:N]
        
        comparison = torch.cat([x, recon])                                                                 # join the two sets of images together side by side for easy visual inspection
        
        save_image(comparison.cpu(), x1_fpath, nrow=N)                                                     # save to file

# ------------------------------------------------------------------------------

    def save_genes_comparison(self, directory, x, xr, desc):


        if DEBUG>2:
          print( "CONFIG:  	INFO:          save_genes_comparison(): about to save gene comparisons  " )

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
