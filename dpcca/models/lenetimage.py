"""=============================================================================
LeNet5 for histology images
============================================================================="""

import cuda
import torch
from   torch import nn
from   models import LNETIMG

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
ARYLIDE='\033[38;2;233;214;107m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
PALE_GREEN='\033[32m'

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1
# ------------------------------------------------------------------------------

class LENETIMAGE(nn.Module):

    def __init__(self, args, cfg, input_mode, nn_type_img, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, latent_dim, em_iters=1 ):

        """Initialize LeNet5 model
        """

        if DEBUG>99:
          print ( "LENETIMAGE:          INFO  \033[38;1mat top of LENET5 before super call\033[m" )

        super(LENETIMAGE, self).__init__()

        if DEBUG>99:
          print ( "LENETIMAGE:          INFO  \033[38;1mafter super call\033[m" )
        
        if DEBUG>88:
          print ( "LENETIMAGE:     INFO:   latent_dim = {:}, cfg.IMG_EMBED_DIM = {:}, cfg.N_GENES = {:}".format( latent_dim, cfg.IMG_EMBED_DIM, cfg.N_GENES  ) )        
        
        if latent_dim >= cfg.IMG_EMBED_DIM or latent_dim >= cfg.N_GENES:
          msg = 'The latent dimension must be smaller than both the image embedding dimensions and genes dimension.'
          raise AttributeError(msg)

        self.cfg        = cfg                                                              #                                                                                                  model.cfg                      = cfg               (as passed in)
        if ( input_mode=='image_rna' ) | ( input_mode=='image' ):  
          if DEBUG>0:
            print ( f"LENETIMAGE:     INFO       about to call model for image net{RESET}", flush=True )      # get_image_net method is in config. Will try to call init on the selected model (e.g. TTVAE) with these parameters 
          self.image_net  = cfg.get_image_net( args, input_mode, nn_type_img, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, tile_size )            # METHOD:   get_image_net will return DCGANAE128(self) so self.image_net = self.DCGANAE128
        if ( input_mode=='image_rna' ) | ( input_mode=='rna' ): 
          if DEBUG>0:
            print ( f"LENETIMAGE:     INFO        about to call model for genes net{RESET}", flush=True )      # get_image_net method is in config. Will try to call init on the selected model (e.g. TTVAE) with these parameters 
          self.genes_net  = cfg.get_genes_net( args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2             )            # METHOD:   get_genes_net will return DENSE(self)   so model.genes_net = get_genes_net(...)
        self.latent_dim = latent_dim                                                       #                                                                                                  model.latent_dim               = latent_dim        (as passed in)

        if DEBUG>99:
          print ( "LENETIMAGE:    INFO        \033[38;1mabout to call LENET5()\033[m" )
        
        self.lnetimg = LNETIMG(latent_dim=latent_dim,                                                      # model.lnetimg = etc
                         dims=[cfg.IMG_EMBED_DIM, cfg.GENE_EMBED_DIM],
                         max_iters=em_iters)

        # This initialization is pulled from the DCGAN implementation:
        #
        #    https://github.com/pytorch/examples/blob/master/dcgan/main.py
        #
        for m in self.modules():
            classname = m.__class__.__name__
            if not classname.find('BasicConv2d') != -1:                                                      # PGD 200219 to fix the issue below. It was catching 'BasicConv2d'
              if classname.find('Conv') != -1:
                  m.weight.data.normal_(0.0, 0.02)                                                           # crashing here for INCEPT3: "AttributeError: 'BasicConv2d' object has no attribute 'weight'"
              elif classname.find('BatchNorm') != -1:
                  m.weight.data.normal_(1.0, 0.02)
                  m.bias.data.fill_(0)

# ------------------------------------------------------------------------------

    def forward(self, x, gpu, encoder_activation):

        """Perform forward pass of images through model.
        'x1' holds images and 'x2' holds genes, if either is defined (int 0 otherwise in each case)
        """

        if DEBUG>88:
          print ( f"LENETIMAGE:     INFO:           forward(): x.type = {type(x)}", flush=True )

        x1, x2, batch_fnames = x
        y1     = 0                                                                                         # int 0 as dummy value to return if we are doing gene  only
        y2     = 0                                                                                         # int 0 as dummy value to return if we are doing image only
        
        if not (type(x1)==int):                                                                            # then it's an image tensor and we should process it
          y1 = self.image_net.forward(x1, batch_fnames)      
        if not (type(x2)==int):                                                                            # then it's a   gene tensor and we should process it
          y2, _, _ = self.genes_net.forward(x2, gpu )
        

        return y1, y2

# ------------------------------------------------------------------------------


    def sample(self, x, n_samples=None):
		
        """Sample from fitted PCCA-VAE model
        """
        x1, x2 = x
        y1 = self.image_net.encode(x1)
        y2 = self.genes_net.encode(x2)

        if not n_samples:
            n_samples = x1.shape[0]

        return self._sample(y1, y2, n_samples, False)

# ------------------------------------------------------------------------------

    def sample_x1_from_x2(self, x2):
		
        """Sample images based on gene expression data
        """
        device = cuda.device()
        y1 = torch.zeros(x2.shape[0], self.cfg.IMG_EMBED_DIM, device=device)
        y2 = self.genes_net.encode(x2)
        x1r, _ = self._sample(y1, y2, n_samples=None, sample_across=True)
        return x1r

# ------------------------------------------------------------------------------

    def sample_x2_from_x1(self, x1):
        """Sample gene expression data from images. 
        """
        device = cuda.device()
        y1 = self.image_net.encode(x1)
        y2 = torch.zeros(x1.shape[0], self.cfg.GENE_EMBED_DIM, device=device)
        _, x2r = self._sample(y1, y2, n_samples=None, sample_across=True)
        return x2r

# ------------------------------------------------------------------------------

    def _sample(self, y1, y2, n_samples, sample_across):
        """Utility function for all sampling methods. Takes a pair of embeddings
        and returns a pair of reconstructed samples.
        """
        assert not y1.requires_grad
        assert not y2.requires_grad
        y1 = y1 - y1.mean(dim=0)
        y2 = y2 - y2.mean(dim=0)
        y  = torch.cat([y1, y2], dim=1)
        y  = y.t()
        if sample_across:
            y1r, y2r = self.lnetimg.sample(y, one_sample_per_y=True)
        else:
            y1r, y2r = self.lnetimg.sample(y, n_samples=n_samples)
        x1r = self.image_net.decode(y1r)
        x2r = self.genes_net.decode(y2r)

        return x1r, x2r

# ------------------------------------------------------------------------------

    def estimate_z_given_x(self, x):
        """Estimate the latent variable z given our data x.
        """
        y = self.encode(x)
        return self.lnetimg.estimate_z_given_y(y).t()

# ------------------------------------------------------------------------------

    def neg_log_likelihood(self, x):
        """Compute the negative log-likelihood of the data given our current
        parameters.
        """
        y = self.encode(x)
        return self.lnetimg.neg_log_likelihood(y)
