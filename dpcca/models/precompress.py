"""=============================================================================
Deep probabilistic CCA (DPCCJ) for histology images and gene expression levels.
============================================================================="""

import cuda
import torch
from   torch import nn
from   models import PCCA

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
ORANGE='\033[38;2;204;85;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
RESET='\033[m'

UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1
# ------------------------------------------------------------------------------

class PRECOMPRESS(nn.Module):

    def __init__(self, args, cfg, input_mode, nn_type, encoder_activation, n_classes, n_genes, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, latent_dim, em_iters=1):
        """Initialize Deep Probabilistic CCA model.
        """

        if DEBUG>2:
          print ( "PRECOMPRESS:          INFO  \033[38;1mat top of DPCCJ before super call\033[m" )

        super(PRECOMPRESS, self).__init__()

        if DEBUG>2:
          print ( "PRECOMPRESS:          INFO  \033[38;1mafter super call\033[m" ) 
          print ( f"PRECOMPRESS:          INFO  latent_dim, cfg.IMG_EMBED_DIM, cfg.N_GENES = {latent_dim}, {cfg.IMG_EMBED_DIM}, {cfg.N_GENES}" )
        
        if latent_dim >= cfg.IMG_EMBED_DIM or latent_dim >= cfg.N_GENES:
            msg = 'The latent dimension must be smaller than the embedding dimension'
            raise AttributeError(msg)

        self.cfg = cfg                                                                                     # VARIABLE: self is DPCCA object model (nn.Module) hence we now have 'model.cfg'
        if ( input_mode=='image_rna' ) | ( input_mode=='image' ):  
          if DEBUG>0:
            print ( f"PRECOMPRESS:          INFO  about to call model for image net{RESET}" )      # get_image_net method is in config. Will try to call init on the selected model (e.g. TTVAE) with these parameters 
          self.image_net  = cfg.get_image_net( args, input_mode, nn_type, encoder_activation, n_classes, n_genes, nn_dense_dropout_1, nn_dense_dropout_2, tile_size )            # METHOD:   get_image_net will return DCGANAE128(self) so self.image_net = self.DCGANAE128
        if ( input_mode=='image_rna' ) | ( input_mode=='rna' ): 
          if DEBUG>0:
            print ( f"PRECOMPRESS:          INFO  about to call model for genes net{RESET}" )      # get_image_net method is in config. Will try to call init on the selected model (e.g. TTVAE) with these parameters 
          self.genes_net  = cfg.get_genes_net( args, input_mode, nn_type, encoder_activation, n_classes, n_genes, nn_dense_dropout_1, nn_dense_dropout_2             )            # METHOD:   get_genes_net will return DENSE(self)   so model.genes_net = get_genes_net(...)

        self.latent_dim = latent_dim                                                                       # VARIABLE: self is DPCCA object model (nn.Module) hence we now have 'model.latent_dim'

        if DEBUG>2:
          print ( "PRECOMPRESS:          INFO  \033[38;1mabout to call PCCJ()\033[m" )
        
        self.pcca = PCCA (
                           latent_dim = latent_dim,                                                            # OBJECT:   PCCA is a class, hence self.pcca = model.pcca
                           dims       = [cfg.IMG_EMBED_DIM, cfg.GENE_EMBED_DIM],
                           max_iters  = em_iters
                         )
      

        # This initialization is pulled from the DCGAN implementation:                                     # PGD 200106 - What does this do exactly?
        #
        #    https://github.com/pytorch/examples/blob/master/dcgan/main.py
        #
        for m in self.modules():                                                                           # METHOD:    self is DPCCA object and modues is a built in function. See https://blog.paperspace.com/pytorch-101-advanced/
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

# ------------------------------------------------------------------------------

    def forward(self, x, encoder_activation):
      
        if DEBUG>9:
          print ( f"PRECOMPRESS:          INFO:    forward(): x.shape     = {CYAN}{x.shape}{RESET}", flush=True   ) 

        x2r, mean, logvar = self.genes_net.forward( x, encoder_activation )                                # self is DPCCA object model (nn.Module), and genes_net is a AELinear object, hence 'model.AELinear.encode(y)'

        #z = self.genes_net.encode( x, encoder_activation )                                                # self is DPCCA object model (nn.Module), and genes_net is a AELinear object, hence 'model.AELinear.encode(y)'

        #if DEBUG>0:
        #  print ( f"AELINEAR:       INFO:    forward(): z.shape     = {CYAN}{z.shape}{RESET}", flush=True   ) 

        #x = self.genes_net.decode(z)                                                                       # self is DPCCA object model (nn.Module), and genes_net is a AELinear object, hence 'model.AELinear.decode(z)'
 
        if DEBUG>9:
          print ( f"PRECOMPRESS:          INFO:    forward(): x.shape     = {CYAN}{x.shape}{RESET}", flush=True   ) 

        return x2r, mean, logvar
        
# ------------------------------------------------------------------------------

    def encode(self, x):  ## NOT USED


        y = self.image_net.encode(x)                                                                       # self is DPCCA object model (nn.Module), and image_net is a DCGANAE128 object hence, 'model.DCGANAE128.encode(x)'

        if DEBUG>9:
          print ( "PRECOMPRESS:          INFO:                encode(): y.shape [encoded version of x (image)] = {:}".format( y.shape ) )
                 
        if DEBUG>9:
          print ( "PRECOMPRESS:          INFO:                encode(): encoded tensor y =\n{:}\n".format( y ) )
                  
        y = y - y.mean(dim=0)

        # PCCJ expects (p-dims, n-samps)-dimensional data.
        y = y.t()

        return y

# ------------------------------------------------------------------------------

    def sample(self, x, n_samples=None):
        """Sample from fitted PCCJ-VAE model.
        """
        x1, x2 = x                                                                                         # extract image and gene components
        y1 = self.image_net.encode(x1)                                                                     # encode: i.e. 'model.DCGANAE128.encode(x1)                                                        
        y2 = self.genes_net.encode(x2)                                                                     # encode: i.e. 'model.AELinear.encode(x2)'

        if not n_samples:                                                                                  # IF n_samples is not defined
            n_samples = x1.shape[0]                                                                        # define it as being the first parameter of x1 (bit weird)

        return self._sample(y1, y2, n_samples, False)

# ------------------------------------------------------------------------------

    def sample_x1_from_x2(self, x2):
        """Sample images based on gene expression data.
        """
        device = cuda.device()
        with torch.no_grad():
          y1 = torch.zeros(x2.shape[0], self.cfg.IMG_EMBED_DIM, device=device)                               # self is DPCCA object model (nn.Module) hence self.cfg.IMG_EMBED_DIM = 'model.cfg.IMG_EMBED_DIM'
          y2 = self.genes_net.encode(x2)                                                                     # self is DPCCA object model (nn.Module), and genes_net is a AELinear object,  hence 'model.AELinear.encode(x2)'
          x1r, _ = self._sample(y1, y2, n_samples=None, sample_across=True)
        return x1r

# ------------------------------------------------------------------------------

    def sample_x2_from_x1(self, x1):
        """Sample gene expression data from images.
        """
        device = cuda.device()
        with torch.no_grad():
          y1 = self.image_net.encode(x1)                                                                     # self is DPCCA object model (nn.Module), and image_net is a DCGANAE128 object hence, 'model.DCGANAE128.encode(x1)
          y2 = torch.zeros(x1.shape[0], self.cfg.GENE_EMBED_DIM, device=device)                              # self is DPCCA object model (nn.Module) hence self.cfg.IMG_EMBED_DIM = 'model.cfg.IMG_EMBED_DIM'
          _, x2r = self._sample(y1, y2, n_samples=None, sample_across=True)                                  # self is DPCCA object model (nn.Module) hence self._sample = 'model._sample'
        return x2r

# ------------------------------------------------------------------------------

    def _sample(self, y1, y2, n_samples, sample_across):

        """Utility function for all sampling methods. Takes a pair of embeddings
        and returns a pair of reconstructed samples.
        """
        
        if DEBUG>0:
          print ( "sampling ..." )

        assert not y1.requires_grad
        assert not y2.requires_grad
        y1 = y1 - y1.mean(dim=0)
        y2 = y2 - y2.mean(dim=0)
        with torch.no_grad():
          y  = torch.cat([y1, y2], dim=1)
          y  = y.t()
        if sample_across:
            y1r, y2r = self.pcca.sample(y, one_sample_per_y=True)                                          # self is DPCCA object model (nn.Module), hence, 'model.pcca.sample(y, one_sample_per_y=True)                             
        else:
            y1r, y2r = self.pcca.sample(y, n_samples=n_samples)                                            # self is DPCCA object model (nn.Module), hence, 'model.pcca.sample(y, n_samples=n_samples)
        x1r = self.image_net.decode(y1r)                                                                   # self is DPCCA object model (nn.Module), and image_net is a DCGANAE128 object hence, 'model.DCGANAE128.decode(x1)
        x2r = self.genes_net.decode(y2r)                                                                   # self is DPCCA object model (nn.Module), and genes_net is a AELinear object,  hence 'model.AELinear.decode(x2)'

        return x1r, x2r

# ------------------------------------------------------------------------------

    def estimate_z_given_x(self, x):
        """Estimate the latent variable z given our data x.
        """
        y = self.encode(x)
        return self.pcca.estimate_z_given_y(y).t()                                                         # self is DPCCA object model (nn.Module), hence, 'model.pcca.estimate_z_given_y(y).t()'

# ------------------------------------------------------------------------------

    def neg_log_likelihood(self, x):
        """Compute the negative log-likelihood of the data given our current
        parameters.
        """
        y = self.encode(x)                                                                                 # self is DPCCA object model (nn.Module), and image_net is a DCGANAE128 object hence, 'model.DCGANAE128.encode(x)
        return self.pcca.neg_log_likelihood(y)                                                             # self is DPCCA object model (nn.Module), hence, 'model.pcca.neg_log_likelihood(y)'
