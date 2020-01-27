"""=============================================================================
LeNet5 for histology images
============================================================================="""

import cuda
import torch
from   torch import nn
from   models import LNETIMG


DEBUG=1
# ------------------------------------------------------------------------------

class LENETIMAGE(nn.Module):

    def __init__(self, cfg, latent_dim, em_iters=1):
        """Initialize LeNet5 model.
        """

        if DEBUG>99:
          print ( "LENETIMAGE:          INFO  \033[38;1mat top of LENET5 before super call\033[m" )

        super(LENETIMAGE, self).__init__()

        if DEBUG>99:
          print ( "LENETIMAGE:          INFO  \033[38;1mafter super call\033[m" )
        
        if DEBUG>0:
          print ( "LENETIMAGE:     INFO:   latent_dim = {:}, cfg.IMG_EMBED_DIM = {:}, cfg.N_GENES = {:}".format( latent_dim, cfg.IMG_EMBED_DIM, cfg.N_GENES  ) )        
        
        if latent_dim >= cfg.IMG_EMBED_DIM or latent_dim >= cfg.N_GENES:
          msg = 'The latent dimension must be smaller than both the image embedding dimensions and genes dimension.'
          raise AttributeError(msg)

        self.cfg        = cfg
        self.image_net  = cfg.get_image_net()                                                              # get_image_net will return DCGANAE128(self) so that self.image_net = self.DCGANAE128
        self.genes_net  = cfg.get_genes_net()                                                              # get_genes_net will return AELinear(self)   so that self.genes_net = self.AELinear
        self.latent_dim = latent_dim

        if DEBUG>99:
          print ( "LENETIMAGE:          INFO  \033[38;1mabout to call LENET5()\033[m" )
        
        self.lnetimg = LNETIMG(latent_dim=latent_dim,
                         dims=[cfg.IMG_EMBED_DIM, cfg.GENE_EMBED_DIM],
                         max_iters=em_iters)

        # This initialization is pulled from the DCGAN implementation:
        #
        #    https://github.com/pytorch/examples/blob/master/dcgan/main.py
        #
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

# ------------------------------------------------------------------------------

    def forward(self, x):

        """Perform forward pass of images and associated signal through model.
        'x' holds a tuple of image and gene tensors (x[0] and x[1]
        """
        
        if DEBUG>9:
          print ( "LENETIMAGE:     INFO:           forward(): x.type = {:}".format( type(x) ) )
          print ( "LENETIMAGE:     INFO:           forward(): x.size = {:}".format( x.size() ) )
          #print ( "LENETIMAGE:    INFO:           forward(): x[0].shape = batch_imagesr.shape = {:}".format( x[0].shape ) )
          #print ( "LENETIMAGE:     INFO:           forward(): x[1].shape = batch_genesr.shape  = {:}".format( x[1].shape ) )

        if DEBUG>9:
          print ( "LENETIMAGE:     INFO:           forward(): image tensor x[0]=\n{:}\nand gene tensor x[1] =\n{:}".format( x[0], x[1] ) )
        
        y = self.encode(x)
        
        #y1 = y[0]
        #y2 = y[1]

        '''
        if DEBUG>0:
          print ( "LENETIMAGE:          INFO:            forward(): y[0].shape = encoded x[0].shape = encoded batch_imagesr.shape = {:}".format( y[0].shape ) )
          print ( "LENETIMAGE:          INFO:            forward(): y[1].shape = encoded x[1].shape = encoded batch_genesr.shape  = {:}".format( y[1].shape ) )
          
        if DEBUG>9:
          print ( "LENETIMAGE:          INFO:            forward(): encoded tensor y =\n{:}\n".format( y[0], y[1] ) )

        y1r, y2r = self.lnetimg.forward(y)

        if DEBUG>9:
          print ( "LENETIMAGE:          INFO:            forward(): line 78  =\n{:}\n".format( y[0], y[1] ) )


        x1r = self.image_net.decode(y1r)
 
        if DEBUG>9:
          print ( "LENETIMAGE:          INFO:            forward(): line 82  =\n{:}\n".format( y[0], y[1] ) )

        x2r = self.genes_net.decode(y1r)
 
        if DEBUG>9:
          print ( "LENETIMAGE:          INFO:            forward(): line 85  =\n{:}\n".format( y[0], y[1] ) )

        return x1r, x2r
        '''

        #return y1, y2
        return y
# ------------------------------------------------------------------------------

    def encode(self, x):
        """Embed data in preparation for LeNet5.
        """
        
        #x1, x2 = x  #NEW


        if DEBUG>9:
          print ( "LENETIMAGE:     INFO:            encode(): x.size = {:}".format( x.size() ) )
          #print ( "LENETIMAGE:          INFO:                encode(): x1.shape = {:}".format( x1.shape ) )
          #print ( "LENETIMAGE:          INFO:                encode(): x2.shape = {:}".format( x2.shape ) )

        if DEBUG>9:
          print ( "LENETIMAGE:          INFO:                encode(): x1 =\n{:}\n".format( x ) )
          #print ( "LENETIMAGE:          INFO:                encode(): x2 =\n{:}\n".format( x2 ) )

        y1 = self.image_net.encode(x)    # image_net will return LENET(self), so self.image_net.encode = LENET.encode(x1)

        if DEBUG>9:
          print ( "LENETIMAGE:     INFO:            encode(): y1.shape [encoded version of x1] = {:}".format( y1.shape ) )
                 
        if DEBUG>9:
          print ( "LENETIMAGE:          INFO:                encode(): encoded tensor y1 =\n{:}\n".format( y1 ) )
        
        #y2 = self.genes_net.encode(x2)    # genes_net will return AELinear(self), so self.genes_net.encode = AELinear.encode(x2) = fc1(x)

        #if DEBUG>0:
          #print ( "LENETIMAGE:          INFO:                encode(): y2.shape [encoded version of x2 (gene) ] = {:}".format( y2.shape ) )
          
        #if DEBUG>9:
         # print ( "LENETIMAGE:          INFO:                encode(): encoded tensor y2 = \n{:}\n".format( y2 ) )
          
        # LeNet5 assumes our data is mean-centered.
        y1 = y1 - y1.mean(dim=0)
        #y2 = y2 - y2.mean(dim=0)

        #y = torch.cat([y1, y2], dim=1)    ## NEW - DON'T CONCATENATE THE ENCODED IMAGES WITH THE ENCODED GENES
        y = y1    ## NEW

        # LeNet5 expects (p-dims, n-samps)-dimensional data.
        y = y.t()

        return y

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
