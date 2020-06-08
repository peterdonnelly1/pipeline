"""=============================================================================
Deep probabilistic CCA (DPCCJ) for histology images and gene expression levels.
============================================================================="""

import cuda
import torch
from   torch import nn
from   models import PCCA


DEBUG=0
# ------------------------------------------------------------------------------

class DPCCA(nn.Module):

    def __init__(self, cfg, latent_dim, em_iters=1):
        """Initialize Deep Probabilistic CCA model.
        """

        if DEBUG>99:
          print ( "DPCCJ:          INFO  \033[38;1mat top of DPCCJ before super call\033[m" )

        super(DPCCA, self).__init__()

        if DEBUG>99:
          print ( "DPCCJ:          INFO  \033[38;1mafter super call\033[m" )

        print ( latent_dim, cfg.IMG_EMBED_DIM, cfg.N_GENES )
        
        if latent_dim >= cfg.IMG_EMBED_DIM or latent_dim >= cfg.N_GENES:
            msg = 'The latent dimension must be smaller than both the image embedding dimensions and genes dimension.'
            raise AttributeError(msg)

        self.cfg        = cfg                                                                              # VARIABLE: self is DPCCA object model (nn.Module) hence we now have 'model.cfg'
        self.image_net  = cfg.get_image_net()                                                              # METHOD:   get_image_net will return DCGANAE128(self) so self.image_net = self.DCGANAE128
        self.genes_net  = cfg.get_genes_net()                                                              # METHOD:   get_genes_net will return AELinear(self)   so self.genes_net = self.AELinear
        self.latent_dim = latent_dim                                                                       # VARIABLE: self is DPCCA object model (nn.Module) hence we now have 'model.latent_dim'

        if DEBUG>99:
          print ( "DPCCJ:          INFO  \033[38;1mabout to call PCCJ()\033[m" )
        
        self.pcca = PCCA(latent_dim=latent_dim,                                                            # OBJECT:   PCCA is a class, hence self.pcca = model.pcca
                         dims=[cfg.IMG_EMBED_DIM, cfg.GENE_EMBED_DIM],
                         max_iters=em_iters)

        # This initialization is pulled from the DCGAN implementation:                                    # PGD 200106 - What does this do exactly?
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

    def forward(self, x):

        """Perform forward pass of images and associated signal through model.
        'x' holds a tuple of image and gene tensors (x[0] and x[1]
        """
        
        if DEBUG>0:
          print ( "DPCCJ:          INFO:            forward(): x[0].shape = batch_imagesr.shape = {:}".format( x[0].shape ) )
          print ( "DPCCJ:          INFO:            forward(): x[1].shape = batch_genesr.shape  = {:}".format( x[1].shape ) )

        if DEBUG>9:
          print ( "DPCCJ:          INFO:            forward(): image tensor x[0]=\n{:}\nand gene tensor x[1] =\n{:}".format( x[0], x[1] ) )
        
        y = self.encode(x)                                                                                 # self is DPCCA object model (nn.Module) hence 'model.encode(x)'

        if DEBUG>0:
          print ( "DPCCJ:          INFO:            forward(): y[0].shape = encoded x[0].shape = encoded batch_imagesr.shape = {:}".format( y[0].shape ) )
          print ( "DPCCJ:          INFO:            forward(): y[1].shape = encoded x[1].shape = encoded batch_genesr.shape  = {:}".format( y[1].shape ) )
          
        if DEBUG>9:
          print ( "DPCCJ:          INFO:            forward(): encoded tensor y =\n{:}\n".format( y[0], y[1] ) )

        y1r, y2r = self.pcca.forward(y)                                                                    # self is DPCCA object model (nn.Module) hence 'model.pcca.forward(y)'

        if DEBUG>9:
          print ( "DPCCJ:          INFO:            forward(): line 78  =\n{:}\n".format( y[0], y[1] ) )


        x1r = self.image_net.decode(y1r)                                                                   # self is DPCCA object model (nn.Module), and image_net is a DCGANAE128 object hence, 'model.DCGANAE128.decode(y1r)'
 
        if DEBUG>9:
          print ( "DPCCJ:          INFO:            forward(): line 82  =\n{:}\n".format( y[0], y[1] ) )

        x2r = self.genes_net.decode(y1r)                                                                   # self is DPCCA object model (nn.Module), and genes_net is a AELinear object, hence 'model.AELinear.decode(y1r)'
 
        if DEBUG>9:
          print ( "DPCCJ:          INFO:            forward(): line 85  =\n{:}\n".format( y[0], y[1] ) )

        return x1r, x2r

# ------------------------------------------------------------------------------

    def encode(self, x):
        """Embed data in preparation for PCCJ.
        """
        
        x1, x2 = x

        if DEBUG>0:
          print ( "DPCCJ:          INFO:                encode(): x1.shape = {:}".format( x1.shape ) )
          print ( "DPCCJ:          INFO:                encode(): x2.shape = {:}".format( x2.shape ) )

        if DEBUG>9:
          print ( "DPCCJ:          INFO:                encode(): x1 =\n{:}\n".format( x1 ) )
          print ( "DPCCJ:          INFO:                encode(): x2 =\n{:}\n".format( x2 ) )

        y1 = self.image_net.encode(x1)                                                                     # self is DPCCA object model (nn.Module), and image_net is a DCGANAE128 object hence, 'model.DCGANAE128.encode(x1)'

        if DEBUG>0:
          print ( "DPCCJ:          INFO:                encode(): y1.shape [encoded version of x1 (image)] = {:}".format( y1.shape ) )
                 
        if DEBUG>9:
          print ( "DPCCJ:          INFO:                encode(): encoded tensor y1 =\n{:}\n".format( y1 ) )
        
        y2 = self.genes_net.encode(x2)                                                                     # self is DPCCA object model (nn.Module), and genes_net is a AELinear object, hence 'model.AELinear.encode(x2)'

        if DEBUG>0:
          print ( "DPCCJ:          INFO:                encode(): y2.shape [encoded version of x2 (gene) ] = {:}".format( y2.shape ) )
          
        if DEBUG>9:
          print ( "DPCCJ:          INFO:                encode(): encoded tensor y2 = \n{:}\n".format( y2 ) )
          
        # PCCJ assumes our data is mean-centered.
        y1 = y1 - y1.mean(dim=0)
        y2 = y2 - y2.mean(dim=0)

        y = torch.cat([y1, y2], dim=1)

        # PCCJ expects (p-dims, n-samps)-dimensional data.
        y = y.t()

        return y

# ------------------------------------------------------------------------------

    def sample(self, x, n_samples=None):
        """Sample from fitted PCCJ-VAE model.
        """
        x1, x2 = x
        y1 = self.image_net.encode(x1)                                                                     # self is DPCCA object model (nn.Module), and image_net is a DCGANAE128 object hence, 'model.DCGANAE128.encode(x1)                                                        
        y2 = self.genes_net.encode(x2)                                                                     # self is DPCCA object model (nn.Module), and genes_net is a AELinear object,  hence 'model.AELinear.encode(x2)'

        if not n_samples:
            n_samples = x1.shape[0]

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
