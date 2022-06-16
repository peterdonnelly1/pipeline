"""=============================================================================
LeNet5 for histology images
============================================================================="""

from   torch import nn

from constants  import *

DEBUG   = 2

# ------------------------------------------------------------------------------

class COMMON( nn.Module ):

  def __init__(self, args, cfg, input_mode, nn_type_img, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2, tile_size, latent_dim, em_iters=1 ):

    """Initialize model
    """

    super(COMMON, self).__init__()

    
    self.cfg  = cfg
    
    if ( input_mode=='image' ):
      
      if DEBUG>10:
        print ( f"COMMON:         INFO:       about to call model for image processing{RESET}", flush=True )                 # get_image_net method is in config. Will try to call init on the selected model (e.g. TTVAE) with these parameters 
      self.image_net  = cfg.get_image_net( args, input_mode, nn_type_img, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2, tile_size )            # METHOD:   get_image_net will return DCGANAE128(self) so self.image_net = self.DCGANAE128

    if ( input_mode=='image_rna' ) | ( input_mode=='rna' ):
      
      if DEBUG>10:
        print ( f"COMMON:             INFO:    about to call model for genes/embeddings processing{RESET}", flush=True )      # get_image_net method is in config. Will try to call init on the selected model (e.g. TTVAE) with these parameters 
      self.genes_net  = cfg.get_genes_net( args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, embedding_dimensions, nn_dense_dropout_1, nn_dense_dropout_2             )            # METHOD:   get_genes_net will return DENSE(self)   so model.genes_net = get_genes_net(...)

    self.latent_dim = latent_dim                                                       #                                                                                                  model.latent_dim               = latent_dim        (as passed in)

    if DEBUG>99:
      print ( "COMMON:    INFO        \033[38;1mabout to call LENET5()\033[m" )
    
    
    # This initialization is pulled from the DCGAN implementation:
    #
    #    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    #
    for m in self.modules():
        classname = m.__class__.__name__
        if not classname.find('BasicConv2d') != -1:                                                        # PGD 200219 to fix the issue below. It was catching 'BasicConv2d'
          if classname.find('Conv') != -1:
              m.weight.data.normal_(0.0, 0.02)                                                             # crashing here for INCEPT3: "AttributeError: 'BasicConv2d' object has no attribute 'weight'"
          elif classname.find('BatchNorm') != -1:
              m.weight.data.normal_(1.0, 0.02)
              m.bias.data.fill_(0)

# ------------------------------------------------------------------------------

  def forward(self, x, gpu, encoder_activation):

    """Perform forward pass of images through model.
    'x1' holds images and 'x2' holds genes, if either is defined (int 0 otherwise in each case)
    """

    if DEBUG>99:
      print ( f"COMMON:         INFO:      forward(): x.type = {type(x)}", flush=True )

    x1, x2, batch_fnames = x
    y1     = 0                                                                                             # int 0 as dummy value to return if we are doing gene  only
    y2     = 0                                                                                             # int 0 as dummy value to return if we are doing image only

    if DEBUG>99:
      print ( f"COMMON:         INFO:      forward(): x1.type   = {MIKADO}{type(x1)}{RESET}",   flush=True )
      print ( f"COMMON:         INFO:      forward(): x2.type   = {MIKADO}{type(x2)}{RESET}",   flush=True )

    if not (type(x1)==int):                                                                                # then it's an image tensor and we should process it
      if DEBUG>99:
        print ( f"{CARRIBEAN_GREEN}COMMON:         INFO:      forward(): type(x1)!=int, therefore process as an image tensor{RESET}", flush=True )
      y1, embedding = self.image_net.forward   ( x1, batch_fnames )      
    if not (type(x2)==int):                                                                                # then it's an rna-seq tensor and we should process it
      if DEBUG>99:
        print ( f"{BITTER_SWEET}COMMON:         INFO:      forward(): type(x2)!=int, therefore process as an rna tensor{RESET}",      flush=True )
      y2, embedding, _ = self.genes_net.forward( x2, gpu, encoder_activation )
   

    return y1, y2, embedding

# ------------------------------------------------------------------------------

  def encode( self, x, input_mode, gpu, encoder_activation ):

    if input_mode=='image':
      z = self.image_net.encode( x, gpu, encoder_activation  )

    if input_mode=='rna':
      z = self.genes_net.encode( x, gpu, encoder_activation   )

    if DEBUG>99:
      print ( f"COMMON:         INFO:       encode(): z.shape [encoded version of x] = {MIKADO}{z.shape}{RESET}"  )
    if DEBUG>99:
      print ( f"COMMON:         INFO:       encode(): z [encoded tensor] =\n{MIKADO}{z}{RESET}" ) 
              
    return z
  
