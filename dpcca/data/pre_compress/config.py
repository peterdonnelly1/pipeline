"""============================================================================
Configuration for the DLBC data set with LENET  
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from   torchvision.utils import save_image

from   models                    import LENET5, AELINEAR, AEDENSE, AEDENSEPOSITIVE, AE3LAYERCONV2D, AEDEEPDENSE, TTVAE, VGG, VGGNN, INCEPT3, DENSE, DENSEPOSITIVE, CONV1D, DCGANAE128
from   data.pre_compress.dataset import pre_compressDataset
from   models.vggnn import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, make_layers, configs
#from   models.incept3 import incept3
from   data.config import Config


WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
ASPARAGUS='\033[38;2;135;169;107m'
CHARTREUSE='\033[38;2;223;255;0m'
COQUELICOT='\033[38;2;255;56;0m'
COTTON_CANDY='\033[38;2;255;188;217m'
HOT_PINK='\033[38;2;255;105;180m'
CAMEL='\033[38;2;193;154;107m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
ARYLIDE='\033[38;2;233;214;107m'
BLEU='\033[38;2;49;140;231m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
BITTER_SWEET='\033[38;2;254;111;94m'
PALE_RED='\033[31m'
DARK_RED='\033[38;2;120;0;0m'
ORANGE='\033[38;2;255;103;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;19;136;8m'
BRIGHT_GREEN='\033[38;2;102;255;0m'
CARRIBEAN_GREEN='\033[38;2;0;204;153m'
PALE_GREEN='\033[32m'
GREY_BACKGROUND='\033[48;2;60;60;60m'


BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'


DEBUG=1

# ------------------------------------------------------------------------------

class pre_compressConfig(Config):

    # Class variables: only parameters that will not change across an entire job (job = many runs of the model)

    ROOT_DIR       = 'data/pre_compress'
    
#    INPUT_MODE     = 'rna'                                                                               # valid values are 'image', 'rna', 'image_rna'
    
    # ~ IMG_SIZE       =  128
    # ~ N_CHANNELS     =  3
    IMG_EMBED_DIM  =  5

#   IMG_SIZE       = 28          # FOR MNIST ONLY
#   N_CHANNELS     = 1           # FOR MNIST ONLY
#   IMG_EMBED_DIM  = 10          # FOR MNIST ONLY

#    N_PIXELS       = N_CHANNELS * IMG_SIZE * IMG_SIZE
    N_GENES              = 60483
    #HIDDEN_LAYER_NEURONS = 8000  now a passed in variable
    #GENE_EMBED_DIM       = 506   now a passed in variable

    LABEL_SWAP_PERUNIT   = 0.0                                                                             # 1.0 =change 100% of labels to a random class                                                            - use for validation
    MAKE_GREY            = 0.0                                                                             # 1.0 =change 100% of RGB images to 3-channel Greyscale etc                                               - use for validation
    JITTER               = [0.0 ,0.0, 0.0 ,0.0]                                                            # torchvision.transforms.ColorJitter(brightness=[0,1], contrast=[0,1], saturation=[0,1], hue=[-0.5,+0.5]) - use for validation

    # Instance variables: parameters that may change from run to run (such as learning rate or batch_size) 

    def __init__(self, lr,  batch_size ):
   
      if DEBUG>1:
        print( f"P_C_CONFIG:         INFO:     at {CYAN} __init__():{RESET}   current learning rate / batch_size  = {MIKADO}{lr}, {batch_size}{RESET} respectively" )

# ------------------------------------------------------------------------------

    def get_image_net( self, args, gpu, rank, input_mode, nn_type_img, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2, tile_size  ):

      if DEBUG>0:
        print( f"P_C_CONFIG:         INFO:     at {CYAN}get_image_net(){RESET}:   nn_type_img  = {CYAN}{nn_type_img}{RESET}" )

      if   nn_type_img=='LENET5':
        return LENET5(self)
      elif nn_type_img=='VGG':
        return VGG(self)
      elif nn_type_img=='VGG11':
        return vgg11_bn(self, n_classes, tile_size )
      elif nn_type_img=='VGG13':
        return vgg13_bn(self, n_classes, tile_size )       
      elif nn_type_img=='VGG16':
        return vgg16_bn(self, n_classes, tile_size )
      elif nn_type_img=='VGG19':
        return vgg19_bn(self, n_classes, tile_size )
      elif nn_type_img=='INCEPT3':
        return INCEPT3(self,  n_classes, tile_size )
      elif nn_type_img=='AE3LAYERCONV2D':
        return AE3LAYERCONV2D ( self, n_classes, tile_size )
      else: 
        print( f"\033[31;1mP_C_CONFIG:         FATAL:  'get_image_net()' Sorry, there is no neural network model called: '{nn_type_img}' ... halting now.\033[m" )        
        exit(0)

# ------------------------------------------------------------------------------

    def get_genes_net( self, args, gpu, rank, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  ):
      
      if DEBUG>2:
        print( "P_C_CONFIG:         INFO:     at \033[35;1m get_genes_net()\033[m:   nn_type_rna  = \033[36;1m{:}\033[m".format( nn_type_rna ) )

      if DEBUG>9:
        print( "P_C_CONFIG:         INFO:     at \033[35;1m get_genes_net()\033[m:   nn_type_rna  = \033[36;1m{:}\033[m".format( nn_type_rna ) )

      if nn_type_rna=='DENSE':
        return DENSE           ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='CONV1D':
        return CONV1D(self)
      elif nn_type_rna=='DENSEPOSITIVE':
        return DENSEPOSITIVE   ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='DCGANAE128':
        return DCGANAE128      ( self, args, gpu, rank, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2)
      elif nn_type_rna=='AELINEAR':
        return AELINEAR        ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='AEDENSE':
        return AEDENSE         ( self, args, gpu, rank, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='AEDENSEPOSITIVE':
        return AEDENSEPOSITIVE ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='AEDEEPDENSE':
        return AEDEEPDENSE     ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='TTVAE':
        ret = TTVAE            ( self, args, gpu, rank, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2   )
        if args.ddp == 'True':
          if DEBUG>0:
            print ( f"{BRIGHT_GREEN}P_C_CONFIG:     INFO:   DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! about to wrap model for multi-GPU processing:{RESET}" )      
            print ( f"P_C_CONFIG:     INFO:     device_ids          = {MIKADO}[{gpu}]{RESET}"           )                
          torch.cuda.set_device(rank)
          return DDP(  ret.to(rank),  device_ids=[rank], find_unused_parameters=True )                     # wrap for parallel processing
        else:
          return ret
      else:
        print( f"\033[31;1mP_C_CONFIG:         FATAL:  'get_genes_net()' Sorry, there is no neural network model called: '{nn_type_rna}' ... halting now.\033[m" )
        exit(0)
# ------------------------------------------------------------------------------

    def get_dataset( self, args, which_dataset, gpu ):

      return pre_compressDataset( self, which_dataset, args )

# ------------------------------------------------------------------------------

    def save_samples(self, directory, model, desc, x1, x2, labels):

      if DEBUG>9:
        print( "P_C_CONFIG:         INFO:       at top of save_samples() and parameter directory = \033[35;1m{:}\033[m".format( directory ) )
        
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

    def save_comparison(self, directory, x, x_recon, desc, is_image ):
        """Save image samples from learned image likelihood.
        """
        if is_image==True:
            self.save_image_comparison(directory, x, x_recon, desc)
        else:
            self.save_genes_comparison(directory, x.squeeze(), x_recon, desc)

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
      
      if DEBUG>2:
        print ( f"P_C_CONFIG:   INFO:      test(): x.shape         = {MIKADO}{x.shape}{RESET}" )
        print ( f"P_C_CONFIG:   INFO:      test(): xr.shape        = {MIKADO}{xr.shape}{RESET}" )
     
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
