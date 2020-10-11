"""============================================================================
Configuration for use with pre-compression
============================================================================"""

import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np
import openslide
from   torch.nn.parallel         import DistributedDataParallel as DDP
from   torchvision.utils         import save_image
from   random                    import randint
from   PIL                       import Image
from   models                    import LENET5, AELINEAR, AEDENSE, AEDENSEPOSITIVE, AE3LAYERCONV2D, AEDEEPDENSE, TTVAE, VGG, VGGNN, INCEPT3, DENSE, DENSEPOSITIVE, CONV1D, DCGANAE128
from   models.vggnn              import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, make_layers, configs
from   data.pre_compress.dataset import pre_compressDataset
from   data.config               import Config

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
    
    IMG_SIZE       =  128
    N_CHANNELS     =  3
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
        print( "P_C_CONFIG:     INFO:     at \033[35;1m __init__()\033[m:   current learning rate / batch_size  = \033[36;1m{:}, {:}\033[m respectively".format( lr,  batch_size ) )

# ------------------------------------------------------------------------------

    def get_image_net(self, args, gpu, rank, cfg, input_mode, nn_type_img, encoder_activation, n_classes, tile_size ):

      if DEBUG>0:
        print( f"P_C_CONFIG:     INFO: at {CYAN}get_image_net(){RESET}:   nn_type_img  = {CYAN}{nn_type_img}{RESET}" )

      if   nn_type_img=='LENET5':
        return LENET5(self)
      elif nn_type_img=='VGG':
        return VGG(self)
      elif nn_type_img=='VGG11':
        return vgg11_bn(self, n_classes, tile_size)
      elif nn_type_img=='VGG13':
        return vgg13_bn(self, n_classes, tile_size)       
      elif nn_type_img=='VGG16':
        return vgg16_bn(self, n_classes, tile_size)
      elif nn_type_img=='VGG19':
        return vgg19_bn(self, n_classes, tile_size)
      elif nn_type_img=='INCEPT3':
        return INCEPT3(self,  n_classes, tile_size)
      elif nn_type_img=='AE3LAYERCONV2D':
        return AE3LAYERCONV2D ( self, n_classes, tile_size )
      else: 
        print( f"{RED}P_C_CONFIG:     FATAL:  sorry, there is no neural network model named: '{CYAN}{nn_type_img}{RESET}{RED}'{RESET}" ) 
        print( f"{RED}P_C_CONFIG:     FATAL:  cannot continue: ... halting now{RESET}" )               
        exit(0)

# ------------------------------------------------------------------------------

    def get_genes_net( self, args, gpu, rank, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  ):

      if DEBUG>0:
        print( f"P_C_CONFIG:     INFO:     at {CYAN}get_genes_net(){RESET}:   nn_type_rna  = {CYAN}{nn_type_rna}{RESET}" )

      if nn_type_rna=='DENSE':
        return DENSE           ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='CONV1D':
        return CONV1D(self)
      elif nn_type_rna=='DENSEPOSITIVE':
        return DENSEPOSITIVE   ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='DCGANAE128':
        return DCGANAE128(self)
      elif nn_type_rna=='AELINEAR':
        return AELINEAR        ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='AEDENSE':
        return AEDENSE         ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='AEDENSEPOSITIVE':
        return AEDENSEPOSITIVE ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2  )
      elif nn_type_rna=='AEDEEPDENSE':
        return AEDEEPDENSE     ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2 )
      elif nn_type_rna=='TTVAE':
        ret = TTVAE            ( self, args, input_mode, nn_type_rna, encoder_activation, n_classes, n_genes, hidden_layer_neurons, gene_embed_dim, nn_dense_dropout_1, nn_dense_dropout_2   )
        if args.ddp == 'True':
          if DEBUG>0:
            print ( f"{BRIGHT_GREEN}P_C_CONFIG:     INFO:   DDP{YELLOW}[{gpu}] {RESET}{BRIGHT_GREEN}! about to wrap model for multi-GPU processing:{RESET}" )      
            print ( f"P_C_CONFIG:     INFO:     device_ids          = {MIKADO}[{gpu}]{RESET}"           )                
          torch.cuda.set_device(rank)
          return DDP(  ret.to(rank),  device_ids=[rank], find_unused_parameters=True )                     # wrap for parallel processing
        else:
          return ret
      else:
        print( f"\033[31;1mA_D_CONFIG:         FATAL:  Sorry, there is no neural network model called: '{nn_type_rna}' ... halting now.\033[m" )
        exit(0)


# ------------------------------------------------------------------------------

    def get_dataset(self, args, gpu):
      if DEBUG>2:
        print ( "P_C_CONFIG:     INFO:   at \033[35;1mget_dataset\033[m")
      return pre_compressDataset(self, args, gpu)

# ------------------------------------------------------------------------------

    def save_samples(self, directory, model, desc, x2, labels):

      if DEBUG>9:
        print( "P_C_CONFIG:     INFO:       at top of save_samples() and parameter directory = \033[35;1m{:}\033[m".format( directory ) )
        
        n_samples = 100
        nc        = self.N_CHANNELS
        w         = self.IMG_SIZE

        # Visualize images.
        # -----------------
        x2r = model.sample([x2], n_samples)
        fname = '%s/sample_images_%s.png' % (directory, desc)
        save_image(x2r.cpu(), fname)

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

    def save_comparison(self, log_dir, x, x_recon, desc, is_x1=None):
        """Save image samples from learned image likelihood.
        """
        
         # note: x is a tensor and it contains a whole batch of tiles
         
        if DEBUG>99:
          print ( f"P_C_CONFIG:                    type(x)   =  {MIKADO}{type(x)}{RESET}"                        )
          print ( f"P_C_CONFIG:                    x.shape   =  {MIKADO}{x.shape}{RESET}"                        )
          print ( f"P_C_CONFIG: ((x.cpu().numpy())[0]).shape =  {MIKADO}{((x.cpu().numpy())[0]).shape}{RESET}"   )
        if DEBUG>99:
          np.set_printoptions(formatter={'float': lambda x: "{:>5.2f}".format(x)})
          print ( f"P_C_CONFIG: x   =  {MIKADO}{(x.cpu().numpy())[0,0,0,0:24]}{RESET}"                           )

        if DEBUG>999:
          shall_we_save = randint(0,3)
          if shall_we_save==1:
            np.set_printoptions(formatter={'int': lambda x: "{:3d}".format(x)})
            print ( f"P_C_CONFIG: INFO: batch  shape                       =  {MIKADO}{x.cpu().numpy().shape}{RESET}" )
            this_tile = randint(0, x.shape[0]-1)
            print ( f"P_C_CONFIG: INFO: randomly selected tile from batch  =  {CARRIBEAN_GREEN}{this_tile}{RESET}" )
            this_tile_npy                = ( x.cpu().numpy()[this_tile] )
            print ( f"P_C_CONFIG: INFO: this_tile_npy.shape                =  {MIKADO}{this_tile_npy.shape}{RESET}" )
            this_tile_npy_255            = ( 255*this_tile_npy )
            print ( f"P_C_CONFIG: INFO: this_tile_npy_255.shape            =  {MIKADO}{this_tile_npy_255.shape}{RESET}" )
            this_tile_npy_255_uint8      = np.uint8( this_tile_npy_255 )
            print ( f"P_C_CONFIG: INFO: this_tile_npy_255_uint8.shape      =  {MIKADO}{this_tile_npy_255_uint8.shape}{RESET}" )
            this_tile_npy_255_uint8_axes =  np.moveaxis( this_tile_npy_255_uint8, 0, -1 )                                 # swap axes from ( c, x, x ) to ( x, x, c ) to suit pytorch
            #print ( f"P_C_CONFIG: this_tile_npy_255_uint8_axes =  {MIKADO}{this_tile_npy_255_uint8_axes}{RESET}" )
            print ( f"P_C_CONFIG: INFO: this_tile_npy_255_uint8_axes.shape =  {MIKADO}{this_tile_npy_255_uint8_axes.shape}{RESET}" )
            tile_pil = Image.fromarray( this_tile_npy_255_uint8_axes )
            now      = datetime.datetime.now()                
            sname    = f"{log_dir}/tile_randomly_saved_during_save_comparison_{now:%y%m%d%H}_{randint(0,1000):04d}.bmp"
            if DEBUG>0:
              print ( f"\r{RESET}{MAGENTA}\033[0C       {sname}       {RESET}")                  
            tile_pil.save( f"{sname}", "BMP")
                  
        #if is_x1:
        self.save_image_comparison(log_dir, x, x_recon, desc)
        #else:
        #    self.save_genes_comparison(log_dir, x, x_recon, desc)

# ------------------------------------------------------------------------------

    def save_image_comparison( self, log_dir, x, x_recon, fnumber ):
        
        nc = self.N_CHANNELS
        w  = self.IMG_SIZE 
        
        if DEBUG>9:
          print ( f"P_C_CONFIG:     INFO:        x     shape     =  {GREEN}{x.shape}{RESET}" )
        fqn = f"{log_dir}/recon_images_{fnumber:06d}.png"
        N = np.minimum( x.shape[0], 8 )                                                                              # Number of images pairs to save for display; can't be larger than batch size
        if DEBUG>9:
          print ( f"P_C_CONFIG:     INFO:                  N     =  {MAGENTA}{N}{RESET}" )

        comparison = torch.cat( [x[0:N], x_recon[0:N] ] )
        save_image(comparison.cpu(), fqn, nrow=N )

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
