"""=============================================================================
Dataset setup and request handling
============================================================================="""

import sys
import torch
import numpy as np
from   random import random
from   random import randint
from   sklearn import preprocessing
from   torch.utils.data import Dataset
from   torchvision import transforms

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
BRIGHT_GREEN='\033[38;2;102;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
RESET='\033[m'

UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'

DEBUG=1

np.set_printoptions( threshold=100000)
np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

# ------------------------------------------------------------------------------

class GTExV6Dataset( Dataset ):

    def __init__(self, cfg, args):

        self.cfg = cfg
        
        input_mode                 = args.input_mode

        print( f"DLBCL_Dataset:  INFO:       loading dataset from {MAGENTA}{cfg.ROOT_DIR }/train.pth{RESET}" )

        #threads=torch.get_num_threads()
        
        #if DEBUG>0:
        #  print ( f"{ORANGE}DLBCL_Dataset:  INFO:     number of threads currently being used by Torch = {threads}{RESET}")
          
        #print( f"{ORANGE}DLBCL_Dataset:  INFO:     test_mode enabled; num_threads will be set to 1 for dataset loading to ensure  dataset maintains patch tiles order {RESET}" )          
        #torch.set_num_threads(1)
        
        data = torch.load('%s/train.pth' % cfg.ROOT_DIR)

        #torch.set_num_threads(threads)
        #if DEBUG>0:
        #  print( f"{ORANGE}DLBCL_Dataset:  INFO:     num_threads has been changed back to original value ({threads}){RESET}" )          
          
          
        if input_mode=='image':
          self.images     = data['images']                                                                 # self.images  contains ALL the image tiles 
          self.genes      = torch.zeros(1)                                                                 # so that we can test in __get_item__ to see if the image tensor exists
          self.fnames     = data['fnames']                                                                 # self.fnames  contains the corresponding (fully qualified) file name of the SVS file from which the tile was exgtracted               
        elif input_mode=='rna':
          self.images     = data['genes']                                                                  # PGD 200613 - CARE ! USING THIS AS A DIRTY SHORTCUT IN RNA MODE
          #self.images    = data['images']                                                                 # PGD 200613
          self.genes     = data['genes']                                                                   # PGD 200613 - it's identical to self.images, but not actually used
          self.fnames     = data['gnames']                                                                 # TODO 200523 temp. Need to populate gene names in generate()           
        elif input_mode=='image_rna':
          self.images     = data['images']                                                                 # self.images  contains ALL the image tiles 
          self.genes      = data['genes']                                                                  # self.
          self.fnames     = data['fnames']                                                                 # TODO 200523 temp. Need to populate gene names in generate()                             
          #self.gnames     = data['gnames']                                                                 # TODO 200523 temp. Need to populate gene names in generate()                             
        else:
          print ( f"{RED}DLBCL_Dataset:  FATAL:    unknown data mode \033[1m'{CYAN}{input_mode}{RESET}{RED} ... quitting{RESET}" )
          sys.exit(0)

        self.tissues    = data['tissues']                                                                  # self.tissues contains true labels for ALL the samples

        print( "DLBCL_Dataset:  INFO:       \033[3mdataset loaded\033[m" )
        
        self.tissues = (self.tissues).long()                                                               # PGD 200129 - We also use self.tissues in DPPCA, where it needs to be a float value. Here it is a truth label and must be of type long
        
        if input_mode=='rna':
          if DEBUG>99:
            print ( f"DLBCL_Dataset:  INFO:     data['genes'][0] shape    = {CYAN}{data['genes'][0].cpu().numpy().shape }{RESET}"  )
          if DEBUG>999:
              np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
              print ( f"DLBCL_Dataset:  INFO:     data['genes'][0]          = \n{CYAN}{data['genes'][0:5].cpu().numpy()}{RESET}" )
                

        if DEBUG>99:
          print ( "DLBCL_Dataset:  INFO:     self.images shape          = \033[35;1m{:}\033[m".format( self.images.size() ) )
          if DEBUG>9:
            print ( "DLBCL_Dataset:  INFO:     self.images type           = {:}"  .format( type(self.images)    ) )
            print ( "DLBCL_Dataset:  INFO:     self.images                = \n{:}".format(  self.images[0]      ) )

        if DEBUG>9:
          print ( f"DLBCL_Dataset:  INFO:     self.tissues shape         = {CYAN}{self.tissues.size()}{RESET}          ")

        if DEBUG>9:
          np.set_printoptions(formatter={'int': lambda x: "{:>2d}".format(x)})
          print ( f"DLBCL_Dataset:  INFO:     self.tissues               = "     )
          print ( f"{MIKADO}{self.tissues.numpy()}{RESET},", end=""              )
          print ( f"\n",                                     end=""              )

        '''self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.tissues)
        self.labels = self.labelEncoder.transform(self.tissues)
        # `classes` are the unique class names, i.e. tissues.
        self.classes = list(set(self.tissues))
        '''
        
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.tissues)
        self.labels = self.labelEncoder.transform(self.tissues)
        
        # I don't need the above because my classes are already in the correct format for Torch (0-n)

        # `classes` are the unique class names, i.e. tissues.  E.g. there are 7 classes in the case of STAD
        self.classes = list(set(self.tissues))
        

        if DEBUG>9999:
          print( "DLBCL_Dataset:  INFO:        __init__(): self.classes        = \n\033[35;1m{:}\033[m".format(    self.classes      ) )

        InputModeIsRna     = False
        input_size         =  (self.images).size()
        input_dimensions   =  len(input_size)
        if input_dimensions==2:                                                                            # using it as a proxy to find out if we're dealing with RNA, coz don't have access to cfg here
          InputModeIsRna = True
        
        if DEBUG>1:
          print( "DLBCL_Dataset:  INFO:        __init__(): input_size           = \033[35;1m{:}\033[m".format  (   input_size        ) )
          print( "DLBCL_Dataset:  INFO:        __init__(): input_dimensions     = \033[35;1m{:}\033[m".format  (  input_dimensions   ) )
          print( "DLBCL_Dataset:  INFO:        __init__(): InputModeIsRna       = \033[35;1m{:}\033[m".format  (   InputModeIsRna    ) )
        if DEBUG>999:
          print( "DLBCL_Dataset:  INFO:        __init__(): self.tissues        = \n\033[35;1m{:}\033[m".format(    self.tissues     ) )

        labels_length         =  len(self.labels)

        if DEBUG>99:
          print( "DLBCL_Dataset:  INFO:        __init__(): labels_length         = \033[36;1m{:}\033[m".format (    labels_length        ) )

        if InputModeIsRna == False:
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              #transforms.RandomRotation((0, 360)),
              #transforms.RandomCrop(cfg.IMG_SIZE),
              #transforms.RandomHorizontalFlip(),
              transforms.ToTensor()
          ])
        
        make_grey_perunit = cfg.MAKE_GREY
        if not make_grey_perunit==0:
          if DEBUG>0:
            print( "DLBCL_Dataset:  INFO:        __init__(): CAUTION! \033[31;1m\033[3mMAKE_GREY OPTION\033[m IS ACTIVE!; {:3.0f}% OF TILES WILL BE CONVERTED TO 3-CHANNEL GREYSCALE\033[m".format(   make_grey_perunit * 100        ) )  
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              transforms.RandomGrayscale(p=make_grey_perunit),
              transforms.ToTensor()
          ])

        label_swap_perunit = cfg.LABEL_SWAP_PERUNIT
        if not label_swap_perunit==0: 
          if DEBUG>0:
            print( "DLBCL_Dataset:  INFO:        __init__(): CAUTION! \033[31;1m\033[3mLABEL SWAPS MODE\033[m IS ACTIVE!; {:3.0f}% OF TRUTH LABELS WILL BE SWAPPED FOR RANDOM CLASS VALUES\033[m".format(   label_swap_perunit * 100        ) )
          self.tissues = torch.LongTensor([ randint(0,8) if random() < label_swap_perunit  else x for x in self.tissues])

        jitter = cfg.JITTER
        if not sum( jitter )==0:                                                                             # then the user has requested some jitter insertion
          if DEBUG>0:
            print( "DLBCL_Dataset:  INFO:        __init__(): CAUTION! \033[31;1m\033[3mJITTER OPTION\033[m IS ACTIVE!; brightness_jitter=\033[36;1m{:}\033[m contrast_jitter=\033[36;1m{:}\033[m saturation_jitter\033[36;1m{:}\033[m hue_jitter\033[36;1m{:}\033[m".format( jitter[0], jitter[1], jitter[2], jitter[3]        ) )  
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              transforms.transforms.ColorJitter( jitter[0], jitter[1], jitter[2], jitter[3] ),
              transforms.ToTensor()
          ])


        if DEBUG>999:
          print( "DLBCL_Dataset:  INFO:        __init__(): input_dimensions   = \033[35;1m{:}\033[m".format  (  input_dimensions   ) )
          print( "DLBCL_Dataset:  INFO:        __init__(): InputModeIsRna     = \033[35;1m{:}\033[m".format  (   InputModeIsRna    ) )          
        
        
        if DEBUG>9:
          print( "DLBCL_Dataset:  INFO:     returning from \033[35;1m__init__\033[m" )

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        if DEBUG>99:
          print( "DLBCL_Dataset:  INFO:     at __len__, and number of tiles in dataset = \033[36;1m{:}\033[m".format( len(self.images)))
        
        return len(self.images)

# ------------------------------------------------------------------------------

    def __getitem__(self, i ):
        
        if DEBUG>99:
          print ( f"DLBCL_Dataset:  INFO:        __getitem__() ----------------------------------------------------------------- i                 = {i}" )
          print ( f"DLBCL_Dataset:  INFO:        __getitem__() ----------------------------------------------------------------- self.images.dim() = {self.images.dim()}" )

        if not (self.images.dim()==1):                                                                   # if dim==1, then  image tensor does not exist in the dataset
          images          = self.images[i]
          images          = self.subsample_image(images).numpy()
          images          = torch.Tensor(images)
          fnames          = self.fnames[i]
        else:
          images          = self.images[0]

        if not (self.genes.dim()==1):                                                                     # if dim==1, then gene tensor does not exist in the dataset
          genes           = self.genes[i]
          #gnames          = self.gnames[i]
          genes           = torch.Tensor(genes)
        else:
          genes          = self.genes[0]            
      
        labels          = self.tissues[i]                                                                   # labels must always exist in the dataset       


        return images, genes, labels, fnames

# ------------------------------------------------------------------------------

    def sample_ith_image(self, i):
        """Memory-saving utility function when we just want an image.
        """

        print ( "GTExV6Dataset: hello from here in line 83 or so in sample_ith_image generate.py" )

        return self.subsample_image(self.inputs[i])

# ------------------------------------------------------------------------------

    def get_all_tissue(self, label, test_inds):
        """Return all samples of a specific tissue.
        """
        if type(label) is str:
            label = int(self.labelEncoder.transform([label])[0])

        n = 0
        for i in test_inds:
            if self.labels[i] == label:
                n += 1

        nc = self.cfg.N_CHANNELS
        w  = self.cfg.IMG_SIZE
        inputs = torch.Tensor(n, nc, w, w)

        for i in test_inds:
            if self.labels[i] == label:
                inputs[i] = self.subsample_image(self.inputs[i])

        if type(label) is str:
            label = int(self.labelEncoder.transform([label])[0])

        inds = torch.Tensor(self.labels) == label
        inds = inds and test_inds
        inputs_raw = self.inputs[inds]

        n  = len(inputs_raw)
        nc = self.cfg.N_CHANNELS
        w  = self.cfg.IMG_SIZE

        inputs = torch.Tensor(n, nc, w, w)
        for i, img in enumerate(inputs_raw):
            inputs[i] = self.subsample_image(img)

        tissues  = self.tissues[inds]
        return inputs, tissues
