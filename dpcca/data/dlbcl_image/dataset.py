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

np.set_printoptions( threshold=100000)
np.set_printoptions( edgeitems=5000  )
np.set_printoptions( linewidth=5000 )

# ------------------------------------------------------------------------------

class GTExV6Dataset( Dataset ):

    def __init__(self, cfg, args):

        self.cfg = cfg
        
        input_mode                 = args.input_mode

        if DEBUG>0:
          print( f"DATASET:        INFO:   loading dataset from                {MAGENTA}{cfg.ROOT_DIR }/train.pth{RESET}" )

        #threads=torch.get_num_threads()
        
        #if DEBUG>0:
        #  print ( f"{ORANGE}DATASET:        INFO:     number of threads currently being used by Torch = {threads}{RESET}")
          
        #print( f"{ORANGE}DATASET:        INFO:     test_mode enabled; num_threads will be set to 1 for dataset loading to ensure  dataset maintains patch tiles order {RESET}" )          
        #torch.set_num_threads(1)
        
        data = torch.load('%s/train.pth' % cfg.ROOT_DIR)

        #torch.set_num_threads(threads)
        #if DEBUG>0:
        #  print( f"{ORANGE}DATASET:        INFO:     num_threads has been changed back to original value ({threads}){RESET}" )          
          
          
        if input_mode=='image':
          self.images      = data['images']                                                                # self.images  contains ALL the image tiles 
          self.genes       = torch.zeros(1)                                                                # so that we can test in __get_item__ to see if the image tensor exists
          self.fnames      = data['fnames']                                                                # fnames  contains the corresponding (fully qualified) file name of the SVS file from which the tile was extracted               
          self.img_labels  = (data['img_labels']).long()                                                   # PGD 200129 - We also use self.labels in DPPCA, where it needs to be a float value. Here it is a truth label and must be of type long
          self.rna_labels  = (data['img_labels']).long()                                                   # so that __len__ will produce the correct dataset length regardless of whether we're in 'image' or 'rna' mode
        elif  ( input_mode=='rna' ) | ( input_mode=='image_rna' ) :
          self.images      = torch.zeros(1)                                                                # so that we can test in __get_item__ to see if the image tensor exists
          self.genes       = data['genes']                                                                 
          self.fnames      = data['fnames']                                                                # fnames  contains the corresponding (fully qualified) file name of the SVS file from which the tile was extracted               
          self.gnames      = data['gnames']                                                                # TODO 200523 temp. Need to populate gene names in generate()           
          self.img_labels  = (data['rna_labels']).long()                                                   # so that __len__ will produce the dataset length regardless of whether we're in 'image' or 'rna' mode
          self.rna_labels  = (data['rna_labels']).long()                                                   # PGD 200129 - We also use self.labels in DPPCA, where it needs to be a float value. Here it is a truth label and must be of type long
        else:
          print ( f"{RED}DATASET:        FATAL:    unknown data mode \033[1m'{CYAN}{input_mode}{RESET}{RED} ... quitting{RESET}" )
          sys.exit(0)

        if DEBUG>2:
          print( f"DATASET:        INFO:       {WHITE}dataset loaded{RESET}" )
            
                                                                    
        if input_mode=='image':
          if DEBUG>2:
            print ( f"DATASET:        INFO:     images     size            = {MIKADO}{(self.images).size()}{RESET}"                  )
            print ( f"DATASET:        INFO:     fnames     size            = {MIKADO}{(self.fnames).size()}{RESET}"                  )
            print ( f"DATASET:        INFO:     img_labels size            = {MIKADO}{(self.img_labels).size()}{RESET}"              )
          if DEBUG>9:
            print ( f"DATASET:        INFO:     self.images                = \n{self.images[0]}"                                     )
                    
        if ( input_mode=='rna' ) |  ( input_mode=='image_rna' ):
          if DEBUG>2:
            print ( f"DATASET:        INFO:     genes      size            = {MIKADO}{(self.genes).size()}{RESET}"                   )
            print ( f"DATASET:        INFO:     fnames     size            = {MIKADO}{(self.fnames).size()}{RESET}"                  )
            print ( f"DATASET:        INFO:     gnames     size            = {MIKADO}{(self.gnames).size()}{RESET}"                  )
            print ( f"DATASET:        INFO:     rna_labels size            = {MIKADO}{(self.rna_labels).size()}{RESET}"              )
          if DEBUG>6:
            print ( f"DATASET:        INFO:     rna_labels                 = \n{MIKADO}{(self.rna_labels).numpy()}{RESET}"            )
            print ( f"DATASET:        INFO:     rna_labels.shape           = \n{MIKADO}{(self.rna_labels).numpy().shape}{RESET}"            )
            #print ( f"DATASET:        INFO:     rna_labels                 = \n{MIKADO}{(data['rna_labels']).long()}{RESET}"         )            
          if DEBUG>999:
              np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
              print ( f"DATASET:        INFO:     data['genes'][0]          = \n{CYAN}{data['genes'][0:5].cpu().numpy()}{RESET}"     )
          if DEBUG>88:
              np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
              print ( f"DATASET:        INFO:     data['fnames'][0]          = \n{CYAN}{data['fnames']}{RESET}"     )
              


        if DEBUG>9:
          np.set_printoptions(formatter={'int': lambda x: "{:>2d}".format(x)})
          if ( input_mode=='image' ):
              print ( f"DATASET:        INFO:     self.img_labels               = "     )
              print ( f"{MIKADO}{self.img_labels.numpy()}{RESET},", end=""          )
              print ( f"\n",                                        end=""          )
          if ( input_mode=='rna' ) | ( input_mode=='image_rna' ):
              print ( f"DATASET:        INFO:     self.rna_labels               = "     )
              print ( f"{MIKADO}{self.rna_labels.numpy()}{RESET},", end=""          )
              print ( f"\n",                                        end=""          )


        '''self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.labels)
        self.labels = self.labelEncoder.transform(self.labels)
        # `classes` are the unique class names, i.e. labels.
        self.classes = list(set(self.labels))
        '''
        # I don't need the above because my classes are already in the correct format for Torch (0-n with no gaps)



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
            print( "DATASET:        INFO:     __init__(): CAUTION! \033[31;1m\033[3mMAKE_GREY OPTION\033[m IS ACTIVE!; {:3.0f}% OF TILES WILL BE CONVERTED TO 3-CHANNEL GREYSCALE\033[m".format(   make_grey_perunit * 100        ) )  
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              transforms.RandomGrayscale(p=make_grey_perunit),
              transforms.ToTensor()
          ])

        label_swap_perunit = args.label_swap_perunit
        if not label_swap_perunit==0: 
          if DEBUG>0:
            print( f"{RED}DATASET:        INFO:        __init__(): CAUTION! LABEL SWAP MODE IS ACTIVE!; {MIKADO}{label_swap_perunit*100:3.0f}{RESET}{RED}% OF TRUTH LABELS WILL BE SWAPPED FOR RANDOM CLASS VALUES\033[m"  )
          if ( input_mode=='image' ):
            self.img_labels = torch.LongTensor([ randint(0,len(args.class_names)-1) if random() < label_swap_perunit  else x for x in self.img_labels])
          if ( input_mode=='rna'   ) | ( input_mode=='image_rna' ):
            self.rna_labels = torch.LongTensor([ randint(0,len(args.class_names)-1) if random() < label_swap_perunit  else x for x in self.rna_labels])


        jitter = cfg.JITTER
        if not sum( jitter )==0:                                                                             # then the user has requested some jitter insertion
          if ( input_mode=='image' ):          
            if DEBUG>0:
              print( "DATASET:        INFO:        __init__(): CAUTION! \033[31;1m\033[3mJITTER OPTION\033[m IS ACTIVE!; brightness_jitter=\033[36;1m{:}\033[m contrast_jitter=\033[36;1m{:}\033[m saturation_jitter\033[36;1m{:}\033[m hue_jitter\033[36;1m{:}\033[m".format( jitter[0], jitter[1], jitter[2], jitter[3]        ) )  
            self.subsample_image = transforms.Compose([
                transforms.ToPILImage(),
                transforms.transforms.ColorJitter( jitter[0], jitter[1], jitter[2], jitter[3] ),
                transforms.ToTensor()
            ])


# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        
        
        if 'self.images' in locals():
          if DEBUG>88:
            print ( f"DATASET:        INFO:        __len__() ----------------------------------------------------------------- len(self.img_labels) = {MIKADO}{len(self.img_labels)}{RESET}" )        
          return len(self.img_labels)
        else:
          if DEBUG>88:
            print ( f"DATASET:        INFO:        __len__() ----------------------------------------------------------------- len(self.rna_labels) = {MIKADO}{len(self.rna_labels)}{RESET}" )  
          return len(self.rna_labels)
          
# ------------------------------------------------------------------------------

    def __getitem__(self, i ):
  
        if DEBUG>88:
          print ( f"self.images.dim() = {MIKADO}{self.images.dim()}{RESET}" )
        
        if not ( self.images.dim()==1) :                                                                   # if dim!=1, then image tensor does not exist in the dataset, so skip
          images          = self.images[i]
          images          = self.subsample_image(images).numpy()
          images          = torch.Tensor( images )                                                         # convert to Torch tensor
          fnames          = self.fnames[i]                                                                
          img_labels      = self.img_labels[i]
          if DEBUG>88:
            print ( f"DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- type(self.img_labels[{MIKADO}{i:3d}{RESET}]) = {MIKADO}{type(self.img_labels[i])}{RESET}" )
            print ( f"DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- self.img_labels[{MIKADO}{i:3d}{RESET}] = {MIKADO}{self.img_labels[i]}{RESET}" )
        else:
          images          = self.images     [0]                                                            # return a dummy
          img_labels      = self.img_labels [0]                                                            # return a dummy


        if DEBUG>88:
          print ( f"self.genes.dim()  = {MIKADO}{self.genes.dim()}{RESET}" )

        if not (self.genes.dim()==1):                                                                      # if dim==1, then gene tensor does not exist in the dataset, so skip
          genes           = self.genes[i]
          fnames          = self.fnames[i]                                                                
          #gnames          = self.gnames[i]
          genes           = torch.Tensor( genes )                                                          # convert to Torch tensor
          rna_labels      = self.rna_labels[i]       

          if DEBUG>88:
            print ( f"DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- type(self.rna_labels[{MIKADO}{i:3d}{RESET}]) = {MIKADO}{type(self.rna_labels[i])}{RESET}" )
            print ( f"DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- self.rna_labels[{MIKADO}{i:3d}{RESET}] = {MIKADO}{self.rna_labels[i]}{RESET}" )

        else:
          genes           = self.genes      [0]                                                            # return a dummy          
          rna_labels      = self.rna_labels [0]                                                            # return a dummy


        return images, genes, img_labels, rna_labels, fnames
        
        
        
