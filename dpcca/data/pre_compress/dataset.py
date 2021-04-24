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
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
ASPARAGUS='\033[38;2;135;169;107m'
CHARTREUSE='\033[38;2;223;255;0m'
COQUELICOT='\033[38;2;255;56;0m'
COTTON_CANDY='\033[38;2;255;188;217m'
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

class pre_compressDataset( Dataset ):

    def __init__(self, cfg, which_dataset, args):

        self.cfg = cfg
        
        input_mode = args.input_mode
        
        peer_noise = self.peer_noise
        
        fqn = f"{cfg.ROOT_DIR }/{which_dataset}.pth"
          
        if DEBUG>0:
          print( f"P_C_DATASET:    INFO:      loading {CYAN}{which_dataset}{RESET} dataset from {MAGENTA}{fqn}{RESET}{CLEAR_LINE}" )
        try:
          data             = torch.load(fqn)
        except Exception as e:
          print ( f"{RED}P_C_DATASET:         FATAL:    could not open file  {MAGENTA}{fqn}{RESET}{RED} - it probably doesn't exist. Cannot continue without a valid Pytorch dataset file to use for training"  )
          print ( f"{RED}P_C_DATASET:         FATAL:    explanation: did you use a shell script or python user argument which suppresses tiling or dataset generation? {RESET}" )                
          print ( f"{RED}P_C_DATASET:         FATAL:        e.g. the script '{CYAN}only_run.sh{RESET}{RED}',                  suppresses tile generation{RESET}" )                 
          print ( f"{RED}P_C_DATASET:         FATAL:        e.g. the script '{CYAN}just_test_dont_tile.sh{RESET}{RED}',       suppresses tile generation{RESET}" )                 
          print ( f"{RED}P_C_DATASET:         FATAL:        e.g. the script '{CYAN}generate_and_run.sh{RESET}{RED}',          suppresses tile generation{RESET}" )                 
          print ( f"{RED}P_C_DATASET:         FATAL:        e.g. the option '{CYAN}--skip_tiling     = 'True'{RESET}{RED}'    suppresses tile generation if invoked by any shell script{RESET}" )                 
          print ( f"{RED}P_C_DATASET:         FATAL:        e.g. the option '{CYAN}--skip_generation = 'True'{RESET}{RED}'    suppresses dataset generation even if tiles have been generated{RESET}" )                 
          print ( f"{RED}P_C_DATASET:         FATAL:    halting now...{RESET}" )
          sys.exit(0)

        #torch.set_num_threads(threads)
        #if DEBUG>0:
        #  print( f"{ORANGE}P_C_DATASET:        INFO:     num_threads has been changed back to original value ({threads}){RESET}" )          
          
  
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
          print ( f"{RED}P_C_DATASET:        FATAL:    unknown data mode \033[1m'{CYAN}{input_mode}{RESET}{RED} ... quitting{RESET}" )
          sys.exit(0)

        if DEBUG>2:
          print( f"P_C_DATASET:        INFO:       {WHITE}dataset loaded{RESET}" )
            
                                                                    
        if input_mode=='image':
          if DEBUG>2:
            print ( f"P_C_DATASET:        INFO:     images     size            = {MIKADO}{(self.images).size()}{RESET}"                  )
            print ( f"P_C_DATASET:        INFO:     fnames     size            = {MIKADO}{(self.fnames).size()}{RESET}"                  )
            print ( f"P_C_DATASET:        INFO:     img_labels size            = {MIKADO}{(self.img_labels).size()}{RESET}"              )
          if DEBUG>9:
            print ( f"P_C_DATASET:        INFO:     self.images                = \n{self.images[0]}"                                     )
                    
        if ( input_mode=='rna' ) |  ( input_mode=='image_rna' ):
          if DEBUG>2:
            print ( f"P_C_DATASET:        INFO:     genes      size            = {MIKADO}{(self.genes).size()}{RESET}"                   )
            print ( f"P_C_DATASET:        INFO:     fnames     size            = {MIKADO}{(self.fnames).size()}{RESET}"                  )
            print ( f"P_C_DATASET:        INFO:     gnames     size            = {MIKADO}{(self.gnames).size()}{RESET}"                  )
            print ( f"P_C_DATASET:        INFO:     rna_labels size            = {MIKADO}{(self.rna_labels).size()}{RESET}"              )
          if DEBUG>6:
            print ( f"P_C_DATASET:        INFO:     rna_labels                 = \n{MIKADO}{(self.rna_labels).numpy()}{RESET}"            )
            print ( f"P_C_DATASET:        INFO:     rna_labels.shape           = \n{MIKADO}{(self.rna_labels).numpy().shape}{RESET}"            )
            #print ( f"P_C_DATASET:        INFO:     rna_labels                 = \n{MIKADO}{(data['rna_labels']).long()}{RESET}"         )            
          if DEBUG>999:
              np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
              print ( f"P_C_DATASET:        INFO:     data['genes'][0]          = \n{CYAN}{data['genes'][0:5].cpu().numpy()}{RESET}"     )
          if DEBUG>88:
              np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
              print ( f"P_C_DATASET:        INFO:     data['fnames'][0]          = \n{CYAN}{data['fnames']}{RESET}"     )
              


        if DEBUG>9:
          np.set_printoptions(formatter={'int': lambda x: "{:>2d}".format(x)})
          if ( input_mode=='image' ):
              print ( f"P_C_DATASET:        INFO:     self.img_labels               = "     )
              print ( f"{MIKADO}{self.img_labels.numpy()}{RESET},", end=""          )
              print ( f"\n",                                        end=""          )
          if ( input_mode=='rna' ) | ( input_mode=='image_rna' ):
              print ( f"P_C_DATASET:        INFO:     self.rna_labels               = "     )
              print ( f"{MIKADO}{self.rna_labels.numpy()}{RESET},", end=""          )
              print ( f"\n",                                        end=""          )


        '''self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.labels)
        self.labels = self.labelEncoder.transform(self.labels)
        # `classes` are the unique class names, i.e. labels.
        self.classes = list(set(self.labels))
        '''
        # We don't use the above because our classes are already in the correct format for Torch (0-n with no gaps)


        self.subsample_image = transforms.Compose([                                                        # i.e. compose the following transforms
            transforms.ToPILImage(),
            #transforms.RandomRotation((0, 360)),
            #transforms.RandomCrop(cfg.IMG_SIZE),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
 
        peer_noise_per_unit = args.peer_noise_per_unit
        if not peer_noise_per_unit==0:
          if DEBUG>0:
            print( f"P_C_DATASET:    INFO:    CAUTION! {RED}{BOLD}PEER NOSE OPTION{RESET} IS ACTIVE!; ALL TILES WILL RECIEVE {MIKADO}{peer_noise_per_unit * 100:3.0f}%{RESET} NOISE FROM RANDOMLY SELECTED PEER IMAGES{RESET}" )  
          self.subsample_image = peer_noise()
      
        make_grey_perunit = args.make_grey_perunit
        if not make_grey_perunit==0:
          if DEBUG>0:
            print( f"P_C_DATASET:    INFO:    CAUTION! {RED}{BOLD}MAKE_GREY OPTION{RESET} IS ACTIVE!; {MIKADO}{make_grey_perunit * 100:3.0f}%{RESET} OF TILES WILL BE CONVERTED TO 3-CHANNEL GREYSCALE{RESET}" )  
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              transforms.RandomGrayscale(p=make_grey_perunit),
              transforms.ToTensor()
          ])

        label_swap_perunit = args.label_swap_perunit
        if not label_swap_perunit==0: 
          if DEBUG>0:
            print( f"{RED}P_C_DATASET:        INFO:        {RED}{BOLD}CAUTION! LABEL SWAP MODE{RESET} IS ACTIVE!; {MIKADO}{label_swap_perunit*100:3.0f}%{RESET} OF TRUTH LABELS WILL BE SWAPPED FOR RANDOM CLASS VALUES{RESET}"  )
          if ( input_mode=='image' ):
            self.img_labels = torch.LongTensor([ randint(0,len(args.class_names)-1) if random() < label_swap_perunit  else x for x in self.img_labels])
          if ( input_mode=='rna'   ) | ( input_mode=='image_rna' ):
            self.rna_labels = torch.LongTensor([ randint(0,len(args.class_names)-1) if random() < label_swap_perunit  else x for x in self.rna_labels])


        jitter = cfg.JITTER
        if not sum( jitter )==0:                                                                             # then the user has requested some jitter insertion
          if ( input_mode=='image' ):          
            if DEBUG>0:
              print( f"P_C_DATASET:        INFO:        {RED}{BOLD}CAUTION! JITTER OPTION{RESET} IS ACTIVE!; brightness_jitter={MIKADO}{jitter[0]}{RESET} contrast_jitter={MIKADO}{jitter[1]}{RESET} saturation_jitter={MIKADO}{jitter[2]}{RESET} hue_jitter={MIKADO}{jitter[3]}{RESET}" )  
            self.subsample_image = transforms.Compose([
                transforms.ToPILImage(),
                transforms.transforms.ColorJitter( jitter[0], jitter[1], jitter[2], jitter[3] ),
                transforms.ToTensor()
            ])

# ------------------------------------------------------------------------------

    def peer_noise(self):
          
          if DEBUG>0:
            print ( f"P_C_DATASET:        INFO:    PLACEHOLDER FOR peer_noise method  type(self) = {type(self)}" )
            return self
          
# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        
        
        if 'self.images' in locals():
          if DEBUG>88:
            print ( f"P_C_DATASET:        INFO:        __len__() ----------------------------------------------------------------- len(self.img_labels) = {MIKADO}{len(self.img_labels)}{RESET}" )        
          return len(self.img_labels)
        else:
          if DEBUG>88:
            print ( f"P_C_DATASET:        INFO:        __len__() ----------------------------------------------------------------- len(self.rna_labels) = {MIKADO}{len(self.rna_labels)}{RESET}" )  
          return len(self.rna_labels)
          
# ------------------------------------------------------------------------------

    def __getitem__(self, i ):
  
        if DEBUG>88:
          print ( f"P_C_DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- {CYAN}self.images.dim(){RESET} = {MIKADO}{self.images.dim()}{RESET}" )
        
        if not ( self.images.dim()==1) :                                                                   # if dim!=1, then image tensor does not exist in the dataset, so skip
          images          = self.images[i]
          images          = self.subsample_image(images).numpy()
          images          = torch.Tensor( images )                                                         # convert to Torch tensor
          fnames          = self.fnames[i]                                                                
          img_labels      = self.img_labels[i]
          if DEBUG>88:
            # ~ print ( f"P_C_DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- type(self.img_labels[{MIKADO}{i:3d}{RESET}]) = {MIKADO}{type(self.img_labels[i])}{RESET}" )
            print ( f"P_C_DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- self.img_labels[{MIKADO}{i:3d}{RESET}] = {MIKADO}{self.img_labels[i]}{RESET}" )
        else:
          images          = self.images     [0]                                                            # return a dummy
          img_labels      = self.img_labels [0]                                                            # return a dummy


        if DEBUG>2:
          print ( f"P_C_DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- {CYAN}self.genes.dim(){RESET}  = {MIKADO}{self.genes.dim()}{RESET}", flush=True  )

        if not (self.genes.dim()==1):                                                                      # if dim==1, then gene tensor does not exist in the dataset, so skip
          genes           = self.genes[i]
          fnames          = self.fnames[i]                                                                
          #gnames          = self.gnames[i]
          genes           = torch.Tensor( genes )                                                          # convert to Torch tensor
          rna_labels      = self.rna_labels[i]       

          if DEBUG>2:
            print (   f"P_C_DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- {CYAN}genes     [{MIKADO}{i:3d}{RESET}{CYAN}].shape = {MIKADO}{(self.genes[i]).shape}{RESET}", flush=True )

          if DEBUG>2:
            print (   f"P_C_DATASET:        INFO:        __getitem__() ----------------------------------------------------------------- {CYAN}rna_labels[{MIKADO}{i:3d}{RESET}{CYAN}]       = {MIKADO}{self.rna_labels[i]}{RESET}",    flush=True )

        else:
          genes           = self.genes      [0]                                                            # return a dummy          
          rna_labels      = self.rna_labels [0]                                                            # return a dummy


        return images, genes, img_labels, rna_labels, fnames
        
        
        
