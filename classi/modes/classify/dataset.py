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

from constants  import *

DEBUG   = 0

np.set_printoptions( threshold=100000)
np.set_printoptions( edgeitems=5000  )
np.set_printoptions( linewidth=5000 )

# ------------------------------------------------------------------------------

class classifyDataset( Dataset ):

    def __init__(self, cfg, which_dataset, args):


        DEBUG     = args.debug_level_dataset
        LOG_LEVEL = args.log_level
        
        self.cfg = cfg
        
        input_mode = args.input_mode
        
        fqn = f"{args.base_dir}/{args.application_dir}/modes/{args.mode}/{which_dataset}.pth"

        if DEBUG>0:
          print( f"DATASET:        INFO:  loading {CYAN}{which_dataset:18s}{RESET} \r\033[50C dataset from {MAGENTA}{fqn}{RESET}{CLEAR_LINE}" )
        try:
          data             = torch.load(fqn)
        except Exception as e:
          print ( f"{RED}DATASET:        FATAL:    could not open file  {MAGENTA}{fqn}{RESET}{RED} - it probably doesn't exist. Cannot continue without a valid Pytorch dataset file to use for training"  )
          print ( f"{RED}DATASET:        FATAL:    explanation: did you use a shell script option or python user argument which suppresses tiling or dataset generation? {RESET}" )                
          print ( f"{RED}DATASET:        FATAL:        e.g. the option '{CYAN}-s True{RESET}{RED}' (python '{CYAN}--skip_tiling     = 'True'{RESET}{RED})'    suppresses tile generation{RESET}" )                 
          print ( f"{RED}DATASET:        FATAL:        e.g. the option '{CYAN}-g True{RESET}{RED}' (python '{CYAN}--skip_tiling     = 'True'{RESET}{RED})'    ssuppresses dataset generation even if tiles exit{RESET}" )                 
          print ( f"{RED}DATASET:        FATAL:    halting now...{RESET}" )
          sys.exit(0)

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




        make_grey_pct = args.make_grey_pct
        if not make_grey_pct==0:
          if DEBUG>0:
            print( f"DATASET:        INFO:    CAUTION! {RED}{BOLD}MAKE_GREY OPTION{RESET} IS ACTIVE!; {MIKADO}{make_grey_pct * 100:3.0f}{RESET}% OF TILES WILL BE CONVERTED TO 3-CHANNEL GREYSCALE{RESET}" )  
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              transforms.RandomGrayscale(p=make_grey_pct),
              transforms.ToTensor()
          ])

        label_swap_pct = args.label_swap_pct
        if not label_swap_pct==0: 
          if DEBUG>0:
            print( f"{RED}DATASET:        INFO:  {RED}{BOLD}CAUTION! LABEL SWAP MODE{RESET} IS ACTIVE!; {MIKADO}{label_swap_pct*100:3.0f}{RESET}% OF TRUTH LABELS WILL BE SWAPPED FOR RANDOM CLASS VALUES{RESET}"  )
          if ( input_mode=='image' ):
            self.img_labels = torch.LongTensor([ randint(0,len(args.class_names)-1) if random() < label_swap_pct  else x for x in self.img_labels])
          if ( input_mode=='rna'   ) | ( input_mode=='image_rna' ):
            self.rna_labels = torch.LongTensor([ randint(0,len(args.class_names)-1) if random() < label_swap_pct  else x for x in self.rna_labels])


        jitter = cfg.JITTER
        if not sum( jitter )==0:                                                                             # then the user has requested some jitter insertion
          if ( input_mode=='image' ):          
            if DEBUG>0:
              print( f"DATASET:        INFO:        {RED}{BOLD}CAUTION! JITTER OPTION{RESET} IS ACTIVE!; brightness_jitter={MIKADO}{jitter[0]}{RESET} contrast_jitter={MIKADO}{jitter[1]}{RESET} saturation_jitter={MIKADO}{jitter[2]}{RESET} hue_jitter={MIKADO}{jitter[3]}{RESET}" )  
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
          images          = self.subsample_image( images ).numpy()
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
        
        
        
