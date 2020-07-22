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
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
DULL_YELLOW='\033[38;2;179;179;0m'
BLUE='\033[38;2;0;0;255m'
DULL_BLUE='\033[38;2;0;102;204m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
ORANGE='\033[38;2;255;127;0m'
DULL_ORANGE='\033[38;2;127;63;0m'
GOLD='\033[38;2;255;215;0m'
GREEN='\033[38;2;0;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

DEBUG=1

np.set_printoptions( threshold=100000)
np.set_printoptions( edgeitems=25  )
np.set_printoptions( linewidth=240 )

# ------------------------------------------------------------------------------

class pre_compressDataset( Dataset ):

    def __init__(self, cfg, args):

        self.cfg = cfg
        
        input_mode                 = args.input_mode

        print( f"pre_compressDataset:  INFO:     loading dataset from {MAGENTA}{cfg.ROOT_DIR}/train.pth{RESET}" )

        print( f"{ORANGE}pre_compressDataset:  INFO:     args.nn_mode = {MAGENTA}{args.nn_mode}{RESET}" )        
        
        data = torch.load('%s/train.pth' % cfg.ROOT_DIR)
                 
          
        if input_mode=='image':
          self.images     = data['images']                                                                 # self.images  contains ALL the image tiles 
          self.genes      = torch.zeros(1)                                                                 # so that we can test in __get_item__ to see if the image tensor exists
          self.fnames     = data['fnames']                                                                 # self.fnames  contains the corresponding (fully qualified) file name of the SVS file from which the tile was exgtracted               
        elif input_mode=='rna':
          if ( args.nn_mode=='pre_compress' ) | ( args.nn_mode=='analyse_data' ):
            print( f"{ORANGE}pre_compressDataset:     INFO:  CAUTION! 'pre_compress' mode is set{RESET}" )            
            self.genes      = np.squeeze(data['genes'])                                                    # PGD 200714 - TEMP
          else:
            self.images     = data['genes']                                                                # PGD 200613 - CARE ! USING THIS AS A DIRTY SHORTCUT IN RNA MODE
            #self.images    = data['images']                                                               # PGD 200613
            self.genes     = data['genes']                                                                 # PGD 200613 - it's identical to self.images, but not actually used
            self.fnames     = data['gnames']                                                               # TODO 200523 temp. Need to populate gene names in generate()           

        elif input_mode=='image_rna':
          self.images     = data['images']                                                                 # self.images  contains ALL the image tiles 
          self.genes      = data['genes']                                                                  # self.
          self.fnames     = data['fnames']                                                                 # TODO 200523 temp. Need to populate gene names in generate()                             
          #self.gnames     = data['gnames']                                                                # TODO 200523 temp. Need to populate gene names in generate()                             
        else:
          print ( f"{RED}pre_compressDataset:  FATAL:    unknown data mode \033[1m'{CYAN}{input_mode}{RESET}{RED} ... quitting{RESET}" )
          sys.exit(0)

        self.tissues    = data['tissues']                                                                  # self.tissues contains true labels for ALL the samples

        print( "pre_compressDataset:  INFO:     \033[3mdataset loaded\033[m" )
        
        self.tissues = (self.tissues).long()                                                               # PGD 200129 - We also use self.tissues in DPPCA, where it needs to be a float value. Here it is a truth label and must be of type long
        
        if input_mode=='rna':
          if DEBUG>99:
            print ( f"pre_compressDataset:  INFO:     data['genes'][0] shape    = {CYAN}{data['genes'][0].cpu().numpy().shape }{RESET}"  )
          if DEBUG>999:
              np.set_printoptions(formatter={'float': lambda x: "{:>10.2f}".format(x)})
              print ( f"pre_compressDataset:  INFO:     data['genes'][0]          = \n{CYAN}{data['genes'][0:5].cpu().numpy()}{RESET}" )
                

        if DEBUG>99:
          print ( "pre_compressDataset:  INFO:     self.images shape          = \033[35;1m{:}\033[m".format( self.images.size() ) )
          if DEBUG>9:
            print ( "pre_compressDataset:  INFO:     self.images type           = {:}"  .format( type(self.images)    ) )
            print ( "pre_compressDataset:  INFO:     self.images                = \n{:}".format(  self.images[0]      ) )

        if DEBUG>9:
          print ( f"pre_compressDataset:  INFO:     self.tissues shape         = {CYAN}{self.tissues.size()}{RESET}          ")

        if DEBUG>0:
          np.set_printoptions(formatter={'int': lambda x: "{:>2d}".format(x)})
          print ( f"pre_compressDataset:  INFO:     self.tissues               = "     )
          print ( f"{self.tissues.numpy()},", end=""                            )
          print ( f"\n",                      end=""                            )

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
        

        if DEBUG>999:
          print( "pre_compressDataset:  INFO:        __init__(): self.classes        = \n\033[35;1m{:}\033[m".format(    self.classes      ) )
        
        if DEBUG>999:
          print( "pre_compressDataset:  INFO:        __init__(): self.tissues        = \n\033[35;1m{:}\033[m".format(    self.tissues     ) )

        labels_length         =  len(self.labels)

        if DEBUG>99:
          print( "pre_compressDataset:  INFO:        __init__(): labels_length         = \033[36;1m{:}\033[m".format (    labels_length        ) )
        
        

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        if DEBUG>99:
          print( "pre_compressDataset:  INFO:     at __len__, and number of tiles in dataset = \033[36;1m{:}\033[m".format( len(self.labels)))
        
        return len(self.labels)

# ------------------------------------------------------------------------------

    def __getitem__(self, i ):
        
        if DEBUG>9:
          print ( f"pre_compressDataset:  INFO:        __getitem__() ----------------------------------------------------------------- i                 = {i}" )
          print ( f"pre_compressDataset:  INFO:        __getitem__() ----------------------------------------------------------------- self.genes.dim()  = {self.genes.dim()}" )
          
        genes           = self.genes[i]       

        return genes

# ------------------------------------------------------------------------------

    def sample_ith_image(self, i):
        """Memory-saving utility function when we just want an image.
        """

        print ( "pre_compressDataset: hello from here in line 83 or so in sample_ith_image generate.py" )

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
