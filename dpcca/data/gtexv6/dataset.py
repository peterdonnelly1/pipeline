"""=============================================================================
GTEx V6 data set of histology images and gene expression levels.
============================================================================="""

import sys
import torch
import numpy as np
from   sklearn import preprocessing
from   torch.utils.data import Dataset
from   torchvision import transforms

WHITE='\033[37;1m'
DIM_WHITE='\033[37;2m'
CYAN='\033[36;1m'
MAGENTA='\033[38;2;255;0;255m'
YELLOW='\033[38;2;255;255;0m'
BLUE='\033[38;2;0;0;255m'
RED='\033[38;2;255;0;0m'
PINK='\033[38;2;255;192;203m'
PALE_RED='\033[31m'
ORANGE='\033[38;2;255;127;0m'
PALE_ORANGE='\033[38;2;127;63;0m'
GREEN='\033[38;2;0;255;0m'
PALE_GREEN='\033[32m'
BOLD='\033[1m'
ITALICS='\033[3m'
RESET='\033[m'

DEBUG=1


np.set_printoptions(edgeitems=300)
np.set_printoptions(linewidth=300)

# ------------------------------------------------------------------------------

class GTExV6Dataset(Dataset):

    def __init__(self, cfg):
		
        self.cfg = cfg

        if DEBUG>1:
          print( "GTExV6Dataset:  INFO:     at top if \033[33;1m__init__\033[m" )

        print( "GTExV6Dataset:  INFO:     loading Torch dataset from \033[33;1m{:}/train.pth\033[m".format( cfg.ROOT_DIR )  )
        data = torch.load('/home/peter/git/pipeline/dpcca/data/dlbcl_image/train.pth') ########################################## PGD 200609
        #data = torch.load('%s/train.pth' % cfg.ROOT_DIR)
        self.images     = data['images']        # self.images is ALL the images
        self.genes      = data['genes']         # self.genes is ALL the images
        self.tissues    = data['tissues']
        self.names      = data['fnames']
        self.gene_names = data['gnames']

        print( "GTExV6Dataset:  INFO:     Torch dataset loaded" )
        
        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     data['images'][0] shape     = \033[35;1m{:}\033[m".format( data['images'][0].shape ) )
          if DEBUG>99:
              print ( "GTExV6Dataset:  INFO:     data['images'][0]           = \n{:}".format(  data['images'][0]      ) )
        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     data['genes'][0]      = \033[35;1m{:}\033[m".format( data['genes'][0].shape ) )
          if DEBUG>0:
              print ( "GTExV6Dataset:  INFO:     data['genes'][0]            = \n{:}".format(  data['genes'][0][0:20]     ) )

        if DEBUG>99:
            print ( "GTExV6Dataset:  INFO:     data['tissues'][0]            = {:}".format(  data['tissues'][0:9]      ) )


        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     self.images shape     = \033[35;1m{:}\033[m".format( self.images.size() ) )
          if DEBUG>9:
              print ( "GTExV6Dataset:  INFO:     self.images type      = {:}"  .format( type(self.images) ) )
              print ( "GTExV6Dataset:  INFO:     self.images           = \n{:}".format(  self.images[0]      ) )

        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     self.gnorthwest to northeasterly 15 to 20 genes shape      = \033[35;1m{:}\033[m".format( self.genes.size() ) )
          if DEBUG>9:
              print ( "GTExV6Dataset:  INFO:     self.genes type       = {:}"  .format( type(self.genes) ) )
              print ( "GTExV6Dataset:  INFO:     self.genes            = \n{:}".format(  self.genes[0]      ) )

        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     self.names shape      = \033[35;1m{:}\033[m".format( self.names.size() ) )
          if DEBUG>9:
              print ( "GTExV6Dataset:  INFO:     self.names type       = {:}"  .format( type(self.names) ) )
              print ( "GTExV6Dataset:  INFO:     self.names            = \n{:}".format(  self.names      ) )

        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     self.tissues shape    = \033[35;1m{:}\033[m".format( self.tissues.size() ) )
          if DEBUG>9:
              print ( "GTExV6Dataset:  INFO:     self.tissues type     = {:}"  .format( type(self.tissues) ) )
              print ( "GTExV6Dataset:  INFO:     self.tissues          = \n{:}".format(  self.tissues      ) )

        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     self.gene_names shape = \033[35;1m{:}\033[m".format( self.gene_names.size() ) )
          if DEBUG>9:
              print ( "GTExV6Dataset:  INFO:     self.gene_names type  = {:}"  .format( type(self.gene_names) ) )
              print ( "GTExV6Dataset:  INFO:     self.gene_names       = \n{:}".format(  self.gene_names      ) )
               
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.tissues)
        self.labels = self.labelEncoder.transform(self.tissues)

        # `classes` are the unique class names, i.e. tissues.
        self.classes = list(set(self.tissues))

        # this defines the transfort 'subsample_image()' for later use. It doesn't operate on any data. It doesn't operate on any data at this point.
        self.subsample_image = transforms.Compose([
            transforms.ToPILImage(),
#            transforms.RandomRotation((0, 360)),
#            transforms.RandomCrop(cfg.IMG_SIZE),
#            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])        
        
        
        if DEBUG>99:
          print( "GTExV6Dataset:  INFO:     returning from \033[33;1m__init__\033[m" )

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        if DEBUG>0:
          print( "GTExV6Dataset:  INFO:     at __len__, and number of tiles in dataset = \033[35;1m{:}\033[m".format( len(self.images)))
        
        return len(self.images)
       
# ------------------------------------------------------------------------------

    def __getitem__(self, i):
		
        """Return the `idx`-th (image, metadata)-pair from the dataset.
        """
        pixels = self.images[i]
        genes  = self.genes[i]


        if DEBUG>99:
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():       type(self.images[{CYAN}{i:4d}{RESET}])  = {CYAN}{type(self.images[i])}{RESET}" )
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():       self.images[{CYAN}{i:4d}{RESET}].shape  = {CYAN}{self.images[i].shape}{RESET}" )

        if DEBUG>99:
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():       type(self.genes[{CYAN}{i:4d}{RESET}])  = {CYAN}{type(self.genes[i])}{RESET}"   )
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():            self.genes[{CYAN}{i:4d}{RESET}].shape  = {CYAN}{self.genes[i].shape}{RESET}"  )
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():            self.genes[{CYAN}{i:4d}{RESET}]        = {CYAN}{self.genes[i,0:50].cpu().numpy()}{RESET}"  )
          
          
        if DEBUG>99:
          print( "GTExV6Dataset:  INFO:        at \033[33;1m__getitem__\033[m with parameters i=\033[35;1m{:}\033[m, \033[35;1m{:}\033[m".format (i, self) )

        # PGD 191230 - NEW CODE TO REPLACE THE bad_crop code. SEE COMMENTS BELOW
        
        image = self.subsample_image(pixels).numpy()
        image = torch.Tensor(image)
        
        #genes = (genes - torch.mean(genes) ) / torch.std(genes)   # PGD 200127 Normalize the genes vector, which at least in the case of upper quartle statistics contains very large numbers
        # PGD 200610 - THE ABOVE CAUSES A CRASH "RuntimeError: cholesky_cuda: U(1,1) is zero, singular U." - DON'T KNOW WHY

        if DEBUG>99:
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():       type(self.genes[{MAGENTA}{i:4d}{RESET}])  = {MAGENTA}{type(self.genes[i])}{RESET}"   )          
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():            self.genes[{MAGENTA}{i:4d}{RESET}].shape  = {MAGENTA}{self.genes[i].shape}{RESET}"  )
          print ( f"DATASET_DPCCA:     INFO:      __getitem__():            self.genes[{MAGENTA}{i:4d}{RESET}]        = {MAGENTA}{self.genes[i,0:50].cpu().numpy()}{RESET}"  )
       
        if DEBUG>99:
          print ( "----------------------------------------------------------------------->           length gene[{:}]{:}".format( i, genes.size()   ) )
          print ( "----------------------------------------------------------------------->  (raw UQ values) gene[{:}]{:}".format( i, genes          ) )
			
        # PGD 191230 - NOT NECESSARY FOR ME BECAUSE I REMOVED ALL THE "BAD CROPS" (AS DEFINED) AT THE TILE GENERATION STAGE

        '''bad_crop = True

        while bad_crop:
            image = self.subsample_image(pixels).numpy()

            # We want to avoid all black crops because it prevents us from feature normalization
            if image.min() == image.max():
                print( "GTExV6Dataset:  INFO:        \033[33;1m__getitem__\033[m: \033[38;2;255;0;0mcompletely black tile detected - will be ignored\033[m")
                continue

            # We also want to avoid crops that are majority black.
            if (image == 0).sum() / image.size > 0.5:
                print( "GTExV6Dataset:  INFO:        \033[33;1m__getitem__\033[m: \033[38;2;255;0;0mmmajority black tile detected - will be ignored\033[m")
                continue

            bad_crop = False
            image = torch.Tensor(image)'''

        return image, genes

# ------------------------------------------------------------------------------

    def sample_ith_image(self, i):
        """Memory-saving utility function when we just want an image.
        """

        return self.subsample_image(self.images[i])

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
        images = torch.Tensor(n, nc, w, w)

        for i in test_inds:
            if self.labels[i] == label:
                images[i] = self.subsample_image(self.images[i])

        if type(label) is str:
            label = int(self.labelEncoder.transform([label])[0])

        inds = torch.Tensor(self.labels) == label
        inds = inds and test_inds
        images_raw = self.images[inds]

        n  = len(images_raw)
        nc = self.cfg.N_CHANNELS
        w  = self.cfg.IMG_SIZE

        images = torch.Tensor(n, nc, w, w)
        for i, img in enumerate(images_raw):
            images[i] = self.subsample_image(img)

        genes  = self.genes[inds]
        return images, genes
