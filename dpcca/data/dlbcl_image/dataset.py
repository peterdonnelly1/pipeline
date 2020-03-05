"""=============================================================================
GTEx V6 data set of histology images and gene expression levels.
============================================================================="""

import sys
import torch
import numpy as np
from   random import random
from   random import randint
from   sklearn import preprocessing
from   torch.utils.data import Dataset
from   torchvision import transforms

DEBUG=1

np.set_printoptions(edgeitems=18000)
np.set_printoptions(linewidth=6000)

# ------------------------------------------------------------------------------

class GTExV6Dataset(Dataset):

    def __init__(self, cfg):

        self.cfg = cfg

        if DEBUG>1:
          print( "GTExV6Dataset:  INFO:     at top of \033[33;1m__init__\033[m" )

        print( "GTExV6Dataset:  INFO:     loading dataset from \033[35;1m{:}/train.pth\033[m".format( cfg.ROOT_DIR )  )
        
        data = torch.load('%s/train.pth' % cfg.ROOT_DIR)
        
        if cfg.INPUT_MODE=='rna':
          self.images     = data['genes']                                                                  # self.genes  contains ALL the genes
        elif cfg.INPUT_MODE=='image':
          self.images     = data['images']                                                                 # self.images  contains ALL the images
        else:
          print ( "GTExV6Dataset:  FATAL:    unknown data mode \033[1m'{:}'\033[m ... quitting".format( cfg.INPUT_MODE ) )
          sys.exit(0)

        self.tissues    = data['tissues']                                                                  # self.tissues contains the truth value for ALL the images

        print( "GTExV6Dataset:  INFO:     \033[3mdataset loaded\033[m" )
        
        self.tissues = (self.tissues).long()     # PGD 200129 - We also use self.tissues in DPPCA, where it needs to be a float value. Here it is a truth label and must be of type long

        if DEBUG>99:
          print ( "GTExV6Dataset:  INFO:     data['images'][0] shape     = \033[3;1m{:}\033[m".format( data['images'][0].shape ) )
          if DEBUG>99:
              print ( "GTExV6Dataset:  INFO:     data['images'][0]           = \n{:}".format(  data['images'][0]      ) )
        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     type(data['tissues'].numpy()[0] = {:}".format(  type(data['tissues'].numpy()[0])     ) )
          print ( "GTExV6Dataset:  INFO:     data['tissues'][sample]           = {:}".format(  data['tissues'].numpy()[1000:1200]  ) )                     


        if DEBUG>99:
          print ( "GTExV6Dataset:  INFO:     self.images shape               = \033[35;1m{:}\033[m".format( self.images.size() ) )
          if DEBUG>9:
              print ( "GTExV6Dataset:  INFO:     self.images type      = {:}"  .format( type(self.images) ) )
              print ( "GTExV6Dataset:  INFO:     self.images           = \n{:}".format(  self.images[0]      ) )

        if DEBUG>9:
          print ( "GTExV6Dataset:  INFO:     self.tissues shape              = \033[35;1m{:}\033[m".format( self.tissues.size() ) )
          if DEBUG>9:
              print ( "GTExV6Dataset:  INFO:     self.tissues type     = {:}"  .format( type(self.tissues.numpy().shape) ) )
              print ( "GTExV6Dataset:  INFO:     self.tissues          = \n{:}".format(  self.tissues.numpy()[0:17700]      ) )


        '''self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.tissues)
        self.labels = self.labelEncoder.transform(self.tissues)

        # `classes` are the unique class names, i.e. tissues.
        self.classes = list(set(self.tissues))
        '''
        
        self.labelEncoder = preprocessing.LabelEncoder()
        self.labelEncoder.fit(self.tissues)
        self.labels = self.labelEncoder.transform(self.tissues)

        # `classes` are the unique class labels                                                            # PGD200304 I use self.tissues to hold classes for historical reasons. Will need to fix this up.
        self.classes = list(set(self.tissues))

        if DEBUG>9999:
          print( "GTExV6Dataset:  INFO:        __init__(): self.classes        = \n\033[35;1m{:}\033[m".format(    self.classes      ) )

        InputModeIsRna     = False
        input_size         =  (self.images).size()
        input_dimensions   =  len(input_size)
        if input_dimensions==2:                                                                            # using it as a proxy to find out if we're dealing with RNA, coz don't have access to cfg here
          InputModeIsRna = True
        
        if DEBUG>99:
          print( "GTExV6Dataset:  INFO:        __init__(): input_size         = \033[35;1m{:}\033[m".format  (   input_size        ) )
          print( "GTExV6Dataset:  INFO:        __init__(): input_dimensions   = \033[35;1m{:}\033[m".format  (  input_dimensions   ) )
          print( "GTExV6Dataset:  INFO:        __init__(): InputModeIsRna     = \033[35;1m{:}\033[m".format  (   InputModeIsRna    ) )
        if DEBUG>999:
          print( "GTExV6Dataset:  INFO:        __init__(): self.tissues        = \n\033[35;1m{:}\033[m".format(    self.tissues     ) )

        labels_length         =  len(self.labels)

        if DEBUG>99:
          print( "GTExV6Dataset:  INFO:        __init__(): labels_length         = \033[36;1m{:}\033[m".format (    labels_length        ) )

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
            print( "GTExV6Dataset:  INFO:        __init__(): CAUTION! \033[31;1m\033[3mMAKE_GREY OPTION\033[m IS ACTIVE!; {:3.0f}% OF TILES WILL BE CONVERTED TO 3-CHANNEL GREYSCALE\033[m".format(   make_grey_perunit * 100        ) )  
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              transforms.RandomGrayscale(p=1.0),
              transforms.ToTensor()
          ])

        label_swap_perunit = cfg.LABEL_SWAP_PERUNIT
        if not label_swap_perunit==0: 
          if DEBUG>0:
            print( "GTExV6Dataset:  INFO:        __init__(): CAUTION! \033[31;1m\033[3mLABEL SWAPS MODE\033[m IS ACTIVE!; {:3.0f}% OF TRUTH LABELS WILL BE SWAPPED FOR RANDOM CLASS VALUES\033[m".format(   label_swap_perunit * 100        ) )
          self.tissues = torch.LongTensor([ randint(0,8) if random() < label_swap_perunit  else x for x in self.tissues])

        jitter = cfg.JITTER
        if not sum( jitter )==0:                                                                             # then the user has requested some jitter insertion
          if DEBUG>0:
            print( "GTExV6Dataset:  INFO:        __init__(): CAUTION! \033[31;1m\033[3mJITTER OPTION\033[m IS ACTIVE!; brightness_jitter=\033[36;1m{:}\033[m contrast_jitter=\033[36;1m{:}\033[m saturation_jitter\033[36;1m{:}\033[m hue_jitter\033[36;1m{:}\033[m".format( jitter[0], jitter[1], jitter[2], jitter[3]        ) )  
          self.subsample_image = transforms.Compose([
              transforms.ToPILImage(),
              transforms.transforms.ColorJitter( jitter[0], jitter[1], jitter[2], jitter[3] ),
              transforms.ToTensor()
          ])



        if DEBUG>999:
          print( "GTExV6Dataset:  INFO:        __init__(): input_dimensions   = \033[35;1m{:}\033[m".format  (  input_dimensions   ) )
          print( "GTExV6Dataset:  INFO:        __init__(): InputModeIsRna     = \033[35;1m{:}\033[m".format  (   InputModeIsRna    ) )          
        
        
        if DEBUG>0:
          print( "GTExV6Dataset:  INFO:     returning from \033[35;1m__init__\033[m" )

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        if DEBUG>99:
          print( "GTExV6Dataset:  INFO:     at __len__, and number of samples in dataset = \033[36;1m{:}\033[m".format( len(self.images)))
        
        return len(self.images)

# ------------------------------------------------------------------------------

    def __getitem__(self, i):

        """Return the `idx`-th (image, metadata)-pair from the dataset.
        """
        pixels       = self.images[i]
        tissue_type  = self.tissues[i]

        if DEBUG>99:
          print( "GTExV6Dataset:  INFO:        at \033[33;1m__getitem__\033[m with parameters i=\033[35;1m{:}\033[m, \033[35;1m{:}\033[m".format (i, self) )

        # PGD 191230 - NEW CODE TO REPLACE THE bad_crop code. SEE COMMENTS BELOW
        
        InputModeIsRna = False
        if len(pixels.shape)==1:                                                                           # using it as a proxy to find out if we're dealing with RNA, coz don't have access to cfg here
          InputModeIsRna = True

        if DEBUG>999:
          print( "GTExV6Dataset:  INFO:        __getitem__(): InputModeIsRna =\033[35;1m{:}\033[m".format ( InputModeIsRna ) )

        if InputModeIsRna:
          image  = torch.Tensor(pixels)
        else:                                                                                              # do 'image only' stuff
          image = self.subsample_image(pixels).numpy()
          image = torch.Tensor(image)
       
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

        return image, tissue_type

# ------------------------------------------------------------------------------

    def sample_ith_image(self, i):
        """Memory-saving utility function when we just want an image.
        """

        print ( "GTExV6Dataset: hello from here in line 83 or so in sample_ith_image generate.py" )

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

        tissues  = self.tissues[inds]
        return images, tissues
