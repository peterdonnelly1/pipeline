"""============================================================================================================================================================

From:    "Realize image clustering with PyTorch"

Author:  Anders Ohrn Compilation|VK Source|Towards Data Science

Notes:   "The encoder compresses the image through features, which is the starting point for clustering"
 
         "Unlike the canonical application of VGG, the code is not input into the classification layer. The last two layers of vgg.classifier and vgg.avgpool are discarded.:

============================================================================================================================================================="""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ~ from torch import nn
from torchvision import models

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
CHARTREUSE='\033[38;2;223;255;0m'
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

DEBUG=0

class EncoderVGG(nn.Module):

    '''VGG16 image encoder with batch normalization.

    Args:
        pretrained_params (bool, optional): If the network should be populated with pre-trained VGG parameters. Defaults to True.

    '''
    
    channels_in = 3
    channels_code = 512

    def __init__( self, pretrained_params=True ):
      
        super(EncoderVGG, self).__init__()
        
        use_pretrained=False
        model = models.vgg16_bn(pretrained=use_pretrained)
        # ~ set_parameter_requires_grad(model, feature_extracting=False)        

        # ~ vgg = models.vgg16_bn(pretrained=pretrained_params)
        vgg = models.vgg16_bn(pretrained=False)
        del vgg.classifier
        del vgg.avgpool

        self.encoder = self._encodify_(vgg)


    def forward(self, x):
      
        '''Execute the encoder on the image input

        Args:
            x (Tensor): image tensor

        Returns:
            x_code (Tensor): code tensor
            pool_indices (list): Pool indices tensors in order of the pooling modules

        '''
        pool_indices = []
        x_current = x
        
        if DEBUG>0:
          print ( f"EncoderVGG:     INFO:         encode():  x_current.size()                = {ARYLIDE}{x_current.size()}{RESET}", flush=True     )

              
        for module_encode in self.encoder:
          
            output = module_encode( x_current )

            # If the module is pooling, there are two outputs, the second the pool indices
            
            if isinstance(output, tuple) and len(output) == 2:
                x_current = output[0]
                pool_indices.append(output[1])
            else:
                x_current = output

            if DEBUG>0:
              print ( f"AEVGG16:        INFO:         forward(): x.size() after module_encode    = {BITTER_SWEET}{x_current.size()}{RESET}", flush=True     )  

        return x_current, pool_indices
        

    def _encodify_(self, encoder):
      
        '''Create list of modules for encoder based on the architecture in VGG template model.

        In the encoder-decoder architecture, the unpooling operations in the decoder require pooling
        indices from the corresponding pooling operation in the encoder. In VGG template, these indices
        are not returned. Hence the need for this method to extend the pooling operations.

        Args:
            encoder : the template VGG model

        Returns:
            modules : the list of modules that define the encoder corresponding to the VGG model

        '''
        modules = nn.ModuleList()
        for module in encoder.features:
            if isinstance(module, nn.MaxPool2d):
                module_add = nn.MaxPool2d(kernel_size=module.kernel_size,
                                          stride=module.stride,
                                          padding=module.padding,
                                          return_indices=True)
                modules.append(module_add)
            else:
                modules.append(module)

        return modules
        

    @staticmethod
    def dim_code(img_dim):
        '''Convenience function to provide dimension of code given a square image of specified size. The transformation
        is defined by the details of the VGG method. The aim should be to resize the image to produce an integer
        code dimension.

        Args:
            img_dim (int): Height/width dimension of the tentative square image to input to the auto-encoder

        Returns:
            code_dim (float): Height/width dimension of the code
            int_value (bool): If False, the tentative image dimension will not produce an integer dimension for the
                code. If True it will. For actual applications, this value should be True.

        '''
        value = img_dim / 2**5
        int_value = img_dim % 2**5 == 0
        return value, int_value




class DecoderVGG(nn.Module):
    '''Decoder of code based on the architecture of VGG-16 with batch normalization.

    The decoder is created from a pseudo-inversion of the encoder based on VGG-16 with batch normalization. The
    pesudo-inversion is obtained by (1) replacing max pooling layers in the encoder with max un-pooling layers with
    pooling indices from the mirror image max pooling layer, and by (2) replacing 2D convolutions with transposed
    2D convolutions. The ReLU and batch normalization layers are the same as in the encoder, that is subsequent to
    the convolution layer.

    Args:
        encoder: The encoder instance of `EncoderVGG` that is to be inverted into a decoder

    '''
    channels_in = EncoderVGG.channels_code
    channels_out = 3

    def __init__(self, encoder):
        super(DecoderVGG, self).__init__()

        self.decoder = self._invert_(encoder)
        
        

    def forward(self, x, pool_indices):
        '''Execute the decoder on the code tensor input

        Args:
            x (Tensor): code tensor obtained from encoder
            pool_indices (list): Pool indices Pytorch tensors in order the pooling modules in the encoder

        Returns:
            x (Tensor): decoded image tensor

        '''
        x_current = x

        k_pool = 0
        reversed_pool_indices = list(reversed(pool_indices))
        for module_decode in self.decoder:

            # If the module is unpooling, collect the appropriate pooling indices
            if isinstance(module_decode, nn.MaxUnpool2d):
                x_current = module_decode(x_current, indices=reversed_pool_indices[k_pool])
                k_pool += 1
            else:
                x_current = module_decode(x_current)

            if DEBUG>0:
              print ( f"AEVGG16:        INFO:         forward(): x.size() after module_decode    = {CARRIBEAN_GREEN}{x_current.size()}{RESET}", flush=True     )  

        return x_current
        
        

    def _invert_(self, encoder):
      
        '''Invert the encoder in order to create the decoder as a (more or less) mirror image of the encoder

        The decoder is comprised of two principal types: the 2D transpose convolution and the 2D unpooling. The 2D transpose
        convolution is followed by batch normalization and activation. Therefore as the module list of the encoder
        is iterated over in reverse, a convolution in encoder is turned into transposed convolution plus normalization
        and activation, and a maxpooling in encoder is turned into unpooling.

        Args:
            encoder (ModuleList): the encoder

        Returns:
            decoder (ModuleList): the decoder obtained by "inversion" of encoder

        '''
        modules_transpose = []
        for module in reversed(encoder):

            if isinstance(module, nn.Conv2d):
                kwargs = {'in_channels' : module.out_channels, 'out_channels' : module.in_channels,
                          'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.ConvTranspose2d(**kwargs)
                module_norm = nn.BatchNorm2d(module.in_channels)
                module_act = nn.ReLU(inplace=True)
                modules_transpose += [module_transpose, module_norm, module_act]

            elif isinstance(module, nn.MaxPool2d):
                kwargs = {'kernel_size' : module.kernel_size, 'stride' : module.stride,
                          'padding' : module.padding}
                module_transpose = nn.MaxUnpool2d(**kwargs)
                modules_transpose += [module_transpose]

        # Discard the final normalization and activation, so final module is convolution with bias
        modules_transpose = modules_transpose[:-2]

        return nn.ModuleList(modules_transpose)



class AEVGG16(nn.Module):
  
    '''Auto-Encoder based on the VGG-16 with batch normalization. The class is compriseS an encoder and a decoder.

    Args:
        pretrained_params (bool, optional): If the network should be populated with pre-trained VGG parameters. Defaults to True.

    '''
    channels_in   = EncoderVGG.channels_in
    channels_code = EncoderVGG.channels_code
    channels_out  = DecoderVGG.channels_out

    def __init__(  self, cfg, args, n_classes, tile_size, pretrained_params=True ):
      
        super(AEVGG16, self).__init__()

        self.encoder = EncoderVGG( pretrained_params=pretrained_params )
        self.decoder = DecoderVGG( self.encoder.encoder                )

        if 0<=tile_size<32:
          print( f"{RED}VGGNN:           FATAL:  for the VGG models '{CYAN}TILE_SIZE{RESET}{RED}' (corresponding to python argument '{CYAN}--tile_size{RESET}{RED}') is not permitted to be less than {MIKADO}32{RESET}", flush=True)
          print( f"{RED}VGGNN:           FATAL: ... halting now{RESET}" )
          sys.exit(0)
        elif 32<=tile_size<64:
          first_fc_width=512
        elif 64<=tile_size<95:
          first_fc_width=2048
        elif 96<=tile_size<128:
          first_fc_width=4608
        elif 128<=tile_size<160:
          first_fc_width=8192 
        elif 160<=tile_size<192:
          first_fc_width=12800
        elif 192<=tile_size<223:
          first_fc_width=18432 
        elif 224<=tile_size<1024:
          first_fc_width=25088
                
        self.fc1 = nn.Linear(first_fc_width, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, args.gene_embed_dim[0] )        

        self.Dropout = nn.Dropout()
                

    def forward( self, args, x, gpu, encoder_activation ):
    
        '''Forward the autoencoder for image input

        Args:
            x (Tensor): image tensor

        Returns:
            x_prime (Tensor): image tensor following encoding and decoding

        '''
        
        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         forward(): x.size() at input               = {AMETHYST}{x.size()}{RESET}", flush=True     ) 
                  
        x, pool_indices = self.encoder(x)

        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         forward(): x.size() after encode           = {BRIGHT_GREEN}{x.size()}{RESET}", flush=True     ) 

        x = self.decoder(x, pool_indices)

        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         forward(): x.size() after decode           = {BRIGHT_GREEN}{x.size()}{RESET}", flush=True     )    

        return x, 0, 0
        
        

    def encode( self, x, gpu, encoder_activation ):                                                        # adapter function added by PGD to re-introduce the VGG fully connected layers when creating an embedding
    
        '''Forward the autoencoder for image input

        Args:
            x (Tensor): image tensor

        Returns:
            z (Tensor): image tensor following encoding

        '''
        x, pool_indices = self.encoder.forward( x )

        x = x.view(x.size(0), -1)                                                                          # flatten
        # ~ x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = F.relu(x) 
        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         forward(): x.size() after         fc1/relu = {BLEU}{x.size()}{RESET}{CLEAR_LINE}" )
        x = self.Dropout(x)
        x = self.fc2(x)
        embedding = x
        x = F.relu(x)
        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         forward(): x.size() after dropout/fc2/relu = {BLEU}{x.size()}{RESET}{CLEAR_LINE}" )
        if DEBUG>88:
          print ( f"AEVGG16:        INFO:         forward(): x[:,0:20]                                 = {BLEU}{x[:,0:20]}{RESET}{CLEAR_LINE}" )
        x = self.Dropout(x)
        x = self.fc3(x)
        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         forward(): x.size() after dropout/fc3/relu = {BLEU}{x.size()}{RESET}{CLEAR_LINE}" )

        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         forward(): x.size() after all FC layers    = {CARRIBEAN_GREEN}{x.size()}{RESET}{CLEAR_LINE}" )
          

        if DEBUG>0:
          print ( f"AEVGG16:        INFO:         encode():  x.size() after  x.view          = {ARYLIDE}{x.size()}{RESET}", flush=True     )              
        
        return x
        
        
    @staticmethod
    def dim_code( img_dim ):
      
        '''Convenience function to provide dimension of code given a square image of specified size. The transformation
        is defined by the details of the VGG method. The aim should be to resize the image to produce an integer
        code dimension.

        Args:
            img_dim (int): Height/width dimension of the tentative square image to input to the auto-encoder

        Returns:
            code_dim (float): Height/width dimension of the code
            int_value (bool): If False, the tentative image dimension will not produce an integer dimension for the
                code. If True it will. For actual applications, this value should be True.

        '''
        return EncoderVGG.dim_code(img_dim)

    @staticmethod
    def state_dict_mutate(encoder_or_decoder, ae_state_dict):
        '''Mutate an auto-encoder state dictionary into a pure encoder or decoder state dictionary

        The method depends on the naming of the encoder and decoder attribute names as defined in the auto-encoder
        initialization. Currently these names are "encoder" and "decoder".

        The state dictionary that is returned can be loaded into a pure EncoderVGG or DecoderVGG instance.

        Args:
            encoder_or_decoder (str): Specification if mutation should be to an encoder state dictionary or decoder
                state dictionary, where the former is denoted with "encoder" and the latter "decoder"
            ae_state_dict (OrderedDict): The auto-encoder state dictionary to mutate

        Returns:
            state_dict (OrderedDict): The mutated state dictionary that can be loaded into either an EncoderVGG
                or DecoderVGG instance

        Raises:
            RuntimeError : if state dictionary contains keys that cannot be attributed to either encoder or decoder
            ValueError : if specified mutation is neither "encoder" or "decoder"

        '''
        if not (encoder_or_decoder == 'encoder' or encoder_or_decoder == 'decoder'):
            raise ValueError('State dictionary mutation only for "encoder" or "decoder", not {}'.format(encoder_or_decoder))

        keys = list(ae_state_dict)
        for key in keys:
            if 'encoder' in key or 'decoder' in key:
                if encoder_or_decoder in key:
                    key_new = key[len(encoder_or_decoder) + 1:]
                    ae_state_dict[key_new] = ae_state_dict[key]
                    del ae_state_dict[key]

                else:
                    del ae_state_dict[key]

            else:
                raise RuntimeError('State dictionary key {} is neither part of encoder or decoder'.format(key))

        return ae_state_dict



