"""================================================================================================================================================

Normalization scripts for digital pathology tasks

Source: https://github.com/tand826/stain_normalization
            
Author: Takumiando, University of Tokyo, Tokyo, Japan

================================================================================================================================================"""

import sys
import cv2
import numpy as np
import spams

np.set_printoptions(edgeitems=38)
np.set_printoptions(linewidth=350)

DEBUG=0

class Normalizer(object):
    """
    Normalize the patches with some methods implemented.

    Parameters
    ----------
    method : str
        Specify the stain normalization method. Method can be selected from None, reinhard, spcn,
        gan, nct
    source : pathlib.PosixPath
        Path to the source image to normalize images from.
    target : pathlib.PosixPath
        Path to the target image to USE AS A REFERENCE WHEN NORMALIZING
    """

    def __init__(self, method, target):
        self.method = method
        self.normalizer = NormalizerReinhard(target)                                                   # Norm.normalizer = NormalizerReinhard(parameters)

        """
        if method == "NONE":
            self.normalizer = NormalizerNone(target)
        elif method == "reinhard":
            self.normalizer = NormalizerReinhard(target)                                                   # Norm.normalizer = NormalizerReinhard(parameters)
        elif method == "spcn":
            self.normalizer = NormalizerSPCN(target)
        elif method == "staingan":
            print(sys.exc_info())
        elif method == "nct":
            print(sys.exc_info())
        else:
            raise(f"Method {method} is not defined.")
        """
    
    
    def __call__(self, source):
        normalized = self.normalizer(source)
        return normalized

    def __repr__(self):
        return self.__class__.__name__ + f"(method={self.method})"


    def activate_normalizer( norm, tile ):

        if norm == 'reinhard':
            print( f"NORMS.PY:                 INFO:    NormalizerReinhard: get_normalizer(): normalizer = \033[36;1m{norm}\033[m" )
            return NormalizerReinhard( tile )
        elif norm == 'spcn':
            print( f"NORMS.PY:                 INFO:    NormalizerReinhard: get_normalizer(): normalizer = \033[36;1m{norm}\033[m" )
            return NormalizerSPCN( tile )
        else:
            print( f"NORMS.PY:                 INFO:   NormalizerReinhard: get_normalizer(): defaulting to 'reinhart' stain normalizer")
            return NormalizerReinhard( tile )


class NormalizerNone:

    def __init__(self, target):
        pass

    def __call__(self, source):
        return cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB)


class NormalizerReinhard:

    def __init__(self, target):
        """
        if target.shape == (2,3) of np.array,
        the first row is the rgb mean values in LAB color space,
        the second row is the rgb std values in LAB color space.
        Both have to be calculated in float32
        """
        
        if isinstance(target, np.ndarray):         # if target is a numpy array
            self.target_mean = target[0]
            self.target_std  = target[1]
            if (DEBUG>0 ):
              print("NORMS.PY:                 INFO:    NormalizerReinhard: __init__(): target.shape = \033[35m{:}\033[m".format   ( target.shape       ));
              print("NORMS.PY:                 INFO:    NormalizerReinhard: __init__(): type(target) = \033[35m{:}\033[m".format   ( type(target)       ));
              #print("NORMS.PY:                 INFO:    NormalizerReinhard: __init__(): target_mean  = \033[35m{:}\033[m".format ( self.target_mean   ));
              #print("NORMS.PY:                 INFO:    NormalizerReinhard: __init__(): target_std   = \033[35m{:}\033[m".format ( self.target_std    ));
        else:
            self.set_target(target)
            if (DEBUG>0):  
              print("NORMS.PY:                 INFO:    NormalizerReinhard: __init__(): user provided image to us as target = ".format( target ));


    def __call__(self, source):

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerReinhard: __call__(): source = ".format( source ));
        normalized = self.normalize(source)
        return normalized

    def set_target(self, target):

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerReinhard: set_target(): target = ".format( target ));

        target_img = self.preprocess(cv2.imread(target, 1))
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
        self.target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
        self.target_std = np.std(target_lab.reshape(-1, 3), axis=0)

    def preprocess(self, img):

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerReinhard: preprocess(): ");

        return (img / 255).astype(np.float32)


    #def normalize(self, source: np.array) -> np.float32:
    def normalize(self, source ):
        """
        [description]
            Transfer the distribution of target image to the distribution of source image in the Lab color space.
        [Arguments]
            src {np.array}
                -- Source image which has BGR components.
        [Returns]
            normalized {np.float32}
                -- Transferred result which has BGR components.
        """

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerReinhard: at top of normalize(): ".format( source ) );

        #source_img = self.preprocess(cv2.imread(source, 1))
        source_img = self.preprocess(source)
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)

        source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
        source_std = np.std(source_lab.reshape(-1, 3), axis=0)

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerReinhard: normalize(): source_img  = ".format ( source_img  ) );
          print("NORMS.PY:                 INFO:   NormalizerReinhard: normalize(): source_lab  = ".format ( source_lab  ) );
          print("NORMS.PY:                 INFO:   NormalizerReinhard: normalize(): source_mean = ".format ( source_mean ) );
          print("NORMS.PY:                 INFO:   NormalizerReinhard: normalize(): source_std  = ".format ( source_std  ) );

        source_norm = (source_lab - source_mean) / source_std
        transferred = (source_norm * self.target_std + self.target_mean)

        normalized = cv2.cvtColor(transferred.astype(np.float32), cv2.COLOR_LAB2BGR)

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerReinhard: normalized version = ".format ( normalized ) );

        return normalized


class NormalizerSPCN:
    
    
    def __init__(self, target):
        """
        if target.shape == (2,3) of np.array, target is the target_matrix. Otherwise, you can get the matrix from an image, by giving the path to the image.
        """

        if isinstance(target, np.ndarray):
            self.target_mat = target
            if (DEBUG>9):  
              print("NORMS.PY:                 INFO:   NormalizerSPCN: __init__() user defined target".format( target ));
        else:
            if (DEBUG>9):  
              print("NORMS.PY:                 INFO:   NormalizerSPCN: __init__() 'you have set stain_matrix'".format( target ));
            self.set_target(target)

    def set_target(self, target):

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerSPCN: set_target()")

        target_img = cv2.imread(target, 1)
        target_od = self.beer_lambert(target_img).reshape([-1, 3])
        _, self.target_mat = self.snmf(target_od)

    def __call__(self, source):

        if (DEBUG>0):  
          print("NORMS.PY:                 INFO:   NormalizerSPCN: __call__(): now processing".format( source.shape ));

        print ( source )

#        source_img = cv2.imread(source, 1)
#        source_od = self.beer_lambert(source_img).reshape([-1, 3])
#        source_dict, _ = self.snmf(source_od)
#        w, h, _ = source_img.shape

        source_od = self.beer_lambert(source).reshape([-1, 3])
        source_dict, _ = self.snmf(source_od)
        w, h, _ = source.shape

        norm = np.dot(source_dict, self.target_mat)
        norm_rgb = self.beer_lambert_reverse(norm)
        normalized = norm_rgb.reshape([w, h, 3])

        return normalized

    def beer_lambert(self, img):
        """Convert image into OD space refering to Beer-Lambert law."""

        if (DEBUG>0):  
          print("NORMS.PY:                   INFO: NormalizerSPCN: beer_lambert()")

        return np.log(255/(img + 1e-6))

    def snmf(self, img):
        """Sparse Non-negative Matrix Factorization with spams."""

        if (DEBUG>0):  
          print("NORMS.PY:                   INFO: NormalizerSPCN: snmf()")
          
        img = np.asfortranarray(img)
        (W, H) = spams.nmf(img, K=2, return_lasso=True)
        H = np.array(H.todense())
        return W, H

    def beer_lambert_reverse(self, img_od):
        """Reverse calculation of beer_lambert()"""
        
        if (DEBUG>0):  
          print("NORMS.PY:                   INFO: NormalizerSPCN: beer_lambert_reverse()")
                  
        img = 255 / np.exp(img_od) - 1e-6
        return img.astype(np.uint8)
