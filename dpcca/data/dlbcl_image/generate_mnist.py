"""=============================================================================
Script to generate MNIST dataset for IMAGE ONLY analysis.
============================================================================="""

import numpy as np

import torch
from   torch.distributions.multivariate_normal import MultivariateNormal
from   torchvision import transforms
import torchvision.datasets as datasets

import random
from   data.mnist.config import MnistConfig

DEBUG=1

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)


output_dir = '/home/peter/git/pipeline/dpcca/data/dlbcl_image/train.pth'


# ------------------------------------------------------------------------------

def main(cfg):

    train_set = datasets.MNIST(root=cfg.ROOT_DIR,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
    
    images = train_set.data.numpy()
    labels = train_set.targets.numpy()

    inds      = (labels == 0) | (labels == 1) | (labels == 2) | (labels == 3) | (labels == 4) | (labels == 5) | (labels == 6) | (labels == 7) | (labels == 8) | (labels == 9)
    images    = images[inds]
    labels    = labels[inds]
    n_samples = len(labels)

    images_new = np.empty((n_samples, 28, 28))
    labels_new = np.empty((n_samples,))
    genes_new  = np.empty((n_samples, cfg.N_GENES))

    if DEBUG>0:
      print ( "number of samples = {:}".format( n_samples ))


    for i, (img, lab) in enumerate(zip(images, labels)):

      print( i+1 )
      print( "\033[1A", end="" )

      images_new[i] = img
      labels_new[i] = lab
      genes_new[i]  = 1773

      if DEBUG>9:
        print ( "\nimages_new [{:}].shape = {:}".format(i,  images_new[i].shape))
        print ( "type(images_new [{:}]) = {:}".format(i,  type(images_new[i])))
        print ( "images_new[{:}] = \n{:}".format(i, images_new[i]))
        print ( "\nlabels_new [{:}].shape = {:}".format(i,  labels_new[i].shape))
        print ( "type(labels_new [{:}]) = {:}".format(i,  type(labels_new[i])))            
        print ( "labels_new[{:}] = {:}".format(i, labels_new[i]))
        print ( "\ngenes_new [{:}].shape = {:}".format(i,  genes_new[i].shape))
        print ( "type(genes_new [{:}]) = {:}".format(i,  type(genes_new[i])))                                
        print ( "genes_new [{:}] = \n{:}".format(i,  genes_new[i]))            

    images_new = torch.Tensor(images_new)
    labels_new = torch.Tensor(labels_new)
    genes_new  = torch.Tensor(genes_new)
    
    print( "\033[1B", end="" )


    if DEBUG>0:
      print ( "saving torch formatted data file to {:}".format( output_dir ) )
    
    torch.save({
        'images': images_new,
        'tissues': labels_new,
        'genes':  genes_new
    }, output_dir )

    print('all processing complete')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = MnistConfig()
    main(cfg)
