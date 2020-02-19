"""=============================================================================
Script to generate multimodal MNIST dataset.
============================================================================="""

import numpy as np

import torch
from   torch.distributions.multivariate_normal import MultivariateNormal
from   torchvision import transforms
import torchvision.datasets as datasets

import random
from   data.mnist.config import MnistConfig

DEBUG=0

#np.set_printoptions(edgeitems=200)
np.set_printoptions(linewidth=1000)

# ------------------------------------------------------------------------------

def main(cfg):

    train_set = datasets.MNIST(root=cfg.ROOT_DIR,
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

    
    images = train_set.data.numpy()
    labels = train_set.targets.numpy()

    # Select only 0s, 1s, and 2s.
    inds      = (labels == 0) | (labels == 1) | (labels == 2)
    images    = images[inds]
    labels    = labels[inds]
    n_samples = len(labels)

    p2 = cfg.N_GENES

    # Map MNIST class to multivariate normal.
    multimodal_mvn_map = {
        0: MultivariateNormal(torch.ones(p2),       torch.eye(p2)),
        1: MultivariateNormal(torch.ones(p2) * 10,  torch.eye(p2) * 5)
    }

    images_new = np.empty((n_samples, 28, 28))
    labels_new = np.empty((n_samples,))
    genes_new  = np.empty((n_samples, cfg.N_GENES))

    j = 0
    while len(images):

        if j % 1000 == 0:
            print('%s / 18600' % j)

        # The latent variable is the MNIST class, which is associated with one
        # of three multivariate Gaussian random variables.

        # Pick one of three classes.
        r1 = random.randint(0, 2)

        # Pick one of the two gene-specific distributions.
        idx = 0 if r1 <= 1 else 1

        # Find an image with the correct class.
        for i, (img, lab) in enumerate(zip(images, labels)):

            if r1 != lab:
                continue

            images_new[j] = img
            labels_new[j] = lab

            # Sample from gene distribution that correlates with digit class.
            genes_new[j]  = multimodal_mvn_map[idx].sample()

            # Remove image from list of candidates.
            new_inds = list(range(len(images)))
            new_inds.pop(i)
            images = images[new_inds, :]
            labels = labels[new_inds]

            if DEBUG>1:
              print ( "\nimages_new [{:}].shape = {:}".format(j,  images_new[j].shape))
              print ( "type(images_new [{:}]) = {:}".format(j,  type(images_new[j])))
              print ( "images_new[{:}] = \n{:}".format(j, images_new[j]))
              print ( "\nlabels_new [{:}].shape = {:}".format(j,  labels_new[j].shape))
              print ( "type(labels_new [{:}]) = {:}".format(j,  type(labels_new[j])))            
              print ( "labels_new[{:}] = {:}".format(j, labels_new[j]))
              print ( "\ngenes_new [{:}].shape = {:}".format(j,  genes_new[j].shape))
              print ( "type(genes_new [{:}]) = {:}".format(j,  type(genes_new[j])))                                
              print ( "genes_new [{:}] = \n{:}".format(j,  genes_new[j]))            

            j += 1

            break

    images_new = torch.Tensor(images_new)
    labels_new = torch.Tensor(labels_new)
    genes_new  = torch.Tensor(genes_new)
    
    examples = torch.Tensor(64, 1, 28, 28)
    
    for i in range(64):
        r1 = random.randint(0, n_samples)
        img = images_new[r1]
        img = img.unsqueeze(0)
        examples[i] = img
 
    print ( {
        'images': images_new,
        'labels': labels_new,
        'genes':  genes_new
    } )
    
    torch.save({
        'images': images_new,
        'labels': labels_new,
        'genes':  genes_new
    }, '%s/train.pth' % cfg.ROOT_DIR)

    print('all processing complete')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = MnistConfig()
    main(cfg)
