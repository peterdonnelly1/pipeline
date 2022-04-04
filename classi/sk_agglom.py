"""
=============================================================================
Various Agglomerative Clustering on a 2D embedding of digits
=============================================================================

An illustration of various linkage option for agglomerative clustering <snip>

The goal of this example is to show intuitively how the metrics behave, and not to find good clusters for the digits. This is why the example works on a 2D embedding.

What this example shows us is the behavior "rich getting richer" of agglomerative clustering that tends to create uneven cluster sizes. 
This behavior is especially pronounced for the average linkage strategy, that ends up with a couple of singleton clusters.

"""

# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

import sys
import torch
import random
import argparse
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import matplotlib.pyplot as plt
import matplotlib.colors

from scipy import ndimage

# ~ print(__doc__)

from time            import time
from matplotlib      import pyplot as plt
from sklearn         import manifold, datasets
from sklearn.cluster import AgglomerativeClustering


from constants  import *

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}


DEBUG   = 1

np.set_printoptions(edgeitems=100000)
np.set_printoptions(linewidth=100000)

def sk_agglom( args, pct_test):
  
  
  n_clusters   = args.n_clusters
  is_embedding   = args.use_autoencoder_output=='True'
    
  if args.use_autoencoder_output=='True':
    
    fqn = f"../logs/ae_output_features.pt"
      
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}SK_SPECTRAL:     INFO:  about to load autoencoder generated is_embedding from input file '{MAGENTA}{fqn}{RESET}'", flush=True )
    try:
      dataset  = torch.load( fqn )
      if DEBUG>0:
        print( f"{BRIGHT_GREEN}SK_SPECTRAL:     INFO:  dataset successfully loaded{RESET}" ) 
    except Exception as e:
      print ( f"{RED}SK_SPECTRAL:     ERROR:  could not load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
      print ( f"{RED}SK_SPECTRAL:     ERROR:  the exception was: {CYAN}'{e}'{RESET}" )
      print ( f"{RED}SK_SPECTRAL:     ERROR:  halting now" )
      sys.exit(0)
  
    samples_npy  = dataset['embeddings'].cpu().numpy().squeeze()                                           # eliminate empty dimensions
    labels       = dataset['labels'    ].cpu().numpy().squeeze()                                           # eliminate empty dimensions
    
    if DEBUG>0:
      print ( f"SK_SPECTRAL:     INFO:  (is_embedding) samples_npy.shape     =  {MIKADO}{samples_npy.shape}{RESET}"      ) 
      print ( f"SK_SPECTRAL:     INFO:  sanity check: np.sum(samples_npy)  =  {MIKADO}{np.sum(samples_npy):.2f}{RESET}"      ) 
    
    if np.sum(samples_npy)==0.0:
      print ( f"{RED}SK_SPECTRAL:     ERROR:  all samples_npy are zero vectors - the input file was completely degenerate{RESET}" )
      print ( f"{RED}SK_SPECTRAL:     ERROR:  not halting, but might as well be{RESET}" )
 
  else:
    
    sample_file = "../logs/images_new.npy" 
    label_file = "../logs/img_labels_new.npy"
    
    samples_npy  =  np.load( sample_file )
    labels       =  np.load( label_file  )
  

  if args.input_mode=='image':
    
    samples = samples_npy
    
    if DEBUG>0:
      print ( f"SK_SPECTRAL:     INFO:  about to flatten channels and r,g,b dimensions"      ) 
      print ( f"SK_SPECTRAL:     INFO:  (flattened) samples.shape          = {MIKADO}{samples.shape}{RESET}"      ) 

  if args.input_mode=='rna': 
    samples = samples_npy
  
    if DEBUG>0:
      print ( f"SK_SPECTRAL:     INFO:  samples.shape          = {MIKADO}{samples.shape}{RESET}"      ) 



  # 2. cluster

  for linkage in ('ward', 'average', 'complete'):
    
    ##############################################################################
    clustering = AgglomerativeClustering( linkage=linkage, n_clusters=n_clusters )
    ##############################################################################   
    
    t0 = time()

    clustering.fit( samples )


    if DEBUG>0:
      print( f"SK_AGGLOM:     INFO:  clustering.labels_ = \n{MIKADO}{clustering.labels_}{RESET}" )
      
      
    all_clusters_unique=sorted(set(clustering.labels_))
    if (DEBUG>0):
      print ( f"SK_AGGLOM:     INFO:  unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
    
    if (DEBUG>0):
      for i in range ( 0, len(all_clusters_unique) ):
        print ( f"SK_AGGLOM:     INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(clustering.labels_==i).sum()}{RESET}" )
        
  
    # 3. plot the results as a scattergram
      
    plot( args, is_embedding, samples_npy.shape, clustering.labels_, labels,  n_clusters, all_clusters_unique,  f"{linkage:s}" )
    
    plt.show()
 

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------


def plot(args, is_embedding, shape, cluster_labels, true_labels, n_clusters, all_clusters_unique, mode ):
  
  # 3. plot the results as a jittergram
    
  figure_width  = 20
  figure_height = 10
  fig, ax       = plt.subplots( figsize = (figure_width, figure_height) )
  # ~ fig.tight_layout()
  
  # ~ color_palette         = sns.color_palette('bright', 100)
  # ~ cluster_colors        = [color_palette[x] for x in clustering.cluster_labels]
  # ~ cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clustering.probabilities_)]
  
  colors  = [f"C{i}" for i in np.arange(1, cluster_labels.max()+2)]
  if (DEBUG>1):
    print ( f"SK_SPECTRAL:      INFO:  colors    = {MIKADO}{colors}{RESET}" )
  cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, cluster_labels.max()+3), colors )
  
  X = cluster_labels
  Y = true_labels
  
  X_jitter = np.zeros_like(X)
  X_jitter = [ random.uniform( -0.45, 0.45 ) for i in range( 0, len(X) ) ]
 
  X = X + X_jitter
  
  N=true_labels.shape[0]
  title    = f"Unsupervised Agglomerative Clustering of {N:,} TCGA {args.dataset.upper()} {args.input_mode}s;  X=cluster number (jittered), Y=true subtype"
  subtitle = f"mode = {mode}'  n_clusters={n_clusters};  input dims = {shape[1:]};  autoencoder input used={is_embedding}"
  
  plt.title ( title, fontsize=16 )
  plt.text  ( -.2, 0.2, subtitle, ha='left', fontsize=12 )


  xx     = np.arange(0, len(all_clusters_unique), step=1)
  true_labels = all_clusters_unique
  plt.xticks( xx, labels=true_labels )
  
  yy     = [ i for i in range (0, len(args.class_names) )]
  true_labels = args.class_names
  plt.yticks(yy, labels=true_labels )

  s = ax.scatter( X, Y, s=5, linewidth=0, marker="s", c=cluster_labels, cmap=cmap, alpha=1.0)
  legend1 = ax.legend(*s.legend_elements(), loc="upper left", title="cluster number")
  ax.add_artist(legend1)

  if (DEBUG>1):
    offset=.5
    for i, label in enumerate( true_labels ):
      plt.annotate( args.class_names[label][0], ( X[i]-.25, Y[i]-.5), fontsize=5, color='black' )
  
      if (DEBUG>1):  
        print ( f"i={i:4d} label={MIKADO}{label}{RESET}  args.class_names[label]={MIKADO}{ args.class_names[label]:16s}{RESET} args.class_names[label][0]={MIKADO}{args.class_names[label][0]}{RESET}" )

  if DEBUG>1:
    print( f"SK_SPECTRAL:     INFO: X = \n{MIKADO}{X}{RESET}" )
    print( f"SK_SPECTRAL:     INFO: Y = \n{MIKADO}{Y}{RESET}" )

    
  