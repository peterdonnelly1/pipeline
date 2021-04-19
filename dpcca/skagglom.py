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

import argparse

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import matplotlib.pyplot as plt
import matplotlib.colors

print(__doc__)

from time            import time
from scipy           import ndsample
from matplotlib      import pyplot as plt
from sklearn         import manifold, datasets
from sklearn.cluster import AgglomerativeClustering


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
HOT_PINK='\033[38;2;255;105;180m'
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
GREY_BACKGROUND='\033[48;2;60;60;60m'

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

FAIL    = 0
SUCCESS = 1

DEBUG   = 1

np.set_printoptions(edgeitems=100000)
np.set_printoptions(linewidth=100000)

def agglom( args, pct_test):
  
  n_clusters   = 7

  
  # 1. load and prepare data

  sample_file = "../logs/images_new.npy" 
  label_file = "../logs/img_labels_new.npy"
  
  samples = np.load( sample_file )
  labels  = np.load( label_file  )

  if DEBUG>0:
    print( f"\n{GREY_BACKGROUND}SKAGGLOM:      INFO: {WHITE}{CHARTREUSE}SKAGGLOM{WHITE}: samples_file={MAGENTA}{sample_file}{WHITE}, labels_file={MAGENTA}{label_file}{WHITE} clusters={MIKADO}{clusters}                                                                                                                        {RESET}" )  

  x_npy = samples.reshape( samples.shape[0], samples.shape[1]*samples.shape[2]*samples.shape[3] )
  
  print("Computing embedding using sklearn manifold.SpectralEmbedding")
  X_2d = manifold.SpectralEmbedding(n_components=2).fit_transform(x_npy)
  print("Done.")
  
  if DEBUG>0:
    print( f"SKAGGLOM:      INFO:  sample file shape = {MIKADO}{samples.shape}{RESET}" )
    print( f"SKAGGLOM:      INFO:  x_npy shape       = {MIKADO}{x_npy.shape}{RESET}"         )
    print( f"SKAGGLOM:      INFO:  X_2d  shape       = {MIKADO}{X_2d.shape}{RESET}"          )
    print( f"SKAGGLOM:      INFO:  label file        = {CYAN}{labels}{RESET} \r\033[60Ccontains {MIKADO}{labels.shape[0]}{RESET} labels", flush=True)
 

  if DEBUG>2:
    print( f"SKAGGLOM:      INFO:  samples[0]        = \n{MIKADO}{samples[0,2,40:80,90:100]}{RESET}" )  
    print( f"SKAGGLOM:      INFO:  x_npy [0]         =  {MIKADO}{x_npy[0,1000:1100]}{RESET}" )  



  # 2. cluster

  
  if DEBUG>0:
    print( f"SKAGGLOM:       INFO:  about to configure {CYAN}Sk-learn AgglomerativeClustering {RESET}object with: metric='{CYAN}{metric}{RESET}', n_iter={MIKADO}{n_iter}{RESET}, n_components={MIKADO}{n_components}{RESET}, perplexity={MIKADO}{perplexity}{RESET}, n_jobs={MIKADO}{n_jobs}{RESET}", flush=True )
  

  for linkage in ('ward', 'average', 'complete'):
    
    ######################################################
    clustering = AgglomerativeClustering( linkage=linkage, n_clusters=n_clusters )
    ######################################################   

    t0 = time()

    clustering.fit(X_2d)

    print("%s : %.2fs" % (linkage, time() - t0))
    
    plot_clustering(X_2d, clustering.labels_, "%s linkage" % linkage)
    
    
  plt.show()  
  
  # 3. plot the results as a scattergram
 
 

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def plot_clustering(X_2d, labels, title=None):
  
  x_min, x_max = np.min(X_2d, axis=0), np.max(X_2d, axis=0)
  X_2d = (X_2d - x_min) / (x_max - x_min)

  plt.figure(figsize=(6, 4))
  for i in range(X_2d.shape[0]):
      plt.text(X_2d[i, 0], X_2d[i, 1], str(y[i]),
               color=plt.cm.spectral(labels[i] / 10.),
               fontdict={'weight': 'bold', 'size': 9})

  plt.xticks([])
  plt.yticks([])
  if title is not None:
      plt.title(title, size=17)
  plt.axis('off')
  plt.tight_layout()
 
  
  # ~ figure_width  = 20
  # ~ figure_height = 10
  # ~ fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

  # ~ if (DEBUG>2):
    # ~ np.set_printoptions(formatter={'int': lambda x:   "{:>2d}".format(x)})
    # ~ print ( f"SKAGGLOM:      INFO:  labels    = {MIKADO}{labels}{RESET}" )
  # ~ c = labels
  # ~ if (DEBUG>2):
    # ~ print ( f"SKAGGLOM:      INFO:  labels+1  = {MIKADO}{c}{RESET}" )
  # ~ colors  = MACOSKO_COLORS
  # ~ if (DEBUG>2):
    # ~ print ( f"SKAGGLOM:      INFO:  colors               = {MIKADO}{colors}{RESET}" )
    # ~ print ( f"SKAGGLOM:      INFO:  np.unique(labels)    = {MIKADO}{np.unique(labels)}{RESET}" )

  # ~ if (DEBUG>2):
    # ~ print ( f"SKAGGLOM:      INFO:  labels               = {MIKADO}{labels}{RESET}" )

  # ~ plot( embedding_train, labels, args.class_names, ax=ax )
  # ~ plt.show()

