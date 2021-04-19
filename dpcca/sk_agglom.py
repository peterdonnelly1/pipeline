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

from scipy import ndimage

# ~ print(__doc__)

from time            import time
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

def sk_agglom( args, pct_test):
  
  n_clusters   = 8

  
  # 1. load and prepare data

  sample_file = "../logs/images_new.npy" 
  label_file = "../logs/img_labels_new.npy"
  
  samples = np.load( sample_file )
  labels  = np.load( label_file  )

  if DEBUG>9:
    print( f"SK_AGGLOM:     INFO:  label file        = {CYAN}{labels}{RESET} \r\033[60Ccontains {MIKADO}{labels.shape[0]}{RESET} labels", flush=True)
    
  if DEBUG>0:
    print( f"\n{GREY_BACKGROUND}SK_AGGLOM:     INFO: {WHITE}{CHARTREUSE}SKAGGLOM{WHITE}: samples_file={MAGENTA}{sample_file}{WHITE}, labels_file={MAGENTA}{label_file}{WHITE} n_clusters={MIKADO}{n_clusters}                                                                                                                        {RESET}" )  

  x_npy = samples.reshape( samples.shape[0], samples.shape[1]*samples.shape[2]*samples.shape[3] )
  
  
  if DEBUG>0:
    print( f"SK_AGGLOM:     INFO:  sample file shape = {MIKADO}{samples.shape}{RESET}" )
    print( f"SK_AGGLOM:     INFO:  x_npy shape       = {MIKADO}{x_npy.shape}{RESET}"         )
 

  if DEBUG>2:
    print( f"SK_AGGLOM:     INFO:  samples[0]        = \n{MIKADO}{samples[0,2,40:80,90:100]}{RESET}" )  
    print( f"SK_AGGLOM:     INFO:  x_npy [0]         =  {MIKADO}{x_npy[0,1000:1100]}{RESET}" )  



  # 2. cluster


  for linkage in ('ward', 'average', 'complete'):
    
    ##############################################################################
    clustering = AgglomerativeClustering( linkage=linkage, n_clusters=n_clusters )
    ##############################################################################   
    
    t0 = time()

    clustering.fit( x_npy )


    if DEBUG>0:
      print( f"SK_AGGLOM:     INFO:  clustering.labels_ = \n{MIKADO}{clustering.labels_}{RESET}" )
      
      
    if (DEBUG>0):
      all_clusters_unique=sorted(set(clustering.labels_))
      print ( f"HDBSCAN:         INFO:  unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
    
    if (DEBUG>0):
      for i in range ( 0, len(all_clusters_unique) ):
        print ( f"HDBSCAN:         INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(clustering.labels_==i).sum()}{RESET}" )
        
  
    # 3. plot the results as a scattergram
      
    plot( x_npy, labels, clustering.labels_, n_clusters, f"{linkage:s}" )
    
    
  plt.show()
 

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def plot(x, labels, labels_, n_clusters, title=None, ):
  
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    x        = (x - x_min) / (x_max - x_min)

    plt.figure(figsize=(20,14))
    
    for i in range(x.shape[0]):

      if DEBUG>2:
        print( f"SK_AGGLOM:     INFO: x= {MIKADO}{x[i,0]:4.3f}{RESET}  y= {MIKADO}{x[i,1]:4.3f}{RESET}  cluster label[{MIKADO}{i}{RESET}] = {CARRIBEAN_GREEN}{labels_[i]}{RESET}" )
        
      plt.text(
        x[i, 0],                                                                                     # x ordinate
        x[i, 1],                                                                                     # y ordinate
        str(labels_[i]),                                                                                 # text to place at x,y         
        color = plt.cm.get_cmap("Spectral") (labels_[i] / n_clusters ),                                  # color of this text element
        fontdict={'weight': 'bold', 'size': 6 }                                                          # constant attributes of text
        )

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()
  
"""  
def plot(
    x,
    y,
    class_names,
    title=None,    
    ax=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        plt, ax = matplotlib.pyplot.subplots(figsize=(14,14))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)

    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
    point_colors = list(map(colors.get, y))

    if (DEBUG>2):
      print ( f"SKTSNE:         INFO: plot()  class_names           = {BITTER_SWEET}{class_names}{RESET}" )
      print ( f"SKTSNE:         INFO: plot()  classes               = {BITTER_SWEET}{classes}{RESET}" )
      print ( f"SKTSNE:         INFO: plot()  colors                = {BITTER_SWEET}{colors}{RESET}" )
      print ( f"SKTSNE:         INFO: plot()  colors.get            = {BITTER_SWEET}{colors.get}{RESET}" )
      print ( f"SKTSNE:         INFO: plot()  point_colors          = {BITTER_SWEET}{point_colors}{RESET}" )

    # ~ lim = ( x.min(), x.max() )
    
    if (DEBUG>2):
      print ( f"SKTSNE:         INFO: plot()  x[:, 0].min()               = {BITTER_SWEET}{x[:, 0].min()}{RESET}" )
      print ( f"SKTSNE:         INFO: plot()  x[:, 0].max()               = {BITTER_SWEET}{x[:, 0].max()}{RESET}" )
      print ( f"SKTSNE:         INFO: plot()  x[:, 1].min()               = {BITTER_SWEET}{x[:, 1].min()}{RESET}" )
      print ( f"SKTSNE:         INFO: plot()  x[:, 1].max()               = {BITTER_SWEET}{x[:, 1].max()}{RESET}" )      

    x1 = x[:, 0]
    x2 = x[:, 1]
    std_devs=4
    ax.set_xlim( [ np.median(x1)-std_devs*np.std(x1), np.median(x1)+std_devs*np.std(x1) ] )
    ax.set_ylim( [ np.median(x2)-std_devs*np.std(x2), np.median(x2)+std_devs*np.std(x2) ] )
    
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params) 
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True) 
    ax.scatter( x1, x2, c=point_colors, s=4, marker="s")

"""
