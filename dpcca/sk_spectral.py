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

# ~ print(__doc__)

from time            import time
from matplotlib      import pyplot as plt
from sklearn         import manifold, datasets
from sklearn.cluster import SpectralClustering


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

def sk_spectral( args, pct_test):
  
  n_clusters   = args.n_clusters
  eigen_solver = 'arpack'
  affinity     = "nearest_neighbors" 
  embeddings   = args.use_autoencoder_output=='True'
  
  # 1. load and prepare data

  if args.use_autoencoder_output=='True':
    
    fqn = f"../logs/ae_output_features.pt"
      
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}SK_SPECTRAL:     INFO:  about to load autoencoder generated embeddings from input file '{MAGENTA}{fqn}{RESET}'", flush=True )
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
      print ( f"SK_SPECTRAL:     INFO:  (embeddings) samples_npy.shape     =  {MIKADO}{samples_npy.shape}{RESET}"      ) 
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
    
    samples = samples_npy.reshape(samples_npy.shape[0], samples_npy.shape[1]*samples_npy.shape[2]*samples_npy.shape[3])
    
    if DEBUG>0:
      print ( f"SK_SPECTRAL:     INFO:  about to flatten channels and r,g,b dimensions"      ) 
      print ( f"SK_SPECTRAL:     INFO:  (flattened) samples.shape          = {MIKADO}{samples.shape}{RESET}"      ) 

  if args.input_mode=='rna': 
    samples = samples_npy
  
    if DEBUG>0:
      print ( f"SK_SPECTRAL:     INFO:  samples.shape          = {MIKADO}{samples.shape}{RESET}"      ) 



  # 2. cluster

  ###############################################################################################################################
  clustering = SpectralClustering( n_clusters=n_clusters, eigen_solver=eigen_solver, affinity=affinity )
  ###############################################################################################################################   

  
  t0 = time()

  clustering.fit( samples )


  if DEBUG>2:
    print( f"SK_SPECTRAL:     INFO:  clustering.labels_ = {MIKADO}{clustering.labels_}{RESET}" )
    

  all_clusters_unique=sorted(set(clustering.labels_))    
  if (DEBUG>0):
    print ( f"SK_SPECTRAL:     INFO:  n_clusters (user specified)        = {MIKADO}{n_clusters}{RESET}" )
    print ( f"SK_SPECTRAL:     INFO:  unique classes represented         = {MIKADO}{all_clusters_unique}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( 0, len(all_clusters_unique) ):
      print ( f"SK_SPECTRAL:     INFO:    count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(clustering.labels_==i).sum()}{RESET}" )


  if (DEBUG>1):
    print ( f"SK_SPECTRAL:      INFO:  true_labels        = \n{MIKADO}{clustering.labels_}{RESET}" )
    print ( f"SK_SPECTRAL:      INFO:  cluster_labels     = \n{MIKADO}{labels}{RESET}" )


  # 3. plot the results as a scattergram
  
  plot( args, embeddings, samples_npy.shape, clustering.labels_, labels,  n_clusters, all_clusters_unique, eigen_solver, affinity )
    
  plt.show()  



# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def plot(args, embeddings, shape, cluster_labels, true_labels, n_clusters, all_clusters_unique, eigen_solver, affinity ):
  
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
  title=f"Unsupervised Spectral Clustering of {N:,} TCGA {args.dataset.upper()} {args.input_mode}s;  X=cluster number (jittered), Y=true subtype"
  subtitle=f"n_clusters={n_clusters};  input dims = {shape[1:]};  autoencoder input used={embeddings};  eigen_solver={eigen_solver};  affinity={affinity}"
  
  plt.title ( title, fontsize=16 )
  plt.text  ( -.2, 0.1, subtitle, ha='left', fontsize=12 )


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

