
import sys
import torch
import argparse
import matplotlib.colors
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

import hdbscan
from   IPython.display import display

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

def h_dbscan( args, pct_test):

 
  # 1. load and prepare data
  
  if args.use_autoencoder_output=='True':
    
    fqn = f"../logs/ae_output_features.pt"
      
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}HDBSCAN:         INFO:  about to load autoencoder generated feature file from input file '{MAGENTA}{fqn}{RESET}'", flush=True )
    try:
      dataset  = torch.load( fqn )
      if DEBUG>0:
        print( f"{BRIGHT_GREEN}HDBSCAN:         INFO:  dataset successfully loaded{RESET}" ) 
    except Exception as e:
      print ( f"{RED}HDBSCAN:         ERROR:  could not load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
      print ( f"{RED}HDBSCAN:         ERROR:  the exception was: {CYAN}'{e}'{RESET}" )
      print ( f"{RED}HDBSCAN:         ERROR:  halting now" )
      sys.exit(0)
  
    embeddings  = dataset['embeddings'].cpu().numpy().squeeze()
    labels      = dataset['labels']    .cpu().numpy().squeeze()
    
    if DEBUG>0:
      print ( f"HDBSCAN:         INFO:  np.sum(embeddings)      =  {MIKADO}{np.sum(embeddings)}{RESET}"      ) 
    
    if np.sum(embeddings)==0.0:
      print ( f"{RED}HDBSCAN:         ERROR:  all embeddings are zero vectors - the input file was completely degenerate{RESET}" )
      print ( f"{RED}HDBSCAN:         ERROR:  not halting, but might as well be{RESET}" )
    
    if DEBUG>0:
      print ( f"HDBSCAN:         INFO:  about to flatten channels and r,g,b dimensions"      ) 
    
    x_npy = embeddings.reshape(embeddings.shape[0], embeddings.shape[1]*embeddings.shape[2]*embeddings.shape[3])
    
    if DEBUG>0:
      print ( f"HDBSCAN:         INFO:  x_npy.shape          = {MIKADO}{x_npy.shape}{RESET}"      ) 
      print ( f"HDBSCAN:         INFO:  about to convert to pandas dataframe"                     )  
 
  else:
    
    image_file = "../logs/images_new.npy" 
    label_file = "../logs/img_labels_new.npy"
    
    embeddings = np.load( image_file )
    labels     = np.load( label_file )
  
    if DEBUG>0:
      print( f"\n{GREY_BACKGROUND}HDBSCAN:  INFO: {WHITE}{CHARTREUSE}HDBSCAN clustering{WHITE}: samples_file={MAGENTA}{image_file}{WHITE}, labels_file={MAGENTA}{label_file}{WHITE}, pct_test={MIKADO}{pct_test}{WHITE}, metric={CYAN}{args.metric}{WHITE}, iterations={MIKADO}{args.n_epochs}{WHITE}, perplexity={MIKADO}{args.perplexity}{WHITE}, momentum={MIKADO}{args.momentum}                                                                                                                        {RESET}" )  
  
    x_npy = embeddings.reshape( embeddings.shape[0], embeddings.shape[1]*embeddings.shape[2]*embeddings.shape[3] )
    
    if DEBUG>0:
      print( f"HDBSCAN:        INFO:  image file shape {MIKADO}{x_npy.shape}{RESET}" )
      print( f"HDBSCAN:        INFO:  label file shape {MIKADO}{labels.shape}{RESET}" )  
      print( f"HDBSCAN:        INFO:  image file {CYAN}{image_file}{RESET} \r\033[60Ccontains {MIKADO}{x_npy.shape[0]}{RESET} samples each with {MIKADO}{x_npy.shape[1]}{RESET} features", flush=True)
      print( f"HDBSCAN:        INFO:  label file {CYAN}{label_file}{RESET} \r\033[60Ccontains {MIKADO}{x_npy.shape[0]}{RESET} labels", flush=True)
  
    if DEBUG>0:
      print( f"HDBSCAN:        INFO:  x_npy.shape     = {MIKADO}{x_npy.shape}{RESET}" )  
      # ~ print( f"HDBSCAN:        INFO:  x_npy[0].shape  = {MIKADO}{x_npy[0].shape}{RESET}" )  
  
    if DEBUG>2:
      print( f"HDBSCAN:        INFO:  embeddings[0] = \n{MIKADO}{embeddings[0,2,40:80,90:100]}{RESET}" )  
      print( f"HDBSCAN:        INFO:  x_npy [0]  =  {MIKADO}{x_npy[0,1000:1100]}{RESET}" )  



  # 2. cluster
  
  if DEBUG>0:
    print ( f"HDBSCAN:        INFO:  about to create an {CYAN}HDBSCAN{RESET} clusterer object"      ) 
    
  algorithm            = 'best'
  metric               = args.metric  
  alpha                = 1.4142
  min_cluster_size     = 50
  approx_min_span_tree = True
  gen_min_span_tree    = False
  leaf_size            = 40
  p                    = None
  
  ######################################################
  # ~ clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=3.0, approx_min_span_tree=True, gen_min_span_tree=False, leaf_size=10000, metric='braycurtis', min_cluster_size=3, min_embeddings=None, p=None).fit(x_npy)
  # ~ clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=3.0, approx_min_span_tree=True, gen_min_span_tree=False, leaf_size=10000, metric='canberra', min_cluster_size=3, min_embeddings=None, p=None).fit(x_npy)
  clusterer = hdbscan.HDBSCAN(algorithm=algorithm, alpha=alpha, approx_min_span_tree=approx_min_span_tree, gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, metric=metric, min_cluster_size=min_cluster_size, p=p).fit(x_npy)
  # ~ clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(data)
  ######################################################
  
  if DEBUG>0:
    print ( f"HDBSCAN:         INFO:  about to cluster        {CYAN}x_npy{RESET} using {CYAN}clusterer.fit(x_npy){RESET}"     ) 
    print ( f"HDBSCAN:         INFO:  now finished clustering {CYAN}x_npy{RESET}"                                             ) 

  if DEBUG>2:
    print ( f"HDBSCAN:         INFO:  clusterer.labels_    = {MIKADO}{clusterer.labels_}{RESET}"                              ) 
  
  if (DEBUG>0):
    all_clusters_unique=sorted(set(clusterer.labels_))
    print ( f"HDBSCAN:         INFO:  unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( -1, len(all_clusters_unique) ):
      print ( f"HDBSCAN:         INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(clusterer.labels_==i).sum()}{RESET}" )

  if (DEBUG>0):
    all_clusters_unique=sorted(set(clusterer.labels_))
    print ( f"HDBSCAN:         INFO:  unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( -1, len(all_clusters_unique) ):
      print ( f"HDBSCAN:         INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(clusterer.labels_==i).sum()}{RESET}" )
  

  
  # 3. plot the results as a scattergram
    
  figure_width  = 20
  figure_height = 10
  fig, ax = plt.subplots( figsize = (figure_width, figure_height) )
  fig.tight_layout()
  
  # ~ color_palette         = sns.color_palette('bright', 100)
  # ~ cluster_colors        = [color_palette[x] for x in clusterer.labels_]
  # ~ cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]

  if (DEBUG>1):
    print ( f"HDBSCAN:         INFO:  labels    = {MIKADO}{clusterer.labels_}{RESET}" )
  c = clusterer.labels_ + 1
  if (DEBUG>1):
    print ( f"HDBSCAN:         INFO:  labels+1  = {MIKADO}{c}{RESET}" )
  colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
  if (DEBUG>1):
    print ( f"HDBSCAN:         INFO:  colors    = {MIKADO}{colors}{RESET}" )
  cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, c.max()+3), colors )
  
  X = x_npy[:,0]
  Y = x_npy[:,1]
  
  N=x_npy.shape[0]
  title=f"Hierarchical Unsupervised clustering using Density Based Spatial Clustering of Applications with Noise (HDBSCAN)\n(cancer type={args.dataset}, N={N}, colour=cluster, letter=true subtype)"
  
  plt.title( title,fontsize=15)
  s = ax.scatter( X, Y, s=20, linewidth=0, marker="s", c=c, cmap=cmap, alpha=1.0)
  legend1 = ax.legend(*s.legend_elements(), loc="upper left", title="cluster number")
  ax.add_artist(legend1)

  offset=.5
  for i, label in enumerate( labels ):
    plt.annotate( args.class_names[label][0], ( X[i]-.035, Y[i]-.06), fontsize=4, color='black' )

    if (DEBUG>1):  
      print ( f"i={i:4d} label={MIKADO}{label}{RESET}  args.class_names[label]={MIKADO}{ args.class_names[label]:16s}{RESET} args.class_names[label][0]={MIKADO}{args.class_names[label][0]}{RESET}" )
    
    
  # ~ ax.set_xlim(( -1, 1 ))
  # ~ ax.set_ylim(( -1, 1 ))
  
  lim = (x_npy.min()-12, x_npy.max()+12)
  
  plt.show()



  
