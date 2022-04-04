import sys
import torch
import argparse
import matplotlib.colors
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import random

import hdbscan

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

def h_dbscan( args, pct_test, min_cluster_size ):
 
  # 1. load and prepare data

  algorithm            = 'best'
  metric               = args.metric  
  alpha                = 2.0
  min_cluster_size     = min_cluster_size
  approx_min_span_tree = True
  gen_min_span_tree    = False
  leaf_size            = 200
  p                    = None
  
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
    
    if args.input_mode=='image': 
      x_npy = embeddings

    if args.input_mode=='rna': 
      x_npy = embeddings
    
    if DEBUG>0:
      print ( f"HDBSCAN:         INFO:  x_npy.shape          = {MIKADO}{x_npy.shape}{RESET}"      ) 
      print ( f"HDBSCAN:         INFO:  about to convert to pandas dataframe"                     )  
 
  else:
    
    image_file = "../logs/images_new.npy" 
    label_file = "../logs/img_labels_new.npy"
    
    embeddings = np.load( image_file )
    labels     = np.load( label_file )
  
    if DEBUG>0:
      print( f"\n{GREY_BACKGROUND}HDBSCAN:  INFO: {WHITE}{CHARTREUSE}HDBSCAN clustering{WHITE}: samples_file={MAGENTA}{image_file}{WHITE}, labels_file={MAGENTA}{label_file}{WHITE}, alpha={MIKADO}{alpha}{WHITE}, metric={CYAN}{args.metric}{WHITE}, min_cluster_size={MIKADO}{min_cluster_size}{WHITE}, leaf_size={MIKADO}{leaf_size}                                                          {RESET}" )  
  
    x_npy = embeddings
    
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
    print ( f"HDBSCAN:         INFO:  about to create an {CYAN}HDBSCAN{RESET} clusterer object"      ) 
    

  
  ######################################################
  clusterer = hdbscan.HDBSCAN(algorithm=algorithm, alpha=alpha, approx_min_span_tree=approx_min_span_tree, gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, metric=metric, min_cluster_size=min_cluster_size, p=p, core_dist_n_jobs=-1 ).fit(x_npy)
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


  c = clusterer.labels_
  
  if (DEBUG>1):
    print ( f"HDBSCAN:         INFO:  labels             = {MIKADO}{labels}{RESET}" )
    print ( f"HDBSCAN:         INFO:  clusterer.labels_  = {MIKADO}{c}{RESET}" )

  if (DEBUG>2):
    print ( f"HDBSCAN:         INFO:  labels             = {MIKADO}{labels.shape}{RESET}" )
    print ( f"HDBSCAN:         INFO:  clusterer.labels_  = {MIKADO}{c.shape}{RESET}" )
    
    


  # 3. plot the results as a jittergram
    
  figure_width  = 20
  figure_height = 10
  fig, ax       = plt.subplots( figsize = (figure_width, figure_height) )
  # ~ fig.tight_layout()
  
  # ~ color_palette         = sns.color_palette('bright', 100)
  # ~ cluster_colors        = [color_palette[x] for x in clusterer.labels_]
  # ~ cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]

  if (DEBUG>1):
    print ( f"HDBSCAN:         INFO:  labels    = \n{MIKADO}{clusterer.labels_}{RESET}" )
  c = clusterer.labels_ + 1
  if (DEBUG>1):
    print ( f"HDBSCAN:         INFO:  labels+1  = \n{MIKADO}{c}{RESET}" )
  colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
  if (DEBUG>1):
    print ( f"HDBSCAN:         INFO:  colors    = {MIKADO}{colors}{RESET}" )
  cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, c.max()+3), colors )
  
  X = c
  Y = labels
  
  X_jitter = np.zeros_like(X)
  X_jitter = [ random.uniform( -0.45, 0.45 ) for i in range( 0, len(X) ) ]
 
  X = X + X_jitter
  
  N=x_npy.shape[0]
  title=f"Unsupervised Clustering using HDBSCAN ('Hierarchical Density Based Spatial Clustering of Applications with Noise')\n(cancer type={args.dataset}, N={N:,}, X=cluster number (jittered), Y=true subtype, min_cluster_size={min_cluster_size}, letter=true subtype)"
  
  plt.title( title,fontsize=15 )

  xx     = np.arange(0, len(all_clusters_unique), step=1)
  labels = all_clusters_unique
  plt.xticks( xx, labels=labels )
  
  yy     = [ i for i in range (0, len(args.class_names) )]
  labels = args.class_names
  plt.yticks(yy, labels=labels )

  s = ax.scatter( X, Y, s=5, linewidth=0, marker="s", c=c, cmap=cmap, alpha=1.0)
  legend1 = ax.legend(*s.legend_elements(), loc="upper left", title="cluster number")
  ax.add_artist(legend1)

  if (DEBUG>1):
    offset=.5
    for i, label in enumerate( labels ):
      plt.annotate( args.class_names[label][0], ( X[i]-.25, Y[i]-.5), fontsize=5, color='black' )
  
      if (DEBUG>1):  
        print ( f"i={i:4d} label={MIKADO}{label}{RESET}  args.class_names[label]={MIKADO}{ args.class_names[label]:16s}{RESET} args.class_names[label][0]={MIKADO}{args.class_names[label][0]}{RESET}" )

  if DEBUG>1:
    print( f"HDBSCAN:        INFO: X = \n{MIKADO}{X}{RESET}" )
    print( f"HDBSCAN:        INFO: Y = \n{MIKADO}{Y}{RESET}" )
  
  lim = (x_npy.min(), x_npy.max())
  
  plt.show()