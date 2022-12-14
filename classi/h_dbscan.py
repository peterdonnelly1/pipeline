import sys
import torch
import datetime
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

def h_dbscan( args, class_names, pct_test, min_cluster_size, super_title, descriptor_clustering  ):
 
  # 1. load and prepare data

  input_mode           = args.input_mode
  algorithm            = 'best'
  metric               = args.metric  
  alpha                = 2.0
  min_cluster_size     = min_cluster_size
  approx_min_span_tree = True
  gen_min_span_tree    = False
  leaf_size            = 200
  p                    = None
  
      
  # 1. load and prepare data

  if args.use_autoencoder_output=='True':
    
    fqn = f"../logs/ae_output_features.pt"
      
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}H_DBSCAN:       INFO:  about to load autoencoder generated embeddings from input file '{MAGENTA}{fqn}{RESET}'", flush=True )
    try:
      dataset  = torch.load( fqn )
      if DEBUG>0:
        print( f"{BRIGHT_GREEN}H_DBSCAN:       INFO:  dataset successfully loaded{RESET}" ) 
    except Exception as e:
      print ( f"{RED}H_DBSCAN:         FATAL:  could not load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
      print ( f"{RED}H_DBSCAN:         FATAL:  the exception was: {CYAN}'{e}'{RESET}" )
      print ( f"{RED}H_DBSCAN:         FATAL:  halting now" )
      sys.exit(0)
  
    samples      = dataset['embeddings'].cpu().detach().numpy().squeeze()                                           # eliminate empty dimensions
    labels       = dataset['labels'    ].cpu().detach().numpy().squeeze()                                           # eliminate empty dimensions
    
    if DEBUG>0:
      print ( f"H_DBSCAN:       INFO:  (embeddings) samples.shape     =  {MIKADO}{samples.shape}{RESET}"      ) 
      print ( f"H_DBSCAN:       INFO:  sanity check: np.sum(samples)  =  {MIKADO}{np.sum(samples):.2f}{RESET}"      ) 
    
    if np.sum(samples)==0.0:
      print ( f"{RED}H_DBSCAN:         FATAL:  all samples are zero vectors - the input file was completely degenerate{RESET}" )
      print ( f"{RED}H_DBSCAN:         FATAL:  not halting, but might as well be{RESET}" )
 
  else:
  
    if input_mode=='image':
  
      sample_file = "../logs/all_images_from_last_run_of_generate.npy" 
      label_file = "../logs/all_image_labels__from_last_run_of_generate.npy"
      
      try:
        samples      =  np.load( sample_file )
      except Exception as e:
        print( f"{RED}H_DBSCAN:       INFO:  could not load file:  {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}H_DBSCAN:       INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
              
      try:
        labels       =  np.load( label_file  )
      except Exception as e:
        print( f"{RED}H_DBSCAN:       INFO:  could not load file: {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}H_DBSCAN:       INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
      
      if DEBUG>0:
        print ( f"H_DBSCAN:       INFO:  input                  = {MIKADO}{input_mode}{RESET}",                flush=True   ) 
        print ( f"H_DBSCAN:       INFO:  about to flatten channels and r,g,b dimensions",                      flush=True   ) 
        print ( f"H_DBSCAN:       INFO:  (flattened) samples.shape          = {MIKADO}{samples.shape}{RESET}", flush=True   ) 
  
    if input_mode=='rna': 
  
      sample_file = "../logs/all_rna_seq_vectors_from_last_run_of_generate.npy" 
      label_file = "../logs/all_rna_seq_vector_labels_from_last_run_of_generate.npy"
      
      try:
        samples      =  np.load( sample_file ).squeeze()
      except Exception as e:
        print( f"{RED}H_DBSCAN:       INFO:  could not load file:  {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}H_DBSCAN:       INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
              
      try:
        labels       =  np.load( label_file  )
      except Exception as e:
        print( f"{RED}H_DBSCAN:       INFO:  could not load file:  {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}H_DBSCAN:       INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
      
      if DEBUG>0:
        print ( f"H_DBSCAN:       INFO:  input                  = {MIKADO}{input_mode}{RESET}",                flush=True   ) 
        print ( f"H_DBSCAN:       INFO:  samples.shape          = {MIKADO}{samples.shape}{RESET}",             flush=True   )



  # 2. cluster
  
  if DEBUG>0:
    print ( f"H_DBSCAN:       INFO:  about to create an {CYAN}HDBSCAN{RESET} clusterer object"      ) 
    
    x_npy = samples
     

  
  ######################################################
  clusterer = hdbscan.HDBSCAN(algorithm=algorithm, alpha=alpha, approx_min_span_tree=approx_min_span_tree, gen_min_span_tree=gen_min_span_tree, leaf_size=leaf_size, metric=metric, min_cluster_size=min_cluster_size, p=p, core_dist_n_jobs=-1 ).fit(x_npy)
  ######################################################
  
  if DEBUG>0:
    print ( f"H_DBSCAN:       INFO:  about to cluster        {CYAN}x_npy{RESET} using {CYAN}clusterer.fit(x_npy){RESET}"     ) 
    print ( f"H_DBSCAN:       INFO:  now finished clustering {CYAN}x_npy{RESET}"                                             ) 

  if DEBUG>2:
    print ( f"H_DBSCAN:       INFO:  clusterer.labels_    = {MIKADO}{clusterer.labels_}{RESET}"                              ) 
  
  if (DEBUG>0):
    all_clusters_unique=sorted(set(clusterer.labels_))
    print ( f"H_DBSCAN:       INFO:  unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( -1, len(all_clusters_unique) ):
      print ( f"H_DBSCAN:       INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(clusterer.labels_==i).sum()}{RESET}" )


  c = clusterer.labels_
  
  if (DEBUG>1):
    print ( f"H_DBSCAN:       INFO:  labels             = {MIKADO}{labels}{RESET}" )
    print ( f"H_DBSCAN:       INFO:  clusterer.labels_  = {MIKADO}{c}{RESET}" )

  if (DEBUG>2):
    print ( f"H_DBSCAN:       INFO:  labels             = {MIKADO}{labels.shape}{RESET}" )
    print ( f"H_DBSCAN:       INFO:  clusterer.labels_  = {MIKADO}{c.shape}{RESET}" )
    
    


  # 3. plot the results as a jittergram
    
  figure_width  = 20
  figure_height = 10
  fig, ax       = plt.subplots( figsize = (figure_width, figure_height) )
  # ~ fig.tight_layout()
  
  # ~ color_palette         = sns.color_palette('bright', 100)
  # ~ cluster_colors        = [color_palette[x] for x in clusterer.labels_]
  # ~ cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]

  if (DEBUG>1):
    print ( f"H_DBSCAN:       INFO:  labels    = \n{MIKADO}{clusterer.labels_}{RESET}" )
  c = clusterer.labels_ + 1
  if (DEBUG>1):
    print ( f"H_DBSCAN:       INFO:  labels+1  = \n{MIKADO}{c}{RESET}" )
  colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
  if (DEBUG>1):
    print ( f"H_DBSCAN:       INFO:  colors    = {MIKADO}{colors}{RESET}" )
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
  
  yy     = [ i for i in range (0, len(class_names) )]
  labels = class_names
  plt.yticks(yy, labels=labels )

  s = ax.scatter( X, Y, s=5, linewidth=0, marker="s", c=c, cmap=cmap, alpha=1.0)
  legend1 = ax.legend(*s.legend_elements(), loc="upper left", title="cluster number")
  ax.add_artist(legend1)

  if (DEBUG>1):
    offset=.5
    for i, label in enumerate( labels ):
      plt.annotate( class_names[label][0], ( X[i]-.25, Y[i]-.5), fontsize=5, color='black' )
  
      if (DEBUG>1):  
        print ( f"i={i:4d} label={MIKADO}{label}{RESET}  class_names[label]={MIKADO}{ class_names[label]:16s}{RESET} class_names[label][0]={MIKADO}{class_names[label][0]}{RESET}" )

  if DEBUG>1:
    print( f"H_DBSCAN:       INFO: X = \n{MIKADO}{X}{RESET}" )
    print( f"H_DBSCAN:       INFO: Y = \n{MIKADO}{Y}{RESET}" )
  
  lim = (x_npy.min(), x_npy.max())
  
  plt.show()
