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

# ~ from otsne_utils import plot
# ~ from otsne_utils import MACOSKO_COLORS
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

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

def sk_tsne( args, class_names, pct_test):
    
  n_components = 2
  n_iter       = args.n_iterations
  perplexity   = args.perplexity[0]                                                                        # only one value of perplexity allowed for sk_tsne runs (cf. cuda_tsnet, which provides for multiple values
  metric       = args.metric
  n_jobs       = -1                                                                                        # -1 means use all available processors
  verbose      =  2  
  
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
    
    samples = samples_npy
    
    if DEBUG>0:
      print ( f"SK_SPECTRAL:     INFO:  about to flatten channels and r,g,b dimensions"      ) 
      print ( f"SK_SPECTRAL:     INFO:  (flattened) samples.shape          = {MIKADO}{samples.shape}{RESET}"      ) 

  if args.input_mode=='rna': 
    samples = samples_npy
  
    if DEBUG>0:
      print ( f"SK_SPECTRAL:     INFO:  samples.shape          = {MIKADO}{samples.shape}{RESET}"      ) 



  # 2. cluster

  
  if DEBUG>0:
    print( f"SK_TSNE:         INFO:  about to configure {CYAN}SKLEARN TSNE {RESET}object with: metric='{CYAN}{metric}{RESET}', n_iter={MIKADO}{n_iter}{RESET}, n_components={MIKADO}{n_components}{RESET}, perplexity={MIKADO}{perplexity}{RESET}, n_jobs={MIKADO}{n_jobs}{RESET}", flush=True )

  if DEBUG>0:
    print( f"SK_TSNE:        INFO:  {CYAN}type(perplexity){RESET} ={MIKADO}{type(perplexity)}{RESET}", flush=True )
    print( f"SK_TSNE:        INFO:  {CYAN}type(n_samples) {RESET} ={MIKADO}{type(perplexity)}{RESET}", flush=True )

    
  embedding_train = TSNE(                                                                                             # create and configure TSNE object
      n_components = n_components,
      n_iter       = n_iter,
      perplexity   = perplexity,
      metric       = metric,
      n_jobs       = n_jobs,
      verbose      = verbose,
  ).fit_transform( samples )
    
  if DEBUG>0:
    print( f"SK_TSNE:        INFO:  finished {CYAN}tsne.fit{RESET}", flush=True )
    print( f"SK_TSNE:        INFO:  {CYAN}embedding_train.shape{RESET} ={MIKADO}{embedding_train.shape}{RESET}", flush=True )
    print( f"SK_TSNE:        INFO:  {CYAN}labels.shape{RESET}         ={MIKADO}{labels.shape}{RESET}",         flush=True )
    # ~ print( f"SK_TSNE:        INFO:  {CYAN}embedding_train{RESET}       =\n{MIKADO}{embedding_train}{RESET}",     flush=True )


  if (DEBUG>0):
    all_clusters_unique=sorted(set(labels))
    print ( f"SK_TSNE:        INFO:   unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( 0, len(all_clusters_unique) ):
      print ( f"SK_TSNE:        INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(labels==i).sum()}{RESET}" )
  

  
  # 3. plot the results as a scattergram
  
  figure_width  = 20
  figure_height = 10
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

  if (DEBUG>2):
    np.set_printoptions(formatter={'int': lambda x:   "{:>2d}".format(x)})
    print ( f"SK_TSNE:        INFO:  labels    = {MIKADO}{labels}{RESET}" )
  c = labels
  if (DEBUG>2):
    print ( f"SK_TSNE:        INFO:  labels+1  = {MIKADO}{c}{RESET}" )
  # ~ colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
  colors  = MACOSKO_COLORS
  if (DEBUG>2):
    print ( f"SK_TSNE:        INFO:  colors               = {MIKADO}{colors}{RESET}" )
    print ( f"SK_TSNE:        INFO:  np.unique(labels)    = {MIKADO}{np.unique(labels)}{RESET}" )

  if (DEBUG>2):
    print ( f"SK_TSNE:        INFO:  labels               = {MIKADO}{labels}{RESET}" )
    
  # ~ cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, c.max()+3), colors )

  N=labels.shape[0]
  title=f"Unsupervised Clustering using sklearn T-SNE \n(cancer type={args.dataset}, N={N:,}, n_iter={n_iter:,}, n_components={n_components}, perplexity={perplexity}, metric={metric})"

  # ~ plot( embedding_train, labels, colors=MACOSKO_COLORS )
  plot( embedding_train, labels, class_names, ax=ax, title=title  )
  plt.show()



# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def plot(
    x,
    y,
    class_names,
    ax=None,
    title=None,
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
      print ( f"SK_TSNE:        INFO: plot()  class_names           = {BITTER_SWEET}{class_names}{RESET}" )
      print ( f"SK_TSNE:        INFO: plot()  classes               = {BITTER_SWEET}{classes}{RESET}" )
      print ( f"SK_TSNE:        INFO: plot()  colors                = {BITTER_SWEET}{colors}{RESET}" )
      print ( f"SK_TSNE:        INFO: plot()  colors.get            = {BITTER_SWEET}{colors.get}{RESET}" )
      print ( f"SK_TSNE:        INFO: plot()  point_colors          = {BITTER_SWEET}{point_colors}{RESET}" )

    # ~ lim = ( x.min(), x.max() )
    
    if (DEBUG>2):
      print ( f"SK_TSNE:        INFO: plot()  x[:, 0].min()               = {BITTER_SWEET}{x[:, 0].min()}{RESET}" )
      print ( f"SK_TSNE:        INFO: plot()  x[:, 0].max()               = {BITTER_SWEET}{x[:, 0].max()}{RESET}" )
      print ( f"SK_TSNE:        INFO: plot()  x[:, 1].min()               = {BITTER_SWEET}{x[:, 1].min()}{RESET}" )
      print ( f"SK_TSNE:        INFO: plot()  x[:, 1].max()               = {BITTER_SWEET}{x[:, 1].max()}{RESET}" )      

    x1 = x[:, 0]
    x2 = x[:, 1]
    std_devs=2
    ax.set_xlim( [ np.median(x1)-std_devs*np.std(x1), np.median(x1)+std_devs*np.std(x1) ] )
    ax.set_ylim( [ np.median(x2)-std_devs*np.std(x2), np.median(x2)+std_devs*np.std(x2) ] )
    
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params) 
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True) 
    ax.scatter( x1, x2, c=point_colors, s=4, marker="s")
  
    
    # ~ offset=.5
    # ~ for i, label in enumerate( y ):
      
      # ~ ax.annotate( class_names[label][0:1], ( x1[i]-.035, x2[i]-.02), fontsize=5, color='white' )
  
      # ~ if (DEBUG>0):  
        # ~ print ( f"i={i:4d} label={MIKADO}{label}{RESET}  class_names[label]={MIKADO}{ class_names[label]:16s}{RESET} class_names[label][0]={MIKADO}{class_names[label][0]}{RESET}" )
      

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=100, marker="x", alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)


  
