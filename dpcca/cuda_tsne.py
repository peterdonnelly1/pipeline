import sys
import torch
import random
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors

from sklearn.model_selection import train_test_split

from tsnecuda import TSNE

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


def cuda_tsne( args, pct_test):
    
  n_components     =  2
  n_jobs           = -1                                                                                        # -1 means use all available processors
  verbose          =  2
  learning_rate    = 10
  n_iter           = args.n_iterations
  perplexity       = args.perplexity
  grid_size        = args.supergrid_size
  class_names      = args.class_names
  

  if  grid_size**2 < len(perplexity):
  
    print ( f"CUDA_TSNE:       WARN:  the selected grid size ({MIKADO}{grid_size}x{grid_size}{RESET} isn't large enough to hold the number of plots required for {MIKADO}{len(perplexity)}{RESET} values of perplexity)"        ) 
    grid_size = (int(len(perplexity)**0.5))  if (len(perplexity)**0.5)**2==len(perplexity) else (int(len(perplexity)**0.5)+1)
    print ( f"CUDA_TSNE:       WARN:  grid size has been changed to {MIKADO}{grid_size}x{grid_size}{RESET}" )
  
  nrows        = grid_size
  ncols        = grid_size
  num_subplots = grid_size * grid_size
         
  if DEBUG>0:
    print ( f"CUDA_TSNE:       INFO:  perplexity          = {MIKADO}{perplexity}{RESET}"      ) 
    
      
  # 1. load and prepare data

  if args.use_autoencoder_output=='True':
    
    fqn = f"../logs/ae_output_features.pt"
      
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}CUDA_TSNE:       INFO:  about to load autoencoder generated embeddings from input file '{MAGENTA}{fqn}{RESET}'", flush=True )
    try:
      dataset  = torch.load( fqn )
      if DEBUG>0:
        print( f"{BRIGHT_GREEN}CUDA_TSNE:       INFO:  dataset successfully loaded{RESET}" ) 
    except Exception as e:
      print ( f"{RED}CUDA_TSNE:       INFO:could not load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
      print ( f"{RED}CUDA_TSNE:       INFO:the exception was: {CYAN}'{e}'{RESET}" )
      print ( f"{RED}CUDA_TSNE:       INFO:halting now" )
      sys.exit(0)
  
    samples  = dataset['embeddings'].cpu().numpy().squeeze()                                           # eliminate empty dimensions
    labels       = dataset['labels'    ].cpu().numpy().squeeze()                                           # eliminate empty dimensions
    
    if DEBUG>0:
      print ( f"CUDA_TSNE:       INFO:  (embeddings) samples.shape     =  {MIKADO}{samples.shape}{RESET}"      ) 
      # ~ print ( f"CUDA_TSNE:       INFO:  sanity check: np.sum(samples)  =  {MIKADO}{np.sum(samples):.2f}{RESET}"      ) 
    
    if np.sum(samples)==0.0:
      print ( f"{RED}CUDA_TSNE:       INFO:all samples are zero vectors - the input file was completely degenerate{RESET}" )
      print ( f"{RED}CUDA_TSNE:       INFO:not halting, but might as well be{RESET}" )
 
  else:
    
    sample_file = "../logs/images_new.npy" 
    label_file = "../logs/img_labels_new.npy"
    
    samples  =  np.load( sample_file )
    labels       =  np.load( label_file  )
  


  # 2. cluster & plot

  figure_width  = 20
  figure_height = 10
  
  figsize=( figure_width, figure_height )
  
  # ~ fig, ax = plt.subplots( figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=True )
  fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=True )
  
  

  for r in range(0, nrows):
  
    for c in range(0, ncols ):

      subplot_index = r*nrows+c  
  
      if DEBUG>0:
        print( f"CUDA_TSNE:       INFO:  about to configure {CYAN}cuda TSNE {RESET}object with: n_components={MIKADO}{n_components}{RESET}, perplexity={MIKADO}{perplexity[subplot_index]}{RESET}", flush=True )
    
  
      if DEBUG>0:
        print ( f"CUDA_TSNE:       INFO:  subplot_index         = {MIKADO}{subplot_index}{RESET}"      ) 
  
        
      embedding_train = TSNE(                                                                                             # create and configure TSNE object
          n_components = n_components,
          n_iter       = n_iter,
          perplexity   = perplexity[subplot_index],
        learning_rate=learning_rate
          # ~ verbose      = verbose
      ).fit_transform( samples )
            
      
      if DEBUG>0:
        print( f"CUDA_TSNE:       INFO:  finished {CYAN}tsne.fit{RESET}", flush=True )
        print( f"CUDA_TSNE:       INFO:  {CYAN}embedding_train.shape{RESET} = {MIKADO}{embedding_train.shape}{RESET}", flush=True )
        print( f"CUDA_TSNE:       INFO:  {CYAN}labels.shape{RESET}          = {MIKADO}{labels.shape}{RESET}",         flush=True )
        # ~ print( f"CUDA_TSNE:       INFO:  {CYAN}embedding_train{RESET}       =\n{MIKADO}{embedding_train}{RESET}",     flush=True )
    
    
      if (DEBUG>0):
        all_clusters_unique=sorted(set(labels))
        print ( f"CUDA_TSNE:       INFO:  unique classes represented in samples (truth labels) = {MIKADO}{all_clusters_unique}{RESET}" )
      
      if (DEBUG>0):
        for i in range ( 0, len(all_clusters_unique) ):
          print ( f"CUDA_TSNE:       INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(labels==i).sum()}{RESET}" )
      
    
  
      # 3. plot the results as a scattergram

      if DEBUG>2:
        print( f"CUDA_TSNE:       INFO:  r             {BLEU}{r}{RESET}", flush=True )
        print( f"CUDA_TSNE:       INFO:  c             {BLEU}{c}{RESET}", flush=True )
        print( f"CUDA_TSNE:       INFO:  num_subplots  {BLEU}{num_subplots}{RESET}", flush=True )
        print( f"CUDA_TSNE:       INFO:  subplot_index {BLEU}{subplot_index}{RESET}", flush=True )
              
      N=labels.shape[0]
      title=f"unsupervised clustering using cuda t-sne \ncancer type={args.dataset}, N={N:,}, n_components={n_components}, perplexity={perplexity[subplot_index]}"
  
      plot( num_subplots, subplot_index, embedding_train, labels, class_names, axes[r,c], title  )
    
  
  
  plt.show()




# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def plot( num_subplots, subplot_index, x, y, class_names, ax, title, draw_legend=True, draw_centers=False, draw_cluster_labels=False, colors=None, legend_kwargs=None, label_order=None, **kwargs ):


    ax.set_title( title, fontsize="10" )

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
      print ( f"CUDA_TSNE:       INFO: plot()  class_names           = {BITTER_SWEET}{class_names}{RESET}" )
      print ( f"CUDA_TSNE:       INFO: plot()  classes               = {BITTER_SWEET}{classes}{RESET}" )
      print ( f"CUDA_TSNE:       INFO: plot()  colors                = {BITTER_SWEET}{colors}{RESET}" )
      print ( f"CUDA_TSNE:       INFO: plot()  colors.get            = {BITTER_SWEET}{colors.get}{RESET}" )
      print ( f"CUDA_TSNE:       INFO: plot()  point_colors          = {BITTER_SWEET}{point_colors}{RESET}" )

    # ~ lim = ( x.min(), x.max() )
    
    if (DEBUG>2):
      print ( f"CUDA_TSNE:       INFO: plot()  x[:, 0].min()               = {BITTER_SWEET}{x[:, 0].min()}{RESET}" )
      print ( f"CUDA_TSNE:       INFO: plot()  x[:, 0].max()               = {BITTER_SWEET}{x[:, 0].max()}{RESET}" )
      print ( f"CUDA_TSNE:       INFO: plot()  x[:, 1].min()               = {BITTER_SWEET}{x[:, 1].min()}{RESET}" )
      print ( f"CUDA_TSNE:       INFO: plot()  x[:, 1].max()               = {BITTER_SWEET}{x[:, 1].max()}{RESET}" )      

    x1 = x[:, 0]
    x2 = x[:, 1]
    std_devs=2
    ax.set_xlim( [ np.median(x1)-std_devs*np.std(x1), np.median(x1)+std_devs*np.std(x1) ] )
    ax.set_ylim( [ np.median(x2)-std_devs*np.std(x2), np.median(x2)+std_devs*np.std(x2) ] )
    
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params) 
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True)
       
    ax.scatter( x1, x2, c=point_colors, s=1, marker="s")

    
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


  
