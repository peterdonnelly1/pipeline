import sys
import torch
import random
import datetime
from   cycler import cycler

import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.colors

import pandas as pd
from sklearn.datasets        import load_digits
from sklearn.preprocessing   import StandardScaler

from sklearn.model_selection import train_test_split

from tsnecuda import TSNE

from constants  import *


DEBUG   = 1

np.set_printoptions(edgeitems=5)
np.set_printoptions(linewidth=20)


def cuda_tsne( args, class_names, pct_test, super_title, descriptor_clustering ):
  
  input_mode        = args.input_mode
  n_iter            = args.n_iterations
  perplexity        = args.perplexity
  grid_size         = args.supergrid_size
  render_clustering = args.render_clustering                                                               # 'True'   or 'False'. if 'True', show plots on terminal (they are always be saved to logs)
  
  n_components      =  2
  n_jobs            = -1                                                                                   # -1 means use all available processors
  verbose           =  2
  # ~ learning_rate     = args.tsne_learning_rate
  learning_rate     = 10.


  if DEBUG>0:
    print ( f"CUDA_TSNE:      INFO:  perplexity                  = {MIKADO}{perplexity}{RESET}"             ) 
    print ( f"CUDA_TSNE:      INFO:  render_clustering           = {MIKADO}{render_clustering}{RESET}"      ) 
    
      
  # 1. load and prepare data

  if args.use_autoencoder_output=='True':
    
    fqn = f"../logs/ae_output_features.pt"
      
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}CUDA_TSNE:      INFO:  about to load autoencoder generated embeddings from input file '{MAGENTA}{fqn}{RESET}'", flush=True )
    try:
      dataset  = torch.load( fqn )
      if DEBUG>0:
        print( f"{BRIGHT_GREEN}CUDA_TSNE:      INFO:  dataset successfully loaded{RESET}" ) 
    except Exception as e:
      print ( f"{RED}CUDA_TSNE:        FATAL:  could not load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
      print ( f"{RED}CUDA_TSNE:        FATAL:  the exception was: {CYAN}'{e}'{RESET}" )
      print ( f"{RED}CUDA_TSNE:        FATAL:  halting now" )
      sys.exit(0)
  
    samples      = dataset['embeddings'].cpu().detach().numpy().squeeze()                                           # eliminate empty dimensions
    labels       = dataset['labels'    ].cpu().detach().numpy().squeeze()                                           # eliminate empty dimensions
    
    if DEBUG>0:
      print ( f"CUDA_TSNE:      INFO:  (embeddings) samples.shape     =  {MIKADO}{samples.shape}{RESET}"      ) 
      print ( f"CUDA_TSNE:      INFO:  sanity check: np.sum(samples)  =  {MIKADO}{np.sum(samples):.2f}{RESET}"      ) 
    
    if np.sum(samples)==0.0:
      print ( f"{RED}CUDA_TSNE:        FATAL:  all samples are zero vectors - the input file was completely degenerate{RESET}" )
      print ( f"{RED}CUDA_TSNE:        FATAL:  not halting, but might as well be{RESET}" )
 
  else:
  
    if input_mode=='image':
  
      sample_file = "../logs/all_images_from_last_run_of_generate.npy" 
      label_file = "../logs/all_image_labels__from_last_run_of_generate.npy"
      
      try:
        samples      =  np.load( sample_file )
      except Exception as e:
        print( f"{RED}CUDA_TSNE:      INFO:  could not load file:  {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}CUDA_TSNE:      INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
              
      try:
        labels       =  np.load( label_file  )
      except Exception as e:
        print( f"{RED}CUDA_TSNE:      INFO:  could not load file: {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}CUDA_TSNE:      INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
      
      if DEBUG>0:
        print ( f"CUDA_TSNE:      INFO:  input                  = {MIKADO}{input_mode}{RESET}",                flush=True   ) 
        print ( f"CUDA_TSNE:      INFO:  about to flatten channels and r,g,b dimensions",                      flush=True   ) 
        print ( f"CUDA_TSNE:      INFO:  (flattened) samples.shape          = {MIKADO}{samples.shape}{RESET}", flush=True   ) 
  
    if input_mode=='rna': 
  
      sample_file = "../logs/all_rna_seq_vectors_from_last_run_of_generate.npy" 
      label_file = "../logs/all_rna_seq_vector_labels_from_last_run_of_generate.npy"
      
      try:
        samples      =  np.load( sample_file ).squeeze()
      except Exception as e:
        print( f"{RED}CUDA_TSNE:      INFO:  could not load file:  {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}CUDA_TSNE:      INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
              
      try:
        labels       =  np.load( label_file  )
      except Exception as e:
        print( f"{RED}CUDA_TSNE:      INFO:  could not load file:  {CYAN}{sample_file}{RESET}", flush=True)
        print( f"{RED}CUDA_TSNE:      INFO:  can't continue --- halting{RESET}",         flush=True)
        time.sleep(4)
        sys.exit(0)
      
      if DEBUG>0:
        print ( f"CUDA_TSNE:      INFO:  input                  = {MIKADO}{input_mode}{RESET}",                flush=True   ) 
        print ( f"CUDA_TSNE:      INFO:  samples.shape          = {MIKADO}{samples.shape}{RESET}",             flush=True   ) 





  # 2. cluster & plot

  figure_width  = 20
  figure_height = 10
  
  figsize = ( figure_width, figure_height )
  
  
  
  if len( perplexity ) != 1:

    if  grid_size**2 < len(perplexity):
    
      print ( f"{ORANGE}CUDA_TSNE:      WARN:  the selected grid size ({MIKADO}{grid_size}x{grid_size}{RESET}{ORANGE} isn't large enough to hold the number of plots required for {MIKADO}{len(perplexity)}{RESET}{ORANGE} values of perplexity)"        ) 
      grid_size = ( int(len(perplexity)**0.5))  if  int(len(perplexity)**0.5)**2==len(perplexity) else (int(len(perplexity)**0.5)+1)
      print ( f"{ORANGE}CUDA_TSNE:      WARN:  grid size has been changed to {MIKADO}{grid_size}x{grid_size}{RESET}{ORANGE}{RESET}" )
      
    nrows        = grid_size
    ncols        = grid_size
    num_subplots = grid_size * grid_size
    
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=True )
  
    if len( perplexity ) <= 4:
      title_font_size = 12
      labelspacing    = 0.4
      handletextpad   = 0.2
      marker_size     = 1
      ms              = 10 
    elif len( perplexity ) <= 9:
      title_font_size = 10
      labelspacing    = 0.2
      handletextpad   = 0.2
      marker_size     = 1
      ms              = 7
    elif len( perplexity ) <= 16:
      title_font_size = 9
      labelspacing    = 0.2
      handletextpad   = 0.2
      marker_size     = 1
      ms              = 9 
    elif len( perplexity ) <= 25:
      title_font_size = 8
      labelspacing    = 0.2
      handletextpad   = 0.2
      marker_size     = 1
      ms              = 7 
    elif len( perplexity ) <= 49:
      title_font_size = 6
      labelspacing    = 0.2
      handletextpad   = 0.2
      marker_size     = 1
      ms              = 6 
    else:
      title_font_size = 7
      labelspacing    = 0.2
      handletextpad   = 0.2      
      marker_size     = 1
      ms              = 6 
        
    
    # remove borders from all subplots (otherwise any empty subplots will have a black border)
    for r in range(0, nrows):
    
      for c in range(0, ncols ):
              
        axes[r,c].spines["top"]   .set_visible(False)                                                           # 
        axes[r,c].spines["right"] .set_visible(False)
        axes[r,c].spines["left"]  .set_visible(False)
        axes[r,c].spines["bottom"].set_visible(False)   
        
  
    for r in range(0, nrows):
    
      for c in range(0, ncols ):
  
        subplot_index = r*nrows+c
        if  subplot_index >= len(perplexity):
          break
    
        if DEBUG>0:
          print( f"CUDA_TSNE:      INFO:  about to configure {CYAN}cuda TSNE {RESET}object with: n_components={MIKADO}{n_components}{RESET} learning_rate={MIKADO}{learning_rate}{RESET} n_iter={MIKADO}{n_iter:,}{RESET} perplexity={MIKADO}{perplexity[subplot_index]}{RESET}", flush=True )
    
        if DEBUG>0:
          print ( f"CUDA_TSNE:      INFO:  subplot_index               = {MIKADO}{subplot_index}{RESET}"      ) 
    
          
        embedding_train = TSNE(                                                                            # create and configure TSNE object
            n_components = n_components,
            n_iter       = n_iter,
            perplexity   = perplexity[subplot_index],
          learning_rate  = learning_rate
            # ~ verbose      = verbose
        ).fit_transform( samples )
              
        
        if DEBUG>0:
          print( f"CUDA_TSNE:      INFO:  finished {CYAN}tsne.fit{RESET}", flush=True )
          print( f"CUDA_TSNE:      INFO:  {CYAN}embedding_train.shape{RESET} = {MIKADO}{embedding_train.shape}{RESET}", flush=True )
          print( f"CUDA_TSNE:      INFO:  {CYAN}labels.shape{RESET}          = {MIKADO}{labels.shape}{RESET}",         flush=True )
          # ~ print( f"CUDA_TSNE:      INFO:  {CYAN}embedding_train{RESET}       =\n{MIKADO}{embedding_train}{RESET}",     flush=True )
      
      
        if (DEBUG>0):
          all_clusters_unique=sorted(set(labels))
          print ( f"CUDA_TSNE:      INFO:  unique classes represented in samples (truth labels) = {MIKADO}{all_clusters_unique}{RESET}" )
        
        if (DEBUG>0):
          for i in range ( 0, len(all_clusters_unique) ):
            print ( f"CUDA_TSNE:      INFO:  number of examples of class label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(labels==i).sum()}{RESET}" )
        
      
    
        # 3. plot the results as a scattergram
  
        if DEBUG>2:
          print( f"CUDA_TSNE:      INFO:  r             {BLEU}{r}{RESET}", flush=True )
          print( f"CUDA_TSNE:      INFO:  c             {BLEU}{c}{RESET}", flush=True )
          print( f"CUDA_TSNE:      INFO:  num_subplots  {BLEU}{num_subplots}{RESET}", flush=True )
          print( f"CUDA_TSNE:      INFO:  subplot_index {BLEU}{subplot_index}{RESET}", flush=True )
                
        fig.suptitle( f"(cuda) t-sne Clustering   {super_title}  Embedding Dims={samples.shape[1]}" )
    
        N=labels.shape[0]
        title=f"N={N}  iters={n_iter:,}  perplexity={perplexity[subplot_index]}"
       
        plot( num_subplots, subplot_index, grid_size, embedding_train, labels, class_names, axes[r,c], title, title_font_size, marker_size, labelspacing, handletextpad, ms  )
        

  else:
    
    fig, axes = plt.subplots( figsize=figsize, nrows=1, ncols=1  )
    title_font_size = 14
    marker_size     = 2
    labelspacing    = 0.1
    handletextpad   = 0.8
    ms              = 12 
    
    if DEBUG>0:
      print( f"CUDA_TSNE:      INFO:  about to configure {CYAN}cuda TSNE {RESET}object with: n_iter={MIKADO}{n_iter}{RESET} perplexity={MIKADO}{perplexity[0]}{RESET}", flush=True )

      
    embedding_train = TSNE(                                                                                             # create and configure TSNE object
        n_components = n_components,
        n_iter       = n_iter,
        perplexity   = perplexity[0],
      learning_rate  = learning_rate
        # ~ verbose      = verbose
    ).fit_transform( samples )
          
    
    if DEBUG>0:
      print( f"CUDA_TSNE:      INFO:  finished {CYAN}tsne.fit{RESET}", flush=True )
      print( f"CUDA_TSNE:      INFO:  {CYAN}embedding_train.shape{RESET} = {MIKADO}{embedding_train.shape}{RESET}", flush=True )
      print( f"CUDA_TSNE:      INFO:  {CYAN}labels.shape{RESET}          = {MIKADO}{labels.shape}{RESET}",         flush=True )
      # ~ print( f"CUDA_TSNE:      INFO:  {CYAN}embedding_train{RESET}       =\n{MIKADO}{embedding_train}{RESET}",     flush=True )
  
  
    if (DEBUG>0):
      all_clusters_unique=sorted(set(labels))
      print ( f"CUDA_TSNE:      INFO:  unique classes represented in samples (truth labels) = {MIKADO}{all_clusters_unique}{RESET}" )
    
    if (DEBUG>0):
      for i in range ( 0, len(all_clusters_unique) ):
        print ( f"CUDA_TSNE:      INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(labels==i).sum()}{RESET}" )
    
  

    # 3. plot the results as a scattergram
            
    N=labels.shape[0]
    title=f"unsupervised clustering using cuda t-sne \n{args.dataset.upper()} dataset:  {N}=samples   iterations={n_iter}   perplexity={perplexity[0]}"

    plot( 1, 1, grid_size, embedding_train, labels, class_names, axes, title, title_font_size, marker_size, labelspacing, handletextpad, ms  )  
  
  now = datetime.datetime.now()  
  fqn = f"{args.log_dir}/{now:%y%m%d_%H%M}__________{descriptor_clustering}_dims_{samples.shape[1]}____cuda_tsne_clustering.png"
  fig.savefig(fqn)

  if render_clustering=="True":
    plt.show()  


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def plot( num_subplots, subplot_index, grid_size, x, y, class_names, ax, title, title_font_size, marker_size, labelspacing, handletextpad, ms, draw_legend=True, draw_centers=False, draw_cluster_labels=False, colors=None, legend_kwargs=None, label_order=None, **kwargs ):

    if grid_size>1:
      plt.tight_layout(rect=[0, 0, 1, 0.945])                                                                 # see https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
    else:
      pass

    ax.set_title( title, fontsize=title_font_size )

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)

    if len(classes)<10:
      colors = list(plt.cm.tab10(np.arange(10)))                                                           # colors for pan cancer (hard-wired)

    else:                                                                                                          
      colors = [ "black",         "paleturquoise",   "silver", 
               "red",             "royalblue",      "lightsteelblue",  "yellow",   "lightcoral",
               "thistle",         "palevioletred", 
               "lightgreen",      "limegreen",      "green",   "greenyellow",                                 
               "peachpuff",       "orange", 
               "powderblue",      "cornflowerblue", 
               "khaki",           "peru", 
               "blueviolet",       "teal", 
               "yellow",          "fuchsia",         "cyan",   
               "yellowgreen",     "chartreuse",      "mediumspringgreen",
               "firebrick",       "green",
               "magenta",         "blue",
               "coral",           "darkseagreen",    "darkturquoise",
               "deepskyblue",     "deeppink"
                              ] 
                              
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)                                                 # see https://matplotlib.org/stable/tutorials/introductory/customizing.html?highlight=axes.prop_cycle


    default_colors = matplotlib.rcParams["axes.prop_cycle"]
    colors =  {k: v["color"] for k, v in zip(classes, default_colors())}
    point_colors = list( map(colors.get, y ) )                                                             # get the color for each y and create a python list from same

    if (DEBUG>0):
      print ( f"CUDA_TSNE:      INFO: plot()  class_names           = {BITTER_SWEET}{class_names}{RESET}"   )
      print ( f"CUDA_TSNE:      INFO: plot()  classes               = {BITTER_SWEET}{classes}{RESET}"       )
      print ( f"CUDA_TSNE:      INFO: plot()  colors                = {BITTER_SWEET}{colors}{RESET}"        )      # dictionary mapping colors to class
    if (DEBUG>99):
      print ( f"CUDA_TSNE:      INFO: plot()  point_colors          = {BITTER_SWEET}{point_colors}{RESET}"  )      # the color each individual point will receive

    
    if (DEBUG>2):
      print ( f"CUDA_TSNE:      INFO: plot()  x[:, 0].min()         = {BITTER_SWEET}{x[:, 0].min()}{RESET}" )
      print ( f"CUDA_TSNE:      INFO: plot()  x[:, 0].max()         = {BITTER_SWEET}{x[:, 0].max()}{RESET}" )
      print ( f"CUDA_TSNE:      INFO: plot()  x[:, 1].min()         = {BITTER_SWEET}{x[:, 1].min()}{RESET}" )
      print ( f"CUDA_TSNE:      INFO: plot()  x[:, 1].max()         = {BITTER_SWEET}{x[:, 1].max()}{RESET}" )      

    if (DEBUG>99):
      print ( f"CUDA_TSNE:       x[:, 0]                             = {BITTER_SWEET}{x[:, 0]}{RESET}"       )
      print ( f"CUDA_TSNE:       x[:, 1]                             = {BITTER_SWEET}{x[:, 1]}{RESET}"       )

    x1 = x[:, 0]
    x2 = x[:, 1]
    std_devs=2


    if (DEBUG>0):
      print ( f"CUDA_TSNE:      INFO: plot()  np.median(x1)           = {BITTER_SWEET}{np.median(x1)}{RESET}"   )
      print ( f"CUDA_TSNE:      INFO: plot()  np.median(x2)           = {BITTER_SWEET}{np.median(x2)}{RESET}"   )
      print ( f"CUDA_TSNE:      INFO: plot()  np.std(x1)              = {BITTER_SWEET}{np.std(x1)}{RESET}"   )
      print ( f"CUDA_TSNE:      INFO: plot()  np.std(x2)              = {BITTER_SWEET}{np.std(x2)}{RESET}"   )
      
      
    x_lim_left  = np.median(x1)-std_devs*np.std(x1)
    x_lim_right = np.median(x1)+std_devs*np.std(x1)
    y_lim_left  = np.median(x2)-std_devs*np.std(x2)
    y_lim_right = np.median(x2)+std_devs*np.std(x2)    
    
    x_lim_left  = x_lim_left  if not np.isnan(x_lim_left)  else -10
    x_lim_right = x_lim_right if not np.isnan(x_lim_right) else  10
    y_lim_left  = y_lim_left  if not np.isnan(y_lim_left)  else -10
    y_lim_right = y_lim_right if not np.isnan(y_lim_right) else  10

    ax.set_xlim( [ x_lim_left, x_lim_right ] )
    ax.set_ylim( [ y_lim_left, y_lim_right ] )
    
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params) 
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True)
       
    ax.scatter( x1, x2, c=point_colors, s=marker_size, marker="s")

    
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

    fontsize=5 if (len(class_names)>2 and grid_size>2) else 6 if (len(class_names)>2 and grid_size>1) else 9 if (len(class_names)>2 and grid_size>1) else 12
    
    
    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=ms,
                alpha=1,
                linewidth=0,
                # ~ label=class_names[yi],
                label = f"{class_names[yi]} ({(y==yi).sum()})",
                # ~ markeredgecolor="k",
            )
            for yi in classes
        ]
        
        bbox_to_anchor=( 1., 0.5) if grid_size>1 else (0.9, 0.5)
        
        legend_kwargs_ = dict(loc="center right", bbox_to_anchor=bbox_to_anchor, frameon=False, labelspacing=0 if (len(class_names)>2 and grid_size>1) else 0 if len(class_names)>2 else 0.1, handletextpad=handletextpad )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)

        ax.legend(handles=legend_handles, **legend_kwargs_, fontsize=fontsize )


  
