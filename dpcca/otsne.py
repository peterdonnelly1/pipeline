
import argparse
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import matplotlib.pyplot as plt
import matplotlib.colors

# ~ from otsne_utils import plot
# ~ from otsne_utils import MACOSKO_COLORS
from openTSNE                import TSNE
from openTSNE.callbacks      import ErrorLogger
from sklearn.model_selection import train_test_split


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

def otsne( args, pct_test):
  
  
  # 1. load and prepare data

  image_file = "../logs/images_new.npy" 
  label_file = "../logs/img_labels_new.npy"
  
  images = np.load( image_file )
  labels = np.load( label_file )

  if DEBUG>0:
    print( f"\n{GREY_BACKGROUND}OTSNE_SIMPLE:   INFO: {WHITE}{CHARTREUSE}OTSNE clustering{WHITE}: samples_file={MAGENTA}{image_file}{WHITE}, labels_file={MAGENTA}{label_file}{WHITE}, pct_test={MIKADO}{pct_test}{WHITE}, metric={CYAN}{args.metric}{WHITE}, iterations={MIKADO}{args.n_iterations}{WHITE}, perplexity={MIKADO}{args.perplexity}{WHITE}, momentum={MIKADO}{args.momentum}                                                                                                                        {RESET}" )  


  x_npy = images.reshape( images.shape[0], images.shape[1]*images.shape[2]*images.shape[3] )
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:   INFO:  image file shape {MIKADO}{x_npy.shape}{RESET}" )
    print( f"OTSNE_SIMPLE:   INFO:  label file shape {MIKADO}{labels.shape}{RESET}" )  
    print( f"OTSNE_SIMPLE:   INFO:  image file {CYAN}{image_file}{RESET} \r\033[60Ccontains {MIKADO}{x_npy.shape[0]}{RESET} samples each with {MIKADO}{x_npy.shape[1]}{RESET} features", flush=True)
    print( f"OTSNE_SIMPLE:   INFO:  label file {CYAN}{label_file}{RESET} \r\033[60Ccontains {MIKADO}{x_npy.shape[0]}{RESET} labels", flush=True)

  if DEBUG>0:
    # ~ print( f"OTSNE_SIMPLE:   INFO:  labels = {MIKADO}{labels}{RESET}" )  
    print( f"OTSNE_SIMPLE:   INFO:  x_npy.shape     = {MIKADO}{x_npy.shape}{RESET}" )  
    print( f"OTSNE_SIMPLE:   INFO:  x_npy[0].shape  = {MIKADO}{x_npy[0].shape}{RESET}" )  

  if DEBUG>2:
    print( f"OTSNE_SIMPLE:   INFO:  images[0] = \n{MIKADO}{images[0,2,40:80,90:100]}{RESET}" )  
    print( f"OTSNE_SIMPLE:   INFO:  x_npy [0]  =  {MIKADO}{x_npy[0,1000:1100]}{RESET}" )  
    
  x_train, x_test, y_train, y_test = train_test_split( x_npy, labels, test_size=pct_test, random_state=42 )
  
  training_examples = x_train.shape[0]
  test_examples     = x_test .shape[0]
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:   INFO:  after splitting:" )
    print( f"OTSNE_SIMPLE:   INFO:    training set comprises {MIKADO}{training_examples}{RESET} samples" )
    print( f"OTSNE_SIMPLE:   INFO:    test     set comprises {MIKADO}{test_examples}{RESET}     samples" )


  # 2. cluster
      
  n_iter       = args.n_iterations
  perplexity   = args.perplexity
  momentum     = args.momentum
  metric       = args.metric
  n_jobs       = -1                                                                                        # -1 means use all available processors
  random_state = 42
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:   INFO:  about to configure {CYAN}TSNE{RESET} with: n_iter={MIKADO}{n_iter}{RESET}, perplexity={MIKADO}{perplexity}{RESET}, momentum={MIKADO}{momentum}{RESET},  metric='{CYAN}{metric}{RESET}', n_jobs={MIKADO}{n_jobs}{RESET}, random_state={MIKADO}{random_state}{RESET}", flush=True )
    
  tsne = TSNE(                                                                                             # create and configure TSNE object
      n_iter       = n_iter,
      perplexity   = perplexity,
      metric       = metric,
      callbacks    = ErrorLogger(),
      n_jobs       = n_jobs,
      random_state = random_state,
  )
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:   INFO:  about to run {CYAN}tsne.fit{RESET}", flush=True )
    
  embedding_train = tsne.fit( x_train )
  embedding_test  = tsne.fit( x_test  )

  # ~ n_classes = len(args.class_names)
  # ~ confusion_matrix    =  np.zeros( (n_classes, n_classes), dtype=int )
  
  # ~ for j in range(0, len(embedding_test) ):
    # ~ if DEBUG>0:
      # ~ print( f"OTSNE_SIMPLE:   INFO:  {CYAN}y_test        [{j}]{RESET} = {MIKADO}{y_test[j]}{RESET}", flush=True )
      # ~ print( f"OTSNE_SIMPLE:   INFO:  {CYAN}embedding_test[{j}]{RESET} = {MIKADO}{embedding_test[j]}{RESET}", flush=True )
    # ~ confusion_matrix[ y_test[j], embedding_test[j] ] +=1
    
  # ~ if DEBUG>0:
    # ~ print( f"OTSNE_SIMPLE:   INFO:  {CYAN}confusion_matrix.shape{RESET} ={MIKADO}{confusion_matrix.shape}{RESET}", flush=True )
    # ~ print( f"OTSNE_SIMPLE:   INFO:  {CYAN}confusion_matrix{RESET} ={MIKADO}{CARRIBEAN_GREEN}{RESET}", flush=True )
    
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:   INFO:  finished {CYAN}tsne.fit{RESET}", flush=True )
    print( f"OTSNE_SIMPLE:   INFO:  {CYAN}embedding_train.shape{RESET} ={MIKADO}{embedding_train.shape}{RESET}", flush=True )
    print( f"OTSNE_SIMPLE:   INFO:  {CYAN}y_train.shape{RESET}         ={MIKADO}{y_train.shape}{RESET}",         flush=True )
    print( f"OTSNE_SIMPLE:   INFO:  {CYAN}embedding_train{RESET}       =\n{MIKADO}{embedding_train}{RESET}",     flush=True )


  if (DEBUG>0):
    all_clusters_unique=sorted(set(y_train))
    print ( f"OTSNE_SIMPLE:   INFO:   unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( -1, len(all_clusters_unique) ):
      print ( f"OTSNE_SIMPLE:   INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(y_train==i).sum()}{RESET}" )
  
  
  
  # 3. plot the results as a scattergram
  
  figure_width  = 20
  figure_height = 10
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

  if (DEBUG>2):
    np.set_printoptions(formatter={'int': lambda x:   "{:>2d}".format(x)})
    print ( f"OTSNE_SIMPLE:   INFO:  labels    = {MIKADO}{y_train}{RESET}" )
  c = y_train
  if (DEBUG>2):
    print ( f"OTSNE_SIMPLE:   INFO:  labels+1  = {MIKADO}{c}{RESET}" )
  # ~ colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
  colors  = MACOSKO_COLORS
  if (DEBUG>2):
    print ( f"OTSNE_SIMPLE:   INFO:  colors               = {MIKADO}{colors}{RESET}" )
    print ( f"OTSNE_SIMPLE:   INFO:  np.unique(y_train)   = {MIKADO}{np.unique(y_train)}{RESET}" )

  if (DEBUG>2):
    print ( f"OTSNE_SIMPLE:   INFO:  y_train              = {MIKADO}{y_train}{RESET}" )
    
  # ~ cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, c.max()+3), colors )

  # ~ plot( embedding_train, y_train, colors=MACOSKO_COLORS )
  plot( embedding_train, y_train, args.class_names, ax=ax )
  plt.show()
  
  plot( embedding_test,  y_test, args.class_names )
  plt.show()    



"""    
    # ~ ===
  figure_width  = 20
  figure_height = 10
  fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  
  # ~ tsne_result_df = pd.DataFrame({'tsne_1': embedding_train[0:training_examples,0], 'tsne_2': embedding_train[0:training_examples,1], 'label': y[0:training_examples] })
  tsne_result_df = pd.DataFrame({'tsne_1': embedding_train[:,0], 'tsne_2': embedding_train[:,1], 'label': y_train} )
  
  sns.scatterplot( x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120 )
  
  lim = ( embedding_train[:,0].min(), embedding_train[:,0].max() )
  
  plt.show()


"""


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
      print ( f"OTSNE_SIMPLE:   INFO: plot()  class_names           = {BITTER_SWEET}{class_names}{RESET}" )
      print ( f"OTSNE_SIMPLE:   INFO: plot()  classes               = {BITTER_SWEET}{classes}{RESET}" )
      print ( f"OTSNE_SIMPLE:   INFO: plot()  colors                = {BITTER_SWEET}{colors}{RESET}" )
      print ( f"OTSNE_SIMPLE:   INFO: plot()  colors.get            = {BITTER_SWEET}{colors.get}{RESET}" )
      print ( f"OTSNE_SIMPLE:   INFO: plot()  point_colors          = {BITTER_SWEET}{point_colors}{RESET}" )

    # ~ lim = ( x.min(), x.max() )
    
    if (DEBUG>2):
      print ( f"OTSNE_SIMPLE:   INFO: plot()  x[:, 0].min()               = {BITTER_SWEET}{x[:, 0].min()}{RESET}" )
      print ( f"OTSNE_SIMPLE:   INFO: plot()  x[:, 0].max()               = {BITTER_SWEET}{x[:, 0].max()}{RESET}" )
      print ( f"OTSNE_SIMPLE:   INFO: plot()  x[:, 1].min()               = {BITTER_SWEET}{x[:, 1].min()}{RESET}" )
      print ( f"OTSNE_SIMPLE:   INFO: plot()  x[:, 1].max()               = {BITTER_SWEET}{x[:, 1].max()}{RESET}" )      

    x1 = x[:, 0]
    x2 = x[:, 1]
    std_devs=1
    ax.set_xlim( [ np.median(x1)-std_devs*np.std(x1), np.median(x1)+std_devs*np.std(x1) ] )
    ax.set_ylim( [ np.median(x2)-std_devs*np.std(x2), np.median(x2)+std_devs*np.std(x2) ] )
    
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params) 
    # ~ ax.scatter( x[:, 0], x[:, 1], c=point_colors, rasterized=True) 
    ax.scatter( x1, x2, c=point_colors, s=4, marker="s" )
  
    
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


  
