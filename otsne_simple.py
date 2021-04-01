
import argparse
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import matplotlib.pyplot as plt
import matplotlib.colors

from utils import plot
from utils import MACOSKO_COLORS
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

def main(args):
  
  
  # 1. load and prepare data

  image_file = "logs/images_new.npy" 
  label_file = "logs/img_labels_new.npy"
  
  images = np.load( image_file )
  labels = np.load( label_file )

  x_npy = images.reshape( images.shape[0], images.shape[1]*images.shape[2]*images.shape[3] )
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  image file shape {MIKADO}{x_npy.shape}{RESET}" )
    print( f"OTSNE_SIMPLE:     INFO:  label file shape {MIKADO}{labels.shape}{RESET}" )  
    print( f"OTSNE_SIMPLE:     INFO:  image file {CYAN}{image_file}{RESET} \r\033[60Ccontains {MIKADO}{x_npy.shape[0]}{RESET} samples each with {MIKADO}{x_npy.shape[1]}{RESET} features", flush=True)
    print( f"OTSNE_SIMPLE:     INFO:  label file {CYAN}{label_file}{RESET} \r\033[60Ccontains {MIKADO}{x_npy.shape[0]}{RESET} labels", flush=True)

  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  labels = {MIKADO}{labels}{RESET}" )  
    print( f"OTSNE_SIMPLE:     INFO:  x_npy.shape     = {MIKADO}{x_npy.shape}{RESET}" )  
    print( f"OTSNE_SIMPLE:     INFO:  x_npy[0].shape  = {MIKADO}{x_npy[0].shape}{RESET}" )  

  if DEBUG>2:
    print( f"OTSNE_SIMPLE:     INFO:  images[0] = \n{MIKADO}{images[0,2,40:80,90:100]}{RESET}" )  
    print( f"OTSNE_SIMPLE:     INFO:  x_npy [0]  = {MIKADO}{x_npy[0,1000:1100]}{RESET}" )  
    
  x_train, x_test, y_train, y_test = train_test_split( x_npy, labels, test_size=.1, random_state=42 )
  
  training_examples = x_train.shape[0]
  test_examples     = x_test .shape[0]
  
  if DEBUG>0:

    print( f"OTSNE_SIMPLE:     INFO:  after splitting:" )
    print( f"OTSNE_SIMPLE:     INFO:    Training set comprises {MIKADO}{training_examples}{RESET} samples" )
    print( f"OTSNE_SIMPLE:     INFO:    Test     set comprises {MIKADO}{test_examples}{RESET}     samples" )


  # 2. cluster
      
  n_iter       = 100
  perplexity   = 30
  momentum     = 0.8
  metric       = "euclidean"
  n_jobs       = -1                                                                                          # -1 means use all available processors
  random_state = 42
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  about to configure {CYAN}TSNE{RESET} with: n_iter={MIKADO}{n_iter}{RESET}, perplexity={MIKADO}{perplexity}{RESET}, momentum={MIKADO}{momentum}{RESET},  metric='{CYAN}{metric}{RESET}', n_jobs={MIKADO}{n_jobs}{RESET}, random_state={MIKADO}{random_state}{RESET}", flush=True )
    
  tsne = TSNE(                                                                                               # create and configure TSNE object
      n_iter       = n_iter,
      perplexity   = perplexity,
      metric       = metric,
      callbacks    = ErrorLogger(),
      n_jobs       = n_jobs,
      random_state = random_state,
  )
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  about to run {CYAN}tsne.fit{RESET}", flush=True )
    
  embedding_train = tsne.fit( x_train )
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  finished {CYAN}tsne.fit{RESET}", flush=True )
    print( f"OTSNE_SIMPLE:     INFO:  {CYAN}embedding_train.shape{RESET} ={MIKADO}{embedding_train.shape}{RESET}", flush=True )
    print( f"OTSNE_SIMPLE:     INFO:  {CYAN}y_train.shape{RESET}         ={MIKADO}{y_train.shape}{RESET}",         flush=True )
    print( f"OTSNE_SIMPLE:     INFO:  {CYAN}embedding_train{RESET}       =\n{MIKADO}{embedding_train}{RESET}",     flush=True )


  if (DEBUG>0):
    all_clusters_unique=sorted(set(y_train))
    print ( f"HDBSCAN:         INFO:  unique classes represented  = {MIKADO}{y_train}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( -1, len(all_clusters_unique) ):
      print ( f"HDBSCAN:         INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(y_train==i).sum()}{RESET}" )
  
  
  
    # 3. plot the results as a scattergram
    
    # ~ figure_width  = 20
    # ~ figure_height = 10
    # ~ fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  
    # ~ if (DEBUG>0):
      # ~ np.set_printoptions(formatter={'int': lambda x:   "{:>2d}".format(x)})
      # ~ print ( f"OTSNE:           INFO:  labels    = {MIKADO}{labels[0:training_examples]}{RESET}" )
    # ~ c = labels[0:training_examples]
    # ~ if (DEBUG>0):
      # ~ print ( f"OTSNE:           INFO:  labels+1  = {MIKADO}{c}{RESET}" )
    # ~ colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
    # ~ if (DEBUG>0):
      # ~ print ( f"OTSNE:           INFO:  colors    = {MIKADO}{colors}{RESET}" )
    # ~ cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, c.max()+3), colors )

    # ~ plot( embedding_train, y_train, colors=MACOSKO_COLORS)

    # ~ lim = ( embedding_train.min(), embedding_train.max() )

    # ~ plt.show()    
    
      # ~ ===
    figure_width  = 20
    figure_height = 10
    fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
    
    # ~ tsne_result_df = pd.DataFrame({'tsne_1': embedding_train[0:training_examples,0], 'tsne_2': embedding_train[0:training_examples,1], 'label': y[0:training_examples] })
    tsne_result_df = pd.DataFrame({'tsne_1': embedding_train[:,0], 'tsne_2': embedding_train[:,1], 'label': y_train} )
    
    sns.scatterplot( x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120 )
    
    lim = ( embedding_train[:,0].min(), embedding_train[:,0].max() )
    
    plt.show()

    

# --------------------------------------------------------------------------------------------
  
if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--metric',                          type=str,   default="hamming"                          )        
    p.add_argument('--dataset',                         type=str,   default="stad"                             )        
    p.add_argument('--input_file',                      type=str,   default="logs/ae_output_features.pt"       )
    p.add_argument('--class_names',         nargs="*",  type=str,                                              )                 
    
    args, _ = p.parse_known_args()

    main(args)
  
