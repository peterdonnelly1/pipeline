
import argparse
import utils
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import matplotlib.pyplot as plt
import matplotlib.colors

import utils
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

def main(args):
  
  
  # 1. load and prepare data

  image_file = "logs/images_new.npy" 
  label_file = "logs/img_labels_new.npy"
  
  images = np.load( image_file )
  labels = np.load( label_file )

  x_npy = images.reshape(images.shape[0], images.shape[1]*images.shape[2]*images.shape[3])
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  image file shape {MIKADO}{x_npy.shape}{RESET}" )
    print( f"OTSNE_SIMPLE:     INFO:  label file shape {MIKADO}{labels.shape}{RESET}" )  
    print( f"OTSNE_SIMPLE:     INFO:  image file {CYAN}{images}{RESET} contains {MIKADO}{x_npy.shape[0]}{RESET} samples each with {MIKADO}{x_npy.shape[1]}{RESET} features", flush=True)
    print( f"OTSNE_SIMPLE:     INFO:  label file {CYAN}{labels}{RESET} contains {MIKADO}{x_npy.shape[0]}{RESET} labels", flush=True)

  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  labels = {MIKADO}{labels}{RESET}" )  
    
  x_train, x_test, y_train, y_test = train_test_split( x_npy, labels, test_size=.1, random_state=42 )
  
  training_examples = x_train.shape[0]
  test_examples     = x_test .shape[0]
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  after splitting:" )
    print( f"OTSNE_SIMPLE:     INFO:    Training set comprises {MIKADO}{training_examples}{RESET} samples" )
    print( f"OTSNE_SIMPLE:     INFO:    Test     set comprises {MIKADO}{test_examples}{RESET}     samples" )


  # 2. cluster
      
  n_iter       = 1000
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
    
  embedding_train = tsne.fit(x_train)
  
  if DEBUG>0:
    print( f"OTSNE_SIMPLE:     INFO:  finished {CYAN}tsne.fit{RESET}", flush=True )
    print( f"OTSNE_SIMPLE:     INFO:  {CYAN}embedding_train.shape{RESET} ={MIKADO}{embedding_train.shape}{RESET}", flush=True )
    print( f"OTSNE_SIMPLE:     INFO:  {CYAN}y_train.shape{RESET}         ={MIKADO}{y_train.shape}{RESET}",         flush=True )
    print( f"OTSNE_SIMPLE:     INFO:  {CYAN}embedding_train{RESET}       =\n{MIKADO}{embedding_train}{RESET}",     flush=True )
    
  
    # 3. plot the results as a scattergram
    
    figure_width  = 20
    figure_height = 10
    
  
    if (DEBUG>0):
      np.set_printoptions(formatter={'int': lambda x:   "{:>2d}".format(x)})
      print ( f"OTSNE:           INFO:  labels    = {MIKADO}{labels[0:training_examples]}{RESET}" )
    c = labels[0:training_examples]
    if (DEBUG>0):
      print ( f"OTSNE:           INFO:  labels+1  = {MIKADO}{c}{RESET}" )
    colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
    if (DEBUG>0):
      print ( f"OTSNE:           INFO:  colors    = {MIKADO}{colors}{RESET}" )
    cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, c.max()+3), colors )
    
    fig, ax = plt.subplots( figsize = (figure_width, figure_height) )
    # ~ fig.tight_layout()
    X = embedding_train[0:training_examples,0]
    Y = embedding_train[0:training_examples,1]
    N=X.shape[0]
  
    title=f"t-sne Clustering (TSNE)\n(cancer type={args.dataset}, N={N}, colour=cluster, letter=true subtype)"
    
    plt.title( title,fontsize=15)
    s = ax.scatter( X, Y, s=50, linewidth=0, marker="s", c=c, cmap=cmap, alpha=1.0)  
    legend1 = ax.legend(*s.legend_elements(), loc="upper left", title="cluster number")
    ax.add_artist(legend1)

    if DEBUG>0:
      print( f"OTSNE_SIMPLE:     INFO:  labels.shape {MIKADO}{labels.shape}{RESET}" )
      print( f"OTSNE_SIMPLE:     INFO:  X     .shape {MIKADO}{     X.shape}{RESET}" )
      print( f"OTSNE_SIMPLE:     INFO:  Y     .shape {MIKADO}{     Y.shape}{RESET}" )
  
    offset=.5
    for i, label in enumerate( labels[0:training_examples] ):
      
      if (DEBUG>2):  
        print ( f"i={i:4d} label={MIKADO}{label}{RESET}  args.class_names[label]={MIKADO}{ args.class_names[label]:16s}{RESET} args.class_names[label][0]={MIKADO}{args.class_names[label][0]}{RESET}" )

              
      plt.annotate( args.class_names[label][0], ( X[i]-.035, Y[i]-.06), fontsize=10, color='black' )
  

      
    # ~ ax.set_xlim(( -1, 1 ))
    # ~ ax.set_ylim(( -1, 1 ))
    
    lim = (x_npy.min()-1, x_npy.max()+1)
    
    plt.show()
  
  
  # ~ ===
  # ~ figure_width  = 20
  # ~ figure_height = 10
  # ~ fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )
  
  # ~ tsne_result_df = pd.DataFrame({'tsne_1': embedding_train[0:training_examples,0], 'tsne_2': embedding_train[0:training_examples,1], 'label': y[0:training_examples] })
  
  # ~ sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)
  
  # ~ lim = (embedding_train.min()-5, embedding_train.max()+5)
  
  # ~ plt.show()


# --------------------------------------------------------------------------------------------
  
if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--metric',                          type=str,   default="hamming"                          )        
    p.add_argument('--dataset',                         type=str,   default="stad"                             )        
    p.add_argument('--input_file',                      type=str,   default="logs/ae_output_features.pt"       )
    p.add_argument('--class_names',         nargs="*",  type=str,                                              )                 
    
    args, _ = p.parse_known_args()

    main(args)
  
