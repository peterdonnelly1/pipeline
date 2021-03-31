import sys
import torch
import numpy             as np
import argparse
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn           as sns

from sklearn.cluster import DBSCAN

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

pd.set_option('display.max_rows',      35  )
pd.set_option('display.max_columns',   35  )
pd.set_option('display.width',         300 )
pd.set_option('display.max_colwidth',  8   ) 
pd.set_option('display.float_format',  lambda x: '%6.2f' % x)

np.set_printoptions(edgeitems=1000)
np.set_printoptions(linewidth=1000)





def main(args):


  # 1. loading and preparing data
  
  fqn = f"logs/{args.input_file}"
    
  if DEBUG>0:
    print( f"{BRIGHT_GREEN}DBSCAN:          INFO:  about to load autoencoder generated feature file from input file '{MAGENTA}{fqn}{RESET}'", flush=True )
  try:
    dataset  = torch.load( fqn )
    if DEBUG>0:
      print( f"{BRIGHT_GREEN}DBSCAN:          INFO:  dataset successfully loaded{RESET}" ) 
  except Exception as e:
    print ( f"{RED}DBSCAN:          ERROR:  could not load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
    print ( f"{RED}DBSCAN:          ERROR:  the exception was: {CYAN}'{e}'{RESET}" )
    print ( f"{RED}DBSCAN:          ERROR:  halting now" )
    sys.exit(0)

  embeddings  = dataset['embeddings'].cpu().numpy().squeeze()
  labels      = dataset['labels']    .cpu().numpy().squeeze()
  
  if DEBUG>0:
    print ( f"DBSCAN:          INFO:  np.sum(embeddings)      =  {MIKADO}{np.sum(embeddings)}{RESET}"      ) 
  
  if np.sum(embeddings)==0.0:
    print ( f"{RED}DBSCAN:          ERROR:  all embeddings are zero vectors - the input file was completely degenerate{RESET}" )
    print ( f"{RED}DBSCAN:          ERROR:  not halting, but might as well be{RESET}" )
  
  if DEBUG>0:
    print ( f"DBSCAN:          INFO:  about to flatten channels and r,g,b dimensions"      ) 
  
  x_npy = embeddings.reshape(embeddings.shape[0], embeddings.shape[1]*embeddings.shape[2]*embeddings.shape[3])
  
  if DEBUG>0:
    print ( f"DBSCAN:          INFO:  x_npy.shape          = {MIKADO}{x_npy.shape}{RESET}"      ) 
    print ( f"DBSCAN:          INFO:  about to convert to pandas dataframe"      ) 
  
  
  
  # 2. cluster
  
  if DEBUG>0:
    print ( f"DBSCAN:          INFO:  about to create an {CYAN}HDBSCAN{RESET} clusterer object"      ) 

  #############################
  clusterer = DBSCAN(metric='euclidean', eps=11, min_samples=30).fit(x_npy)
  #############################

  
  if DEBUG>0:
    print ( f"DBSCAN:          INFO:  about to cluster        {CYAN}x_pd{RESET} using {CYAN}clusterer.fit(x_pd){RESET}"     ) 
    print ( f"DBSCAN:          INFO:  now finished clustering {CYAN}x_pd{RESET}"     ) 
    print ( f"DBSCAN:          INFO:  clusterer.labels_    = {MIKADO}{clusterer.labels_}{RESET}"      ) 
  
  if (DEBUG>0):
    all_clusters_unique=sorted(set(clusterer.labels_))
    print ( f"DBSCAN:          INFO:  unique classes represented  = {MIKADO}{all_clusters_unique}{RESET}" )
  
  if (DEBUG>0):
    for i in range ( -1, len(all_clusters_unique) ):
      print ( f"DBSCAN:          INFO:  count of instances of cluster label {CARRIBEAN_GREEN}{i:2d}{RESET}  = {MIKADO}{(clusterer.labels_==i).sum()}{RESET}" )
  
  
  
  # 3. plot the results as a scattergram
  
  figure_width  = 20
  figure_height = 10
  
  # ~ color_palette         = sns.color_palette('bright', 100)
  # ~ cluster_colors        = [color_palette[x] for x in clusterer.labels_]
  # ~ cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]

  if (DEBUG>0):
    np.set_printoptions(formatter={'int': lambda x:   "{:>2d}".format(x)})
    print ( f"DBSCAN:          INFO:  labels    = {MIKADO}{clusterer.labels_}{RESET}" )
  c = clusterer.labels_ +1
  if (DEBUG>0):
    print ( f"DBSCAN:          INFO:  labels+1  = {MIKADO}{c}{RESET}" )
  colors  = [f"C{i}" for i in np.arange(1, c.max()+2)]
  if (DEBUG>0):
    print ( f"DBSCAN:          INFO:  colors    = {MIKADO}{colors}{RESET}" )
  cmap, norm = matplotlib.colors.from_levels_and_colors( np.arange(1, c.max()+3), colors )
  
  fig, ax = plt.subplots( figsize = (figure_width, figure_height) )
  # ~ fig.tight_layout()
  X = x_npy[:,0]
  Y = x_npy[:,1]
  
  N=x_npy.shape[0]
  title=f"Unsupervised clustering using Density Based Spatial Clustering of Applications with Noise (DBSCAN)\n(cancer type={args.dataset}, N={N}, colour=cluster, letter=true subtype)"
  
  plt.title( title,fontsize=15)
  s = ax.scatter( X, Y, s=50, linewidth=0, marker="s", c=c, cmap=cmap, alpha=1.0)  
  legend1 = ax.legend(*s.legend_elements(), loc="upper left", title="cluster number")
  ax.add_artist(legend1)

  offset=.5
  for i, label in enumerate( labels ):
    plt.annotate( args.class_names[label][0], ( X[i]-.035, Y[i]-.06), fontsize=10, color='black' )

    if (DEBUG>0):  
      print ( f"i={i:4d} label={MIKADO}{label}{RESET}  args.class_names[label]={MIKADO}{ args.class_names[label]:16s}{RESET} args.class_names[label][0]={MIKADO}{args.class_names[label][0]}{RESET}" )
    
    
  # ~ ax.set_xlim(( -1, 1 ))
  # ~ ax.set_ylim(( -1, 1 ))
  
  lim = (x_npy.min()-12, x_npy.max()+12)
  
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
  
