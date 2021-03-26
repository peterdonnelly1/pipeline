

import sys
import torch
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

import hdbscan
from IPython.display import display

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

pd.set_option('display.max_rows',     35 )
pd.set_option('display.max_columns',  35 )
pd.set_option('display.width',        300 )
pd.set_option('display.max_colwidth',  8 ) 
pd.set_option('display.float_format', lambda x: '%6.2f' % x)

np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=200)
  
fqn = "logs/ae_output_features.pt"
if DEBUG>0:
  print( f"{BRIGHT_GREEN}HDBSCAN:         INFO:  about to load autoencoder generated feature file from {MAGENTA}{fqn}{RESET}", flush=True )
try:
  samples     = torch.load( fqn )
  if DEBUG>0:
    print ( f"HDBSCAN:         INFO:  samples.size         = {MIKADO}{samples .size()}{RESET}"      ) 
  if DEBUG>0:   
    print( f"{BRIGHT_GREEN}HDBSCAN:         INFO:  autoencoder feature file successfully loaded{RESET}" )          
except Exception as e:
  print ( f"{RED}HDBSCAN:         INFO:  could now load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
  if DEBUG>0:
    print ( f"{RED}HDBSCAN:         INFO:  the exception was: {CYAN}'{e}'{RESET}" )
  sys.exit(0)

samples     = samples .numpy().squeeze()

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:  about to flatten channels and r,g,b dimensions"      ) 

x_npy = samples.reshape(samples.shape[0], samples.shape[1]*samples.shape[2]*samples.shape[3])

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:  x_npy.shape         = {MIKADO}{x_npy.shape}{RESET}"      ) 

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:  about to convert to pandas dataframe"      ) 
  
x_pd  = pd.DataFrame ( x_npy )

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:  x_pd.shape          = {MIKADO}{x_pd.shape}{RESET}"      ) 

display(x_pd)

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:  about to create an HDBSCAN clusterer object"      ) 

clusterer = hdbscan.HDBSCAN()

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:  about to cluster {CYAN}x_pd{RESET} using {CYAN}clusterer.fit(x_pd){RESET}"     ) 
  
print ( clusterer.fit(x_pd) )

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:  now finished clustering {CYAN}x_pd{RESET}"     ) 

if DEBUG>0:
  print ( f"HDBSCAN:         INFO:   clusterer.labels_    = {MIKADO}{ clusterer.labels_}{RESET}"      ) 
  

# ~ x_original_shape = samples 
# ~ y = np.ones( 169 )

# ~ if DEBUG>0:
  # ~ print( f"HDBSCAN:         INFO:  Data stats:" )
  # ~ print( f"HDBSCAN:         INFO:   x_original_shape.shape {MIKADO}{x_original_shape.shape}{RESET}" )
  












# ~ plot(embedding_train, y[0:training_examples])

# ~ plt.show()

# plot the results on a scattergram

# ~ figure_width  = 20
# ~ figure_height = 10
# ~ fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

# ~ tsne_result_df = pd.DataFrame({'tsne_1': embedding_train[0:training_examples,0], 'tsne_2': embedding_train[0:training_examples,1], 'label': y[0:training_examples] })

# ~ sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)

# ~ lim = (embedding_train.min()-5, embedding_train.max()+5)

# ~ plt.show()

