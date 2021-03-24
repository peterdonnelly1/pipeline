
from   otsne_utils import  plot
from   otsne_utils import  MACOSKO_COLORS

import sys
import torch
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns


from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE.callbacks import ErrorLogger

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


# ~ img_file        = "logs/images_new.npy" 
# ~ img_labels_file = "logs/img_labels_new.npy"

# ~ x_original_shape = np.load( img_file )

  
fqn = "logs/ae_output_features.pt"
if DEBUG>0:
  print( f"{BRIGHT_GREEN}OTSNE_SIMPLE:     INFO:  about to load autoencoder generated feature file from {MAGENTA}{fqn}{RESET}", flush=True )
try:
  genes_new    = torch.load( fqn )
  if DEBUG>0:
    print ( f"OTSNE_SIMPLE:     INFO:  genes_new.size         = {MIKADO}{genes_new.size()}{RESET}"      ) 
  if DEBUG>0:   
    print( f"{BRIGHT_GREEN}OTSNE_SIMPLE:     INFO:  autoencoder feature file successfully loaded{RESET}" )          
except Exception as e:
  print ( f"{RED}OTSNE_SIMPLE:     INFO:  could now load feature file. Did you remember to run the system with {CYAN}NN_MODE='pre_compress'{RESET}{RED} and an autoencoder such as {CYAN}'AEDENSE'{RESET}{RED} to generate the feature file? ... can't continue, so halting now [143]{RESET}" )
  if DEBUG>0:
    print ( f"{RED}OTSNE_SIMPLE:     INFO:  the exception was: {CYAN}'{e}'{RESET}" )
  sys.exit(0)

genes_new    = genes_new.numpy().squeeze()
  
x_original_shape = genes_new
y = np.ones( 169 )

x = x_original_shape.reshape(x_original_shape.shape[0], x_original_shape.shape[1]*x_original_shape.shape[2]*x_original_shape.shape[3])

# ~ x = x_original_shape.reshape(x_original_shape.shape[0], x_original_shape.shape[1]*x_original_shape.shape[2]*x_original_shape.shape[3])
# ~ y = np.load( img_labels_file )

# ~ x = data["pca_50"]
# ~ y = data["CellType1"].astype(str)

if DEBUG>0:
  print( f"OTSNE_SIMPLE:     INFO:  Data stats:" )
  print( f"OTSNE_SIMPLE:     INFO:    image file shape {MIKADO}{x.shape}{RESET}" )
  print( f"OTSNE_SIMPLE:     INFO:    label file shape {MIKADO}{y.shape}{RESET}" )  
  # ~ print( f"OTSNE_SIMPLE:     INFO:    image file {MAGENTA}{img_file}{RESET}     contains {MIKADO}{x.shape[0]:,}{RESET} samples, each with {MIKADO}{x.shape[1]:,}{RESET} features:", flush=True)
  # ~ print( f"OTSNE_SIMPLE:     INFO:    label file {MAGENTA}{img_labels_file}{RESET} contains {MIKADO}{x.shape[0]:,}{RESET} labels", flush=True)


affinities = affinity.PerplexityBasedNN(
    x,
    perplexity=30,
    n_jobs=-1,
    random_state=42
)

init = initialization.pca(x, random_state=0)

embedding_standard = TSNEEmbedding(
    init,
    affinities,
    negative_gradient_method="fft",
    n_jobs=-1,
)

embedding_standard.optimize(
    n_iter=250,
    exaggeration=12,
    momentum=0.5,
    inplace=True,
    callbacks    = ErrorLogger())


plot(embedding_standard, y)
plt.show()

# ~ embedding_standard.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

embedding_standard.optimize(
    n_iter=750,
    exaggeration=1,
    momentum=0.8,
    inplace=True,
    callbacks    = ErrorLogger())
    
plot(embedding_standard, y)
plt.show()



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=42)

training_examples = x_train.shape[0]
test_examples     = x_test .shape[0]

if DEBUG>0:
  print( f"OTSNE_SIMPLE:     INFO:  after splitting:" )
  print( f"OTSNE_SIMPLE:     INFO:    Training set comprises {MIKADO}{training_examples}{RESET} samples" )
  print( f"OTSNE_SIMPLE:     INFO:    Test     set comprises {MIKADO}{test_examples}{RESET}     samples" )

n_iter       = 400 
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



plot(embedding_train, y[0:training_examples])

plt.show()

# plot the results on a scattergram

# ~ figure_width  = 20
# ~ figure_height = 10
# ~ fig, ax = plt.subplots( figsize=( figure_width, figure_height ) )

# ~ tsne_result_df = pd.DataFrame({'tsne_1': embedding_train[0:training_examples,0], 'tsne_2': embedding_train[0:training_examples,1], 'label': y[0:training_examples] })

# ~ sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)

# ~ lim = (embedding_train.min()-5, embedding_train.max()+5)

# ~ plt.show()

