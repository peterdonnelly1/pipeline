
import numpy as np
import pandas as pd
import matplotlib.pyplot     as plt
import seaborn as sns
from   random          import randint
from sklearn.manifold  import TSNE

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

fqn    =  f"/home/peter/git/pipeline/logs/images_new.npy"
X      = np.load(fqn)

fqn    =  f"/home/peter/git/pipeline/logs/img_labels_new.npy"
labels = np.load(fqn)


if DEBUG>0:
  print( f"SKL_TSNE:     INFO: X.shape          = {MAGENTA}{X.shape}{RESET}"       )

X_flat = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])

if DEBUG>0:
  print( f"SKL_TSNE:     INFO: X_flat.shape     = {MAGENTA}{X_flat.shape}{RESET}"       )



# ~ tsne_result = TSNE(n_components=3,  perplexity=30.0, n_iter=10000,  early_exaggeration=12.0 ).fit_transform(X_flat)
tsne_result = TSNE(n_components=2,  perplexity=30.0, n_iter=10000 ).fit_transform(X_flat)
tsne_result.shape

tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels })

fig, ax = plt.subplots(1)

sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)


lim = (tsne_result.min()-5, tsne_result.max()+5)


ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()
