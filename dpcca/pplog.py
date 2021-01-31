"""=============================================================================
Utility functions for easy and pretty file logging.
============================================================================="""

import logging
import numpy as np
import types
import torch

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
DULL_WHITE='\033[38;2;140;140;140m'
CYAN='\033[36;1m'
MIKADO='\033[38;2;255;196;12m'
AZURE='\033[38;2;0;127;255m'
AMETHYST='\033[38;2;153;102;204m'
CHARTREUSE='\033[38;2;223;255;0m'
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

BOLD='\033[1m'
ITALICS='\033[3m'
UNDER='\033[4m'
BLINK='\033[5m'
RESET='\033[m'

DEBUG=1


# ------------------------------------------------------------------------------

MAIN_LOGGER = 'logger.main'
DIRECTORY   = None

# ------------------------------------------------------------------------------

def set_logfiles(directory, level=logging.INFO):
    """Function setup as many loggers as you want.
    """
    global DIRECTORY
    DIRECTORY = directory

    handler = logging.FileHandler(f'{directory}/log.txt')
    logger = logging.getLogger(MAIN_LOGGER)
    logger.setLevel(level)
    logger.addHandler(handler)

# ------------------------------------------------------------------------------

def log(msg):
    """Print message.
    """
    
    np.set_printoptions(edgeitems=100)
    np.set_printoptions(linewidth=300)
    _log(msg, MAIN_LOGGER)

# ------------------------------------------------------------------------------

def _log(msg, logger_name):
    """Print message to appropriate logger if on cluster, otherwise print to
    stdout.
    """
    # ~ if torch.cuda.is_available():
    logger = logging.getLogger(logger_name)
    logger.info(msg)
    # ~ else:
        # ~ print(msg)

# ------------------------------------------------------------------------------

def log_line(epoch, train_msgs, test_msgs):
    """Print main line in log file, including current epoch and train and test
    data.
    """
    train = '\t'.join(['{:6f}' for _ in train_msgs]).format(*train_msgs)
    test  = '\t'.join(['{:6f}' for _ in test_msgs]).format(*test_msgs)
    msg   = '\t|\t'.join([str(epoch), train, test])
    log(msg)

# ------------------------------------------------------------------------------

def log_section(msg, delim='='):
  
    """Print message with header.
    """
    log(delim * 200)
    log(msg)
    log(delim * 200)

# ------------------------------------------------------------------------------

def log_args(args):
    """Print arguments passed to script.
    """
    fields = [f for f in vars(args)]
    longest = max(fields, key=len)
    format_str = '{:>%s}  {:}' % len(longest)
    for f in fields:
        msg = format_str.format(f, getattr(args, f))
        log(msg)

# ------------------------------------------------------------------------------

def log_config(cfg):
    """Print settings in the configuration object.
    """
    fields = [f for f in dir(cfg) if not f.startswith('__') and
              type(getattr(cfg, f)) != types.MethodType]
    longest = max(fields, key=len)
    format_str = '{:>%s}  {:}' % len(longest)
    for f in fields:
        if type(getattr(cfg, f)) != types.MethodType:
            msg = format_str.format(f, getattr(cfg, f))
        log(msg)

# ------------------------------------------------------------------------------

def log_model(model):
    """Print model specifications.
    """
    log(model)

# ------------------------------------------------------------------------------

def save_test_indices(indices):
    """Save a Python list so we know our random split of test indices.
    """
    
    save_test_directory = f"{DIRECTORY}/testset_indices"
    
    if DEBUG>2:
      print ( f"PPRINT:         INFO:   save_test_indices() = {CYAN}{save_test_directory}{RESET}" )
    
    np.save( save_test_directory, indices )
