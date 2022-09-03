"""=============================================================================
Utility functions for pretty logging.
============================================================================="""

import logging
import types
import datetime
from multimethod import multimethod

from constants  import *

    
# ------------------------------------------------------------------------------


def add_logger( logger_name, log_dir, descriptor, level=logging.INFO  ):
  
    """Establish a logger called 'logger_name' that will log to a file name made up a composite of log_dir, and descriptor something like this for job level logs:
          220902_1527_BATCH__003_SKIN_image_____NOTBLNCD_UNIMODE_CASE_________log.txt
        and this for 'per run' logs:
          220902_1527__01_OF_003_SKIN_image_____NOTBLNCD_UNIMODE_CASE_________RESNET152_______NONE_ADAM_____e_003_N_10015_hi_03_bat_1785_test_020_lr_00.000100_tiles_0001_tlsz_0016__mag_1.0__prob_1.0__________________log.txt
        
    """
    
    now  = datetime.datetime.now()
    if descriptor=="cumulative":
      log_file_name = f'{log_dir}/{descriptor}_log.txt' 
    else:
      log_file_name = f'{log_dir}/{now:%y%m%d_%H%M}_{descriptor}_log.txt' 
    handler = logging.FileHandler( log_file_name)
    logger  = logging.getLogger  ( logger_name )
    logger.setLevel   (level)
    logger.addHandler ( handler )
    
    console = logging.StreamHandler()
    console.setLevel(logging.CRITICAL)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

# ------------------------------------------------------------------------------

def del_logger( logger_name ):


  del logging.Logger.manager.loggerDict[ logger_name ]


# ------------------------------------------------------------------------------


@multimethod
def log( msg: str ):
    
    _log( "run", msg )

# ------------------------------------------------------------------------------

@multimethod
def log( logger_name: str, msg: str ):
    
    _log( logger_name, msg )


# ------------------------------------------------------------------------------

def _log( logger_name, msg ):
  
    """Print message to applicable logger
    """

    logger = logging.getLogger( logger_name )
    logger.info( msg )


# ------------------------------------------------------------------------------
@multimethod
def log_section( msg: str ):
  
    """Print message with header to default logger
    """
    
    log( "run", '=' * 200)
    log( "run", msg)
    log( "run", '=' * 200)


# ------------------------------------------------------------------------------
@multimethod
def log_section( logger_name: str, msg: str ):
  
    """Print message with header to named logger
    """

    _log( logger_name, '=' * 200 )
    _log( logger_name, msg         )
    _log( logger_name, '=' * 200 )







# ~ # ------------------------------------------------------------------------------

# ~ def log_line(epoch, train_msgs, test_msgs ):
  
    # ~ """Print main line in log file, including current epoch and train and test
    # ~ data.
    # ~ """
    # ~ train = '\t'.join(['{:6f}' for _ in train_msgs]).format(*train_msgs)
    # ~ test  = '\t'.join(['{:6f}' for _ in test_msgs]).format(*test_msgs)
    # ~ msg   = '\t|\t'.join([str(epoch), train, test])
    # ~ log(msg)
# ~ # ------------------------------------------------------------------------------


# ~ def log_config(cfg):
    # ~ """Print settings in the configuration object.
    # ~ """
    # ~ fields = [f for f in dir(cfg) if not f.startswith('__') and
              # ~ type(getattr(cfg, f)) != types.MethodType]
    # ~ longest = max(fields, key=len)
    # ~ format_str = '{:>%s}  {:}' % len(longest)
    # ~ for f in fields:
        # ~ if type(getattr(cfg, f)) != types.MethodType:
            # ~ msg = format_str.format(f, getattr(cfg, f))
        # ~ log(msg)

# ~ # ------------------------------------------------------------------------------

# ~ def log_model(model):
  
    # ~ """Print model specifications.
    # ~ """
    # ~ log(model)

# ~ # ------------------------------------------------------------------------------

