import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2
import openslide
from Estimate_W import BLtrans
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

WHITE='\033[37;1m'
PURPLE='\033[35;1m'
DIM_WHITE='\033[37;2m'
AUREOLIN='\033[38;2;253;238;0m'
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


DEBUG=1

from Estimate_W import Wfast


def run_stainsep(filename,nstains,lamb,output_direc="",background_correction=True):
  
  print 
  print ( "Running stain separation on:",filename )

  level=0

  I = openslide.open_slide(filename)
  xdim,ydim=I.level_dimensions[level]
  img=np.asarray(I.read_region((0,0),level,(xdim,ydim)))[:,:,:3]

  if (DEBUG>0):
    print ( f"RUN_STAINSEP:           INFO: img.shape                         {AZURE}{img.shape}{RESET}",  flush=True )
      
  print  ( "Fast stain separation is running...." )
  
  Wi,Hi,Hiv,stains=Faststainsep( I, img, nstains, lamb, level, background_correction )

  #print "Time taken:",elapsed

  print ( "Color Basis Matrix:\n",Wi )

  fname=os.path.splitext(os.path.basename(filename))[0]
  cv2.imwrite(output_direc+fname+"-0_original.png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
  cv2.imwrite(output_direc+fname+"-1_Hstain.png",cv2.cvtColor(stains[0], cv2.COLOR_RGB2BGR))
  cv2.imwrite(output_direc+fname+"-2_Estain.png",cv2.cvtColor(stains[1], cv2.COLOR_RGB2BGR))



def Faststainsep( I_obj, I, nstains, lamb, level, background_correction):
  
  s=I.shape
  ndimsI = len(s)
  if ndimsI!=3:
    print  ( "Input Image I should be 3-dimensional!" )
    sys.exit(0)
  rows = s[0]
  cols = s[1]

  num_patches = 20
  patchsize   = 100

  #Estimate stain color bases + acceleration
  Wi,i0= Wfast ( I_obj, nstains, lamb, num_patches, patchsize, level )


  if background_correction:
    print( "Background intensity:",i0 )
  else:
    i0 = np.array([255.,255.,255.])
    print ( "Background correction disabled, default background intensity assumed" )

  #Beer-Lambert tranformation
  print ( "About to start perform Beer-Lambert transformation" )    
  V,VforW=BLtrans( I, i0 )                                                                                 # V=WH see in paper
  print ( "BLtrans" )    
  Hiv=np.transpose(np.dot(np.linalg.pinv(Wi),np.transpose(V)))  #Pseudo-inverse
  print ( "Pseudo-inverse" ) 
  Hiv[Hiv<0]=0
  print ( "Hiv" ) 
  Hi=np.reshape(Hiv,(rows,cols,nstains))
  print ( "reshape" ) 
  
  #calculate the color image for each stain
  sepstains = []

  print ( "about to enter nstains loops" )   
  for i in range(nstains):
    vdAS =  np.reshape(Hiv[:,i],(rows*cols,1))*np.reshape(Wi[:,i],(1,3))
    print ( "vdAS" ) 
    sepstains.append(np.uint8(i0*np.reshape(np.exp(-vdAS), (rows, cols, 3))))
    print ( "sepstains.append" )
    
  return Wi,Hi,Hiv,sepstains
