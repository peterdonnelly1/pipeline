import numpy as np
import cv2
import openslide
import os
import glob
from PIL import Image
import sys

np.set_printoptions(edgeitems=100)
np.set_printoptions(linewidth=500)

DEBUG  = 1

BB="\033[1m"
RESET="\033[m"

slide_name = sys.argv[1]                                                                                   # location of the svs image
#id = '17039791'
save_dir = sys.argv[2]                                                                                     # this will be the name of the directory used to hold the patches for the image
fq_name = "{:}/{:}".format( save_dir, slide_name )
 
if (DEBUG>0):
  print ( "\n    BACKGROUND_MASK.PY: INFO: argv[1] (slide_name)   = {:}{:}{:}".format( BB, sys.argv[1], RESET ),  flush=True )
  print ( "    BACKGROUND_MASK.PY: INFO: argv[2] (save_dir)     = {:}{:}{:}".format(   BB, sys.argv[2], RESET ),  flush=True )
  print ( "    BACKGROUND_MASK.PY: INFO: fq_name                = {:}{:}{:}".format(   BB, fq_name,     RESET ),  flush=True )

if not os.path.exists(save_dir):                                                                             
    os.mkdir(save_dir);                                                                                    # if patch directory does not exist (and why would it) then create


for f in glob.glob("*.png"):
    os.remove(f) 
    
oslide = openslide.OpenSlide(slide_name)                                                                   # open the svs image using openslide
width  = oslide.dimensions[0]
height = oslide.dimensions[1]

level = oslide.level_count - 1

if (DEBUG>0):
  print ( "    BACKGROUND_MASK.PY: INFO: SVS level count for this slide_name is: {:};  we will scale down to SVS level:  {:}".format(level, oslide.level_count), flush=True)                        # The number of levels in the slide_name. Levels are numbered from 0 (highest resolution) to level_count - 1 (lowest resolution)"                                                                    


scale_down = oslide.level_downsamples[level]                                                               # "A list of downsample factors for each level of the slide_name. level_downsamples[k] is the downsample factor of level k"
w, h = oslide.level_dimensions[level]                                                                      # "A list of (width, height) tuples, one for each level of the slide_name. level_dimensions[k] are the dimensions of level k."

if (DEBUG>9):
  print ( "    BACKGROUND_MASK.PY: INFO:        level dimensions are (w, h):       {:},{:}".format(w,h), flush=True)

#print('level: ', level)
#print('size: {}, {}'.format(w, h))

patch = oslide.read_region((0, 0), level, (w, h));                                                         # "Return an RGBA Image containing the contents of the specified region"

slide_id = slide_name.split('/')[-1].split('.svs')[0]                                                      # extract the filename from 'slide_name' (which is the fully qualified path plus filename with extension)
fname = '{}/{}_mask.png'.format(save_dir, slide_id);                                                       # define a filename for the mask
#fname = '{}/{}_mask.png'.format(save_dir, scale_down);
patch.save('{}/{}_resized.png'.format(save_dir, slide_id));                                                # save the resized version we just read into the patches folder for this slide_name

img = cv2.imread('{}/{}_resized.png'.format(save_dir, slide_id), 0)                                        # get opencv to read the resized version
sq=100
ctl = (h+w)//4
ctr = ctl+sq
cbl = ctl+sq
cbr = ctr+sq

if (DEBUG>9):
  print ("\n    BACKGROUND_MASK.PY: INFO: centre {:} x {:} slice = [{:}:{:},{:}:{:}]".format(sq, sq, ctl,ctr,cbl,cbr))
  print ("    BACKGROUND_MASK.PY: INFO: image size  {:}".format(img.shape))
  print ("    BACKGROUND_MASK.PY: INFO: smallest value in img = {:}".format(np.min(img)))
  print ("    BACKGROUND_MASK.PY: INFO: largest  value in img = {:}".format(np.max(img)))
  print ("    BACKGROUND_MASK.PY: INFO: before Gaussian blur: \n", img[ctl:ctr,cbl:cbr])
img = cv2.GaussianBlur(img, (61, 61), 0)                                                                   # apply a 61x61 gaussian kernel to remove noise
if (DEBUG>9):
  print ("    BACKGROUND_MASK.PY: INFO: smallest value in img = {:}".format(np.min(img)))
  print ("    BACKGROUND_MASK.PY: INFO: largest  value in img = {:}".format(np.max(img)))
  print ("    BACKGROUND_MASK.PY: INFO: after Gaussian blur:  \n", img[ctl:ctr,cbl:cbr])
ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)                                  # threshold it to extract background as a mask
if (DEBUG>9):
  print ("    BACKGROUND_MASK.PY: INFO: smallest value in img = {:}".format(np.min(img)))
  print ("    BACKGROUND_MASK.PY: INFO: largest  value in img = {:}".format(np.max(img)))
  print ("    BACKGROUND_MASK.PY: INFO: after  thresholding:  \n", img[ctl:ctr,cbl:cbr])


#cv2.imshow('dst_rt', imgf)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite(fname, imgf)                                                                                   # save the mask into the patches folder for this slide_name

oslide.close()                                                                                             # close openslide

#imgf = cv2.resize(imgf, (0, 0), fx = 0.3, fy = 0.3)
#cv2.imshow('img', imgf)
