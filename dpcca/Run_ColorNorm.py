import numpy as np
import time
import os
import cv2
import openslide
import pyvips
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

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

CLEAR_LINE='\033[0K'
UP_ARROW='\u25B2'
DOWN_ARROW='\u25BC'
SAVE_CURSOR='\033[s'
RESTORE_CURSOR='\033[u'

INDENT='\r\033[220C'
DEBUG=1


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from Estimate_W import Wfast

#################################3
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


   
def run_batch_colornorm ( slide_type, source_filename, target_filename, nstains, lamb,  dir_path, img_level, background_correction, target_i0,  Wi_target, Htarget_Rmax, config ):  
 
  # there are only two slide_types
  REFERENCE_SLIDE        = 0
  SLIDE_FOR_NORMALISING  = 1
  
  filenames=[ target_filename, source_filename ]
    
  # set up tensorflow
  
  if config is None:
    config = tf.ConfigProto( log_device_placement=False )

  g = tf.Graph()
  
  with g.as_default():
    
    Wis                  = tf.placeholder     ( tf.float32 )                                               # color basis matrix      - source
    Img                  = tf.placeholder     ( tf.float32, shape=(None,None,3)  )
    source_i_0           = tf.placeholder     ( tf.float32 )                                               # maximum intensity value - source

    s                    = tf.shape           ( Img )
    Img_vecd             = tf.reshape         ( tf.minimum    (Img,source_i_0), [s[0]*s[1], s[2]])
    V                    = tf.log             ( source_i_0+1.0) - tf.log    ( Img_vecd+1.0 )
    Wi_inv               = tf.transpose       ( tf.py_func    (np.linalg.pinv, [Wis], tf.float32))         # color basis matrix
    Hiv1                 = tf.nn.relu         ( tf.matmul     (V,Wi_inv))                                  # stain density matrix

    
    Wit                  = tf.placeholder     ( tf.float32 )                                               # color basis matrix - target
    Hiv2                 = tf.placeholder     ( tf.float32 )                                               # stain density matrix
    sav_name             = tf.placeholder     ( tf.string  )
    target_i_0           = tf.placeholder     ( tf.float32 )                                               # maximum intensity value - target
    normalisation_factor = tf.placeholder     ( tf.float32 )
    shape                = tf.placeholder     ( tf.int32   )

    Hsonorm              = Hiv2*normalisation_factor
    source_norm          = tf.cast            ( target_i_0*tf.exp((-1)*tf.reshape(tf.matmul(Hsonorm,Wit),shape)),tf.uint8)
    enc                  = tf.image.encode_png( source_norm   )
    fwrite               = tf.write_file      ( sav_name, enc )

  session=tf.Session( graph=g, config=config )


  if (DEBUG>0):
    print ( f"RUN_COLORNORM:          INFO: file to be normalized ('{BITTER_SWEET}Source{RESET}'):    {DULL_BLUE}{filenames[1:]}{RESET}",  flush=True )
    print ( f"RUN_COLORNORM:          INFO: reference file        ('{CARRIBEAN_GREEN}Target{RESET}'):    {DULL_WHITE}{filenames[0]}{RESET}",  flush=True )


  for filename in filenames:

    if background_correction:
      background_correction="background_corrected"
    else:
      background_correction="background_not_corrected"

    base_target  = os.path.basename ( filenames[0])                                                        # target.svs (stain reference)
    fname_target = os.path.splitext ( base_target)[0]                                                      # target
    base_source  = os.path.basename ( filename )                                                           # source.svs
    fname_source = os.path.splitext ( base_source)[0]                                                      # source
    save_name    = dir_path+base_source.replace(".", "_")+".png"

    tic=time.time()

    I = openslide.open_slide( filename )

    if DEBUG>0:
      print( f"RUN_COLORNORM:          INFO: no. of levels in slide   =   {MIKADO}{I.level_count}{RESET}",    flush=True )  
            
    if img_level >= I.level_count:
      print( "Level", img_level, "unavailable for image, proceeding with level 0" )
      level = 0
    else:
      level = img_level
      
    xdim,ydim = I.level_dimensions [level]
    ds        = I.level_downsamples[level]

    if slide_type==REFERENCE_SLIDE:
      print( f"RUN_COLORNORM:          INFO: {CARRIBEAN_GREEN}Target{RESET} stain separation in progress: svs image dimensions reported by openslide = {MIKADO}{xdim:,} x {ydim:,}{RESET} pixels",  flush=True )
    else:
      print( f"RUN_COLORNORM:          INFO: {BITTER_SWEET}Source{RESET} stain separation in progress: svs image dimensions reported by openslide = {MIKADO}{xdim:,} x {ydim:,}{RESET} pixels",  flush=True )
      


    # 1. DeterninE color basis matrix (wi) and whiteness intensity of background (i0)
    
    # parameters for W estimation
    num_patches = 20
    patchsize   = 1000                                                                                     # length of side of square 

    i0_default=np.array([255.,255.,255.],dtype=np.float32)

    Wi, i0 = Wfast( I, nstains,lamb, num_patches, patchsize, level, background_correction )
    
    if i0 is None:
      print( f"RUN_COLORNORM:          INFO: no white background detected" )
      i0=i0_default

    if not background_correction:
      print( f"RUN_COLORNORM:          INFO: background correction disabled, default background intensity assumed" )
      i0=i0_default

    if Wi is None:
      print( f"RUN_COLORNORM:          INFO: color basis matrix estimation failed ... image normalization will be skipped" )
      continue
      
    print( f"RUN_COLORNORM:          INFO: W estimated {INDENT}{DULL_WHITE}time since processing started: {round(time.time()-tic,3)}{RESET}",  flush=True  )
    
    Wi=Wi.astype( np.float32 )

    if slide_type==REFERENCE_SLIDE:                                                                                       # slide_type 0 is the reference file (target)
      print( f"RUN_COLORNORM:          INFO: {CARRIBEAN_GREEN}Target{RESET} color basis matrix:\n{MIKADO}{Wi}{RESET}"  )
      print( f"RUN_COLORNORM:          INFO: {CARRIBEAN_GREEN}Target{RESET} color basis matrix Size {MIKADO}{Wi.shape}{RESET}",  flush=True )
      
      Wi_target=np.transpose(Wi)
      target_i0 = i0
      print( f"RUN_COLORNORM:          INFO: {CARRIBEAN_GREEN}Target{RESET} image background 'white' intensity = {MIKADO}{i0}{RESET}",  flush=True )
    else:
      print( f"RUN_COLORNORM:          INFO: {BITTER_SWEET}Source{RESET} color basis matrix               = \n{MIKADO}{Wi}{RESET}",  flush=True )
      
      
      print( f"RUN_COLORNORM:          INFO: {BITTER_SWEET}Source{RESET} image background 'white' intensity = {MIKADO}{i0}{RESET}",  flush=True )



    # 2. Normalise colors
                                                                                                           
    _maxtf = 2550                                                                                          # "changed from 3000"
    x_max  = xdim
    y_max  = min(max(int(_maxtf*_maxtf/x_max),1),ydim)
    
    print( "RUN_COLORNORM:          INFO: large image processing..." )
    
    if slide_type==0:                                                                                         # slide_type 0 is the reference file (target)
      Hiv_target        = np.memmap('H_target',  dtype='float32', mode='w+', shape=(xdim*ydim,2))          # DOES NOT APPEAR TO BE USED
    else:                                                                                                  # any other value is a file to be colour normalised
      Hiv_source        = np.memmap('H_source',  dtype='float32', mode='w+', shape=(xdim*ydim,2))
      normalised_source = np.memmap('wsi',       dtype='uint8',   mode='w+', shape=(ydim,xdim,3))
      
    x_tiles = range(0, xdim, x_max)
    y_tiles = range(0, ydim, y_max)
    
    if DEBUG>0:
      print( f"RUN_COLORNORM:          INFO: WSI divided into {MIKADO}{len(x_tiles)}{RESET} x {MIKADO}{len(y_tiles)}{RESET}",  flush=True   )
    count=0
    if DEBUG>0:
      print( f"RUN_COLORNORM:          INFO: patch-wise H calculation in progress...",  flush=True   )
    ind=0
    perc=[]
    
    for x in x_tiles:
      for y in y_tiles:
        count+=1
        xx=min(x_max,xdim-x)
        yy=min(y_max,ydim-y)
        if (DEBUG>0):
          print ( f"RUN_COLORNORM:          INFO: patch size {MIKADO}{xx:,}{RESET} x {MIKADO}{yy:,}{RESET}     {INDENT}{DULL_WHITE}time since processing started: {round(time.time()-tic,3)}{RESET}",  flush=True )
          print( "\033[2A" )

        img=np.asarray( I.read_region(  (int(ds*x), int(ds*y) ), level, (xx,yy)), dtype=np.float32  )  [:,:,:3]    # read region using openslide

        Hiv = session.run( Hiv1, feed_dict={ Img:img, Wis:Wi, source_i_0:i0}  )
        
        if slide_type==0:                                                                                     # slide_type 0 is the reference file (target)
          Hiv_target[ind:ind+len(Hiv),:] = Hiv                                                             # Hiv_target does not appear to be used
          _Htarget_Rmax = np.ones(( nstains,),dtype=np.float32)
          for i in range(nstains):
            t = Hiv[:,i]
            _Htarget_Rmax[i] = np.percentile(t[t>0],q=99.,axis=0)
          perc.append([_Htarget_Rmax[0],_Htarget_Rmax[1]])
          ind+=len(Hiv)
          continue
        else:
          Hiv_source[ind:ind+len(Hiv),:]=Hiv
          _Hsource_Rmax = np.ones((nstains,),dtype=np.float32)
          for i in range(nstains):
            t = Hiv[:,i]
            _Hsource_Rmax[i] = np.percentile(t[t>0],q=99.,axis=0)
          perc.append([_Hsource_Rmax[0],_Hsource_Rmax[1]])
          ind+=len(Hiv)

    if slide_type==0:                                                                                         # slide_type 0 is the reference file (target)
      print( f"{CLEAR_LINE}RUN_COLORNORM:          INFO: {CARRIBEAN_GREEN}Target{RESET} H calculated {INDENT}{DULL_WHITE}time since processing started: {round(time.time()-tic,3)}{RESET}",  flush=True )
      Htarget_Rmax = np.percentile(np.array(perc),50,axis=0)
      # ~ slide_type+=1
      del Hiv_target                                                                                       # Hiv_target does not appear to be used
      ind=0
      continue
    if DEBUG>0:
      print( f"RUN_COLORNORM:          INFO: source H calculated     {INDENT}{DULL_WHITE}time since processing started: {round(time.time()-tic,3)}{RESET}",    flush=True )
    Hsource_Rmax = np.percentile(np.array(perc),50,axis=0)
    print( f"RUN_COLORNORM:          INFO: H percentile calculated     {INDENT}{DULL_WHITE}time since processing started: {round(time.time()-tic,3)}{RESET}",  flush=True )

    _normalisation_factor = np.divide( Htarget_Rmax, Hsource_Rmax).astype(np.float32)


    print( f"RUN_COLORNORM:          INFO: large image color normalization in progress...",  flush=True  )
    count   = 0
    ind     = 0
    np_max  = 1000

    x_max   = xdim
    y_max   = min(max(int(np_max*np_max/x_max),1),ydim)
    x_tiles = range(0,xdim,x_max)
    y_tiles = range(0,ydim,y_max)
    print(  f"RUN_COLORNORM:          INFO: patch-wise color normalization in progress...",  flush=True  )
    total=len(x_tiles)*len(y_tiles)
    
    prev_progress=0
    for x in x_tiles:
      for y in y_tiles:
        count+=1
        xx=min( x_max, xdim-x )
        yy=min( y_max, ydim-y )
        pix=xx*yy
        sh=np.array([yy,xx,3])
        
        # Back projection into spatial intensity space (Inverse Beer-Lambert space)

        if DEBUG>99:
          print( f"RUN_COLORNORM:          INFO: sh                =   {MIKADO}{sh}{RESET}",                                           flush=True )    
          print( f"RUN_COLORNORM:          INFO: Hiv2.shape        =   {MIKADO}{np.array(Hiv_source[ind:ind+pix,:]).shape}{RESET}",    flush=True )    
                
        normalised_source[ y:y+yy, x:x+xx,:3 ] = session.run( source_norm,   feed_dict={  Hiv2:np.array(Hiv_source[ind:ind+pix,:]),  Wit:Wi_target,  normalisation_factor:_normalisation_factor,  shape:sh,  target_i_0:target_i0  }   )

        ind+=pix
        percent=5*int(count*20/total) #nearest 5 percent
        if percent>prev_progress and percent<100:
          print( str(percent)+" percent complete...   time since processing started:",round(time.time()-tic,3) )
          print( "\033[2A" )
          prev_progress=percent


    if DEBUG>0:
      print( f"RUN_COLORNORM:          INFO: color normalization complete     {INDENT}{DULL_WHITE}time since processing started: {round(time.time()-tic,3)}{RESET}",  flush=True )            

    p = time.time()-tic
    

    #
    # NOTE regarding pyramid=True option
    #
    #    "The number of levels created by tiffsave is set by the size of the image. Larger images will need more levels"
    #
    #    A 50,000 x 50,000 pixel image will have these layers:
    #    
    #    0 = 50,000
    #    1 = 25,000
    #    2 = 12,500
    #    3 = 6,250
    #    4 = 3,125
    #    5 = etc.
    #

    save_name = f"{dir_path}/{fname_source}___NORMALISED_TO___{fname_target}___{background_correction}.tif.spcn"
    
    Q           = 90
    tile        = True
    compression = 'jpeg'
    xres        = 2000
    yres        = 2000
    pyramid     = False

    if DEBUG>0:
      print( f"RUN_COLORNORM:          INFO: normalised_source.shape  =   {MIKADO}{normalised_source.shape}{RESET}",    flush=True )    
      print( f"RUN_COLORNORM:          INFO: saving normalized image with parameters xres={MIKADO}{xres:,}{RESET}, yres={MIKADO}{yres:,}{RESET}, compression={MIKADO}{compression}{RESET}, quality={MIKADO}{Q}{RESET}, pyramid={MIKADO}{pyramid}{RESET}", flush=True )
    
    pyimg_source = numpy2vips( normalised_source )
          
    # ~ pyimg_source.tiffsave( save_name, tile=True, compression='lzw', xres=5000, yres=5000, bigtiff=True, pyramid=True, Q=100)  # ORIGINAL PARAMETERS xres and yres should be controlled to produce finer or coarser tif
    
    pyimg_source.tiffsave( save_name, tile=True, compression=compression, xres=xres, yres=yres, bigtiff=True, pyramid=pyramid, Q=Q )    # MY PARAMETERS xres and yres should be controlled to produce finer or coarser tif
    
    
    del normalised_source
    del pyimg_source

    if DEBUG>0:
      print( f"RUN_COLORNORM:          INFO: file written to: {MAGENTA}{save_name}{RESET}",  flush=True )
      display_separator()

  #### end large image processing section
    
    
    
    if os.path.exists("H_target"):
      os.remove("H_target")
    if os.path.exists("H_source"):
      os.remove("H_source")
    if os.path.exists("wsi"):
      os.remove("wsi")


  session.close()

  if slide_type==REFERENCE_SLIDE:                                                                             # return the target (reference)  maximum intensity value, colour density matrix and stain density matrix for use on slides to be stain normalised
    return target_i0, Wi_target, Htarget_Rmax
  else:
    return 0, 0, 0


def numpy2vips(a):
  
    height, width, bands = a.shape
    linear               = a.reshape ( width * height * bands )
    vi = pyvips.Image.new_from_memory( linear.data, width, height, bands, dtype_to_format[str(a.dtype)]  )
    
    return vi


def display_separator():
  print( "__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________\n" )
