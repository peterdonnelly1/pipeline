#!/bin/bash

# exit if any command fails
# set -e

# all code used for spcn stain normalisation is due to:
#
#     D. Anand, G. Ramakrishnan, A. Sethi
#     and is an implementation of spcn described in their paper:
#        "Fast GPU-enabled color normalization for digital pathology"
#        International Conference on Systems, Signals and Image Processing, Osijek, Croatia (2019), pp. 219-224
#
#
# the code (trivially modified by me) may be found here:
#     https://github.com/goutham7r/spcn
#
#
# their spcn implementation is in turn an optimised, GPU based version of the original spcn algorithm which was created by:
#     Vahadane, A. et al.
#     as described in their paper: 
#         "Structure-preserving color normalization and sparse stain separation for histological images" 
#         IEEE Trans. Med. Imaging. 35, 1962–1971 (2016).
#
#
#
#
#  Notes:
#   spcn stain normalisation takes place outside of the CLASSI framework
#   it is run prior to using CLASSI, and creates a new, stain normalised version of each SVS file it finds in the working data directory and places it in the same directory
#   when the spcn option is selected in CLASSI, (-0 spcn), these normalised files are used rather than the SVS files    
#   
#   further:
#   1`characterising the reference file typically takes a long time - perhaps half an hour
#   2`stain normalisation of svs files, which are typically very large, likewise can take a long time - e.g 10-30 minutes per image
#   2 the program performing spcn stain normalisation uses tensorflow rather than pytorch
#   3 since it uses some of the same libraries as CLASSI, but at different version levels, it should be run in a different virtual environment to CLASSI (I use conda)
#   4 here are the dependencies:
#        python              3.6.13
#        tensorflow          1.15.0
#        numpy               1.19.5
#        pyvips              2.1.8
#        openslide           3.4.1
#        openslide-python    1.1.2
#        pillow              8.1.2
#        spams               2.6.1
#        scikit-learn        0.23.2


  
BASE_DIR=/home/peter/git/pipeline                                       # root directory for everything (shell scripts, code, datasets, logs ...)
APPLICATION_DIR=classi
DATA_ROOT=working_data                                                  # holds working copy of the dataset. Cleaned each time if "do_all" script used. Fully regenerated if "regen" option specified. Not cleaned if "just_.. or _only_ scripts used (to save time. Regeneration i particular can take a lot of time)
DATA_DIR=${BASE_DIR}/${DATA_ROOT}

cd ${APPLICATION_DIR}

python normalise_stain.py  --data_dir ${DATA_DIR}  --reference_file "46874009-4b32-43ff-a36a-236a3ca28fef_1/TCGA-VQ-AA69-01Z-00-DX1.1EEF2AD7-6470-44B5-9E87-E77AE47262F0.svs"
