#!/bin/bash

# Train and the test (with held out samples) for TCGA stomach cancer whole slide images
#
#
# PREQEQUISITE:
#
# The image dataset must first be downloaded (from NIH TCGA data repository) and pre-processed. Use the following command.  
#
#    ./gdc-fetch.sh stad TCGA-stad_case_filter GLOBAL_file_filter_SVS
#
# Don't use the TCGA download tool - the downloaded files have to extensively pre-processed, and gdc_fetch does that in addition to downloading them
#
#
# It will extract 22 tiles for each slide, each of tile size 64x64 pixels, and perform 10 training epochs so don't expect much generalisation) using a mini-batch size of 16
# "-A 4" means that Only cancer subtypes 0,1,2,3 and 4, corresponding to tubular, intest_nos, stomach_nos, diffuse, mucinous, will be used.  Subtypes 5 and 6 (, signet, papillary) which have a very small number of examples, are excluded.
#
# A lot of console output will be seen. This is deliberate, to enable you to see what's going on. You can suppress console output using the DEBUG_LEVEL_xxx parameters in 'do_all.sh'
#
#
# OUTPUTS:
#
# Tensorboard outputs are visible via your browser at:                               http://localhost:6006/
# provided you first start the tensorboard server using a command similar to this:   tensorboard --logdir=classi/runs --samples_per_plugin images=0 --reload_interval=10
# *** Note the two tabs in Tensorboard: 'scalar' and 'graphics'

# 
# Log file outputs (including some graphical outputs that aren't visible via tensorboard) will be found in '<BASE_DIR>/logs'
#

# defaults for use if user doesn't set an option

HIGHEST_CLASS_NUMBER=3
BATCH_SIZE=16
DIVIDE_CASES="True"
LEARNING_RATE=.00001
MAKE_BALANCED="level_up"
N_EPOCHS=10
N_SAMPLES=20
NN_TYPE_IMG="VGG11"
#~ REGEN="True"
REGEN="False"
REPEAT=8
SKIP_GENERATION="False"                                                                                    
SKIP_TILING="False"
TILE_SIZE="64"
TILES_PER_IMAGE="10"

while getopts a:A:b:f:g:h:L:o:r:R:s:S:T:v: option     # weird choice of letters is because they're the same as used in do_all.sh, where they are a few out of dozens of parameters
  do
    case "${option}"
    in
    a) NN_TYPE_IMG=${OPTARG};; 
    A) HIGHEST_CLASS_NUMBER=${OPTARG};;     
    b) BATCH_SIZE=${OPTARG};;
    f) TILES_PER_IMAGE=${OPTARG};; 
    g) SKIP_GENERATION=${OPTARG};;                                                                           
    h) MAKE_BALANCED=${OPTARG};;                                                                             
    L) LEARNING_RATE=${OPTARG};;                                                                           
    o) N_EPOCHS=${OPTARG};;                
    r) REGEN=${OPTARG};;                   
    R) REPEAT=${OPTARG};;
    s) SKIP_TILING=${OPTARG};;
    S) N_SAMPLES=${OPTARG};;
    T) TILE_SIZE=${OPTARG};;
    v) DIVIDE_CASES=${OPTARG};;
    esac
  done
  

echo ""
echo "========================================================================================================================================================================"
echo "TRAINING RUN"
echo "========================================================================================================================================================================"
echo ""

./do_all.sh -d stad  -i image -f ${TILES_PER_IMAGE} -a ${NN_TYPE_IMG}  -b ${BATCH_SIZE}  -T ${TILE_SIZE}  -o ${N_EPOCHS}  -c UNIMODE_CASE  -A ${HIGHEST_CLASS_NUMBER}  -v ${DIVIDE_CASES} -r ${REGEN}


sleep 5

# The test run will use the best model produced during training on the held out 20% test examples (20% is the default. Change it with the PCT_TEST option if you want. Eg. for 10%, use '-1 .1'

echo ""
echo "========================================================================================================================================================================"
echo "TEST RUN"
echo "========================================================================================================================================================================"
echo ""

./do_all.sh -d stad  -i image -f -${TILES_PER_IMAGE} -a ${NN_TYPE_IMG}  -b ${BATCH_SIZE}  -T ${TILE_SIZE}       -c UNIMODE_CASE  -A ${HIGHEST_CLASS_NUMBER}   -S ${N_SAMPLES}   -j True 


#
# Meaning of options:
#
# DATASET              -d   Which named dataset to use (eg. ‘stad’). Must exist locally.
# INPUT_MODE           -i   Type of input data: allowed values are image, rna and image_rna (always lower case)
# TILES_PER_IMAGE      -f   Number of tiles to extract from each Whole Slide Image
# NN_TYPE_IMG          -a   neural network model to be used for Image processing (e.g. VGG16, INCEPT4, RESNET152)
# BATCH_SIZE           -b   Number of examples per batch
# TILE_SIZE            -T   Size of tile to be used (pixels x pixels)
# N_EPOCHS             -o   Number of iterations to use during training. Each iteration pushes all N_SAMPLES (or every sample if N_SAMPLES is not specified) through the neural network, in mini-batches of size BATCH_SIZE
# CASES                -c   Specify a named subset of cases to be used for a particular experiment in the same or subsequent experiments. Used in conjunction with the DIVIDE_CASES parameter
# HIGHEST_CLASS_NUMBER -A   Include only classes (cancer subtypes) with class numbers in the range 0 through HIGHEST_CLASS_NUMBER. 
#                           This parameter may be used to instruct the platform to ignore certain classes – for example “Not Otherwise Specified” cancer types. 
#                           This requires planning: classes which the experimenter may later wish to ignore in certain experiments should be given the highest class numbers.
#                           Class numbers are specified in the applicable master spreadsheet, in this case 'stad_mapping_file_MASTER.csv' in folder '<BASE_DIR>/global/stad_global'
# DIVIDE_CASES         -v   Divide the working directory of cases into pre-defined subsets, which may then or subsequently be selected for use in an experiment by means of the ‘CASES flags’
# PCT_TEST             -1   (Numeral 1) Proportion of examples to be held out for testing. (Number in the range 0.0 to 1.0, despite the name)





