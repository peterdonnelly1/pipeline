#!/bin/bash

# exit if any command fails
# set -e

########################################################################################################################################################
#
# NOTES REGARDING PARAMETERS THAT ARE ALLOWED TO HAVE MORE THAN ONE VALUE
#
# More than one value can be specified for the following ...
#
#   COMMON parameters: 
#     N_SAMPLES, BATCH_SIZE, NN_OPTIMIZER, LEARNING_RATE, PCT_TEST, LABEL_SWAP_PCT,  LABEL_SWAP_PCT
#
#   IMAGE parameters: 
#     NN_TYPE_IMG, TILE_SIZE, N_TILES, RANDOM_TILES, STAIN_NORM, JITTER, MAKE_GREY_PCT
#
#   RNA parameters: 
#     NN_TYPE_RNA, HIDDEN_LAYER_NEURONS, NN_DENSE_DROPOUT_1, NN_DENSE_DROPOUT_2, GENE_DATA_NORM, GENE_DATA_TRANSFORM, EMBEDDING_DIMENSIONS, LOW_EXPRESSION_THRESHOLD
#
# If more than one value is specified for any of these, an experiment 'job' will be created and run
# The job will comprise one run for every combination of the specified parameters (Cartesian product of the parameters)
#
#    - values must be quoted & separated by spaces (not commas)  E.g. "3000 3500 4000"
#    -  values must ALWAYS be put in quotes, even if there is only a single value
#
#############################################################################################################################################################
#
# NOTES REGARDING the parameter 'HIDDEN_LAYER_ENCODER_TOPOLOGY', which it specifies number of layers and number of neurons per layers
#
#    (a)  This parameter can only be used with the DEEPDENSE, AEDEEPDENSE and TTVAE models 
#    (b)  there can only be one specification of HIDDEN_LAYER_ENCODER_TOPOLOGY per job
#
#############################################################################################################################################################

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE


MINIMUM_JOB_SIZE=2                                                                                         # Only do a box plot if the job has at least this many runs (otherwise it's a bit meaningless)
CASES_RESERVED_FOR_IMAGE_RNA=5                                                                             # number of cases to be reserved for image+rna testing. <<< HAS TO BE ABOVE ABOUT 5 FOR SOME REASON -- NO IDEA WHY ATM
USE_SAME_SEED="False"                                                                                      # set to TRUE to use the same seed every time for random numbers generation, for reproducability across runs (i.e. so that results can be more validly compared)
JUST_PROFILE="False"                                                                                       # if "True" just analyse slide/tiles then exit
DDP="False"                                                                                                # PRE_COMPRESS mode only: if "True", use PyTorch 'Distributed Data Parallel' to make use of multiple GPUs. (Works on single GPU machines, but is of no benefit and has additional overhead, so should be disabled)


MINIMUM_PERMITTED_GREYSCALE_RANGE=150                                                                      # used in 'save_svs_to_tiles' to filter out tiles that have extremely low information content. Don't set too high
MINIMUM_PERMITTED_UNIQUE_VALUES=150                                                                        # tile must have at least this many unique values or it will be assumed to be degenerate
MIN_TILE_SD=2                                                                                              # Used to cull slides with a very reduced greyscale palette such as background tiles
POINTS_TO_SAMPLE=100                                                                                       # Used for determining/culling background tiles via 'min_tile_sd', how many points to sample on a tile when making determination
MOMENTUM=0.8                                                                                               # for use with t-sne, if desired
BAR_CHART_X_LABELS="case_id"                                                                               # if "case_id" use the case id as the x-axis label for bar charts, otherwise use integer sequence
BAR_CHART_SORT_HI_LO="False"                                                                               # Some less important bar charts will be suppressed if it is set to 'False'
BAR_CHART_SHOW_ALL="False"
RENDER_CLUSTERING="True"
BOX_PLOT="True"                                                                                            # If true, do a Seaborn box plot for the job (one box plot is generated per 'job', not per 'run')
BOX_PLOT_SHOW="True"                                                                                      # If true, present the graphic using pyplot
MAX_CONSECUTIVE_LOSSES=5                                                                                   # training will stop after this many consecutive losses, regardless of nthe value of N_EPOCHS
ZOOM_OUT_MAGS="1"                                                                                          # image only. magnifications (compared to baseline magnification) to be used when selecting areas for tiling, chosen according to the probabilities contained in ZOOM_OUT_PROB
ZOOM_OUT_PROB="1"                                                                                          # image only. Chosen for magnification according to these probabilities, which must add up to 1

COLOUR_MAP="tab20"                                                                                         # see 'https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html' for allowed COLOUR_MAPs (Pastel1', 'Pastel2', 'Accent', 'Dark2' etc.)
#~ COLOUR_MAP="tab40"                                                                                      # see 'https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html' for allowed COLOUR_MAPs (Pastel1', 'Pastel2', 'Accent', 'Dark2' etc.)
CLASS_COLOURS="darkorange       lime      olive      firebrick     dodgerblue    tomato     limegreen   darkcyan  royalblue  lightseagreen    blueviolet  orangered  turquoise darkorchid"


AE_ADD_NOISE="False"
BATCH_SIZE="47"
BATCH_SIZE_TEST="36"
CASES="ALL_ELIGIBLE_CASES"                                                                                 # DON'T CHANGE THIS DEFAULT. OTHER VALUES GENERATE AND LEAVE FLAGS IN PLACE WHICH CAN CAUSE CONFUSION IF FORGOTTEN ABOUT!
CLUSTERING="NONE"                                                                                          # Supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
DATASET="stad"
DIVIDE_CASES="False"                                                                                       # 
ENCODER_ACTIVATION="none"                                                                                  # (no getopts option) activation to used with autoencoder encode state. Supported options are sigmoid, relu, tanh 
EPSILON="0.5"                                                                                         
GENE_DATA_NORM="GAUSSIAN"                                                                                  # supported options are NONE JUST_SCALE GAUSSIAN 
GENE_DATA_TRANSFORM="LOG2PLUS1"                                                                            # supported options are NONE LN LOG2 LOG2PLUS1 LOG10 LOG10PLUS1 RANKED
EMBEDDING_DIMENSIONS="100"
HIDDEN_LAYER_NEURONS="1100"
INPUT_MODE="rna"
JUST_CLUSTER="False"
JUST_TEST="False"
LABEL_SWAP_PCT=0                                                                                           # (no getopts option) Swap this percentage of truth labels to random. Used for testing.
LEARNING_RATE=".0007"
MAKE_GREY_PCT="0.0"                                                                                        # (no getopts option) Proportion of tiles to convert to greyscale. Use to check effect of color on learning. 
METRIC="manhattan"                                                                                         
MIN_CLUSTER_SIZE="10"
MULTIMODE="NONE"                                                                                           # 
NN_DENSE_DROPOUT_1="0.2"                                                                                   # 
NN_DENSE_DROPOUT_2="0.0"                                                                                   # (no getopts option) percent of neurons to be dropped out for certain layers in (AE)DENSE or (AE)DENSEPOSITIVE (parameter 2)
MODE="classify"                                                                                      # 
NN_OPTIMIZER="ADAM"                                                                                        # supported options are ADAM, ADAMAX, ADAGRAD, ADAMW, ADAMW_AMSGRAD, SPARSEADAM, ADADELTA, ASGD, RMSPROP, RPROP, SGD, LBFGS
NN_TYPE_IMG="VGG11"                                                                                        # 
NN_TYPE_RNA="DENSE"                                                                                        # 
N_CLUSTERS="5"                                                                                             # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
N_EPOCHS="150"                                                                                             # 
N_EPOCHS_TEST="1"
N_ITERATIONS="250"                                                                                         # 
N_TESTS="1"                                                                                                # (test mode only) Number of examples to put through the model when just_test=='True'
N_SAMPLES=9876
PCT_TEST=".2"
PCT_TEST___JUST_TEST="1.0"
PCT_TEST___TRAIN="0.2"
PEER_NOISE_PCT="0.0"
PERPLEXITY="30."
PRETRAIN="False"        
SKIP_GENERATION="False"                                                                                    
REGEN="False"
REPEAT=1                                                                                    
SKIP_RNA_PREPROCESSING="False"
SKIP_TILING="False"                                                                                        # supported: any of the sklearn metrics
SKIP_TRAINING="False"
SUPERGRID_SIZE="4"
TILES_PER_IMAGE="10"
TILE_SIZE="32"
USE_AUTOENCODER_OUTPUT="False"

HIDDEN_LAYER_ENCODER_TOPOLOGY="40 20"
STAIN_NORMALIZATION='NONE'

USE_UNFILTERED_DATA="True"                                                      
TARGET_GENES_REFERENCE_FILE="just_hg38_protein_coding_genes"                                               # file specifying genes to be used if USE_UNFILTERED_DATA=False 
TARGET_GENES_REFERENCE_FILE_NAME="pmcc_cancer_genes_of_interest"                                           # To allow "data_comp.sh" to pass in just the file name, so that the user does not need to specify the whole path

RANDOM_GENES_COUNT=0

LOW_EXPRESSION_THRESHOLD="0"                                                                               # Only genes with at least CUTOFF_PERCENTILE % across samples having rna-exp values above LOW_EXPRESSION_THRESHOLD will go into the analysis. Set to zero if you want to include every gene
CUTOFF_PERCENTILE=100                                                                                      # Lower CUTOFF_PERCENTILE -> more genes will be filtered out and higher LOW_EXPRESSION_THRESHOLD ->  more genes will be filtered out. Set low if you only want genes with very high correlation values

DO_COVARIANCE="False"                                                                                      # used by "analyse_data". Should covariance  calculation be performed ? (analyse_data mode only)
DO_CORRELATION="False"                                                                                     # used by "analyse_data". Should correlation calculation be performed ? (analyse_data mode only)    
A_D_USE_CUPY="True"                                                                                        # used by "analyse_data". if True, use cupy linear algrebra library rather than numpy. Only works if computer has a CUDA compatible GPU    
REMOVE_UNEXPRESSED_GENES="True"                                                                            # used by "analyse_data". create and then apply a filter to remove genes whose value is zero                                                 *for every sample*
HIGH_CORRELATION_THRESHOLD=2                                                                               # used by "analyse_data". minimum percentile value highly correlated genes to be displayed. Quite a sensitive parameter so tweak carefully
SHOW_ROWS=1000                                                                                             # used by "analyse_data". 
SHOW_COLS=100                                                                                              # used by "analyse_data". 


HIGHEST_CLASS_NUMBER=98765 

while getopts a:A:b:B:c:C:d:D:e:E:f:F:g:G:H:i:I:j:J:k:K:l:L:m:M:n:N:o:O:p:P:q:Q:r:R:s:S:t:T:u:U:v:V:w:W:x:X:y:Y:z:Z:0:1:2:3:4:5:6:7:8:9: option
  do
    case "${option}"
    in
    a) NN_TYPE_IMG=${OPTARG};;                                                                             
    A) AE_ADD_NOISE=${OPTARG};;                                                                             
    b) BATCH_SIZE=${OPTARG};;                                                                             
    B) BATCH_SIZE_TEST=${OPTARG};;                                                                             
    c) CASES=${OPTARG};;                                                                                   # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    C) MIN_CLUSTER_SIZE=${OPTARG};;
    d) DATASET=${OPTARG};;                                                                                 # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    D) TARGET_GENES_REFERENCE_FILE_NAME=${OPTARG};;
    e) EPSILON=${OPTARG};;                                                                                 # supported: any of the sklearn metrics
    E) EMBEDDING_DIMENSIONS=${OPTARG};;                                                                          # supported: in most cases, one of the sklearn metrics (but not cuda_tsne, which only supports Euclidean)
    f) TILES_PER_IMAGE=${OPTARG};;                                                                         # network mode: supported: 'classify', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    F) HIDDEN_LAYER_ENCODER_TOPOLOGY=${OPTARG};;                                                           # structure of hidden layers (DEEPDENSE, AEDEEPDENSE and TTVAE only. The number of neurons for the final layer is taken from EMBEDDING_DIMENSIONSS
    g) SKIP_GENERATION=${OPTARG};;                                                                         # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
    G) SUPERGRID_SIZE=${OPTARG};;                                                                          
    H) HIDDEN_LAYER_NEURONS=${OPTARG};;                                                                    
    i) INPUT_MODE=${OPTARG};;                                                                              
    I) USE_UNFILTERED_DATA=${OPTARG};;
    j) JUST_TEST=${OPTARG};;                                                                               
    J) JUST_CLUSTER=${OPTARG};;                                                                             
    k) REMOVE_UNEXPRESSED_GENES=${OPTARG};;
    K) HIGH_CORRELATION_THRESHOLD=${OPTARG};;
    l) CLUSTERING=${OPTARG};;                                                                              # supported: NONE, otsne, sk_tsne, cuda_tsne, sk_agglom, sk_spectral, hdbscan, dbscan
    L) LEARNING_RATE=${OPTARG};;                                                                           
    m) MULTIMODE=${OPTARG};;                                                                               # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
    M) METRIC=${OPTARG};;                                                                                  # supported: any of the sklearn metrics. Only 'euclidean' in the case of cuda_tsne
    n) MODE=${OPTARG};;                                                                                    # functional mode: supported: 'classify', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    N) SKIP_TRAINING=${OPTARG};;                                                                           
    o) N_EPOCHS=${OPTARG};;                                                                                # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    O) N_EPOCHS_TEST=${OPTARG};;                                                                           
    p) PERPLEXITY=${OPTARG};;                                                                              
    P) PRETRAIN=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    q) PCT_TEST___TRAIN=${OPTARG};;                                                                        
    Q) SHOW_COLS=${OPTARG};;
    r) REGEN=${OPTARG};;                                                                                   # True or False. If 'True' copies either the entire dataset or just rna-seq files across from the applicable source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT}), depending on the value of INPUT_MODE (if INPUT_MODE is rna, assumption is that uer probably doesn't want to copy across image files, which can take a long time)
    R) REPEAT=${OPTARG};;                                                                                  # number of times to repeat the experiment
    s) SKIP_TILING=${OPTARG};;                                                                             # 'True'   or 'False'. If True,   skip tiling (to save - potentially quite a lot of time - if the desired tiles already exists)
    S) N_SAMPLES=${OPTARG};;                                                                             
    t) N_ITERATIONS=${OPTARG};;                                                                            # Number of iterations. Used by clustering algorithms only (neural networks use N_EPOCHS)
    T) TILE_SIZE=${OPTARG};;
    u) USE_AUTOENCODER_OUTPUT=${OPTARG};;                                                                  # 'True'   or 'False'. if 'True', use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values) 
    U) SHOW_COLS=${OPTARG};; 
    v) DIVIDE_CASES=${OPTARG};;                                                                             
    V) DO_COVARIANCE=${OPTARG};;
    w) PCT_TEST___JUST_TEST=${OPTARG};;                                                                    
    W) DO_CORRELATION=${OPTARG};;
    x) N_CLUSTERS=${OPTARG};;                                                                              # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    X) SKIP_RNA_PREPROCESSING=${OPTARG};;                                                                  
    y) A_D_USE_CUPY=${OPTARG};;
    Y) TARGET_GENES_REFERENCE_FILE=${OPTARG};;    
    z) NN_TYPE_RNA=${OPTARG};;                                                                             
    Z) N_TESTS=${OPTARG};;                                                                             
    0) STAIN_NORMALIZATION=${OPTARG};;                                                                             
    1) PCT_TEST=${OPTARG};;                                                                             
    2) RANDOM_GENES_COUNT=${OPTARG};;                                                                             
    3) PEER_NOISE_PCT=${OPTARG};;                                                                      
    4) NN_OPTIMIZER=${OPTARG};; 
    5) GENE_DATA_TRANSFORM=${OPTARG};; 
    6) GENE_DATA_NORM=${OPTARG};; 
    7) NN_DENSE_DROPOUT_1=${OPTARG};; 
    8) LOW_EXPRESSION_THRESHOLD=${OPTARG};; 
    9) CUTOFF_PERCENTILE=${OPTARG};; 
    esac
  done
  
source conf/variables.sh
  
# The SKIP_TRAINING and JUST_CLUSTER flags are used to allow user to control this script. They aren't passed into any programs.


if [[ ${JUST_CLUSTER} != "True" ]]                                                                         # Skip Autoencoder training and testing if if JUST_CLUSTER flag is true                                                

  then

  if [[ ${SKIP_TRAINING} != "True" ]]
  
    # Do if SKIP_TRAINING flag is False. Trains the Autoencoder.
    
    then
    
      rm logs/lowest_loss_ae_model.pt  > /dev/null 2>&1
      
      ./do_all.sh   -d ${DATASET}                -i ${INPUT_MODE}      -S ${N_SAMPLES}               -o ${N_EPOCHS}          -f ${TILES_PER_IMAGE}    -T ${TILE_SIZE}       -b ${BATCH_SIZE}           \
                    -1 ${PCT_TEST___TRAIN}       -s ${SKIP_TILING}     -X ${SKIP_RNA_PREPROCESSING}  -g ${SKIP_GENERATION}   -j False                 -a ${NN_TYPE_IMG}     -z ${NN_TYPE_RNA}          \
                    -E ${EMBEDDING_DIMENSIONS}   -A ${AE_ADD_NOISE}    -u False                      -v ${DIVIDE_CASES}      -r ${REGEN}
      
      sleep 0.2; echo -en "\007";
  
  fi

     
  # Pushes feature vectors produced during training (the feature vector file MUST exist) through the best model produced during training
  # Key flags: -u True means "USE_AUTOENCODER_OUTPUT"   (that is: the embeddings we just generated and not raw inputs)
  #            -j True means "JUST_TEST"                (that is: use only held out test examples and push them through the optimised/saved model produced during training)
  #            -g False                                 we do need to generate a dataset in this run. Because this is an autoencoder test run, where the point is to generate reduced dimensionality embeddings, (and not a classifier test run), all *training* examples will be pushed through the model (internal logic takes care of this)
   
      rm logs/ae_output_features.pt  > /dev/null 2>&1
 
      ./do_all.sh  -d ${DATASET}                -i ${INPUT_MODE}      -S ${N_SAMPLES}               -o ${N_EPOCHS_TEST}     -f ${TILES_PER_IMAGE}    -T ${TILE_SIZE}       -b ${BATCH_SIZE_TEST}      \
                   -1 ${PCT_TEST___JUST_TEST}   -s True               -X True                       -g False                -j True                  -a ${NN_TYPE_IMG}     -z ${NN_TYPE_RNA}          \
                   -E ${EMBEDDING_DIMENSIONS}   -A False              -u False


sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"

fi






# Perform clustering. Always executed.

if [[ ${CLUSTERING} == "all" ]]

  then
  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne           -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l cuda_tsne         -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral       -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom         -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l dbscan            -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l h_dbscan          -u ${USE_AUTOENCODER_OUTPUT}  

elif [[ ${CLUSTERING} == "sk_spectral" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral  -p 0.1 -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral  -p 1   -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral  -p 7   -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral  -p 10  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral  -p 20  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral  -p 30  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_spectral  -p 50  -u ${USE_AUTOENCODER_OUTPUT}  
  
elif [[ ${CLUSTERING} == "sk_agglom" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom  -p 0.1 -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom  -p 1   -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom  -p 7   -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom  -p 10  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom  -p 20  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom  -p 30  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_agglom  -p 50  -u ${USE_AUTOENCODER_OUTPUT} 

elif [[ ${CLUSTERING} == "sk_tsne" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne  -p 0.1 -u ${USE_AUTOENCODER_OUTPUT}
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne  -p 1   -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne  -p 7   -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne  -p 10  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne  -p 20  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne  -p 30  -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l sk_tsne  -p 50  -u ${USE_AUTOENCODER_OUTPUT}  

elif [[ ${CLUSTERING} == "cuda_tsne" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l cuda_tsne  -p "10 30 400 500"  -G ${SUPERGRID_SIZE} -u ${USE_AUTOENCODER_OUTPUT}


elif [[ ${CLUSTERING} == "dbscan" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l dbscan  -e 0.1    -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l dbscan  -e 0.7    -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l dbscan  -e 1      -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l dbscan  -e 7      -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l dbscan  -e 10     -u ${USE_AUTOENCODER_OUTPUT}  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l dbscan  -e 17     -u ${USE_AUTOENCODER_OUTPUT}  
  
elif [[ ${CLUSTERING} == "h_dbscan" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -X True -s True  -g True  -n classify  -c ${CASES}  -l h_dbscan -C 10    -u ${USE_AUTOENCODER_OUTPUT}  
  
    
fi

sleep 0.2
echo -en "\007"; sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"
