#!/bin/bash

# exit if any command fails
# set -e

echo ""
echo ""

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

DATASET="stad"
INPUT_MODE="image"
BATCH_SIZE="36"
BATCH_SIZE_TEST="36"
PCT_TEST=".2"
PCT_TEST___TRAIN="0.1"
PCT_TEST___JUST_TEST="1.0"
MULTIMODE="NONE"                                                                                           # possibly changed by user '-m' argument if required, but it needs an initial value
TILES_PER_IMAGE="10"
TILE_SIZE="32"
N_EPOCHS="4"                                                                                               # possibly changed by user '-n' argument if required, but it needs an initial value
N_ITERATIONS="250"                                                                                         # possibly changed by user '-n' argument if required, but it needs an initial value
NN_MODE="dlbcl_image"                                                                                      # possibly changed by user '-n' argument if required, but it needs an initial value
NN_TYPE_IMG="AE3LAYERCONV2D"                                                                               # possibly changed by user '-a' argument if required, but it needs an initial value
NN_TYPE_RNA="DENSE"                                                                                        # possibly changed by user '-a' argument if required, but it needs an initial value
CASES="ALL_ELIGIBLE_CASES"                                                                                 # possibly changed by user '-c' argument if required, but it needs an initial value
DIVIDE_CASES="False"                                                                                       # possibly changed by user '-v' argument if required, but it needs an initial value
PRETRAIN="False"        
CLUSTERING="NONE"                                                                                          # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
N_CLUSTERS="5"                                                                                             # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
METRIC="manhattan"                                                                                         
EPSILON="0.5"                                                                                         
HIGHEST_CLASS_NUMBER="7"
USE_AUTOENCODER_OUTPUT="True"
PEER_NOISE_PERUNIT="0.0"
MAKE_GREY_PERUNIT="0.0"
N_SAMPLES="310"
MIN_CLUSTER_SIZE="10"
PERPLEXITY="7 10 20 30 40 60 80 "
AE_ADD_NOISE="False"
SKIP_TRAINING="False"
SKIP_TILING="False"                                                                                        # supported: any of the sklearn metrics
SKIP_GENERATION="False"                                                                                    
JUST_TEST="False"
JUST_CLUSTER="False"
SKIP_RNA_PREPROCESSING="False"
GENE_EMBED_DIM="37"
N_EPOCHS_TEST="1"
SUPERGRID_SIZE="2"

while getopts a:A:b:B:c:C:d:D:e:E:f:g:G:h:i:j:k:l:m:M:n:N:o:O:p:P:q:r:s:S:t:T:u:v:w:x:X:z:1:J:3:4: option
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
    e) EPSILON=${OPTARG};;                                                                                 # supported: any of the sklearn metrics
    E) GENE_EMBED_DIM=${OPTARG};;                                                                                 # supported: any of the sklearn metrics
    f) TILES_PER_IMAGE=${OPTARG};;                                                                         # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    g) SKIP_GENERATION=${OPTARG};;                                                                         # # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
    G) SUPERGRID_SIZE=${OPTARG};;                                                                         # # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
    h) HIGHEST_CLASS_NUMBER=${OPTARG};;                                                                    # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    i) INPUT_MODE=${OPTARG};;                                                                              # supported: image, rna, image_rna
    j) JUST_TEST=${OPTARG};;                                                                               
    T) TILE_SIZE=${OPTARG};;
    l) CLUSTERING=${OPTARG};;                                                                              # supported: otsne, hdbscan, dbscan, NONE
    M) METRIC=${OPTARG};;                                                                                  # supported: any of the sklearn metrics
    m) MULTIMODE=${OPTARG};;                                                                               # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
    n) NN_MODE=${OPTARG};;                                                                                 # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    N) SKIP_TRAINING=${OPTARG};;                                                                           # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    o) N_EPOCHS=${OPTARG};;                                                                                # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    O) N_EPOCHS_TEST=${OPTARG};;                                                                                # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    p) PERPLEXITY=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    P) PRETRAIN=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    q) PCT_TEST___TRAIN=${OPTARG};;                                                                        # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    w) PCT_TEST___JUST_TEST=${OPTARG};;                                                                    # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    r) REGEN=${OPTARG};;                                                                                   # 'regen' or nothing. If 'regen' copy the entire dataset across from the source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT})
    s) SKIP_TILING=${OPTARG};;                                                                             # 'True'   or 'False'. If True, skip tiling (to save - potentially quite a lot of - time if the desired tiles already exists)
    t) N_ITERATIONS=${OPTARG};;                                                                            # Number of iterations. Used by clustering algorithms only (neural networks use N_EPOCHS)
    u) USE_AUTOENCODER_OUTPUT=${OPTARG};;                                                                  # 'True'   or 'False'. # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values) 
    v) DIVIDE_CASES=${OPTARG};;                                                                            # 
    x) N_CLUSTERS=${OPTARG};;                                                                              # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    X) SKIP_RNA_PREPROCESSING=${OPTARG};;                                                                  
    z) NN_TYPE_RNA=${OPTARG};;                                                                             
    1) PCT_TEST=${OPTARG};;                                                                             
    J) JUST_CLUSTER=${OPTARG};;                                                                             
    3) PEER_NOISE_PERUNIT=${OPTARG};;                                                                             
    4) MAKE_GREY_PERUNIT=${OPTARG};;                                                                             
    S) N_SAMPLES=${OPTARG};;                                                                             
    esac
  done
  

source conf/variables.sh ${DATASET}

cd ${NN_APPLICATION_PATH}
  
CUDA_LAUNCH_BLOCKING=1 python cuda_tsne_launcher.py  --use_autoencoder_output ${USE_AUTOENCODER_OUTPUT} --pct_test ${PCT_TEST} --n_iterations ${N_ITERATIONS} --perplexity ${PERPLEXITY} \
--metric ${METRIC} --input_mode ${INPUT_MODE} --dataset ${DATASET} --class_names ${CLASS_NAMES}  --supergrid_size ${SUPERGRID_SIZE}
    
    echo -en "\007"; sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"
