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
PCT_TEST=".2"
PCT_TEST___TRAIN="0.1"
PCT_TEST___JUST_TEST="1.0"
JUST_TEST="False"
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
N_CLUSTERS="5"                                                                                          # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
METRIC="manhattan"                                                                                         
SKIP_TILING="False"                                                                                        # supported: any of the sklearn metrics
SKIP_GENERATION="False"                                                                                    
HIGHEST_CLASS_NUMBER="7"
USE_AUTOENCODER_OUTPUT="False"

while getopts a:b:c:d:e:f:g:h:i:j:k:l:m:n:n_t:o:p:q:r:s:t:u:v:w:x:z:1: option
  do
    case "${option}"
    in
    a) NN_TYPE_IMG=${OPTARG};;                                                                             
    b) BATCH_SIZE=${OPTARG};;                                                                             
    c) CASES=${OPTARG};;                                                                                   # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    d) DATASET=${OPTARG};;                                                                                 # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    e) METRIC=${OPTARG};;                                                                                  # supported: any of the sklearn metrics
    f) TILES_PER_IMAGE=${OPTARG};;                                                                         # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    g) SKIP_GENERATION=${OPTARG};;                                                                         # # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
    h) HIGHEST_CLASS_NUMBER=${OPTARG};;                                                                    # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    i) INPUT_MODE=${OPTARG};;                                                                              # supported: image, rna, image_rna
    j) JUST_TEST=${OPTARG};;                                                                               
    k) TILE_SIZE=${OPTARG};;
    l) CLUSTERING=${OPTARG};;                                                                              # supported: otsne, hdbscan, dbscan, NONE
    m) MULTIMODE=${OPTARG};;                                                                               # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
    n) NN_MODE=${OPTARG};;                                                                                 # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    o) N_EPOCHS=${OPTARG};;                                                                                # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    p) PRETRAIN=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    q) PCT_TEST___TRAIN=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    w) PCT_TEST___JUST_TEST=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    r) REGEN=${OPTARG};;                                                                                   # 'regen' or nothing. If 'regen' copy the entire dataset across from the source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT})
    s) SKIP_TILING=${OPTARG};;                                                                             # 'True'   or 'False'. If True, skip tiling (to save - potentially quite a lot of - time if the desired tiles already exists)
    t) N_ITERATIONS=${OPTARG};;                                                                            # Number of iterations. Used by clustering algorithms only (neural networks use N_EPOCHS)
    u) USE_AUTOENCODER_OUTPUT=${OPTARG};;                                                                  # 'True'   or 'False'. # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values) 
    v) DIVIDE_CASES=${OPTARG};;                                                                            # 
    x) N_CLUSTERS=${OPTARG};;                                                                              # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    z) NN_TYPE_RNA=${OPTARG};;                                                                             
    1) PCT_TEST=${OPTARG};;                                                                             
    esac
  done


./do_all.sh     -d ${DATASET}  -i ${INPUT_MODE}   -o ${N_EPOCHS} -f ${TILES_PER_IMAGE}  -k ${TILE_SIZE}   -b ${BATCH_SIZE}   -1 ${PCT_TEST___TRAIN}     -h ${HIGHEST_CLASS_NUMBER}     -s ${SKIP_TILING}   -g False   -j False  -n pre_compress   -a ${NN_TYPE_IMG} -z ${NN_TYPE_RNA}  -c NOT_A_MULTIMODE_CASE_FLAG  -v ${DIVIDE_CASES}

sleep 0.2; echo -en "\007";



./do_all.sh     -d ${DATASET}  -i ${INPUT_MODE}   -o 1           -f ${TILES_PER_IMAGE}  -k ${TILE_SIZE}   -b 256             -1 ${PCT_TEST___JUST_TEST} -h ${HIGHEST_CLASS_NUMBER}   -s True             -g True    -j True   -n pre_compress   -a ${NN_TYPE_IMG} -z ${NN_TYPE_RNA}  -c NOT_A_MULTIMODE_CASE_FLAG                       -u "True"    # For autoencoder working, the -u flag tells test mode to generate and save the embedded outputs

sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"



./do_all.sh     -d ${DATASET}  -i ${INPUT_MODE}      -t 5000        -x ${N_CLUSTERS}                                -s True             -g True             -n dlbcl_image  -u True  -c NOT_A_MULTIMODE_CASE_FLAG  -l ${CLUSTERING}                                     -u "True"                 # For autoencoder working, the -u flag tells the clusterer to emeddings as the input rather than tiles

echo -en "\007"; sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"



