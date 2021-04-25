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
JUST_TEST="False"
JUST_CLUSTER="False"
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
SKIP_TILING="False"                                                                                        # supported: any of the sklearn metrics
SKIP_GENERATION="False"                                                                                    
HIGHEST_CLASS_NUMBER="7"
USE_AUTOENCODER_OUTPUT="True"
PEER_NOISE_PERUNIT="0.0"
MAKE_GREY_PERUNIT="0.0"
N_SAMPLES="100"
MIN_CLUSTER_SIZE="10"
SKIP_TRAINING="False"
PERPLEXITY="30"

while getopts a:b:B:c:C:d:e:f:g:h:i:j:k:l:m:M:n:N:o:p:P:q:r:s:S:t:T:u:v:w:x:z:1:J:3:4: option
  do
    case "${option}"
    in
    a) NN_TYPE_IMG=${OPTARG};;                                                                             
    b) BATCH_SIZE=${OPTARG};;                                                                             
    B) BATCH_SIZE_TEST=${OPTARG};;                                                                             
    c) CASES=${OPTARG};;                                                                                   # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    C) MIN_CLUSTER_SIZE=${OPTARG};;
    d) DATASET=${OPTARG};;                                                                                 # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    e) EPSILON=${OPTARG};;                                                                                  # supported: any of the sklearn metrics
    f) TILES_PER_IMAGE=${OPTARG};;                                                                         # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    g) SKIP_GENERATION=${OPTARG};;                                                                         # # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
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
    z) NN_TYPE_RNA=${OPTARG};;                                                                             
    1) PCT_TEST=${OPTARG};;                                                                             
    J) JUST_CLUSTER=${OPTARG};;                                                                             
    3) PEER_NOISE_PERUNIT=${OPTARG};;                                                                             
    4) MAKE_GREY_PERUNIT=${OPTARG};;                                                                             
    S) N_SAMPLES=${OPTARG};;                                                                             
    esac
  done
  

if [[ ${JUST_CLUSTER} != "True" ]]

  then

  if [[ ${SKIP_TRAINING} != "True" ]]
  
    then
    
      ./do_all.sh     -d ${DATASET}  -i ${INPUT_MODE}   -S ${N_SAMPLES}  -o ${N_EPOCHS} -f ${TILES_PER_IMAGE}  -T ${TILE_SIZE}   -b ${BATCH_SIZE}       -1 ${PCT_TEST___TRAIN}      -h ${HIGHEST_CLASS_NUMBER}   -s ${SKIP_TILING}   -g False   -j False  -n pre_compress   -a ${NN_TYPE_IMG} -z ${NN_TYPE_RNA}  -c NOT_A_MULTIMODE_CASE_FLAG  -v ${DIVIDE_CASES} -3 ${PEER_NOISE_PERUNIT} -4 ${MAKE_GREY_PERUNIT}
      
      sleep 0.2; echo -en "\007";
  
  fi
  
  ./do_all.sh     -d ${DATASET}  -i ${INPUT_MODE}   -S ${N_SAMPLES}  -o 1               -f ${TILES_PER_IMAGE}  -T ${TILE_SIZE}   -b ${BATCH_SIZE_TEST}  -1 ${PCT_TEST___JUST_TEST}  -h ${HIGHEST_CLASS_NUMBER}   -s True             -g True    -j True   -n pre_compress   -a ${NN_TYPE_IMG} -z ${NN_TYPE_RNA}  -c NOT_A_MULTIMODE_CASE_FLAG                       -u "True"    # For autoencoder working, the -u flag tells test mode to generate and save the embedded outputs
  
  sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"

fi


if [[ ${CLUSTERING} == "all" ]]

  then
  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne           -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_spectral       -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_agglom         -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan            -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan          -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the -u flag tells the clusterer to emeddings as the input rather than tiles
  
elif [[ ${CLUSTERING} == "sk_tsne" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "1"   -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "2"   -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "3"   -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "4"   -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "5"   -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "6"   -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "7"   -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "8"  -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "9"  -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l sk_tsne  -p "10"  -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
  
elif [[ ${CLUSTERING} == "dbscan" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan  -e 0.1    -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the   -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan  -e 0.7    -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the   -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan  -e 1      -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the   -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan  -e 7      -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the   -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan  -e 10     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the   -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l dbscan  -e 17     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the   -u flag tells the clusterer to emeddings as the input rather than tiles
  
elif [[ ${CLUSTERING} == "h_dbscan" ]]

  then

    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 2     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 3     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 4     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 5     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 6     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 7     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 8     -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 10    -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True  -n dlbcl_image  -c NOT_A_MULTIMODE_CASE_FLAG  -l h_dbscan -C 15    -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the  -u flag tells the clusterer to emeddings as the input rather than tiles
  
  else
  
    ./do_all.sh -d ${DATASET}  -i ${INPUT_MODE}  -t 5000  -x ${N_CLUSTERS}  -s True  -g True   -n dlbcl_image    -c NOT_A_MULTIMODE_CASE_FLAG  -l ${CLUSTERING}          -u ${USE_AUTOENCODER_OUTPUT}  # For autoencoder working, the -u flag tells the clusterer to emeddings as the input rather than tiles
    
    echo -en "\007"; sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"

fi


