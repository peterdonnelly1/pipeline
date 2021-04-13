#! /bin/bash

#  See 'conf/variables' for other paramaters.  The following only descibes the shell level user options. Many additional configuration parameters are invoked in the conf file

# -d the dataset to use, specified as a TCGA cancer type code (here, "Stomach/Adenocarcinoma" which has the code "stad")
# -i means "input" mode; "image" indicates that the input will be 3 x W x W tiles. The program itself will generate all tiles from slide image tiles as a first step.
# -a specifies the neural network to use. Here we use AE3LAYERCONV2D (and autoencoder) to generate the image embeddings.  Clustering does not use a neural network, so the -a flag is omitted
# -s if True means SKIP tiling. We use this on the clustering step because the tiles were generated at the preceeding step (line 21)
# -g if True means SKIP generation (of the pytorch dataset)
# -h highest class number to use. See conf/variables CLASS_NAMES. Here we use it to omit the 'normal' subtype (because there are fresh frozen rather than formalin fixed images) as well as the two 'NOS' categories.
# -l clustering algorithm to be used. Here, we apply the 'dbscan' clustering algorithm to the autoencoder embeddings generated in the first to runs  (line 15 and line 21)
# -u if True means use the Autoencoder output rather than the default, which is to use the pytorch datase
# -n = the network mode to use. Here, 'pre_compress initially to generate the embeddings using an Autoencoder, and then the default 'dlbcl_image' mode to run dbscan clustering on the Autoencoder output
# -c designate the subset of the cases to use, by specifying an appropriate flag. At the has to be NOT_A_MULTIMODE_CASE_FLAG for pre_compress mode (for not good reason) 
# -v means segment the cases (case files) to generate the case flags (like NOT_A_MULTIMODE_CASE_FLAG).  Only need to do this one time.

DATASET="stad"                                                                                             # possibly changed by user '-d' argument if required, but it needs an initial value
MULTIMODE="NONE"                                                                                           # possibly changed by user '-m' argument if required, but it needs an initial value
NN_MODE="dlbcl_image"                                                                                      # possibly changed by user '-n' argument if required, but it needs an initial value
DATASET="stad"                                                                                             # possibly changed by user '-i' argument if required, but it needs an initial value
INPUT_MODE="image"                                                                                         # possibly changed by user '-n' argument if required, but it needs an initial value
NN_TYPE_IMG="VGG11"                                                                                        # possibly changed by user '-a' argument if required, but it needs an initial value
CASES="ALL_ELIGIBLE_CASES"                                                                                 # possibly changed by user '-c' argument if required, but it needs an initial value
DIVIDE_CASES="False"                                                                                       # possibly changed by user '-v' argument if required, but it needs an initial value
PRETRAIN="False"        
CLUSTERING="NONE"                                                                                          # supported: 'otsne' (opentsne), 'sktsne' (sklearn t-sne), 'hdbscan', 'dbscan', 'NONE'
METRIC="manhattan"                                                                                         
SKIP_TILING="False"                                                                                        # supported: any of the sklearn metrics
SKIP_GENERATION="False"                                                                                    
SKIP_GENERATION="False"                                                                                    
HIGHEST_CLASS_NUMBER="7"
USE_AUTOENCODER_OUTPUT="False"


while getopts a:c:d:e:g:h:i:l:m:n:p:s:t:r:u:v: option
  do
    case "${option}"
    in
    a) NN_TYPE_IMG=${OPTARG};;                                                                             # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    c) CASES=${OPTARG};;                                                                                   # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    d) DATASET=${OPTARG};;                                                                                 # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    e) METRIC=${OPTARG};;                                                                                  # supported: any of the sklearn metrics
    g) SKIP_GENERATION=${OPTARG};;                                                                         # # 'True'   or 'False'. If True, skip generation of the pytorch dataset (to save time if it already exists)
    i) INPUT_MODE=${OPTARG};;                                                                              # supported: image, rna, image_rna
    h) HIGHEST_CLASS_NUMBER=${OPTARG};;                                                                    # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    l) CLUSTERING=${OPTARG};;                                                                              # supported: otsne, hdbscan, dbscan, NONE
    m) MULTIMODE=${OPTARG};;                                                                               # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
    n) NN_MODE=${OPTARG};;                                                                                 # network mode: supported: 'dlbcl_image', 'gtexv6', 'mnist', 'pre_compress', 'analyse_data'
    p) PRETRAIN=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    r) REGEN=${OPTARG};;                                                                                   # 'regen' or nothing. If 'regen' copy the entire dataset across from the source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT})
    s) SKIP_TILING=${OPTARG};;                                                                             # 'True'   or 'False'. If True, skip tiling (to save - potentially quite a lot of - time if the desired tiles already exists)
    t) JUST_TEST=${OPTARG};;                                                                               # 'test'  or nothing
    u) USE_AUTOENCODER_OUTPUT=${OPTARG};;                                                                  # 'True'   or 'False'. # if "True", use file containing auto-encoder output (which must exist, in log_dir) as input rather than the usual input (e.g. rna-seq values) 
    v) DIVIDE_CASES=${OPTARG};;                                                                            # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    esac
  done

./do_all.sh     -d ${DATASET}  -i ${INPUT_MODE}   -h 4   -s ${SKIP_TILING} -g False   -n pre_compress   -a AE3LAYERCONV2D -u False -c NOT_A_MULTIMODE_CASE_FLAG  -v True  

echo -en "\007"



./just_test.sh  -d ${DATASET}  -i ${INPUT_MODE}   -h 4   -s False          -g False   -n pre_compress   -a AE3LAYERCONV2D -u False -c NOT_A_MULTIMODE_CASE_FLAG

echo -en "\007"; sleep 0.2; echo -en "\007"

echo ${USE_AUTOENCODER_OUTPUT}

./do_all.sh     -d ${DATASET}  -i ${INPUT_MODE}   -h 4   -s True   -g False   -n dlbcl_image  -u True  -c NOT_A_MULTIMODE_CASE_FLAG  -l ${CLUSTERING} 

echo -en "\007"; sleep 0.2; echo -en "\007"; sleep 0.2; echo -en "\007"
