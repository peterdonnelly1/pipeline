#! /bin/bash

#! /bin/bash

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE

MULTIMODE="NONE"                                                                                           # possibly changed by user '-m' argument if required, but it needs an initial value
CASES="ALL_ELIGIBLE_CASES"                                                                                 # possibly changed by user '-c' argument if required, but it needs an initial value
DIVIDE_CASES="False"                                                                                       # possibly changed by user '-v' argument if required, but it needs an initial value
PRETRAIN="False"

while getopts c:d:i:m:p:t:r:v: option
  do
    case "${option}"
    in
    c) CASES=${OPTARG};;                                                                                   # (Flagged) subset of cases to use. At the moment: 'ALL_ELIGIBLE', 'DESIGNATED_UNIMODE_CASES' or 'DESIGNATED_MULTIMODE_CASES'. See user settings DIVIDE_CASES and CASES_RESERVED_FOR_IMAGE_RNA
    d) DATASET=${OPTARG};;                                                                                 # TCGA cancer class abbreviation: stad, tcl, dlbcl, thym ...
    i) INPUT_MODE=${OPTARG};;                                                                              # supported: image, rna, image_rna
    m) MULTIMODE=${OPTARG};;                                                                               # multimode: supported:  image_rna (use only cases that have matched image and rna examples (test mode only)
    p) PRETRAIN=${OPTARG};;                                                                                # pre-train: exactly the same as training mode, but pre-trained model will be used rather than starting with random weights
    t) JUST_TEST=${OPTARG};;                                                                               # 'test'  or nothing
    r) REGEN=${OPTARG};;                                                                                   # 'regen' or nothing. If 'regen' copy the entire dataset across from the source directory (e.g. 'stad') to the working dataset directory (${DATA_ROOT})
    v) DIVIDE_CASES=${OPTARG};;                                                                            # 'yes'   or nothing. If 'true'  carve out (by flagging) CASES_RESERVED_FOR_IMAGE_RNA and CASES_RESERVED_FOR_IMAGE_RNA_TESTING. 
    esac
  done
  
#~ ./do_all.sh             -d stad  -i image                      -c NOT_A_MULTIMODE_CASE_FLAG
#~ ./just_test.sh          -d stad  -i image                      -c NOT_A_MULTIMODE_CASE_FLAG

source conf/variables.sh ${DATASET}

python otsne_simple.py --input_file ae_output_features.pt --dataset ${DATASET} --class_names ${CLASS_NAMES}


