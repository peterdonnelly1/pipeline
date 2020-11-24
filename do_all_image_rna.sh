#!/bin/bash

# exit if any command fails
# set -e

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE


cls
./do_all.sh     -d stad -i image
./just_test.sh  -d stad -i image -m image_rna
./do_all.sh    -d stad -i rna
./just_test.sh -d stad -i rna   -m image_rna

./do_all.sh    -d stad -i image_rna
#./just_test.sh -d stad -i rna
