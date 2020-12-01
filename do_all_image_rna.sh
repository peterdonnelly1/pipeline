#!/bin/bash

# exit if any command fails
# set -e

export MKL_DEBUG_CPU_TYPE=5
export KMP_WARNINGS=FALSE


./do_all.sh     -d stad  -i image                    -c DESIGNATED_UNIMODE_CASE_FLAG
./just_test.sh  -d stad  -i image      -m image_rna  -c DESIGNATED_UNIMODE_CASE_FLAG
./do_all.sh     -d stad  -i rna                      -c DESIGNATED_UNIMODE_CASE_FLAG
./just_test.sh  -d stad  -i rna        -m image_rna  -c DESIGNATED_UNIMODE_CASE_FLAG

./do_all.sh     -d stad  -i image_rna                -c DESIGNATED_UNIMODE_CASE_FLAG

./just_test.sh  -d stad  -i image      -m image_rna  -c DESIGNATED_MULTIMODE_CASE_FLAG
./just_test.sh  -d stad  -i rna        -m image_rna  -c DESIGNATED_MULTIMODE_CASE_FLAG
./just_test.sh  -d stad  -i image_rna  -m image_rna  -c DESIGNATED_MULTIMODE_CASE_FLAG

