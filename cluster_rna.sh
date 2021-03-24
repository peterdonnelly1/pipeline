#! /bin/bash

./do_all.sh         -d stad  -i rna                      -c NOT_A_MULTIMODE_CASE_FLAG  -v True
./just_test.sh      -d stad  -i rna
python otsne_adv.py

