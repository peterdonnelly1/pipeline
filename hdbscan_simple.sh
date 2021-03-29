#! /bin/bash

./do_all.sh             -d stad  -i image                      -c NOT_A_MULTIMODE_CASE_FLAG
./just_test.sh          -d stad  -i image                      -c NOT_A_MULTIMODE_CASE_FLAG
python hdbscan_simple.py --input_file ae_output_features.pt --metric hamming
