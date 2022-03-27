#!/bin/bash
set -x
  
#  -r    = regenerate the dataset                (this only needs to be used when you change dataset     )
#  -X    = suppress RNA-Seq file pre-processing (this only needs to be used one time for a given dataset )
#  -g    = suppress pytotch dataset generation  (this only needs to be used one time for for a given dataset and number of samples )
#  -R 8  = repeats the run 8 times              (running the experiment > MINIMUM_JOB_SIZE times will generate a box plot that shows the spread of performance over the runs )
#
#  all other parameters use the defaults from do_all.sh
# 

./do_all.sh -d kidn -i rna -L  ".00001 .00003 .00007 .0001 .0003 .0007 .001 .003 .007 .01"                                                                     -r True

./do_all.sh -d kidn -i rna -L .00001 -H "300 600 900 1200 1500 1800 2100 2400 2700 3000"                                              -X True  -g True

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 ".00 .05 .10 .15 .20 .25 .30 .35 .40 .45"                                             -X True  -g True

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 .20                                                                                   -X True  -g True  -R 8

./do_all.sh -d kidn -i rna -L .00001  H 2100 -7 .20 -I False -D just_hg38_protein_coding_genes                                        -X True  -g True  -R 8

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 .20 -I False -D pmcc_cancer_genes_of_interest                                         -X True  -g True  -R 8

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 .20 -I False -D KIDN_genes_of_interest                                                -X True  -g True  -R 8

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 .20   -8 "0. .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0 1.1 1.2"   -9 "80 90 95"                  -X True  -g True 

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 .20   -8  0.90  -9 90                                                                 -X True  -g True  -R 8

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 .20   -4 "ADAM ADAMAX ADAGRAD ADAMW ADAMW_AMSGRAD ADADELTA ASGD RMSPROP RPROP SGD"    -X True  -g True

./do_all.sh -d sarc -i rna -L .00001 -H 2100 -7 .20                                                                                                     -R 8   -r True

./do_all.sh -d 0008 -i rna -L .00001 -H 2100 -7 .20                                                                                                     -R 8   -r True
