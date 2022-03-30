#!/bin/bash

#  This script recapitulates Experiment 1, including all of its sub-experiments


# defaults for use if user doesn't set an option

BATCH_SIZE=59
HIDDEN_LAYER_NEURONS=2100
LEARNING_RATE=.00001
N_EPOCHS=150
NN_DENSE_DROPOUT_1=.20
REPEAT=8

while getopts H:L:o:R:7: option                                                                            # weird choice of letters is because they're the same as used in do_all.sh, where they are 5 out of dozens of parameters
  do
    case "${option}"
    in
    H) HIDDEN_LAYER_NEURONS=${OPTARG};;                                                                    
    L) LEARNING_RATE=${OPTARG};;                                                                           
    o) N_EPOCHS=${OPTARG};;                                                                                # Use this parameter to omit classes above HIGHEST_CLASS_NUMBER. Classes are contiguous, start at ZERO, and are in the order given by CLASS_NAMES in conf/variables. Can only omit cases from the top (e.g. 'normal' has the highest class number for 'stad' - see conf/variables). Currently only implemented for unimode/image (not implemented for rna_seq)
    R) REPEAT=${OPTARG};; 
    7) NN_DENSE_DROPOUT_1=${OPTARG};; 
    esac
  done


#  Meaning of Parameters (they are consumed by do_all.sh):   
#
# -R     = regenerate the dataset                       (only need to use one time, after you change dataset)
# -X     = suppress RNA-Seq file pre-processing         (set this to True if using the same dataset as the previous run, otherwise omit)
# -g     = suppress pytotch dataset generation          (set this to True if using the same dataset as the previous run, otherwise omit)
# -R     = repeats the run indicated number of times    

#  all other parameters use the defaults from do_all.sh

# Running an experiment more than MINIMUM_JOB_SIZE times (default is 3) will cause a box plot showing the spread of performance over all runs to be generated and saved in directory logs/
# The box plot will also be displayed if BOX_PLOT_SHOW=True, which is normally handy, but not here, because it will prevent experiment_1.sh from running unattended (it will stop repeatedly to show the graphics)

# 
set -x

./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L  ".00001 .00003 .00007 .0001 .0003 .0007 .001 .003 .007 .01" -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1}                                                                -r True # Base Experiment
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H "300 600 900 1200 1500 1800 2100 2400 2700 3000"                    -7 ${NN_DENSE_DROPOUT_1}                                  -X True  -g True                      # Base Experiment
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ".00 .05 .10 .15 .20 .25 .30 .35 .40 .45"                                                          -X True  -g True                      # Base Experiment
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1}                                                                              -X True  -g True -R ${REPEAT}         # Base Experiment
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1} -I False -D just_hg38_protein_coding_genes                                   -X True          -R ${REPEAT}         # Variant 1.1
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1} -I False -D pmcc_cancer_genes_of_interest                                    -X True          -R ${REPEAT}         # Variant 1.2
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1} -I False -D KIDN_genes_of_interest                                           -X True          -R ${REPEAT}         # Variant 1.3
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1} -4 "ADAM ADAMAX ADAGRAD ADAMW ADAMW_AMSGRAD ADADELTA ASGD RMSPROP RPROP SGD" -X True                               # Variant 2
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1}                                    -4 "RPROP"                                -X True          -R ${REPEAT}         # Variant 2
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1} -I False -D KIDN_genes_of_interest -4 "RPROP"                                 X True          -R ${REPEAT}         # Variant 1.3
./do_all.sh -d stad -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b 91            -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1}                                                                                               -R ${REPEAT} -r True # Additional Experiment 1 (option 1)
./do_all.sh -d sarc -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b 50            -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1}                                                                                               -R ${REPEAT} -r True # Additional Experiment 1 (option 2)
./do_all.sh -d 0008 -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b 107           -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1}                                                                                               -R ${REPEAT} -r True # Additional Experiment 2 
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1} -8 "0. .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0 1.1 1.2" -9 "80 90 95"                                               -r True # Variant 3 - at the end because it takes so long (39 runs)
./do_all.sh -d kidn -i rna -o ${N_EPOCHS} N_BOX_PLOT_SHOW="False" -b ${BATCH_SIZE} -L ${LEARNING_RATE} -H ${HIDDEN_LAYER_NEURONS} -7 ${NN_DENSE_DROPOUT_1} -8  0.90  -9 90                                                                               -R ${REPEAT}         # Variant 3
