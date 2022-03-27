#!/bin/bash
set -x


  
./do_all.sh -d kidn -i rna -L  ".00001 .00003 .00007 .0001 .0003 .0007 .001 .003 .007 .01"                 -r True

./do_all.sh -d kidn -i rna -L .00001 -H "300 600 900 1200 1500 1800 2100 2400 2700 3000"                   -X True  -g True

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 ".00 .05 .10 .15 .20 .25 .30 .35 .40 .45"                  -X True  -g True

./do_all.sh -d kidn -i rna -L .00001 -H 2100 -7 .20                                                        -X True  -g True  -R 8


