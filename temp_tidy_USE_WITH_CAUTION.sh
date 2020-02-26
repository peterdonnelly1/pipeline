#! /bin/bash

for x in */* ; do [ -d $x ] && ( cd $x ; pwd ; mv * .. ; cd ../.. ) ; done 
for x in */* ; do [ -d $x ] && ( cd $x ; pwd ; mv * .. ; cd ../.. ) ; done 
find . -empty -type d -delete
find . -type d | while read d; do if [ $(ls -1 "$d" | wc -l) -lt 2 ]; then rm -rf $d; fi; done
