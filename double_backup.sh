#! /bin/bash


TIMESTAMP=$(date +%Y-%m-%d_%H)
echo $TIMESTAMP

cp -rfv ~/git/dpcca                   /b2/${TIMESTAMP}_dpcca_$1_
cp -rfv ~/git/dpcca /media/peter/Elements/${TIMESTAMP}_dpcca_$1_

cp -rfv ~/biodata                     /b2/${TIMESTAMP}_biodata_$1_
cp -rfv ~/biodata   /media/peter/Elements/${TIMESTAMP}_biodata_$1_
