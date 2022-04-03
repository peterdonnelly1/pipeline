#! /bin/bash

TIMESTAMP=$(date +%Y-%m-%d_%H)

cp -rfv ~/git/pipeline/classi                    /b2/${TIMESTAMP}_classi_$1_
cp -rfv ~/git/pipeline/classi  /media/peter/Elements/${TIMESTAMP}_classi_$1_
