#!/bin/bash

rm -rf dataset
rm -rf 0008
mkdir 0008

find stad -name '*UQ*' -exec cp --parents \{\} 0008 \;
find coad -name '*UQ*' -exec cp --parents \{\} 0008 \;
find kidn -name '*UQ*' -exec cp --parents \{\} 0008 \;
find sarc -name '*UQ*' -exec cp --parents \{\} 0008 \;

mv 0008/stad/* 0008
mv 0008/coad/* 0008
mv 0008/kidn/* 0008
mv 0008/sarc/* 0008

rmdir 0008/stad
rmdir 0008/coad
rmdir 0008/kidn
rmdir 0008/sarc

./create_master.sh 0008
