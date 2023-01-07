#!/bin/bash

# make sure the environment uses python 3.6, NOT 3.7 (specifically  python 3.6.13)

conda install --yes -c conda-forge tensorflow-gpu=1.15.0
conda install --yes                numpy
conda install --yes                pandas
conda install --yes                matplotlib
conda install --yes -c conda-forge opencv
conda install --yes                python-spams
conda install --yes -c anaconda    scikit-learn
conda install --yes                pyvips
conda install --yes -c bioconda    openslide                                                               # apparently need both of these
pip   install                      openslide-python                                                        # apparently need both of these - maybe this is the python interface to openslide
