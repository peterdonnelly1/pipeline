
---
1   Install the NVIDIA Docker Container Runtime on your host machine:

    For GPU support, the NVIDIA Container Runtime is required on the system running Docker.
    Note: there is no  NVIDIA Container Runtime for Windows
    
    Installation instructions (4 steps) follow. 
    From https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker

---
2   Build and run the CLASSI docker image (use 'source' so it will execute within the current shell session) 
 
      source ./____BUILD_AND_RUN_THE_CLASSI_DOCKER_ENVIRONMENT.sh  
      
    You will be left in the CLASSI container in directory 'pipeline'
      
---
3  Use gdc-fetch ('d' option to begin with) to get some TCGA data:  
 
    eg: fetch and pre-process TCGA Sarcoma RNA-Seq data (1.1GB - <60min):  
    
      python ./gdc-fetch.py --debug=9 --dataset sarc --case_filter="filters/TCGA-SARC_case_filter"  --file_filter="filters/GLOBAL_file_filter_UQ" --max_cases=5000 --max_files=10  --output_dir=source_data/sarc  

    eg: fetch and pre-process TCGA stomach cancer dataset (image and RNA-Seq) (~650GB - allow 24hrs+):   
    
      python ./gdc-fetch.py --debug=9 --dataset stad --case_filter="filters/TCGA-STAD_case_filter"  --file_filter="filters/GLOBAL_file_filter_UQ"  --max_cases=5000 --max_files=10  --output_dir=source_data/stad  
      python ./gdc-fetch.py --debug=9 --dataset stad --case_filter="filters/TCGA-STAD_case_filter"  --file_filter="filters/GLOBAL_file_filter_SVS" --max_cases=5000 --max_files=10  --output_dir=source_data/stad  

---
4   Run an experiment. These ones are fully scripted:

     ./do_all_RUN_ME_TO_SEE_RNASEQ_PROCESSING.sh                     <<< requires sarc rna-seq       dataset
     ./do_all_RUN_ME_TO_SEE_IMAGE_PROCESSING.sh                      <<< requires stad image         dataset
     ./do_all_RUN_ME_TO_SEE_CLUSTERING_USING_SCIKIT_SPECTRAL.sh      <<< requires kidn rna-seq       dataset
     ./do_all_RUN_ME_TO_SEE_MULTIMODAL_IMAGE_PLUS_RNA_PROCESSING.sh  <<< requires stad image and rna datasets
     ./experiment_1.sh                                               <<< requires lots of rna-seq    datasets
    
    once you're comfortable, use CLASSI commands to run experiments:
    
    ./do_all.sh -d sarc -i rna   -z DENSE     -b 22 -o 50  -H 2200  -7 0.2                -c UNIMODE_CASE -A 3  -r True
    ./do_all.sh -d sarc -i rna   -z DENSE     -b 22 -o 50  -H 2200  -7 0.2                -c UNIMODE_CASE -A 3  -R 8
    ./do_all.sh -d stad -i rna   -z DENSE     -b 88 -o 100 -H "1000 2000"   -7 ".18 .20"  -c UNIMODE_CASE -A 2
    ./do_all.sh -d stad -i image -a VGG11     -b 64 -o 4  -1 .2  -L .0003 -f 100  -T  64  -c UNIMODE_CASE -A 4
    ./do_all.sh -d stad -i image -a RESNET152 -b 64 -o 4  -1 .2  -L .0003 -f 9    -T 256  -c UNIMODE_CASE -A 4  -r True
     
---
 To monitor experiments and see results:

    during an experiment:
       monitor progress via container console output
       observe learning curves with any browser pointing to http://localhost:6006
       
    after an experiment has completed:
       run 'gimp' inside the container to view images produced by classi. 
        eg. cd logs; gimp 230102_0247__01 ... bar_chart_AL.png &

 To edit configuration files:

    geany                   > /dev/null 2>&1 &
    geany do_all.sh         > /dev/null 2>&1 &
    geany conf/variables.sh > /dev/null 2>&1 &

 To enter running classi container with a bash shell

    sudo docker exec -it classi bash

 To stop/start classi container:

    sudo docker stop  classi
    sudo docker start classi

 To delete classi container:
    sudo docker rm classi

    this does not delete the classi image

 To delete the classi image:

    sudo docker rmi -f classi

    you will have to build it again if you do this. 
    Building should be fast since Docker will have cached everything that did not change since the last build.

---






---
I gratefuly acknowledge the authors of the following software used in CLASSI:

**dpcca**  
The core learning engine of ‘CLASSI’ buids on Gregory Gundersen’s ‘dpcca'. 
      An attractive feature of this software is that it permits neural network models to be dynamically selected at run-time  
Code: https://github.com/gwgundersen/dpcca  
Paper: "End-to-end Training of Deep Probabilistic CCA on Paired Biomedical Observations"  
Paper: http://auai.org/uai2019/proceedings/papers/340.pdf  

**SPCN**  
GPU version of 'Structure Preserving Color Normalisation' by D. Anand, G. Ramakrishnan and A. Sethi  
Code:  https://github.com/goutham7r/spcn  
Paper: "Fast GPU-enabled Color Normalization for Digital pathology, International Conference on Systems, 
        Signals and Image Processing, Osijek, Croatia (2019), pp. 219-224  
Paper: https://ieeexplore.ieee.org/document/8787328/  

**Reinhard Stain Colour Normalisation:**  
Takumi Ando, University of Tokyo, Tokyo, Japan  
Code: https://github.com/tand826  
  
  
  
Each portion of the code is governed by ... respective licenses - however our code is governed by the ...

