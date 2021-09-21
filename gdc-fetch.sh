
clear

DATASET=$1
CASE_FILTER=$2
FILE_FILTER=$3

python ./gdc-fetch.py --debug=999 --base_dir /home/peter/git/pipeline --dataset ${DATASET} --case_filter ${CASE_FILTER} --file_filter ${FILE_FILTER} --max_cases=20000 --max_files=1  --global_max_downloads=20000 --output_dir ${DATASET}
