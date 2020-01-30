# Create files containing the class number (truth value) and save to a file in each case sub-directory( source of renaming information is MAPPING_FILE)

# Notes:
#   Recall that awk will perform the substitution for each line in MAPPING FILE so there is no need for a loop
#   How it works: echo $4 (class value from the MAPPING FILE) into the case subdirectory ($1) (case value from the MAPPING FILE)
#   -v assigns a bash variable to an awk variable so that we can use bash variables within an awk expression
#   -F specifies the field saperator

source conf/variables.sh

bold=$(tput bold)
normal=$(tput sgr0)

echo "SETUP_CLASS_FILES: INFO: mapping file =   "${bold}${MAPPING_FILE}${normal}

#awk -v EXP=$DATA_DIR -F','  'system("echo " $4 "   " EXP "/" $1  "/class.csv" )' ${MAPPING_FILE}
awk -v EXP=$DATA_DIR -F','  'system("echo " $4 " > " EXP "/" $1  "/class.csv" )' ${MAPPING_FILE}

NUMBER_CREATED=$(find ${DATA_DIR} -name "class.csv" | wc -l)
echo "SETUP_CLASS_FILES: INFO: finished creating class files in  "${bold}${DATA_DIR}${normal}
echo "SETUP_CLASS_FILES: INFO: Number of class files created in  = "${bold}${NUMBER_CREATED}${normal}
