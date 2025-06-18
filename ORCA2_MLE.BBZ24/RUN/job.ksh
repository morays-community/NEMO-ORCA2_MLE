#!/bin/ksh
######################
## JEANZAY IDRIS ##
######################
#SBATCH --job-name=ORCA2_ICE_ML
#SBATCH --output=ORCA2_ICE_ML.out
#SBATCH --error=ORCA2_ICE_ML.err
#SBATCH --ntasks=14
#SBATCH --hint=nomultithread # One MPI process per physical core (no hyperthreading)
#SBATCH --time=01:00:00
#SBATCH --account=cli@cpu
#SBATCH --partition=prepost 

# Process distribution
NPROC_NEMO=10
NPROC_PYTHON=4

## -------------------------------------------------------
##   End of user-defined section - modify with knowledge
## -------------------------------------------------------
# Load Environnment
source ~/.bash_profile

# Define and create execution directory and move there
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $CONFIG_DIR/OUT
cd $CONFIG_DIR/OUT

# Get input files for NEMO
DATA1DIR=$CONFIG_DIR/../FORCING
for file in $DATA1DIR/*
do
ln -s $file . || exit 2
done

# Get input namelist  and xml files
for file in $CONFIG_DIR/*namelist*_ref $CONFIG_DIR/*namelist*_cfg $CONFIG_DIR/*.xml
do
    cp $file . || exit 3
done

# Get Executables
cp $CONFIG_DIR/nemo . || exit 5
cp $CONFIG_DIR/*.py . || exit 5

set -x
pwd

# job information 
cat << EOF
------------------------------------------------------------------
Job submit on $SLURM_SUBMIT_HOST by $SLURM_JOB_USER
JobID=$SLURM_JOBID Running_Node=$SLURM_NODELIST 
Node=$SLURM_JOB_NUM_NODES Task=$SLURM_NTASKS
------------------------------------------------------------------
EOF

# Begin of section with executable commands
set -e
ls -l

# run eophis in preproduction mode to generate namcouple
touch namcouple
rm namcouple*
python3 ./main.py --exec preprod

# save eophis preproduction logs
mv eophis.out eophis_preprod.out
mv eophis.err eophis_preprod.err

# check if preproduction did well generate namcouple
namcouple=namcouple
if [ ! -e ${namcouple} ]; then
        echo "namcouple can not be found, preproduction failed"
        exit 1
else
        echo "preproduction successful"
fi

# write multi-prog file
touch run_file
rm run_file
echo 0-$((NPROC_NEMO - 1)) ./nemo >> run_file
echo ${NPROC_NEMO}-$((NPROC_NEMO + NPROC_PYTHON - 1)) python3 ./main.py >> run_file

# run coupled NEMO-Python
time srun --multi-prog ./run_file
