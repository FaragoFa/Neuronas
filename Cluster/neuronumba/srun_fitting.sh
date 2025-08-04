#!/bin/bash -x

#SBATCH --nodes=6
#SBATCH --ntasks-per-node=32
#SBATCH --output=f-out.%j
#SBATCH --error=f-err.%j
#SBATCH --partition=debug
#SBATCH --distribution=cyclic:*:*
#SBATCH --hint=nomultithread

cd /home/cluster/imartin/neuronas/Neuronas/Cluster/neuronumba

source /home/cluster/imartin/.virtualenvs/neuronas/bin/activate

export PYTHONPATH=/home/cluster/imartin/neuronumba/src:$PYTHONPATH

python3 ./prepro_neuronumba.py --fitting
