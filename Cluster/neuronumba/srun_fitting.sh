#!/bin/bash -x

#SBATCH --nodes=6
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --output=f-out.%j
#SBATCH --error=f-err.%j
#SBATCH --partition=debug
#SBATCH --distribution=block:*:*,NoPack
#SBATCH --hint=nomultithread

cd /home/cluster/imartin/neuronas/Neuronas/Cluster/neuronumba

source /home/cluster/imartin/.virtualenvs/neuronas/bin/activate

export PYTHONPATH=/home/cluster/imartin/neuronumba:$PYTHONPATH

python3 ./prepro_cluster.py --fitting
