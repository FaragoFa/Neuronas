#!/bin/bash -x

#SBATCH --nodes=6
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --output=c-out.%j
#SBATCH --error=c-err.%j
#SBATCH --partition=debug
#SBATCH --distribution=cyclic:*:*
#SBATCH --hint=nomultithread

cd /home/cluster/imartin/neuronas/Neuronas/Cluster/neuronumba

source /home/cluster/imartin/.virtualenvs/neuronas/bin/activate

export PYTHONPATH=/home/cluster/imartin/neuronumba:/home/cluster/imartin/neuronas/Neuronas:$PYTHONPATH

python3 ./prepro_neuronumba.py --we-range 1 8 0.25
