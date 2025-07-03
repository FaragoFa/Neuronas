#!/bin/bash -x

#SBATCH --nodes=6
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=f-out.%j
#SBATCH --error=f-err.%j
#SBATCH --partition=debug
#SBATCH --distribution=cyclic:*:*
#SBATCH --hint=nomultithread

cd /home/cluster/imartin/neuronas/Neuronas/Cluster

source /home/cluster/imartin/.virtualenvs/neuronas/bin/activate

export PYTHONPATH=/home/cluster/imartin/WholeBrain:/home/cluster/imartin/neuronas/Neuronas:$PYTHONPATH

python3 ./paramsweep_cluster.py
