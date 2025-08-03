#!/bin/bash -x

#SBATCH --nodes=6
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --output=c-out.%j
#SBATCH --error=c-err.%j
#SBATCH --partition=debug
#SBATCH --distribution=cyclic:*:*
#SBATCH --hint=nomultithread

cd /home/cluster/imartin/neuronas/Neuronas/Cluster/neuronumba

source /home/cluster/imartin/.virtualenvs/neuronas/bin/activate

export PYTHONPATH=/home/cluster/imartin/neuronumba/src:/home/cluster/imartin/neuronumba/examples:$PYTHONPATH
export NUMBA_CACHE_DIR=/tmp/nn_cache

python3 -u ./prepro_neuronumba.py --g-range 0.1 7 0.1 --tmax 600 --tr 2000 --observables 'FC' 'phFCD' 'swFCD' --data-path ../../Datos/Datasets/DataHCP80 --out-path ../../Datos/Results/Results_cluster/test 
