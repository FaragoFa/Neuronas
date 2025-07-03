import glob
import os
import hdf5storage
import numpy as np

print(os.environ['PYTHONPATH'])

import WholeBrain.Observables.phFCD as phFCD
from WholeBrain.Utils.preprocessSignal import processBOLDSignals, processEmpiricalSubjects

dir_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(dir_path, '..', 'Datos', 'Datasets', 'DataHCP80')
out_path = os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster')

sc_filename = 'SC_dbs80HARDIFULL.mat'
sc_scale = 0.1
sc_path = os.path.join(data_path, sc_filename)

fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

print(f"Loading {sc_path}")
sc80 = hdf5storage.loadmat(sc_path)['SC_dbs80FULL']
C = sc80 / np.max(sc80) * sc_scale  # Normalization...


fMRIs = {}
observablesToUse = {'phFCD': (phFCD, True)}


outEmpFileName = out_path + '/fNeuro_emp.mat'
processed = processEmpiricalSubjects(fMRIs, observablesToUse, outEmpFileName)

dir_path = os.path.dirname(os.path.realpath(__file__))
path = os.path.abspath(os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster', 'fit*.mat'))

fit_files = glob.glob(path)

observablesToUse = {'phFCD': (phFCD, True)}
numParms = len(fit_files)
fitting = {}
for ds in observablesToUse:
    fitting[ds] = np.zeros((numParms))

Parms = []
for pos, file in enumerate(fit_files):
    simMeasures = hdf5storage.loadmat(file)
    Parms.append(simMeasures['we'])
    for ds in observablesToUse:
        fitting[ds][pos] = observablesToUse[ds][0].distance(simMeasures[ds], processed[ds])
        print(f" {ds}: {fitting[ds][pos]} for we={Parms[-1]};", flush=True)

for ds in observablesToUse:
    optimValDist = observablesToUse[ds][0].findMinMax(fitting[ds])
    parmPos = [a for a in np.nditer(Parms)][optimValDist[1]]
    print(f"# Optimal {ds} =     {optimValDist[0]} @ {np.round(parmPos, decimals=3)}")
