import glob
import os
import hdf5storage
import numpy as np

import WholeBrain.Observables.phFCD as phFCD
from WholeBrain.Utils.preprocessSignal import processBOLDSignals, processEmpiricalSubjects

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST', 'SOCIAL', 'WM']

dir_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(dir_path, '..', 'Datos', 'Datasets', 'DataHCP80')
out_path = os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster')

subjects = [0, 1]
measure = 'phFCD'

for s in subjects:
    print(f"Staring subject {s}")
    s_path = os.path.join(out_path, f"subj_{s}")
    for task in tasks:
        outEmpFileName = s_path + f'/fNeuro_emp_{task}.mat'
        fMRIs = {}
        observablesToUse = {'phFCD': (phFCD, True)}
        processed = processEmpiricalSubjects(fMRIs, observablesToUse, outEmpFileName)
        fit_files = glob.glob(os.path.join(s_path, f"fitting*{task}.mat"))

        numParms = len(fit_files)
        fitting = {}
        fitting[measure] = {}

        for pos, file in enumerate(fit_files):
            simMeasures = hdf5storage.loadmat(file)
            keys = list(simMeasures.keys())
            k = keys[0] if keys[0] != measure else keys[1]
            fitting[measure][k] = observablesToUse[measure][0].distance(simMeasures[measure], processed[measure])
            # print(f" {measure}: {fitting[measure][k]} for {file};", flush=True)

        optim = 1e20
        optim_key = None
        for k, v in fitting[measure].items():
            if v < optim:
                optim = v
                optim_key = k

        print(f"# Optimal for subject {s} in task {task} -> {measure} = {optim} @ {optim_key}")
