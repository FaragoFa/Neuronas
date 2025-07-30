import glob
import os
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
import pathos.multiprocessing as mp

print(os.environ['PYTHONPATH'])

import WholeBrain.Observables.phFCD as phFCD
from WholeBrain.Utils.preprocessSignal import processBOLDSignals, processEmpiricalSubjects

measure = 'phFCD'

def run(sim_file, proc):
    print(f"Starting processing file {sim_file}", flush=True)
    simMeasures = hdf5storage.loadmat(sim_file)
    dist = phFCD.distance(simMeasures[measure], proc)
    return dist, simMeasures['we']


dir_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(dir_path, '..', 'Datos', 'Datasets', 'DataHCP80')
out_path = os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster')

sc_filename = 'SC_dbs80HARDIFULL.mat'
sc_scale = 0.1
sc_path = os.path.join(data_path, sc_filename)

fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

fMRIs = {}
observablesToUse = {measure: (phFCD, True)}

outEmpFileName = os.path.join(out_path, 'fNeuro_emp.mat')
processed = processEmpiricalSubjects(fMRIs, observablesToUse, outEmpFileName)

path = os.path.abspath(os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster', 'fit*.mat'))

fit_files = glob.glob(path)

observablesToUse = {measure: (phFCD, True)}
numParms = len(fit_files)
fitting = {}
fitting[measure] = {}
Parms = []
t = []
s = []

pool = mp.ProcessPool(nodes=4)
sim_l = [file for file in fit_files]
proc_l = [processed[measure] for _ in fit_files]
results = pool.map(run, sim_l, proc_l)

for pos, r in enumerate(results):
    fitting[measure][pos] = r
    print(f" {measure}: {fitting[measure][pos][0]} for we={r[1]};", flush=True)
    t.append(r[1])
    s.append(r[0])

optim = 1e20
optim_key = None
for k, v in fitting[measure].items():
    we = v[1]
    d = v[0]
    if d < optim:
        optim = d
        optim_key = we

print(f"# Optimal G {measure} = {optim} @ we={optim_key}")

plt.rc('font', family='serif') 
fig, ax = plt.subplots()

tt, ss = zip(*sorted(zip(t, s)))
ax.plot(tt, ss)
ax.set(xlabel='G', ylabel='distance)',
       title='G vs distance')
ax.grid(ls=':')
plt.savefig("fig_g_optim.png", dpi=300)
