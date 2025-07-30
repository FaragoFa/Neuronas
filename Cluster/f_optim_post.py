import glob
import os
import re
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import WholeBrain.Observables.phFCD as phFCD
from WholeBrain.Utils.preprocessSignal import processBOLDSignals, processEmpiricalSubjects

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST', 'SOCIAL', 'WM']

dir_path = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(dir_path, '..', 'Datos', 'Datasets', 'DataHCP80')
out_path = os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster')

subjects = range(0,20)
measure = 'phFCD'

for s in subjects:
    print(f"Starting subject {s}")
    s_path = os.path.join(out_path, f"subj_{s}")
    for task in tasks:
        outEmpFileName = s_path + f'/fNeuro_emp_{task}.mat'
        if not os.path.exists(outEmpFileName):
            continue
        processed = hdf5storage.loadmat(outEmpFileName)
        fit_files = glob.glob(os.path.join(s_path, f"fitting*{task}.mat"))

        if len(fit_files) == 0:
            continue

        numParms = len(fit_files)
        fitting = {}
        fitting[measure] = {}

        mes = set()
        mis = set()
        data = {}

        for pos, file in enumerate(fit_files):
            simMeasures = hdf5storage.loadmat(file)
            keys = list(simMeasures.keys())
            k = keys[0] if keys[0] != measure else keys[1]
            result = re.search(r"we_[0-9]+.[0-9]+_M_e_([0-9]+.[0-9]+)_M_i_([0-9]+.[0-9]+)_[A-Z]+", k) 
            me = result.group(1)
            mi = result.group(2)
            mes.add(me)
            mis.add(mi)

            d = phFCD.distance(simMeasures[measure], processed[measure])
            data[(me, mi)] = d

            fitting[measure][k] = d
            # print(f" {measure}: {fitting[measure][k]} for {file};", flush=True)

        optim = 1e20
        optim_key = None
        for k, v in fitting[measure].items():
            if v < optim:
                optim = v
                optim_key = k

        print(f"Subject {s} ({task}) -> {measure} = {np.round(optim, decimals=5)} @ {optim_key}")

        x_me = list(mes)
        y_mi = list(mis)
        x_me.sort(key=lambda k : float(k))
        y_mi.sort(key=lambda k : float(k), reverse=True)

        mat = np.zeros((len(x_me), len(y_mi)))

        for k, v in data.items():
            x_pos = x_me.index(k[0])
            y_pos = y_mi.index(k[1])
            mat[x_pos, y_pos] = v

        # fig, ax = plt.subplots()
        # im = ax.imshow(mat)

        # # Show all ticks and label them with the respective list entries
        # ax.set_xticks(np.arange(len(x_me)), labels=x_me)
        # ax.set_yticks(np.arange(len(y_mi)), labels=y_mi)
        # for i in range(len(x_me)):
        #     for j in range(len(y_mi)):
        #         text = ax.text(j, i, np.round(mat[i, j], decimals=3),
        #                        ha="center", va="center", color="w")

        ax = sns.heatmap(mat.T, vmin=0, vmax=1, cmap='viridis', annot_kws={"size": 6}, xticklabels=x_me, yticklabels=y_mi, annot=True)
        ax.set_title(f"Subject {s}, task {task}")
        plt.xlabel("M_e")
        plt.ylabel("M_i")
        plt.yticks(rotation=0) 

        plt.savefig(f"fit_{s}_{task}.png", dpi=300)
        plt.clf()

