# --------------------------------------------------------------------------------------
# Full pipeline for applying Leading Eigenvector Dynamics Analysis (LEiDA) to AD data
# using pyLEiDA: https://github.com/PSYMARKER/leida-python
#
# By Gustavo Patow
#
# note: start by configuring this!!!
# --------------------------------------------------------------------------------------
import os
import csv
import random
import numpy as np
import h5py


# --------------------------------------------------------------------------
# functions to select which subjects to process
# --------------------------------------------------------------------------
# ---------------- load a previously saved list
def loadSubjectList(path):
    subjects = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            subjects.append(int(row[0]))
    return subjects


# ---------------- save a freshly created list
def saveSelectedSubjcets(path, subj):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in subj:
            writer.writerow([s])


# ---------------- fix subset of subjects to sample
def selectSubjects(selectedSubjectsF, maxSubj, numSampleSubj):
    if not os.path.isfile(selectedSubjectsF):  # if we did not already select a list...
        listIDs = random.sample(range(0, maxSubj), numSampleSubj)
        saveSelectedSubjcets(selectedSubjectsF, listIDs)
    else:  # if we did, load it!
        listIDs = loadSubjectList(selectedSubjectsF)
    # ---------------- OK, let's proceed
    return listIDs


# --------------------------------------------------------------------------
# functions to load fMRI data for certain subjects
# --------------------------------------------------------------------------
def read_matlab_h5py(filename):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]
        # get the object type for a_group_key: usually group or dataset
        # print(type(f['subjects_idxs']))
        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array

        all_fMRI = {}
        subjects = list(f['subject'])
        for pos, subj in enumerate(subjects):
            print(f'reading subject {pos}')
            group = f[subj[0]]
            dbs80ts = np.array(group['dbs80ts'])
            all_fMRI[pos] = dbs80ts.T

    return all_fMRI


def loadSubjectsData(fMRI_path, listIDs):
    # numSampleSubj = len(listIDs)
    print(f'reading {fMRI_path}')
    fMRIs = read_matlab_h5py(fMRI_path)
    # nNodes, Tmax = fMRIs[next(iter(fMRIs))].shape
    res = {}  # np.zeros((numSampleSubj, nNodes, Tmax))
    for pos, s in enumerate(listIDs):
        res[pos] = fMRIs[s]
    return res




# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
