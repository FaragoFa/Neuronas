import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import subprocess
import sys
import os
import time
import glob
import gc
import re
import matplotlib.pyplot as plt
import seaborn as sns

import h5py
import numpy as np

from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables.measures import (
    KolmogorovSmirnovStatistic, 
    PearsonSimilarity
)
from neuronumba.observables import (
    PhFCD,
    SwFCD,
    FC
)

from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators import EulerDeterministic
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.models import Naskar2021
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator
from neuronumba.observables.accumulators import (
    ConcatenatingAccumulator,
    AveragingAccumulator
)
from neuronumba.tools import hdf

from global_coupling_fitting import (
    process_empirical_subjects, 
    gen_arg_parser, 
    compute_g,
    simulate_single_subject,
    process_bold_signals,
    ObservableConfig,
    create_observable_config
)

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST', 'SOCIAL', 'WM']

subjects_excluded = {515, 330, 778, 140, 77, 74, 335, 591, 978, 596, 406, 729, 282, 667, 157, 224, 290, 355, 930, 742,
                     425, 170, 299, 301, 557, 239, 240, 238, 820, 502, 185, 700, 553, 433, 40}


def read_matlab_h5py(filename, max_subjects=None):
    with h5py.File(filename, "r") as f:
        all_fMRI = {}
        subjects = list(f['subject'])
        for pos, subj in enumerate(subjects):
            group = f[subj[0]]
            if pos in subjects_excluded:
                continue
            try:
                dbs80ts = np.array(group['dbs80ts'])
                all_fMRI[pos] = dbs80ts.T
            except:
                print(f'ignoring register {subj} at {pos}')
            if max_subjects and pos >= max_subjects:
                break

    return all_fMRI


def read_matlab_h5py_single(filename, nsubject):
    with h5py.File(filename, "r") as f:
        if nsubject in subjects_excluded:
            raise Exception(f"Subject {nsubject} is in the excluded list!")
         
        subjects = list(f['subject'])
        try:
            pos = subjects[nsubject]
            subj = f[pos[0]]
            dbs80ts = np.array(subj['dbs80ts'])
            return dbs80ts
        except:
            raise Exception(f"Error loading data for subject {nsubject}")


def loadSubjectsData(fMRI_path, max_subjects=None):
    print(f'Loading {fMRI_path}')
    fMRIs = read_matlab_h5py(fMRI_path, max_subjects)  # ignore the excluded list
    return fMRIs

def extend_command(command, args, params):
    for arg_name, arg_value in vars(args).items():
        if arg_name in params and arg_value is not None:
            if isinstance(arg_value, list):
                command.append(f'--{arg_name}')
                for value in arg_value:
                    command.append(str(value))
            else:
                command.extend([f'--{arg_name.replace("_", "-")}', str(arg_value)])

def executor_simulate_single_subject(n, exec_env, g):
    try:
        return n, simulate_single_subject(exec_env, g)
    except Exception as e:
        print(f"Error simulating subject {n}: {e}", file=sys.stderr)
        raise

def prepro():
    parser = gen_arg_parser()

    dir_path = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--fitting-subj", type=int, help="Start the fitting process for an specific subject")
    parser.add_argument("--fitting", help="Start the fitting process", action=argparse.BooleanOptionalAction)
    parser.add_argument("--recompute-dist", help="Start the fitting process", action=argparse.BooleanOptionalAction)
    parser.add_argument("--multi", help="Use multiple fNeuro_emp files", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max-emp_subjects", help="Maximum empirical subjects to use", type=int, required=False)
    parser.add_argument("--post-g-optim", help="Post process G optimization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--post-fitting", help="Post process G optimization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--process-empirical", help="Process empirical subjects", action=argparse.BooleanOptionalAction)
    parser.add_argument("--subject", help="Subject to explore", type=int, required=False)
    parser.add_argument("--me-range", nargs=3, help="Parameter sweep range for Me", type=float, required=False)
    parser.add_argument("--mi-range", nargs=3, help="Parameter sweep range for Mi", type=float, required=False)
    parser.add_argument("--task", help="Task to compute", type=str, required=False)
    parser.add_argument("--data-path", type=str, help="Input path witj subjects data", required=False, default=os.path.join(dir_path, '..', '..', 'Datos', 'Datasets', 'DataHCP80'))
    parser.add_argument("--use-mp", help="Use multiprocessing for the fitting process", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    data_path = args.data_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')
    emp_filename_pattern = os.path.join(out_path, 'fNeuro_emp_{}.mat')

    sc_scale = args.sc_scaling
    
    tr = args.tr

    bpf = BandPassFilter(tr=tr, k=args.bpf[0], flp=args.bpf[1], fhi=args.bpf[2]) if args.bpf is not None else None

    observables = {}
    for observable_name in args.observables:
        if observable_name == 'FC':
            observables[observable_name] = create_observable_config(observable_name, PearsonSimilarity(), bpf)
        elif observable_name == 'phFCD':
            observables[observable_name] = create_observable_config(observable_name, KolmogorovSmirnovStatistic(), bpf)
        elif observable_name == 'swFCD':
            observables[observable_name] = create_observable_config(observable_name, KolmogorovSmirnovStatistic(), bpf)
        else:
            raise RuntimeError(f"Observable <{observable_name}> not supported!")

    # Process BOLD signals from subjects obrtained with fMRI, and generate the corresponding observable
    if args.process_empirical:
        tasks_to_process = [args.task] if args.task else tasks
        for task in tasks_to_process:
            emp_filename = emp_filename_pattern.format(task)
            if os.path.exists(emp_filename):
                print(f"File <{emp_filename}> already exists!")
                continue

            bold_signals = loadSubjectsData(fMRI_path.format(task))

            processed = process_empirical_subjects(bold_signals, observables, verbose=args.verbose)
            hdf.savemat(emp_filename, processed)

            print(f'Processed empirical subjects, n = {len(bold_signals)}')

        exit(0)

    # Show distances for all G files generated
    if args.post_g_optim:
        file_list = glob.glob(os.path.join(out_path, 'fitting_g_*.mat'))
        y = {}
        for o_name in observables.keys():
            y[o_name] = []
        x = []
        for f in sorted(file_list):
            m = hdf.loadmat(f)
            g = m['g']
            for o_name in y.keys():
                d = m[f'dist_{o_name}']
                print(f"Distance for g={g} and observable {o_name} = {d}", flush=True)
                y[o_name].append(d)
            x.append(g)

        for o_name, ys in y.items():
            fig, ax = plt.subplots()
            ax.plot(x, ys)
            ax.set_title(f"Distance for observable {o_name}")
            plt.savefig(os.path.join(out_path, f"fig_g_optim_{o_name}.png"), dpi=300)

        exit(0)


    task = args.task if args.task else 'REST'

    dt = 0.1
    model = Naskar2021(g=args.g)
    integrator = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 1e-3, 0])
    obs_var = 're'
    out_file_path = args.out_path
    out_file_name_pattern = os.path.join(out_file_path, f'fitting_g_{task}_{{}}.mat')
    # Compute the simulation length according to input data
    t_max = 60 if args.tmax is None else args.tmax
    # Compute simulation time in milliseconds
    t_max_neuronal = t_max * 1000.0
    t_warmup = 0.0
    n_subj = args.nsubj if args.nsubj is not None else 100
    sampling_period = 1.0

    sc_filename = 'SC_dbs80HARDIFULL.mat'
    sc_path = os.path.join(data_path, sc_filename)

    fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

    print(f"Loading {sc_path}")
    sc80 = hdf.loadmat(sc_path)['SC_dbs80FULL']
    C = sc80 / np.max(sc80) * sc_scale  # Normalization...

    n_rois = C.shape[0]
    lengths = np.random.rand(n_rois, n_rois) * 10.0 + 1.0
    speed = 1.0
    dt = 0.1

    print(f"Loading empirical data for task {task}", flush=True)
    processed = hdf.loadmat(emp_filename_pattern.format(task))

    if args.recompute_dist:
        file_list = glob.glob(os.path.join(out_path, 'fitting_g_*.mat'))
        for f in sorted(file_list):
            print(f"Recomputing distances for file {f}", flush=True)
            m = hdf.loadmat(f)
            for ds in observables:
                m[f'dist_{ds}'] = observables[ds].compute_distance(m[ds], processed[ds])

            hdf.savemat(f, m)
            
        exit(0)

    # Compute distance for a specific G value, using the cluster with slurm
    if args.g and not args.use_mp:
        out_file = os.path.join(out_path, f'fitting_g_{args.g}.mat')

        if os.path.exists(out_file):
            exit(0)

        print(f'Computing distance for G={args.g}', flush=True)
        compute_g({
            'verbose':True,
            'model': copy.deepcopy(model),
            'integrator': copy.deepcopy(integrator),
            'weights': C,
            'processed': processed,
            'tr': args.tr,
            'observables': copy.deepcopy(observables),
            'obs_var': obs_var,
            'bold': True,
            'bold_model': BoldStephan2008().configure(),
            'out_file': out_file_name_pattern.format(np.round(args.g, decimals=3)),
            'num_subjects': n_subj,
            't_max_neuronal': t_max_neuronal,
            't_warmup': t_warmup,
            'sampling_period': sampling_period,
            'force_recomputations': False,
        }, args.g)
        exit(0)

    # Compute distance for a specific G value and a given task, NOT USING SLURM, but multiprocessing
    if args.g_range and args.use_mp:

        Gs = np.arange(args.g_range[0], args.g_range[1], args.g_range[2])

        for g in Gs:
            out_file = out_file_name_pattern.format(np.round(g, decimals=3))

            if os.path.exists(out_file):
                print(f"File {out_file} already exists, skipping...")
                continue    

            print(f'Computing distance for task {task} and G={g}', flush=True)

            subjects = list(range(n_subj))    
            results = []
            print(f'Creating process pool with {args.nproc} workers')
            while len(subjects) > 0:
                print(f"EXECUTOR --- START cycle for {len(subjects)} subjects")
                pool = ProcessPoolExecutor(max_workers=args.nproc)
                futures = []
                future2subj = {}
                for n in subjects:
                    exec_env = {
                        'verbose':True,
                        'model': copy.deepcopy(model),
                        'integrator': copy.deepcopy(integrator),
                        'weights': C,
                        'tr': args.tr,
                        'obs_var': obs_var,
                        'bold': True,
                        'bold_model': BoldStephan2008().configure(),
                        't_max_neuronal': t_max_neuronal,
                        't_warmup': t_warmup,
                        'sampling_period': sampling_period
                    }
                    f = pool.submit(executor_simulate_single_subject, n, exec_env, g)
                    future2subj[f] = n
                    futures.append(f)

                print(f"EXECUTOR --- WAITING for {len(futures)} futures to finish")

                subjects = []
                for future in as_completed(futures):
                    try:
                        n, result = future.result()
                        results.append((n, result))
                        print(f"EXECUTOR --- FINISHED subject {n}")
                    except Exception as exc:
                        print(f"EXECUTOR --- FAIL subject {n}. Restarting pool.")
                        pool.shutdown(wait=False)
                        finished = [n for n, _ in results]
                        subjects = [n for n in subjects if n not in finished]
                        break

            simulated_bolds = {}
            for n, r in results:
                simulated_bolds[n] = r
            
            exec_env = {
                        'observables': copy.deepcopy(observables),
                    }
    
            sim_measures = process_bold_signals(simulated_bolds, { 'observables': copy.deepcopy(observables)})
            sim_measures['g'] = g
            for ds in observables:
                sim_measures[f'dist_{ds}'] = observables[ds].compute_distance(sim_measures[ds], processed[ds])

            hdf.savemat(out_file, sim_measures)

        exit(0)    

    # Sweep for G values with the cluster using slurm
    if args.g_range:
        WEs = np.arange(args.g_range[0], args.g_range[1], args.g_range[2])

        srun = ['srun', '-n', '1', '-N', '1', '-c', '1', '--time=9-00']
        # srun = ['srun', '-n1', '--exclusive']

        script = [sys.executable, __file__]

        print('Starting srun sweep', flush=True)
        workers = []
        for g in WEs:
            command = [*srun, *script]
            command.extend(['--g', f'{g:.2f}'])
            extend_command(command, args, ['task', 'data_path', 'out_path', 'sc_scaling', 'tr', 'nsubj', 'tmax', 'observables', 'bpf'])
            workers.append(subprocess.Popen(command))
            print(f"Executing: {command}", flush=True)

        print('Waiting for the sweep to finish', flush=True)
        exit_codes = [p.wait() for p in workers]
        print('Sweep finished!', flush=True)
        print(exit_codes)
        exit(0)

    # Perform the M_e/M_i fitting for a specific subject and for a task
    if args.subject is not None:
        optim_g = 2.05 # This is the G value that was optimized for the Naskar2021 model
        out_path = os.path.join(out_path, f'subj_{args.subject}')
        os.makedirs(out_path, exist_ok=True)

        sc_filename = 'SC_dbs80HARDIFULL.mat'
        sc_path = os.path.join(data_path, sc_filename)
        print(f"Loading {sc_path}")
        sc80 = hdf.loadmat(sc_path)['SC_dbs80FULL']
        C = sc80 / np.max(sc80) * sc_scale  # Normalization...

        n_rois = C.shape[0]
        lengths = np.random.rand(n_rois, n_rois) * 10.0 + 1.0
        speed = 1.0
        dt = 0.1

        mes = np.arange(args.me_range[0], args.me_range[1], args.me_range[2])
        mis = np.arange(args.mi_range[0], args.mi_range[1], args.mi_range[2])

        print(f"Starting fitting process for subject {args.subject} for task {task}, {len(mes)} points for M_e and {len(mis)} points for M_i")
        fMRI = read_matlab_h5py_single(fMRI_path.format(task), args.subject)
        
        processed = process_empirical_subjects({0: fMRI}, observables, verbose=args.verbose)

        for me in mes:
            for mi in mis:
                start_time = time.perf_counter()
                fitting_suffix = f"{task}_{np.round(me, decimals=3)}_{np.round(mi, decimals=3)}"
                out_file = os.path.join(out_path, f'fitting_{fitting_suffix}.mat')
                if os.path.exists(out_file):
                    print(f"File already exists for {task}_{np.round(me, decimals=3)}_{np.round(mi, decimals=3)}")
                    continue

                m = copy.deepcopy(model)
                m.M_e = me
                m.M_i = mi  

                compute_g({
                    'verbose':True,
                    'model': m,
                    'integrator': copy.deepcopy(integrator),
                    'weights': C,
                    'processed': processed,
                    'tr': args.tr,
                    'observables': copy.deepcopy(observables),
                    'obs_var': obs_var,
                    'bold': True,
                    'bold_model': BoldStephan2008().configure(),
                    'out_file': out_file,
                    'num_subjects': n_subj,
                    't_max_neuronal': t_max_neuronal,
                    't_warmup': t_warmup,
                    'sampling_period': sampling_period,
                    'force_recomputations': False,
                }, optim_g)

                gc.collect()
                print(f"Time for subject {args.subject}, point {fitting_suffix} = {time.perf_counter() - start_time}")
        
        exit(0)


    # Perform the M_e/M_i fitting for all subjects, and all tasks using the cluster with slurm
    if args.fitting:
        num_subjects = 1003
        list_subjects = list(set(range(6, num_subjects)) - subjects_excluded)
        srun = ['srun', '-n 1', '-N 1', '-c 1', '--time=4-00', '--exclusive']

        script = [sys.executable, __file__]

        finished = []
        workers = []
        max_proc = 200
        for ns in list_subjects:
            for task in tasks:
                print(f'Starting srun sweep for task {task} for subject {ns}', flush=True)
                for me in np.arange(0.05, 1.5001, 0.05):
                    for mi in np.arange(0.05, 1.5001, 0.05):
                        command = [*srun, *script]
                        command.extend(['--subject', f'{ns}'])
                        command.extend(['--task', task])
                        command.extend(['--me-range', f'{me}', f'{me+0.001}', '0.1'])
                        command.extend(['--mi-range', f'{mi}', f'{mi+0.001}', '0.1'])
                        extend_command(command, args, ['data_path', 'out_path', 'sc_scaling', 'tr', 'nsubj', 'tmax', 'observables', 'bpf'])
                        workers.append(subprocess.Popen(command))
                        print(f"Executing: {command}", flush=True)
                        while len(workers) >= max_proc:
                            print(f"Maximum concurrent process {len(workers)} >= {max_proc} reached. Waiting ...", flush=True)
                            new_workers = []
                            for p in workers:
                                try:
                                    p.wait(timeout=1)
                                    finished.append(p)
                                    print(f"Process {p.pid} finished!", flush=True)
                                except:
                                    new_workers.append(p)
                            workers = new_workers

        print('Waiting for the sweep to finish', flush=True)
        exit_codes = [p.wait() for p in workers]
        print(exit_codes)
        exit(0)

    # Perform the M_e/M_i fitting for a single subject, and all tasks using the cluster with slurm
    if args.fitting_subj is not None:

        ns = args.fitting_subj
        srun = ['srun', '-n 1', '-N 1', '-c 1', '--time=4-00', '--exclusive']

        script = [sys.executable, __file__]

        finished = []
        workers = []
        max_proc = 200
        for task in tasks:
            print(f'Starting srun sweep for task {task} for subject {ns}', flush=True)
            for me in np.arange(0.05, 1.5001, 0.05):
                for mi in np.arange(0.05, 1.5001, 0.05):
                    command = [*srun, *script]
                    command.extend(['--subject', f'{ns}'])
                    command.extend(['--task', task])
                    command.extend(['--me-range', f'{me}', f'{me+0.001}', '0.1'])
                    command.extend(['--mi-range', f'{mi}', f'{mi+0.001}', '0.1'])
                    extend_command(command, args, ['data_path', 'out_path', 'sc_scaling', 'tr', 'nsubj', 'tmax', 'observables', 'bpf'])
                    workers.append(subprocess.Popen(command))
                    print(f"Executing: {command}", flush=True)
                    while len(workers) >= max_proc:
                        print(f"Maximum concurrent process {len(workers)} >= {max_proc} reached. Waiting ...", flush=True)
                        new_workers = []
                        for p in workers:
                            try:
                                p.wait(timeout=1)
                                finished.append(p)
                                print(f"Process {p.pid} finished!", flush=True)
                            except:
                                new_workers.append(p)
                        workers = new_workers

        exit_codes = [p.wait() for p in workers]
        print(f'Sweep finished for task {task} and subject {ns}!', flush=True)
        print(exit_codes)
        exit(0)

    # Generate the heatmaps for the M_e/M_i fitting results
    if args.post_fitting:
        dir_list = glob.glob(os.path.join(out_path, 'subj*'))
        for dir in sorted(dir_list):
            subj_match = re.search(r"subj_([0-9]+)", dir)
            s = int(subj_match.group(1))
            print(f"Starting processing of subject {s}")
            for task in tasks:
                fit_files = glob.glob(os.path.join(dir, f"fitting*_{task}_*.mat"))

                if len(fit_files) == 0:
                    continue

                numParms = len(fit_files)
                fitting = {}

                mes = set()
                mis = set()
                data = {}

                for pos, file in enumerate(fit_files):
                    simMeasures = hdf.loadmat(file)
                    result = re.search(r"_([0-9]+.[0-9]+)_([0-9]+.[0-9]+)\.", file) 
                    me = result.group(1)
                    mi = result.group(2)
                    mes.add(me)
                    mis.add(mi)

                    d = simMeasures['dist']
                    data[(me, mi)] = d

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

                ax = sns.heatmap(mat.T, vmin=0, vmax=1, square=True, cmap='viridis', annot_kws={"size": 4}, xticklabels=x_me, yticklabels=y_mi, annot=True)
                ax.set_title(f"Subject {s}, task {task}")
                plt.xlabel("M_e")
                plt.ylabel("M_i")
                plt.yticks(rotation=0) 

                plt.savefig(os.path.join(dir, f"fit_{s}_{task}.png"), dpi=300)
                plt.clf()



if __name__ == "__main__":
    prepro()
