
import matplotlib as mpl
import pandas as pd
import numpy as np
import math
from gurobipy import *
import pyqubo
from pyqubo import Array, Constraint, LogEncInteger, OneHotEncInteger
import time
import neal
import json
import collections.abc
import os
import copy
import matplotlib.pyplot as plt
from collections import Counter
import scienceplots
import config as config
from itertools import cycle



# Dwaves
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LazyFixedEmbeddingComposite
from dwave.embedding import chain_break_frequency

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm


def generate_instances(num_items_list, bin_capacity):
    """
    Generate a dictionary of instances for Bin Packing Problems (BPP).

    Given a list of 'num_items' and a 'bin_capacity', this function generates a dictionary of BPP instances.
    Each instance is represented as a dictionary with keys 'num_items' and 'bin_capacity', and they are stored in
    the 'instances' dictionary with unique names based on the input values.

    Args:
        num_items_list (list): A list of integers representing the number of items in each instance.
        bin_capacity (int): The capacity of the bins in the instances.

    Returns:
        dict: A dictionary where keys are unique instance names and values are dictionaries representing BPP instances.
    """
    # Create an empty dictionary to store instances
    instances = {}

    # Iterate through each value in the num_items_list
    for n in num_items_list:
        # Create an empty dictionary to represent an instance
        inst = {}

        # Assign the 'num_items' key in the instance dictionary to the current value of 'n'
        inst['num_items'] = n

        # Assign the 'bin_capacity' key in the instance dictionary to the provided 'bin_capacity'
        inst['bin_capacity'] = bin_capacity

        # Generate a unique name for the instance based on 'n' and 'bin_capacity'
        name = f'bpp_{n}_{bin_capacity}'

        # Add the instance dictionary to the 'instances' dictionary with the generated name as the key
        instances[name] = inst

    # Return the dictionary containing all generated instances
    return instances



def append_value(dict_obj, key, value):
    """
    Append a value to a dictionary under the given key.

    If the key already exists in the dictionary, the provided value is added to the existing value.
    If the key is not present, a new key-value pair is created in the dictionary.

    Args:
        dict_obj (dict): The dictionary to which the value should be appended.
        key (hashable): The key under which the value should be appended or created.
        value: The value to append or set for the specified key.

    Returns:
        dict: The updated dictionary containing the appended or newly created key-value pair.
    """
    if key in dict_obj:
        # Key exists in dict, add the value to the existing value
        dict_obj[key] += value
    else:
        # Key is not in dict, create a new key-value pair
        dict_obj[key] = value
    
    return dict_obj


def lb(bin_capacity, item_weights):
    """
    Compute the lower bound on the number of bins required for packing items based on their total volume.

    This function calculates the lower bound on the number of bins needed to pack a set of items with given weights
    into bins with a specified capacity. The lower bound is computed as the total volume of the items divided by the
    bin capacity (assuming uniform capacity for each bin).

    Args:
        bin_capacity (float): The capacity of each bin.
        item_weights (list of float): A list of item weights (volumes).

    Returns:
        int: The lower bound on the number of bins required, rounded up to the nearest integer.
    """
    return int(math.ceil(sum(item_weights) / bin_capacity))



import numpy as np
import pandas as pd

def instance_generator(instances, seed):
    """
    Generate a DataFrame of Bin Packing Problem (BPP) instances for experimentation.

    This function takes a dictionary of instance specifications and generates a DataFrame containing BPP instances
    with specified characteristics. Each instance includes the instance name, random seed, bin capacity, item weights,
    number of items, minimum weight, maximum weight, and a lower bound estimate.

    Args:
        instances (dict): A dictionary of instance specifications, where keys are instance names, and values are
                          dictionaries containing 'num_items' and 'bin_capacity'.
        seed (int): A random seed for reproducible data generation.

    Returns:
        pd.DataFrame: A DataFrame containing BPP instances with the following columns:
            - 'instance_name': Name of the BPP instance.
            - 'seed': The provided random seed for reproducibility.
            - 'c': Bin capacity for the instance.
            - 'w': List of item weights.
            - 'n': Number of items in the instance.
            - 'wmin': Minimum item weight.
            - 'wmax': Maximum item weight.
            - 'lb': Lower bound estimate on the number of bins required for packing.
    """
    bpp_data_set = []
    rng = np.random.default_rng(seed)  # Initialize a random number generator with the given seed

    for name, data in instances.items():
        num_items = data['num_items']
        bin_capacity = data['bin_capacity']
        item_weight_range = [4, max(10, bin_capacity)]
        weights = list(rng.integers(*item_weight_range, num_items))

        bpp_data_set.append({'instance_name': name, 'seed': seed, 'c': bin_capacity, 'w': weights})

    # Create a DataFrame from the generated data
    df = pd.DataFrame(bpp_data_set, columns=['instance_name', 'seed', 'c', 'w'])

    # Calculate additional columns for analysis
    df['n'] = df['w'].apply(len)
    df['wmin'] = df['w'].apply(min)
    df['wmax'] = df['w'].apply(max)
    df['lb'] = df.apply(lambda x: lb(x['c'], x['w']), axis=1)

    return df



def model_bpp(c, w, UB=None, bin_for_item=None, LogToConsole=True, TimeLimit=30):
    """
    Solve the Bin Packing Problem (BPP) using the Gurobi optimization solver.

    This function formulates and solves the BPP as an integer linear programming (ILP) model using the Gurobi solver.
    It aims to minimize the number of bins used to pack a set of items with given weights and a bin capacity 'c'.

    Args:
        c (float): The capacity of each bin.
        w (list of float): A list of item weights.
        UB (int): An upper bound on the number of bins to consider (default is 'n').
        bin_for_item (list of int, optional): A list specifying the initial bin assignment for each item.
        LogToConsole (bool, optional): Whether to log the solver's progress to the console (default is True).
        TimeLimit (int, optional): Time limit in seconds for solving the optimization problem (default is 30).

    Returns:
        tuple: A tuple containing the following information:
            - obj_value (float): The objective value, i.e., the number of bins used.
            - obj_bound (float): The objective bound provided by the solver.
            - bin_for_item (list of int): The final bin assignment for each item.
            - runtime_microseconds (float): Runtime of the solver in microseconds.

    Note:
        The 'bin_for_item' argument allows you to specify initial bin assignments for items. If not provided,
        the solver will determine the assignments.

    Example:
        obj_value, obj_bound, bin_assignment, runtime_microseconds = model_bpp(10, [3, 5, 7, 2, 1])
    """
    n = len(w)
    LB = lb(c, w)
    if UB is None:
        UB = n
    if LogToConsole:
        print('c =', c, '| n =', n, '| LB =', LB, '| UB =', UB)
    model = Model()
    model.params.LogToConsole = LogToConsole
    model.params.TimeLimit = TimeLimit  # seconds
    x = model.addVars(UB, n, vtype=GRB.BINARY)
    y = model.addVars(UB, vtype=GRB.BINARY)
    model.setObjective(quicksum(y[i] for i in range(UB)), GRB.MINIMIZE)  # minimize the number of bins used
    model.addConstrs(quicksum(x[i, j] for i in range(UB)) == 1 for j in range(n))  # each item in exactly one bin
    model.addConstrs(quicksum(w[j] * x[i, j] for j in range(n)) <= c * y[i] for i in range(UB))  # limit total weight in each bin; also, link x_ij with y_j

    if bin_for_item is not None:
        for i in range(n):
            x[bin_for_item[i], i].start = 1
    model.optimize()
    bin_for_item = [-1 for i in range(n)]
    for i in range(UB):
        for j in range(n):
            if x[i, j].X > 0.5:
                bin_for_item[j] = i
    return model.ObjVal, model.ObjBound, bin_for_item, round(model.runtime, 6) * 1000000  # Convert runtime to microseconds


def gurobi_solve(df, num_reads):
    """
    Solve Bin Packing Problems (BPP) using the Gurobi solver for multiple instances.

    This function takes a DataFrame containing BPP instances and solves them using the Gurobi solver. It generates
    solutions for each instance, including the number of bins used, the bin assignment for each item, the runtime,
    and bin filling information.

    Args:
        df (pd.DataFrame): A DataFrame containing BPP instances with columns: 'instance_name', 'c' (bin capacity),
                           and 'w' (item weights).
        num_reads (int): The number of times to run the solver for each instance to obtain multiple solutions.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns for each solved instance:
            - 'gurobi_n_bins': The number of bins used.
            - 'gurobi_bin_for_item': The bin assignment for each item.
            - 'gurobi_runtime': The average runtime (in microseconds) over multiple solver runs.
            - 'bin_filling': Information about item filling in each bin.

    Note:
        The function uses the 'model_bpp' function to solve each BPP instance.

    Example:
        solutions_df = gurobi_solve(df_instances, num_reads=5)
    """
    gurobi_solutions = []
    columns = ['gurobi_n_bins', 'gurobi_bin_for_item', 'gurobi_runtime', 'bin_filling']

    for _, row in df.iterrows():
        bin_item = {}
        instance_name, c, w = row.instance_name, row.c, row.w

        time = []
        for _ in range(num_reads):
            _, _, _, t = model_bpp(c, w, LogToConsole=False, TimeLimit=300)
            time.append(t)

        n_bins, _, bin_for_item, _ = model_bpp(c, w, LogToConsole=False, TimeLimit=300)

        for i in range(len(w)):
            bin_filling = append_value(bin_item, bin_for_item[i], w[i])

        gurobi_solutions.append({'gurobi_n_bins': n_bins, 'gurobi_bin_for_item': bin_for_item,
                                 'gurobi_runtime': np.mean(time), 'bin_filling': bin_filling})

    df_temp = pd.DataFrame(gurobi_solutions, columns=columns)

    return df_temp


def generate_QUBO_AL(n, UB, w, c, penalties=None, al_gamma_term=False):
    """
    Generate a Quadratic Unconstrained Binary Optimization (QUBO) model using the Augmented Lagrangian method.

    This function constructs a QUBO model to solve the Bin Packing Problem (BPP) using the Augmented Lagrangian (AL) method.
    The QUBO model is used to minimize the number of bins used for packing items with given weights and a bin capacity.

    Args:
        n (int): The number of items to be packed.
        UB (int): An upper bound on the number of bins to consider.
        w (list of float): A list of item weights.
        c (float): The capacity of each bin.
        penalties (dict, optional): Dictionary containing penalty parameters (delta, theta, lmbd, rho, gamma) used in
                                    the AL method (default is None).
        al_gamma_term (bool, optional): Whether to include a gamma term in the QUBO model for linking constraints
                                       (default is False).

    Returns:
        Tuple: A tuple containing the following elements:
            - H (pyqubo.Operator): The QUBO Hamiltonian representing the objective and constraints.
            - num_var (int): The total number of decision variables in the QUBO model.

    Note:
        The function uses the pyqubo library to create the QUBO model.

    Example:
        H, num_var = generate_QUBO_AL(5, 10, [3, 4, 2, 5, 1], 10, penalties={'delta': 1.0, 'theta': 1.0, 'lmbd': 1.0,
                                                                       'rho': 1.0, 'gamma': 1.0}, al_gamma_term=True)
    """
    # Generate problem variables
    y = Array.create('y', shape=UB, vartype="BINARY")
    x = Array.create('x', shape=(UB, n), vartype="BINARY")

    num_var = len(y) + (x.shape[0] * x.shape[1])

    # Define the objective: minimize the number of used bins
    H_obj = sum(y[i] for i in range(UB))

    # Define the constraint: one item in exactly one bin
    ha = np.array(sum([x[i, :] for i in range(UB)]) - 1) ** 2
    H_A = sum(ha[j] for j in range(n))

    # Define linear and quadratic constraints on bin capacity
    hb = np.array(sum([w[j] * x[:, j] for j in range(n)]))
    H_B1 = sum(hb[i] - c * y[i] for i in range(UB))
    H_B2 = sum((hb[i] - c * y[i]) ** 2 for i in range(UB))

    delta, theta, lmbd, rho, gamma = penalties.values() if penalties else (1.0, 1.0, 1.0, 1.0, 1.0)

    if al_gamma_term:
        # Linking constraint between x_ij and y_i
        hb3 = np.array(sum([x[:, j] for j in range(n)]))
        H_B3 = sum((1 - y[i]) * hb3[i] for i in range(UB))

    if al_gamma_term:
        H = delta * H_obj + theta * Constraint(H_A, label='H_A') + lmbd * Constraint(H_B1, label='H_B1') + \
            rho * Constraint(H_B2, label='H_B2') + gamma * Constraint(H_B3, label='H_B3')
    else:
        H = delta * H_obj + theta * Constraint(H_A, label='H_A') + lmbd * Constraint(H_B1, label='H_B1') + \
            rho * Constraint(H_B2, label='H_B2')

    return H, num_var


def generate_QUBO_pseudopoly(n, UB, w, c, penalties=None, gamma_term=False):
    """
    Generate a pseudo-polynomial QUBO formulation for the Bin Packing Problem based on the Lodewijks model.

    This function constructs a Quadratic Unconstrained Binary Optimization (QUBO) model that represents the Bin Packing
    Problem (BPP) using a pseudo-polynomial approach inspired by Lodewijks. The QUBO model is used to minimize the number
    of bins used for packing items with given weights and a bin capacity.

    Args:
        n (int): The number of items to be packed.
        UB (int): An upper bound on the number of bins to consider.
        w (list of float): A list of item weights.
        c (float): The capacity of each bin.
        penalties (dict, optional): Dictionary containing penalty parameters (A, B) used in the pseudo-polynomial
                                    formulation (default is None).
        gamma_term (bool, optional): Whether to include a gamma term in the QUBO model for linking constraints
                                     (default is False).

    Returns:
        Tuple: A tuple containing the following elements:
            - H (pyqubo.Operator): The QUBO Hamiltonian representing the objective and constraints.
            - num_var (int): The total number of decision variables in the QUBO model.

    Note:
        The function uses the pyqubo library to create the QUBO model.

    Example:
        H, num_var = generate_QUBO_pseudopoly(5, 10, [3, 4, 2, 5, 1], 10, penalties={'A': 1.0, 'B': 0.5}, gamma_term=True)
    """
    # Generate problem variables
    y = Array.create('y', shape=UB, vartype="BINARY")
    x = Array.create('x', shape=(UB, n), vartype="BINARY")
    z = Array.create('z', shape=(UB, c), vartype="BINARY")

    num_var = len(y) + (x.shape[0] * x.shape[1]) + (z.shape[0] * z.shape[1])

    # Define objective: minimize the number of used bins
    H_obj = sum(y[i] for i in range(UB))

    # Define constraint: each bin is filled up to the capacity level k
    ha1 = np.array(sum([z[:, k] for k in range(c)]))
    H_A1 = sum((y[i] - ha1[i]) ** 2 for i in range(UB))

    # Define constraint: one item in exactly one bin
    ha2 = np.array(sum([x[i, :] for i in range(UB)]) - 1) ** 2
    H_A2 = sum(ha2[j] for j in range(n))

    # Define quadratic constraint on bin capacity - makes use of 1-hot encoding of slack variable
    ha31 = np.array(sum([(k + 1) * z[:, k] for k in range(c)]))
    ha32 = np.array(sum([w[j] * x[:, j] for j in range(n)]))
    H_A3 = sum((ha31[i] - ha32[i]) ** 2 for i in range(UB))

    # Define linking constraint between xij and yi
    ha4 = np.array(sum([x[:, j] for j in range(n)]))
    H_A4 = sum((1 - y[i]) * ha4[i] for i in range(UB))

    if penalties is None:
        B = 1
        A = 2 * B + 1
    else:
        A, B = penalties.values()

    H = B * H_obj + A * Constraint(H_A1, label='H_A1') + A * Constraint(H_A2, label='H_A2') + \
        A * Constraint(H_A3, label='H_A3') + A * Constraint(H_A4, label='H_A4')

    return H, num_var



def solve_model_SA(bqm, num_reads):
    """
    Solve a Binary Quadratic Model (BQM) using Simulated Annealing.

    This function uses the Simulated Annealing sampler to find approximate solutions for a given BQM. It returns the
    best sample, the average runtime per read in microseconds, and the full sample set.

    Args:
        bqm (dimod.BinaryQuadraticModel): The Binary Quadratic Model to be solved.
        num_reads (int): The number of reads (samples) to generate.
        model: Unused argument (reserved for future use).

    Returns:
        Tuple: A tuple containing the following elements:
            - best_sample (dict): The best sample found.
            - avg_runtime_microseconds (float): Average runtime per read in microseconds.
            - sample_set (dimod.SampleSet): The full set of samples obtained.

    Note:
        The function uses the Neal library's SimulatedAnnealingSampler for solving.
    """
    sa = neal.SimulatedAnnealingSampler()
    start = time.time()
    sampleset = sa.sample(bqm, seed=42, num_reads=num_reads, beta_range=(0.1, 100.0), beta_schedule_type='geometric')
    runtime = time.time() - start

    # return sampleset.first, (runtime / num_reads) * 1000000, sampleset
    return sampleset.first, runtime * 1000000, sampleset


def solve_model_QA(bqm, num_reads):
    """
    Solve a Binary Quadratic Model (BQM) using Quantum Annealing.

    This function uses a Quantum Annealing sampler (D-Wave) to find approximate solutions for a given BQM. It returns
    the best sample, the total runtime in microseconds, and the full sample set.

    Args:
        bqm (dimod.BinaryQuadraticModel): The Binary Quadratic Model to be solved.
        num_reads (int): The number of reads (samples) to generate.
        model: Unused argument (reserved for future use).

    Returns:
        Tuple: A tuple containing the following elements:
            - best_sample (dict): The best sample found.
            - total_runtime_microseconds (float): Total runtime in microseconds.
            - sample_set (dimod.SampleSet): The full set of samples obtained.

    Note:
        The function uses the D-Wave Quantum Annealing sampler via the EmbeddingComposite.
    """
    sampler = EmbeddingComposite(DWaveSampler(token=config.dimod_token))
    sample_set = sampler.sample(bqm, num_reads=num_reads, return_embedding=True)
    embedding = sample_set.info['embedding_context']['embedding']
    cbf = sample_set.first.chain_break_fraction
    logiqu = len(embedding.keys())
    physiqu = sum(len(chain) for chain in embedding.values())
    runtime = sample_set.info['timing']['qpu_sampling_time']
    runtime_metrics = sample_set.info['timing']

    return sample_set.first, runtime, sample_set, runtime_metrics, cbf, logiqu, physiqu


def solve_model_Ex(bqm, model):
    """
    Solve a Binary Quadratic Model (BQM) using Exact Solver.

    This function uses an exact solver to find the optimal solution for a given BQM. It returns the best sample, the
    total runtime in microseconds, and the full sample set.

    Args:
        bqm (dimod.BinaryQuadraticModel): The Binary Quadratic Model to be solved.
        model: Unused argument (reserved for future use).

    Returns:
        Tuple: A tuple containing the following elements:
            - best_sample (dict): The best sample found (optimal solution).
            - total_runtime_microseconds (float): Total runtime in microseconds.
            - sample_set (dimod.SampleSet): The full set of samples obtained.

    Note:
        The function uses dimod's ExactSolver for exact solving, with a specified time limit.
    """
    exact_solver = dimod.ExactSolver()
    start = time.time()
    sample_set = exact_solver.sample(bqm, time_limit=10)
    runtime = time.time() - start

    return sample_set.first, runtime * 1000000, sample_set


def sampleset_to_json(sampleset, path, instance_name):
    """
    Save a Dimod SampleSet to a JSON file.

    This function takes a Dimod SampleSet, a specified file path, and an instance name, and saves the SampleSet to a
    JSON file at the specified location. If the folder directory does not exist, it creates the directory.

    Args:
        sampleset (dimod.SampleSet): The SampleSet to be saved.
        path (str): The directory path where the JSON file will be saved.
        instance_name (str): The name of the BPP instance, used as part of the JSON file's name.

    Returns:
        None

    Note:
        The function converts the SampleSet to a JSON-serializable object before saving it.

    Example:
        sampleset_to_json(sampleset, 'results/', 'example_instance')
    """
    if not os.path.exists(path):
        # If the folder directory is not present, then create it.
        os.makedirs(path)

    # Convert solutions to a JSON-serializable object and save to a JSON file
    with open(f'{path}{instance_name}_{int(time.time())}.json', "w") as fp:
        json.dump(sampleset.to_serializable(), fp)


def solve_model(H, num_reads, solver: list):
    """
    Solve a Quadratic Unconstrained Binary Optimization (QUBO) model with selected solvers.

    This function takes a QUBO model, a specified number of reads (samples) to generate, and a list of selected solvers
    ('Ex' for Exact Solver, 'SA' for Simulated Annealing, and 'QA' for Quantum Annealing). It compiles the model and
    solves it using the chosen solvers. The function returns a dictionary containing the best sample, runtime, and
    SampleSet for each solver used.

    Args:
        H (qubovert.Operator): The QUBO Hamiltonian to be solved.
        num_reads (int): The number of reads (samples) to generate for each solver.
        solver (list): A list of selected solvers among 'Ex' (Exact Solver), 'SA' (Simulated Annealing),
                       and 'QA' (Quantum Annealing).

    Returns:
        dict: A dictionary containing solver-specific results. The keys are solver names ('Ex', 'SA', 'QA'), and the
              values are dictionaries with the following keys:
              - 'best_sample' (dict): The best sample found.
              - 'runtime' (float): Total runtime in microseconds.
              - 'sampleset' (dimod.SampleSet): The full set of samples obtained.

    Note:
        The function internally uses the 'solve_model_Ex', 'solve_model_SA', and 'solve_model_QA' functions to solve
        the QUBO model with the selected solvers.

    Example:
        sol = solve_model(H, num_reads=1000, solver=['SA', 'QA'])
    """
    assert len(solver) > 0, 'Please select at least one solver among QA and SA'

    # Compile the model BQM to solve it with the desired solver
    model = H.compile()
    bqm = model.to_bqm()

    sol = {}

    if 'Ex' in solver:
        # Solving with Exact Solver
        best_sample_Ex, runtime_Ex, sampleset_Ex = solve_model_Ex(bqm, model)
        sol['Ex'] = {'best_sample': best_sample_Ex, 'runtime': runtime_Ex, 'sampleset': sampleset_Ex}

    if 'SA' in solver:
        # Solving with Simulated Annealing
        best_sample_SA, runtime_SA, sampleset_SA = solve_model_SA(bqm, num_reads)
        sol['SA'] = {'best_sample': best_sample_SA, 'runtime': runtime_SA, 'sampleset': sampleset_SA}

    if 'QA' in solver:
        # Solving with Quantum Annealing
        best_sample_QA, runtime_QA, sampleset_QA, runtime_metrics, cbf, logiqu, physiqu = solve_model_QA(bqm, num_reads)
        sol['QA'] = {'best_sample': best_sample_QA, 'runtime': runtime_QA, 'sampleset': sampleset_QA, 'runtime_metrics': runtime_metrics, 'cbf': cbf, 'logiqu': logiqu, 'physiqu': physiqu}

    return sol



def compute_used_bins(sample):
    """
    Compute the list of used bins based on a sample from a QUBO model.

    Given a sample obtained from solving a QUBO model, this function extracts and computes the list of used bins
    for items and bins represented in the sample.

    Args:
        sample (dict): A sample obtained from solving a QUBO model.

    Returns:
        Tuple: A tuple containing two lists:
            - yi (list): A list of indices representing bins with at least one item.
            - xij (list): A list of indices representing items placed in bins.

    Example:
        sample = {'y0': 1, 'y1': 0, 'x[0][0]': 1, 'x[1][0]': 0, 'x[0][1]': 1, 'x[1][1]': 0}
        yi, xij = compute_used_bins(sample)
        # yi = [0], xij = ['0']
    """
    # Extract 'y' variables (bins) from the sample
    sol = {key: value for key, value in sample.sample.items() if 'y' in key.lower()}

    # Extract 'x' variables (items in bins) that are set to 1 in the sample
    a = {key: value for key, value in sample.sample.items() if value == 1 and 'x' in key.lower()}

    # Extract indices representing items placed in bins ('xij')
    b = [v.split('][')[0].split('[')[1] for v in a.keys()]
    xij = list(set(b))

    # Extract indices representing bins with at least one item ('yi')
    yi = list(set([k.split(']')[0].split('[')[1] for k, v in sol.items() if v == 1]))

    return yi, xij


def compute_eigenval_prob(min_eigenvalue, sampleset):
    """
    Compute the probability of observing a specific minimum eigenvalue in a sample set.

    Given a minimum eigenvalue and a sample set obtained from solving a QUBO or Ising problem, this function computes
    the probability of observing the specified minimum eigenvalue within the sample set's energy values.

    Args:
        min_eigenvalue (float): The minimum eigenvalue to compute the probability for.
        sampleset (dimod.SampleSet): The sample set containing energy values.

    Returns:
        float: The probability of observing the specified minimum eigenvalue in the sample set.

    Example:
        min_eigenvalue = -5.0
        sampleset = <a Dimod SampleSet with energy values>
        prob = compute_eigenval_prob(min_eigenvalue, sampleset)
    """
    energies = [datum.energy for datum in sampleset.data(['energy'], sorted_by=None)]

    # Count occurrences of the minimum eigenvalue in the energy values
    evs = [ev for ev in energies if ev == min_eigenvalue]

    # Calculate the probability of observing the specified minimum eigenvalue
    prob = len(evs) / len(energies)

    return prob


def compute_assignment(best_sample, weights):
    """
    Compute the assignment of items to bins based on the best sample from a QUBO model.

    Given the best sample obtained from solving a bin packing problem represented as a QUBO model, and a list of
    weights for each item, this function computes the assignment of items to bins and returns the assignment as a
    dictionary. Additionally, it provides a list of used items.

    Args:
        best_sample (dict): The best sample obtained from solving a QUBO model.
        weights (list): A list of weights, where the weight at index 'i' corresponds to the 'i-th' item.

    Returns:
        Tuple: A tuple containing two elements:
            - assignment (dict): A dictionary where keys are bin indices, and values are the total weights of items
                                assigned to each bin.
            - used_items (list): A list of item indices that have been assigned to bins.

    Example:
        best_sample = {'x[0][0]': 1, 'x[1][0]': 1, 'x[2][1]': 1}
        weights = [3, 4, 5]
        assignment, used_items = compute_assignment(best_sample, weights)
    """
    assignment = dict()
    used_items = []

    # Extract 'x' variables (items in bins) that are set to 1 in the best sample
    a = {key: value for key, value in best_sample.sample.items() if value == 1 and 'x' in key.lower()}

    for var in a.keys():
        bin = str(var.split('][')[0].split('[')[1])
        item = int(var.split('][')[1].split(']')[0])
        used_items.append(item)

        if bin in assignment.keys():
            z = assignment[bin] + weights[item]
            assignment[bin] = z
        else:
            assignment[bin] = weights[item]

    return assignment, used_items


def parse_solutions(solutions, weights, c, save_to_json, path, instance_name, penalties_dict, num_reads, compute_tts, run_sampleset, limit_reached):
    """
    Parse and organize solutions from multiple solvers for a bin packing problem.

    This function takes solutions obtained from different solvers for a bin packing problem, along with relevant
    information, and organizes them into a structured dictionary format. It computes various metrics and checks the
    feasibility of each solution.

    Args:
        solutions (dict): A dictionary containing solutions from different solvers.
        weights (list): A list of item weights.
        c (int): The bin capacity.
        save_to_json (bool): Whether to save solutions to JSON files.
        path (str): The directory path to save JSON files.
        instance_name (str): The name of the problem instance.
        penalties_dict (dict): A dictionary of penalties for constraint terms.
        num_reads (int): The number of reads (samples) used for solving.
        compute_tts (bool): Whether to compute Time-to-Solution (TTS) metrics.
        run_sampleset (dimod.SampleSet): A SampleSet containing additional data for computing TTS.
        limit_reached (bool): Whether the computation limit was reached for TTS.

    Returns:
        dict: A structured dictionary containing parsed and organized solutions for each solver. Each solver's results
              include information such as the minimum eigenvalue, eigenvalue probability, assignment of items to bins,
              feasibility, runtime, and more.

    Example:
        solutions = {'QA': {'best_sample': <best sample data>, 'runtime': 123.45, 'sampleset': <SampleSet>},
                     'SA': {'best_sample': <best sample data>, 'runtime': 67.89, 'sampleset': <SampleSet>}}
        parsed_results = parse_solutions(solutions, weights, c, True, '/output/', 'problem_instance',
                                         penalties_dict, 1000, True, run_sampleset, False)
    """
    parsed_solutions = {}

    for solver, results in solutions.items():
        parsed_solutions[solver] = {}
        if solver == 'QA':
          best_sample, runtime, sampleset, runtime_metrics, cbf, logiqu, physiqu = results.values()
        else:
          best_sample, runtime, sampleset = results.values()

        # Save solutions to JSON files if requested
        if save_to_json:
            sampleset_to_json(sampleset, path, instance_name)

        # Compute Time-to-Solution (TTS) if enabled
        if compute_tts:
            if limit_reached:
                tts = '???'
            else:
                num_feasibles = count_feasibles(run_sampleset, weights, c)
                if solver == 'QA':
                    tts = compute_time_to_solution(num_feasibles, num_reads, sampleset.info['timing']['qpu_anneal_time_per_sample'])
                elif solver == 'SA':
                    tts = compute_time_to_solution(num_feasibles, num_reads, runtime / num_reads)

        min_eigenvalue = best_sample.energy
        yi, xij = compute_used_bins(best_sample)

        eigenval_prob = compute_eigenval_prob(min_eigenvalue, sampleset)
        assignment, used_items = compute_assignment(best_sample, weights)
        real_weights_sum = sum(weights)
        placed_weights_sum = sum(assignment.values())

        feasible, reason = check_feasibility(len(yi), len(xij), assignment, real_weights_sum, placed_weights_sum, used_items, c)

        # Convert assignment dict into a list of tuples for easier handling
        assignment = [(k, v) for k, v in assignment.items()]

        # Create a structured dictionary for the parsed solution
        d = {
            'min_eigenvalue': min_eigenvalue,
            'eigenval_prob': eigenval_prob,
            'yi': yi,
            'xij': xij,
            'runtime': runtime,
            'assignment': assignment,
            'real_weights_sum': real_weights_sum,
            'placed_weights_sum': placed_weights_sum,
            'used_items': used_items,
            'feasible': feasible,
            'reason': reason,
            'penalties_dict': penalties_dict,
        }

        # Include TTS in the result if computed
        if compute_tts:
          d['TTS'] = tts

        if solver == 'QA':
          d['runtime_metrics'] = runtime_metrics
          d['cbf'] = cbf
          d['logiqu'] = logiqu
          d['physiqu'] = physiqu

        parsed_solutions[solver] = d

    return parsed_solutions


def check_feasibility(num_yi, num_xij, assignment, real_weights_sum, placed_weights_sum, used_items, c):
    """
    Check the feasibility of a bin packing solution and provide reasons for infeasibility.

    This function checks the feasibility of a bin packing solution by comparing various properties, such as the number of
    bins used (yi), the number of items placed in bins (xij), the total real weights sum, the total placed weights sum,
    and the bin capacities. It returns a feasibility status and a list of reasons for infeasibility, if any.

    Args:
        num_yi (int): The number of bins used (yi).
        num_xij (int): The number of items placed in bins (xij).
        assignment (dict): A dictionary representing the assignment of items to bins.
        real_weights_sum (float): The total sum of item weights in the problem.
        placed_weights_sum (float): The total sum of item weights placed in bins.
        used_items (list): A list of item indices that have been assigned to bins.
        c (float): The bin capacity.

    Returns:
        Tuple: A tuple containing two elements:
            - feasible (bool): A boolean indicating whether the solution is feasible (True) or infeasible (False).
            - reason (list): A list of strings providing reasons for infeasibility, if applicable.
    """
    feasible = True
    reason = []

    # Check if the number of bins used (yi) matches the number of items placed in bins (xij)
    if num_yi != num_xij:
        feasible = False
        reason.append(f'yi:{num_yi} != xij:{num_xij}; ')

    # Check if the total real weights sum matches the total placed weights sum
    if real_weights_sum != placed_weights_sum:
        feasible = False
        reason.append(f'real_weights_sum:{real_weights_sum} != placed_weights_sum:{placed_weights_sum}; ')

    # Check if any bins are overfilled
    if any(fill > c for fill in assignment.values()):
        feasible = False
        reason.append('Some bins are overfilled; ')

    # Check if some items are used multiple times
    if len(used_items) > len(set(used_items)):
        reason.append('Some items are used multiple times.')

    return feasible, reason


def flatten(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary into a flat dictionary with concatenated keys.

    This function recursively flattens a nested dictionary into a flat dictionary by concatenating keys using a separator.
    It preserves the hierarchy of keys by adding the separator between them.

    Args:
        d (dict): The input nested dictionary to be flattened.
        parent_key (str, optional): The parent key for recursive calls. Defaults to an empty string ('').
        sep (str, optional): The separator used to concatenate keys. Defaults to '_'.

    Returns:
        dict: A flat dictionary with concatenated keys.

    Example:
        nested_dict = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        flat_dict = flatten(nested_dict)
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_AL_default_penalty(num_bin, bin_capacity, min_weight):
    """
    Calculate default penalty values for the Augmented Lagrangian method based on problem parameters.

    This function computes default penalty values (delta, theta, lmbd, rho and gamma) for the Augmented Lagrangian method
    used in this problem. The penalties are determined based on the number of bins, bin capacity and minimum item weight.

    Args:
        num_bin (int): The number of bins in the problem.
        bin_capacity (int): The capacity of each bin.
        min_weight (int): The minimum weight among the items.

    Returns:
        dict: A dictionary containing default penalty values for delta, theta, lmbd, rho, and gamma.

    Example:
        num_bin = 5
        bin_capacity = 20.0
        min_weight = 5.0
        default_penalties = get_AL_default_penalty(num_bin, bin_capacity, min_weight)
    """
    delta = 0.15
    theta = 2
    lmbd = round(bin_capacity / (min_weight * (2 * min_weight + bin_capacity)), 4)
    rho = round(lmbd * 2 / bin_capacity, 4)
    gamma = 1

    return {'delta': delta, 'theta': theta, 'lmbd': lmbd, 'rho': rho, 'gamma': gamma}


def solve_QUBOs(df, models, solvers, num_reads, al_penalties, pp_penalties, al_gamma_term, save_to_json=False, path=None):
    """
    Solve QUBO optimization problems for given instances using specified models and solvers.

    This function takes a DataFrame containing bin packing problem instances and solves them using the specified optimization
    models (Augmented Lagrangian - AL and Pseudo-Polynomial - PP) and solvers (e.g., QA, SA). It generates QUBO formulations
    for each instance based on the selected models and penalties, solves them, and collects the results. Optionally, it can
    save the results to JSON files.

    Args:
        df (pd.DataFrame): A DataFrame containing bin packing problem instances.
        models (list): A list of models to use ('AL' for Augmented Lagrangian, 'PP' for Pseudo-Polynomial).
        solvers (list): A list of solvers to use ('QA' for Quantum Annealing, 'SA' for Simulated Annealing, 'Ex' for exact method).
        num_reads (int): The number of reads (samples) to collect for each problem instance.
        al_penalties (dict): Penalty values for the Augmented Lagrangian model (delta, theta, lmbd, rho, gamma).
        pp_penalties (float): Penalty value for the Pseudo-Polynomial model (A and B).
        al_gamma_term (bool): Whether to include the gamma term in the Augmented Lagrangian model.
        save_to_json (bool, optional): Whether to save solutions to JSON files. Defaults to False.
        path (str, optional): The directory path to save JSON files if save_to_json is True. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed QUBO optimization results.

    """
    df_temp = df.copy()
    sol_dict = {}
    QUBO_solutions = []
    print(f'Set penalty as: {"Default" if al_penalties is None else al_penalties} for AL and {"Default" if pp_penalties is None else pp_penalties} for PP')
    
    for i, _ in df_temp.iterrows():
        instance_name = df_temp.loc[i,'instance_name'] + '_' + str(df_temp.loc[i,'seed'])
        n = df_temp.loc[i,'n']
        UB = n.item()
        w = df_temp.loc[i,'w']
        c = df_temp.loc[i,'c']
        wmin = df_temp.loc[i,'wmin']

        if 'AL' in models:
            if al_penalties is None:
                # print('Selecting default penalty for AL model')
                penalties = get_AL_default_penalty(n.item(), c.item(), wmin.item())
            H_lagrangian, num_var_lagrangian = generate_QUBO_AL(n, UB, w, c, penalties, al_gamma_term=al_gamma_term)
            solutions = solve_model(H_lagrangian, num_reads, solvers)

            parsed_solutions = parse_solutions(solutions, w, c, save_to_json, path, instance_name, penalties, num_reads,
                                               compute_tts=False, run_sampleset=None, limit_reached=False)
            sol_dict['AL'] = parsed_solutions

        if 'PP' in models:
            H_pseudopoly, num_var_pseudopol = generate_QUBO_pseudopoly(n, UB, w, c, pp_penalties)
            solutions = solve_model(H_pseudopoly, num_reads, solvers)

            parsed_solutions = parse_solutions(solutions, w, c, save_to_json, path, instance_name, pp_penalties,
                                               num_reads, compute_tts=False, run_sampleset=None, limit_reached=False)
            sol_dict['PP'] = parsed_solutions

        flat_parsed_solutions = flatten(sol_dict)
        columns = list(flat_parsed_solutions.keys())

        QUBO_solutions.append(flat_parsed_solutions)
    
    df_temp = pd.DataFrame(QUBO_solutions, columns=columns)

    return df_temp



def plot_num_bins(df_num_bins):
    """
    Plot a comparison of the number of bins in solutions obtained by different solvers.

    This function creates a bar chart to compare the number of bins in solutions generated by Gurobi, Simulated Annealing (SA), and
    Quantum Annealing (QA) solvers for a set of instances. The function takes a DataFrame with relevant data and generates the
    plot, showing how these solvers perform in terms of the number of bins used.

    Args:
        df_num_bins (pd.DataFrame): A DataFrame containing data about the number of bins used by different solvers for each instance.

    """
    # mpl.rcParams['figure.dpi'] = config.dpi #50 #300
    N = 40
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25        # the width of the bars
    font = 11

    plt.style.use(['science', 'nature'])

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    plt.grid(axis='y', alpha=0.5, zorder=1)
    rects1 = ax.bar(ind, df_num_bins['gurobi_n_bins'], width, color=config.color_scheme['Gurobi'], zorder=3)
    rects2 = ax.bar(ind + width, df_num_bins['AL_SA_n_bins'], width, color=config.color_scheme['SA'], zorder=3)
    rects3 = ax.bar(ind - width, df_num_bins['AL_QA_n_bins'], width, color=config.color_scheme['QA'], zorder=3, alpha=0.8)

    # Add labels and titles
    ax.set_ylabel('Number of bins', fontsize=font)
    ax.set_ylim(-0.1, 11)
    # ax.set_title('Comparison of the number of bins in solutions', fontsize=font + 2)
    ax.set_xticks(ind)
    ax.set_xticklabels('('+df_num_bins['instance_name'].apply(lambda x: x.split('_')[1])+','+df_num_bins['seed'].astype(str)+')', fontsize=font)
    plt.xticks(rotation=60)
    plt.yticks(fontsize=font)

    plt.gca().margins(x=0.01)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    ax.legend((rects1[0], rects2[0], rects3[0]), ('Gurobi', 'Simulated Annealing', 'Quantum Annealing'), loc='upper left', fontsize=font)
    plt.tight_layout()

    plt.savefig("num_bins.png", dpi=config.dpi)
    plt.show()


def plot_complexity():
    """
    Generate a plot illustrating the model complexity with respect to the number of items and bin capacity.

    This function creates a plot to visualize the model complexity of different approaches, including Pseudo-Polynomial (PP)
    and Augmented Lagrangian (AL), with varying numbers of items and bin capacities. The plot shows the number of variables in
    the models as a function of the number of items, providing insights into the computational complexity.

    """
    plt.style.use(['science', 'nature'])

    # colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']

    font = 11
    c = [10, 25, 50]
    pseudopol_10 = []
    pseudopol_25 = []
    pseudopol_50 = []
    al = []
    nitem = []

    # Calculate model complexities for different numbers of items
    for n in range(0, 30, 5):
        nitem.append(n)
        pseudopol_10.append(n * (n + 1 + c[0]))
        pseudopol_25.append(n * (n + 1 + c[1]))
        pseudopol_50.append(n * (n + 1 + c[2]))
        al.append(n * (n + 1))

    # Create the plot
    plt.figure(figsize=(6, 4.5))

    plt.plot(nitem, pseudopol_10, label='Pseudo-Pol C=10', color=config.color_scheme['color6'])
    plt.plot(nitem, pseudopol_25, label='Pseudo-Pol C=25', color=config.color_scheme['color7'])
    plt.plot(nitem, pseudopol_50, label='Pseudo-Pol C=50', color=config.color_scheme['color8'])
    plt.plot(nitem, al, label='Augm. Lagrangian', color=config.color_scheme['color9'])
    plt.axhline(180, color='darkred', linestyle='dotted')

    plt.xlabel(r'Number of items $(n)$', fontsize=font)
    plt.ylabel('Number of variables', fontsize=font)
    plt.ylim(-5, 250)
    plt.grid(alpha=0.3)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend(fontsize=font)
    plt.xticks(np.arange(0, 30, 5), fontsize=font)
    plt.yticks(np.arange(0, 1000, 100), fontsize=font)
    plt.tick_params(labelsize=9)
    # plt.title('Model complexity with respect to number of items and bin capacity', fontsize=font)
    plt.savefig('num_vars_n.png', dpi=config.dpi)
    plt.show()


def plot_runtime(df):
    """
    Generate a plot to visualize the Time to Solution (TTS) performance of QAL-BP for different solvers.

    This function creates a plot to illustrate the Time to Solution (TTS) performance of the QAL-BP (Quantum Augmented Lagrangian
    for the Bin Packing Problem) solver for various problem instances. It compares the TTS results for different solvers,
    including Gurobi, Simulated Annealing (SA) with Augmented Lagrangian (AL), and Quantum Annealing (QA) with Augmented Lagrangian (AL).

    Args:
        df (DataFrame): A DataFrame containing mean and standard deviation of runtimes for different solvers
            and problem instances.

    """

    plt.style.use(['science', 'nature'])
    font = 11
    runtime = [col for col in df.columns if '_mean' in col]

    # Legend labels for different solvers
    leg = {
        'gurobi_runtime_mean': 'Gurobi',
        'AL_SA_runtime_mean': 'Simulated Annealing - AL',
        'AL_QA_runtime_mean': 'Quantum Annealing - AL'
    }

    color_link = {
        'gurobi_runtime_mean': 'Gurobi',
        'AL_SA_runtime_mean': 'SA',
        'AL_QA_runtime_mean': 'QA'
    }

    # colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']

    plt.figure(figsize=(6, 4.5))

    # Plot TTS for different solvers
    for i, r in enumerate(runtime[:3]):
        plt.plot(df.loc[:, 'n'], df.loc[:, r], color=config.color_scheme[color_link[r]], label=leg[r])
        r_std = f'{r[:-5]}_std'
        upper_stock = np.sum([np.round(df.loc[:, r], 0), np.round(df.loc[:, r_std], 0)], 0)
        lower_stock = np.sum([np.round(df.loc[:, r], 0), -np.round(df.loc[:, r_std], 0)], 0)
        plt.plot(df.loc[:, 'n'], upper_stock, color=config.color_scheme[color_link[r]], linestyle='dotted')
        plt.plot(df.loc[:, 'n'], lower_stock, color=config.color_scheme[color_link[r]], linestyle='dotted')

        plt.fill_between(df.loc[:, 'n'], upper_stock, lower_stock, alpha=0.1)

    plt.xlim(2, 11)
    plt.grid(alpha=0.3)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend(fontsize=font)
    plt.tick_params(labelsize=9)
    plt.xlabel('Number of items (n)', fontsize=font)
    plt.ylabel('Time to solution (TTS)', fontsize=font)
    # plt.title('Time performance of QAL-BP in $\mu$s', fontsize=font + 2)
    plt.savefig("tts.png", dpi=config.dpi)
    plt.show()


def plot_runtime_logscale(df):
    """
    Generate a log-scale plot to visualize the Time to Solution (TTS) performance of QAL-BP for different solvers.

    This function creates a log-scale plot to illustrate the Time to Solution (TTS) performance of the QAL-BP (Quantum Augmented Lagrangian
    for the Bin Packing Problem) solver for various problem instances. It compares the TTS results for different solvers,
    including Gurobi, Simulated Annealing (SA) with Augmented Lagrangian (AL), Quantum Annealing (QA) with Augmented Lagrangian (AL),
    and an Exact Solver with Augmented Lagrangian (AL).

    Args:
        df (DataFrame): A DataFrame containing mean and standard deviation of runtimes for different solvers
            and problem instances.

    """
    plt.style.use(['science', 'nature'])

    runtime = [col for col in df.columns if '_mean' in col]

    # Legend labels for different solvers
    leg = {
        'gurobi_runtime_mean': 'Gurobi',
        'AL_SA_runtime_mean': 'Simulated Annealing - AL',
        'AL_QA_runtime_mean': 'Quantum Annealing - AL',
        'AL_Ex_runtime_mean': 'Exact Solver - AL'
    }

    color_link = {
        'gurobi_runtime_mean': 'Gurobi',
        'AL_SA_runtime_mean': 'SA',
        'AL_QA_runtime_mean': 'QA',
        'AL_Ex_runtime_mean': 'Ex'
    }

    # Define custom colors for the plot
    # colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']

    font = 14
    plt.figure(figsize=(6, 4.5))

    for i, r in enumerate(runtime[:4]):
        plt.plot(df.loc[:, 'n'], df.loc[:, r], color=config.color_scheme[color_link[r]], label=leg[r])
        r_std = f'{r[:-5]}_std'
        upper_stock = np.sum([np.round(df.loc[:, r], 0), np.round(df.loc[:, r_std], 0)], 0)
        lower_stock = np.sum([np.round(df.loc[:, r], 0), -np.round(df.loc[:, r_std], 0)], 0)
        plt.plot(df.loc[:, 'n'], upper_stock, color=config.color_scheme[color_link[r]], linestyle='dotted')
        plt.plot(df.loc[:, 'n'], lower_stock, color=config.color_scheme[color_link[r]], linestyle='dotted')

    plt.fill_between(df.loc[:, 'n'], upper_stock, lower_stock, alpha=0.1)

    plt.yscale('log')
    plt.ylim(0, 10 ** 12)

    plt.xlim(2, 11)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, bbox_to_anchor=(1.01, .8), ncol=1)
    plt.tick_params(labelsize=9)
    plt.xlabel('Number of items (n)', fontsize=9)
    plt.ylabel('Time to solution (TTS)', fontsize=9)
    # plt.title('Log time performance of QAL-BP in $\mu$s', fontsize=11)
    plt.savefig("tts_log.png", dpi=config.dpi)
    plt.show()


def plot_eigenval(df_eigen):
    """
    Generate a bar plot to compare the energy of the best solutions found by Simulated, Quantum, and Exact solvers.

    This function creates a bar plot to visually compare the energy (minimum eigenvalue) of the best solutions found by Simulated Annealing (SA)
    with Augmented Lagrangian (AL) and Quantum Annealing (QA) with Augmented Lagrangian for different problem instances. It provides insights
    into the relative quality of solutions obtained by these solvers.

    Args:
        df_eigen (DataFrame): A DataFrame containing minimum eigenvalues for Simulated Annealing (SA) and Quantum Annealing (QA) results
            for different problem instances.

    """
    N = 40
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25        # the width of the bars
    font = 9

    plt.style.use(['science', 'nature'])

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind - width / 2, df_eigen['AL_SA_min_eigenvalue'], width, color=config.color_scheme['SA'])
    rects2 = ax.bar(ind + width / 2, df_eigen['AL_QA_min_eigenvalue'], width, color=config.color_scheme['QA'])

    ax.set_ylabel('Energy', fontsize=font)
    ax.set_xlabel('Instance', fontsize=font)
    ax.set_ylim(-.1, 3)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.set_title('Comparison of the energy of the best solution found by Simulated and Quantum solvers', fontsize=font+2)
    ax.set_xticks(ind)
    ax.set_xticklabels('('+df_eigen['instance_name'].apply(lambda x: x.split('_')[1])+','+df_eigen['seed'].astype(str)+')')
    plt.xticks(rotation=90)
    ax.legend((rects1[0], rects2[0]), ('Simulated Annealing', 'Quantum Annealing'), fontsize=font)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig("minimum_eigenvals.png", dpi=config.dpi)
    plt.show()


def plot_feasibility_density(df_feasible_density):
    """
    Generate a bar plot to visualize the probability of finding feasible solutions for different instance sizes.

    This function creates a bar plot to compare the probability of finding feasible solutions by Simulated Annealing (SA) and Quantum Annealing (QA)
    for different problem instance sizes. It provides insights into the likelihood of obtaining feasible solutions for various problem complexities.

    Args:
        df_feasible_density (DataFrame): A DataFrame containing feasibility probabilities for SA and QA results for different instance sizes.

    """
    N = 8
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    plt.style.use(['science', 'nature'])
    font = 11
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, df_feasible_density['AL_SA_feasible'], width, color=config.color_scheme['SA'], alpha=.9)
    rects2 = ax.bar(ind + width + 0.03, df_feasible_density['AL_QA_feasible'], width, color=config.color_scheme['QA'], alpha=.9)

    ax.set_ylabel('Probability', fontsize=font)
    ax.set_xlabel('Instance size', fontsize=font)
    # ax.set_title('Probability for the found minimum to be feasible', fontsize=font+2)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(df_feasible_density.index.to_series().apply(lambda x: x.split('_')[1]), fontsize=font)
    plt.yticks(fontsize=font)
    plt.xticks(rotation=0)
    ax.legend((rects1[0], rects2[0]), ('Simulated Annealing', 'Quantum Annealing'), fontsize=font, bbox_to_anchor=(1.01, .8), ncol=1)
    plt.tight_layout()
    plt.savefig("feasible_density.png", dpi=config.dpi)
    plt.show()


def find_first_feasible(sampleset, weights, bin_capacity):
    """
    Find the index of the first feasible solution in a given sample set.

    This function iterates through the solutions in a sample set and checks each one for feasibility based on the bin packing problem constraints.
    It returns the index of the first feasible solution encountered in the sample set or -1 if no feasible solution is found.

    Args:
        sampleset (dimod.SampleSet): A sample set containing solutions to the bin packing problem.
        weights (list): A list of item weights.
        bin_capacity (int): The capacity of each bin.

    Returns:
        int: The index of the first feasible solution in the sample set, or -1 if no feasible solution is found.

    """
    feasible = False
    i = -1
    for datum in sampleset.data(fields=['sample']):
        i += 1
        yi, xij = compute_used_bins(datum)
        assignment, used_items = compute_assignment(datum, weights)
        real_weights_sum = sum(weights)
        placed_weights_sum = sum(assignment.values())
        feasible, reason = check_feasibility(len(yi), len(xij), assignment, real_weights_sum, placed_weights_sum, used_items, bin_capacity)
        if feasible:
            return i
    return -1


def plot_enumerate(results, title=None, first_feasible=None, save_fig=False):
    """
      Plot the energy values of solutions in a results object.

      Args:
          results (dimod.SampleSet): A SampleSet object containing solutions.
          title (str): A title for the plot (optional).
          first_feasible (int): Index of the first feasible solution (optional).
          save_fig (bool): Whether to save the plot as a PNG file (optional).

      Returns:
          None
    """
    plt.style.use(['science','nature'])

    plt.figure(figsize=(10,4))

    energies = [datum.energy for datum in results.data(
        ['energy'], sorted_by='energy')]

    if results.vartype == 'Vartype.BINARY':
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        plt.xlabel('bitstring for solution')

    else:
        samples = np.arange(len(energies))
        plt.xlabel('Solutions', fontsize=9)

    plt.bar(samples,energies)
    plt.xticks(rotation=90)
    plt.ylabel('Energy', fontsize=9)
    # plt.title(str(title), fontsize=12)

    if first_feasible is not None:
      plt.axhline(energies[first_feasible], color='darkred')

    if save_fig:
      plt.savefig('plot_enumerate.png', bbox_inches='tight', dpi=config.dpi)
    plt.show()
    return


def plot_energies(results, title=None, first_feasible=None):
  """
    Plot the energy distribution of solutions in a results object.

    Args:
        results (dimod.SampleSet): A SampleSet object containing solutions.
        title (str): A title for the plot (optional).
        first_feasible (int): Index of the first feasible solution (optional).

    Returns:
        None
  """
  plt.figure(figsize=(20,4))

  energies = results.data_vectors['energy']
  occurrences = results.data_vectors['num_occurrences']
  counts = Counter(energies)
  total = sum(occurrences)
  counts = {}
  for index, energy in enumerate(energies):
      if energy in counts.keys():
          counts[energy] += occurrences[index]
      else:
          counts[energy] = occurrences[index]
  for key in counts:
      counts[key] /= total
  df = pd.DataFrame.from_dict(counts, orient='index').sort_index()

  plt.bar(df.index.astype(str), df[0])
  plt.xticks(df.index.astype(str), df.index, rotation=90)
  plt.xlabel('Energy', fontsize=9)
  plt.ylabel('Probabilities', fontsize=9)
#   plt.title(str(title), fontsize=12)
  # print(energies[first_feasible])
  if first_feasible is not None:
    plt.axvline(df.index[first_feasible].astype(str), color='darkred')
  plt.show()
  return


def count_feasibles(sampleset, weights, bin_capacity):
  """
    Count the number of feasible solutions in a SampleSet.

    Args:
        sampleset (dimod.SampleSet): A SampleSet object containing solutions.
        weights (list): A list of item weights.
        bin_capacity (float): The capacity of each bin.

    Returns:
        int: The number of feasible solutions in the SampleSet.
  """
  feasible = False
  num_feasibles = 0

  for datum in sampleset.data(fields=['sample', 'num_occurrences','energy']):
    yi, xij = compute_used_bins(datum)
    assignment, used_items = compute_assignment(datum, weights)
    real_weights_sum = sum(weights)
    placed_weights_sum = sum(assignment.values())
    feasible, reason = check_feasibility(len(yi), len(xij), assignment, real_weights_sum, placed_weights_sum, used_items, bin_capacity)
    if feasible:
      num_feasibles+=1*datum.num_occurrences
  return num_feasibles

def compute_time_to_solution(n_solve, n_a, qpu_anneal_time_per_sample):
  """
    Compute the Time to Solution (TTS) for a quantum annealing run.

    Args:
        n_solve (int): Number of successfully solved instances.
        n_a (int): Number of attempted instances.
        qpu_anneal_time_per_sample (float): Annealing time per sample in microseconds.

    Returns:
        float: The estimated TTS in microseconds.
  """
  p_solve = n_solve/n_a
  if np.log(1-p_solve) != 0:
    tts = (np.log(1-.99)/np.log(1-p_solve))*qpu_anneal_time_per_sample
  else:
    tts = qpu_anneal_time_per_sample
  return tts

def solve_QUBOs_QA_TTS(df, models, num_reads, al_penalties, pp_penalties, al_gamma_term, save_to_json=False, path=None):
  """
    Solve Quadratic Unconstrained Binary Optimization Problems (QUBOs) using Quantum Annealing (QA) and estimate Time to Solution (TTS).

    Args:
        df (pd.DataFrame): DataFrame containing problem instances.
        models (list): List of modeling approaches ('AL' for Augmented Lagrangian, 'PP' for Pseudo-Polynomial).
        num_reads (int): Number of reads or samples to generate.
        al_penalties (dict): Penalties for Augmented Lagrangian modeling (optional).
        pp_penalties (float): Penalty for Pseudo-Polynomial modeling (optional).
        al_gamma_term (bool): Augmented Lagrangian gamma term (optional).
        save_to_json (bool): Whether to save results to JSON files (optional).
        path (str): Path to save JSON files (optional).

    Returns:
        pd.DataFrame: DataFrame containing solution details and estimated TTS.

    Note:
        This function solves QUBOs using Quantum Annealing (QA) and estimates the Time to Solution (TTS)
        by finding the first feasible solution within a limited number of iterations (iter_limit).
  """
  df_temp = df.copy()
  sol_dict = {}
  QUBO_solutions = []
  iter_limit = 5

  print(f'Set penalty as: {"Default" if al_penalties is None else al_penalties} for AL and {"Default" if pp_penalties is None else pp_penalties} for PP')
  for i, row in df_temp.iterrows():

    instance_name = df_temp.loc[i,'instance_name']
    n = df_temp.loc[i,'n']
    UB = n.item()
    w = df_temp.loc[i,'w']
    c = df_temp.loc[i,'c']
    wmin = df_temp.loc[i,'wmin']

    print(f'Solving instance: {instance_name}')
    if 'AL' in models:
      if al_penalties is None:
        penalties = get_AL_default_penalty(n.item(), c.item(), wmin.item())

      H_lagrangian, _ = generate_QUBO_AL(n, UB, w, c, penalties, al_gamma_term=al_gamma_term)

      first_feasible = -1
      solutions = None
      sampleset = None
      run_sampleset = []
      limit_reached = False
      j=0

      # Loop to find the first feasible solution across several samples
      while first_feasible==-1 and j<iter_limit:
        print('iteration: ',j)
        solutions = solve_model(H_lagrangian, num_reads, ['QA'])
        sampleset = solutions['QA']['sampleset']
        first_feasible = find_first_feasible(sampleset, w, c)
        if j>0:
          run_sampleset = dimod.concatenate((run_sampleset, sampleset))
        else:
          run_sampleset = sampleset
        j+=1
      if j==iter_limit:
        limit_reached = True

      parsed_solutions = parse_solutions(solutions, w, c, save_to_json, path, instance_name, penalties, num_reads, compute_tts=True, run_sampleset=run_sampleset, limit_reached=limit_reached)

      sol_dict['AL'] = parsed_solutions

    flat_parsed_solutions = flatten(sol_dict)
    columns = list(flat_parsed_solutions.keys())

    QUBO_solutions.append(flat_parsed_solutions)
  df_temp = pd.DataFrame(QUBO_solutions, columns=columns)

  return df_temp

def solve_QUBOs_SA_TTS(df, models, num_reads, al_penalties, pp_penalties, al_gamma_term, save_to_json=False, path=None):
  """
    Solve Quadratic Unconstrained Binary Optimization Problems (QUBOs) using Simulated Annealing (SA) and estimate Time to Solution (TTS).

    Args:
        df (pd.DataFrame): DataFrame containing problem instances.
        models (list): List of modeling approaches ('AL' for Augmented Lagrangian, 'PP' for Pseudo-Polynomial).
        num_reads (int): Number of reads or samples to generate.
        al_penalties (dict): Penalties for Augmented Lagrangian modeling (optional).
        pp_penalties (float): Penalty for Pseudo-Polynomial modeling (optional).
        al_gamma_term (float): Augmented Lagrangian gamma term (optional).
        save_to_json (bool): Whether to save results to JSON files (optional).
        path (str): Path to save JSON files (optional).

    Returns:
        pd.DataFrame: DataFrame containing solution details and estimated TTS.

    Note:
        This function solves QUBOs using Simulated Annealing (SA) and estimates the Time to Solution (TTS)
        by finding the first feasible solution within a limited number of iterations (iter_limit).
  """
  df_temp = df.copy()
  sol_dict = {}
  QUBO_solutions = []
  iter_limit = 5

  print(f'Set penalty as: {"Default" if al_penalties is None else al_penalties} for AL and {"Default" if pp_penalties is None else pp_penalties} for PP')
  for i, row in df_temp.iterrows():

    instance_name = df_temp.loc[i,'instance_name']
    n = df_temp.loc[i,'n']
    UB = n.item()
    w = df_temp.loc[i,'w']
    c = df_temp.loc[i,'c']
    wmin = df_temp.loc[i,'wmin']
    print(f'Solving instance: {instance_name}')

    if 'AL' in models:
      if al_penalties is None:
        print('Selecting default penalty for AL model')
        penalties = get_AL_default_penalty(n.item(), c.item(), wmin.item())
      H_lagrangian, num_var_lagrangian = generate_QUBO_AL(n, UB, w, c, penalties, al_gamma_term=al_gamma_term)

      first_feasible = -1
      solutions = None
      sampleset = None
      run_sampleset = []
      limit_reached = False

      # Loop to find the first feasible solution across several samples
      j=0
      while first_feasible==-1 and j<iter_limit:
        print('iteration: ',j)
        solutions = solve_model(H_lagrangian, num_reads, ['SA'])
        sampleset = solutions['SA']['sampleset']
        first_feasible = find_first_feasible(sampleset, w, c)
        if j>0:
          run_sampleset = dimod.concatenate((run_sampleset, sampleset))
        else:
          run_sampleset = sampleset
        j+=1
      if j==iter_limit:
        limit_reached = True

      parsed_solutions = parse_solutions(solutions, w, c, save_to_json, path, instance_name, penalties, num_reads, compute_tts=True, run_sampleset=run_sampleset, limit_reached=limit_reached)

      sol_dict['AL'] = parsed_solutions

    flat_parsed_solutions = flatten(sol_dict)
    columns = list(flat_parsed_solutions.keys())

    QUBO_solutions.append(flat_parsed_solutions)
  df_temp = pd.DataFrame(QUBO_solutions, columns=columns)

  return df_temp


def plot_tts(df_mean_std_tts):
  
  plt.style.use(['science','nature'])

  runtime = [col for col in df_mean_std_tts.columns if '_mean' in col]
  leg = {
      'gurobi_runtime_mean': 'Gurobi',
      'AL_SA_TTS_mean': 'Simulated Annealing - AL',
      'AL_QA_TTS_mean': 'Quantum Annealing - AL'
  }
  color_link = {
      'gurobi_runtime_mean': 'Gurobi',
      'AL_SA_TTS_mean': 'SA',
      'AL_QA_TTS_mean': 'QA'
  }

#   colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']

  plt.figure(figsize=(6,4.5))

  for i, r in enumerate(runtime):
    plt.plot(df_mean_std_tts.loc[:,'n'], df_mean_std_tts.loc[:,r], color=config.color_scheme[color_link[r]], label=leg[r] )
    r_std = r.replace("_mean", "_std" )
    upper_stock = np.sum([np.round(df_mean_std_tts.loc[:,r],0), np.round(df_mean_std_tts.loc[:,r_std],0)],0)
    lower_stock = np.sum([np.round(df_mean_std_tts.loc[:,r],0), -np.round(df_mean_std_tts.loc[:,r_std],0)],0)
    plt.plot(df_mean_std_tts.loc[:,'n'], upper_stock, color=config.color_scheme[color_link[r]], linestyle = 'dotted')
    plt.plot(df_mean_std_tts.loc[:,'n'], lower_stock, color=config.color_scheme[color_link[r]], linestyle = 'dotted')

    plt.fill_between(df_mean_std_tts.loc[:,'n'], upper_stock, lower_stock, alpha = .1)

  plt.xlim(2, 11)
  plt.grid(alpha=0.3)
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  plt.legend(fontsize=9)
  plt.tick_params(labelsize=9)
  plt.xlabel('Number of items (n)', fontsize=9)
  plt.ylabel('Time to solution (TTS)', fontsize=9)
#   plt.title('Time to solution of QAL-BP in $\mu$s', fontsize=11)
  plt.savefig("tts.png", dpi=config.dpi)
  plt.show()

def plot_all_runtime_metrics(runtime_df):
#   mpl.rcParams['figure.dpi'] = config.dpi
  plt.style.use(['science', 'nature'])
  font = 11
  runtime = [col for col in runtime_df.columns if '_mean' in col]

  # Legend labels for different solvers
  leg = {
      
      'AL_QA_runtime_metrics_qpu_sampling_time_mean' : 'mean_qpu_sampling_time',
      'AL_QA_runtime_metrics_qpu_anneal_time_per_sample_mean' : 'mean_qpu_anneal_time_per_sample',
      'AL_QA_runtime_metrics_qpu_readout_time_per_sample_mean' : 'mean_qpu_readout_time_per_sample',
      'AL_QA_runtime_metrics_qpu_access_time_mean' : 'mean_qpu_access_time',
      'AL_QA_runtime_metrics_qpu_access_overhead_time_mean' : 'mean_qpu_access_overhead_time',
      'AL_QA_runtime_metrics_qpu_programming_time_mean' : 'mean_qpu_programming_time',
      'AL_QA_runtime_metrics_qpu_delay_time_per_sample_mean' : 'mean_qpu_delay_time_per_sample',
      'AL_QA_runtime_metrics_total_post_processing_time_mean' : 'mean_total_post_processing_time',
      'AL_QA_runtime_metrics_post_processing_overhead_time_mean' : 'mean_post_processing_overhead_time'
  }

#   colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e', '#FF9e47', '#9e47FF']
  lines = ["-","--","-.",":"]
  linecycler = cycle(lines)

  plt.figure(figsize=(10, 7))

  # Plot runtime metrics
  for i, r in enumerate(runtime):
    plt.plot(runtime_df.loc[:, 'n'], runtime_df.loc[:, r], label=leg[r], linestyle=next(linecycler), color=config.color_scheme.values[i])

  plt.xlim(2, 11)
  plt.grid(alpha=0.3)
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
  plt.legend(fontsize=font)
  plt.tick_params(labelsize=9)
  plt.xlabel('Number of items (n)', fontsize=font)
  plt.ylabel('Runtime in $\mu$s', fontsize=font)
  plt.legend(fontsize=font, bbox_to_anchor=(1.01, .8), ncol=1)
  plt.savefig("runtime_metrics.png", dpi=config.dpi)
  plt.show()

  return

def plot_chain_breaks(chain_breaks_df):
  N = 40
  ind = np.arange(N)  # the x locations for the groups
  # width = 0.35       # the width of the bars
  plt.style.use(['science', 'nature'])
  font = 11
  fig = plt.figure(figsize=(10, 5))

  ax = fig.add_subplot(111)
  plt.grid(axis='y', alpha=0.5, zorder=1)
  ax.bar(ind, chain_breaks_df['AL_QA_cbf'], zorder=3)#, color='royalblue', alpha=.9, width=1)

  plt.ylabel('Chain Break Frequency', fontsize=font)
  plt.xlabel('Instance', fontsize=font)
  ax.set_ylabel('Number of bins', fontsize=font)
  ax.set_ylim(0, np.max(chain_breaks_df['AL_QA_cbf'])+.01)
  ax.set_xticks(ind)
  ax.set_xticklabels('('+chain_breaks_df['instance_name'].apply(lambda x: x.split('_')[1])+','+chain_breaks_df['seed'].astype(str)+')', fontsize=font)

  plt.gca().margins(x=0.01)
  plt.gcf().canvas.draw()
  tl = plt.gca().get_xticklabels()
  maxsize = max([t.get_window_extent().width for t in tl])
  m = 0.2  # inch margin
  s = maxsize / plt.gcf().dpi * N + 2 * m
  margin = m / plt.gcf().get_size_inches()[0]

  plt.gcf().subplots_adjust(left=margin, right=1. - margin)
  plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
  plt.tight_layout()
  
  plt.savefig("chain_breaks_frequency.png", dpi=config.dpi)
  plt.show()
  return

def plot_phys_log_qubits(logphys_df):
#   mpl.rcParams['figure.dpi'] = config.dpi
  N = 40
  ind = np.arange(N)  # the x locations for the groups
  width = 0.45        # the width of the bars
  font = 11

  plt.style.use(['science', 'nature'])

  fig = plt.figure(figsize=(12, 7))
  ax = fig.add_subplot(111)
  plt.grid(axis='y', alpha=0.5, zorder=1)
  rects1 = ax.bar(ind, logphys_df['AL_QA_logiqu'], width, color='royalblue', zorder=3)
  rects2 = ax.bar(ind + width, logphys_df['AL_QA_physiqu'], width, color='seagreen', zorder=3)

  # Add labels and titles
  ax.set_ylabel('Number of variables/qubits', fontsize=font)
  # ax.set_ylim(-0.1, 11)
  ax.set_xticks(ind + width/2)
  ax.set_xticklabels('('+logphys_df['instance_name'].apply(lambda x: x.split('_')[1])+','+logphys_df['seed'].astype(str)+')', fontsize=font)
  plt.xticks(rotation=0)
  plt.yticks(fontsize=font)

  plt.gca().margins(x=0.01)
  plt.gcf().canvas.draw()
  tl = plt.gca().get_xticklabels()
  maxsize = max([t.get_window_extent().width for t in tl])
  m = 0.2  # inch margin
  s = maxsize / plt.gcf().dpi * N + 2 * m
  margin = m / plt.gcf().get_size_inches()[0]

  plt.gcf().subplots_adjust(left=margin, right=1. - margin)
  plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

  ax.legend((rects1[0], rects2[0]), ('Logical variables', 'Physical qubits'), loc='upper left', fontsize=font)
  plt.tight_layout()

  plt.savefig("num_bins.png", dpi=config.dpi)
  plt.show()
  return