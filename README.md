# QAL-BP: An Augmented Lagrangian Quantum Approach for Bin Packing
This repository contains the code to reproduce the results presented in the paper *QAL-BP: An Augmented Lagrangian Quantum Approach for Bin Packing*, currently under peer review for publication in Scientific Reports.

# Contents
[Description](#desc)

[Methods](#methods)

[Results](#results)

[Usage](#use)


<a name="desc"></a>
## Description

We propose QAL-BP (*Quantum Augmented Lagrangian method for Bin Packing*), a novel QUBO formulation for the BPP based on the Augmented Lagrangian method.
In particular, we reformulate the BPP in terms of a Quadratic Unconstrained Binary Optimization
(QUBO) problem so that it is possible to leverage existing quantum algorithms to obtain the best solution.
Thus we perform a comparative analysis in terms of time complexity between the proposed quantum algorithm and the most popular classical baselines: *branch-and-bound* and *simulated annealing*.
Finally, since QUBO problems can be solved operating with quantum annealing, we run
QAL-BP on small-size problems using a real quantum annealer device ([Dwave](https://www.dwavesys.com/)).


<a name="methods"></a>
## Methods
The code is organized in a single notebook where four different solvers are implemented to solve and benchmark a set of small-sized bin packing instances.
A brief description of them are as follows:
- *Gurobi*: uses the [**Gurobi**](https://www.gurobi.com/) solver to solve the ILP formulation of the Bin Packing problem
- *Quantum Annealing*: uses the [**D-Wave**](https://www.dwavesys.com/) quantum annealer to solve the input QUBO problem.
- *Simulated Annealing*: uses the [**D-Wave**](https://www.dwavesys.com/) simulated annealer to solve the input QUBO problem.
- *Exact Solver*: uses enumearion to solve the QUBO formulation.

<a name="results"></a>
## Results

The results reported in the paper are contained in the results folder. 
In particular in the path results/AL/ it is possible to
check the results of the four methods mentioned above.

There are three csv files, one for eache QUBO solver, and all the three contain also the results of branch-and-bound.


<a name="use"></a>
## Usage

The only thing needed to run the notebook is to write a D'Wave token in the config file.

At the moment is not possible to change QUBO penalties through the config file, but this will be updated in the future.

If specified, some function can also save the sampleset in the *solutions* folder



## Issues

For any issues or questions related to the code, open a new git issue or send a mail to
[lorenzo.cellini3@studio.unibo.it](lorenzo.cellini3@studio.unibo.it).
