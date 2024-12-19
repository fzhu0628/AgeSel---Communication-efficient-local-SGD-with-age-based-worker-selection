# AgeSel: Communication-Efficient Local SGD with Age-Based Worker Selection

## Overview

- This is my first journal paper titled _Communication-Efficient Local SGD with Age-Based Worker Selection_.
- The paper was published in **The Journal of Supercomputing** in 2023.
- **About this paper**:
  - The goal is to enhance the _communication efficiency_ of distributed learning systems under _intermittent communication_ between the server and workers.
  - The study considers a distributed setup with:
    - **Partial participation** of workers.
    - **Heterogeneous local datasets** (datasets vary in distribution and size across workers).
  - A simple yet effective age-based method, **_AgeSel_**, is proposed. It leverages the "ages" of workers to balance their participation frequencies.
  - The paper establishes rigorous **convergence guarantees** for AgeSel and demonstrates its effectiveness through numerical results.

## Key Ideas of AgeSel

### Key Parameters

- The system consists of \$M\$ workers. Each worker \$m\$ has an age parameter \$\\tau_m\$ maintained by the server, representing the number of consecutive rounds the worker has not communicated with the server.
- A threshold \$\\tau_{max}\$ is predefined to identify workers with low participation frequency.
- In each round, the server selects \$S\$ workers to perform local computations.

### Worker Selection

1. The server first selects all workers with \$\\tau_m \\geq \\tau_{max}\$ in _age-descending order_, ensuring that low-frequency workers are included.
2. If fewer than \$S\$ workers are selected:
   - The remaining workers are chosen with probabilities proportional to the sizes of their datasets.
3. The selection process stops when exactly \$S\$ workers are chosen.

### Age Update

- The server broadcasts the global model to the selected workers for local computations.
- After the workers complete their tasks, the server updates the age parameters.

## Results

### Convergence Analysis

- Assuming smoothness, lower boundedness of the objective function, unbiased gradients, and bounded variance, we derive an upper bound of order: $${O}\\left(\\frac{1}{\\eta UJ} + \\frac{\\eta}{SU} + \\frac{1}{\\eta U}\\right)$$ for the average expected squared gradient norm with nonconvex objectives, where:
  - \$J\$ is the total number of communication rounds.
  - \$U\$ is the number of local steps per round.
  - \$\\eta\$ is the local step size.
  - \$S\$ is the number of participating workers per round.

### Simulation Results

- Numerical examples demonstrate the effectiveness of AgeSel in terms of communication cost and training rounds:

<img src="https://github.com/user-attachments/assets/96ddd0e9-fae0-41d1-959a-649be0be64bc" width="50%" /> <img src="https://github.com/user-attachments/assets/4d150d88-4083-4254-a1d5-1fe4c510cf57" width="50%" />

## Code Description

- **`AgeSel_EMNIST_MC.py`**:
  - Implements an image classification task on the EMNIST dataset using a two-layer fully connected neural network.
  - Compares AgeSel against state-of-the-art algorithms such as FedAvg, Optimal Client Sampling (OCS), and Round Robin (RR).
  - Results are averaged over 10 Monte Carlo runs for robustness.

- **`AgeSel_CIFAR_MC.py`**:
  - Similar to the above but applied to the CIFAR-10 dataset using a CNN with 2 convolutional layers and 3 fully connected layers.
  - Demonstrates AgeSel's superiority over benchmarks.

- **`AgeSel_S.py`**:
  - Explores the impact of the hyper-parameter \$S\$ (the number of workers participating per round).

For more details, please refer to our published paper: [Springer Link](https://link.springer.com/article/10.1007/s11227-023-05190-7) or [arXiv](https://arxiv.org/abs/2210.17073).
