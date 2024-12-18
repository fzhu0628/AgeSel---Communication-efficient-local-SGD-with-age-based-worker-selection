# AgeSel---Communication-efficient-local-SGD-with-age-based-worker-selection
## Overview
- This is my first journal paper titled _Communication-efficient local SGD with age-based worker selection_.
- This paper was published at **The Journal of Supercomputing** in 2023.
- About this paper:
  * This work aims to improve the _communication efficiency_ of distributed learning systems with _intermittent communication_ between the server and the workers.
  * We consider a distributed setting with **partial participation** of workers and **heterogeneous local datasets** across workers (heteregeneous in the sense that datasets of different workers are generated from different distributions and vary in size.)
  * An simple, yet effective age-based method called **_AgeSel_** is proposed, which uses the ''_ages_'' of workers to balance their participation frequencies.
  * Rigorous **convergence results** of Agesel are established and numerical results are provided to validate the superiority of our algorithm.
## Results
### Convergence analysis
- Assuming smoothness and lower boundedness of the objective function, and assuming unbiasedness and bounded variance, we manage to provide an upper-bound of order ${O}(\frac{1}{\eta UJ}+\frac{\eta}{SU}+\frac{1}{\eta U})$ for the average of expected sum of squared norm of gradients with nonconvex objectives, where $J$ is the total number of communication rounds, $U$ is the number of local steps taken by each worker in each round, $\eta$ is the loca stepsize, and $S$ the number of participating workers in each round.
### Simulation results
- Numerical examples:
  

<img src="https://github.com/user-attachments/assets/96ddd0e9-fae0-41d1-959a-649be0be64bc" width="50%" /> <img src="https://github.com/user-attachments/assets/4d150d88-4083-4254-a1d5-1fe4c510cf57" width="50%" />


## Codes
- In the file AgeSel_EMNIST_MC.py, we solve the image classification task of the EMNIST dataset with a two-layer fully connected NN. We compare the proposed AgeSel algorithm with state-of-the-art algorithms such as FedAvg, Optimal Client Sampling (OCS) and Round Robin (RR) in terms of training rounds and communication cost. We perform 10 Monte Carlo runs to increase the stability of the results.
- In the file AgeSel_CIFAR_MC.py, we do the same thing as above, but using the CIFAR-10 dataset and a CNN with 2 conv layers and 3 fc layers. Both results demonstrate the superiority of our algorithm.
- In the file AgeSel_S.py, we explore the impact of the hyper-parameter $S$ in the algorithm, which is the number of workers participating in each round.

We direct the reader to our published paper for more information https://link.springer.com/article/10.1007/s11227-023-05190-7 or https://arxiv.org/abs/2210.17073.
