# AgeSel---Communication-efficient-local-SGD-with-age-based-worker-selection
## Overview
- This is my first journal paper titled _Communication-efficient local SGD with age-based worker selection_.
- This paper was published at **The Journal of Supercomputing** in 2023.
- About this paper:
  * This work aims to improve the _communication efficiency_ of distributed learning systems with _intermittent communication_ between the server and the workers.
  * We consider a distributed setting with **partial participation** of workers and **heterogeneous local datasets** across workers (heteregeneous in the sense that datasets of different workers are generated from different distributions and vary in size.)
  * An simple, yet effective age-based method called **AgeSel** is proposed, which uses the ''_ages_'' of workers to balance their participation frequencies.
  * Rigorous **convergence rates** of Agesel is established and numerical results are provided to validate the superiority of our algorithm.
