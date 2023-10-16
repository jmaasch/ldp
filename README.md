# Local Discovery by Partitioning: Polynomial-Time Causal Discovery around Exposure-Outcome Pairs

### Files
- `ldp.py`: The main function of class `LDP` is `partition_z()`.
- `ldp_utils.py`: Class `LDPUtils` provides some basic helper functions for experiments.
- `data_generation.py`: Class `DataGeneration` provides a basic linear-Gaussian data generating process for demonstration purposes.
- `ldp_demo.py`: This script provides a demo of LDP functionality on a linear-Gaussian DAG.

### Demo

We provide a script to demo LDP on a linear-Gaussian DAG using the Fisher-z independence test. This DAG can optionally contain an M-structure, a butterfly structure, or both. X can be a direct cause of Y, or have no direct effect.

```bash
python ldp_demo.py -x=1 -m=0 -b=0 -n=5000 -a=0.005 -r=10 -e=0
```

**Arguments:**
- `-x` (int): whether X directly causes Y or not (1 = True, 0 = False).
- `-m` (int): whether the DAG contains an M-structure or not (1 = True, 0 = False).
- `-b` (int): whether the DAG contains a butterfly structure or not (1 = True, 0 = False).
- `-n` (int): sample size.
- `-a` (float): alpha for p-value of independence test.
- `-r` (int): total replicate DAGs to run.
- `-e` (int): whether to export results or not (1 = True, 0 = False).
