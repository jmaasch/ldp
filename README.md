# Local Discovery by Partitioning: Polynomial-Time Causal Discovery around Exposure-Outcome Pairs

### Demo
To demo LDP on a linear-Gaussian DAG, run `python ldp_demo.py -x=1 -m=0 -b=1 -n=5000 -a=0.005 -r=10`.

### Files
- `ldp.py`: The main function of class `LDP` is `partition_z()``.
- `ldp_utils.py`: Class `LDPUtils` provides some basic helper functions for experiments.
- `data_generation.py`: Class `DataGeneration` provides a basic linear-Gaussian data generating process for demonstration purposes.
- `ldp_demo.py`: This script provides a demo of LDP functionality on a linear-Gaussian DAG.
