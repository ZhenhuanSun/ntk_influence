# Extending Kernel Trick to Influence Functions

## How to run the experiments

Two or more GPUs are required to run all experiments described in the paper.

### 1. Download the repository
### 2. Create required conda environment
```bash
conda env create -f environment.yml
conda activate nt
```
### 3. Run experiments from the project root directory
Available experiment options can be found in `run_experiments.py`
```bash
# Run all experiments
python run_experiments.py --all

# Run all experiments required to plot a figure (e.g., figure 1)
python run_experiments.py --figure fig1

# Run a single experiment
python run_experiments.py --experiments mnist_fcnn_lin

# Run multiple experiments
python run_experiments.py --experiments mnist_fcnn_lin cifar10_cnn_inf
```
### 4. Plot figures by running the jupyter notebooks in `./plots/`