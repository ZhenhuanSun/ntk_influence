# Extending Kernel Trick to Influence Functions

## How to run the experiments

### 1. Download the repository
### 2. Create required conda environment
```bash
conda env create -f environment.yml
conda activate nt
```
### 3. Run experiments from the project root directory
Available experiment options can be found in `run_experiments.py`.

- The experiments shown in Figures 1 and 3(a) require two GPUs.
- The experiment shown in Figure 2 requires only one GPU.
- The experiment shown in Figure 3(b) can be run using either one or two GPUs. When using only one GPU, make sure the `shard`
  argument is set to `True` in `prepare_solve_delta_alpha`, `influence_on_outputs_delta_alpha`, and `influence_on_loss_delta_alpha`.

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