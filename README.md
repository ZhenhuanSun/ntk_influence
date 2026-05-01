# Extending Kernel Trick to Influence Functions

## How to run the experiments

Two or more GPUs are required to run all experiments described in the paper.

1. Download the repository
2. Create required conda environment by
```bash
conda env create -f environment.yml
conda activate nt
```
3. Run all experiments at project root directory by
```bash
python run_experiments.py --all
```
4. Plot figures by running the jupyter notebooks in `./plots/`


Experiments corresponding to each figure can be run by
```bash
# run all experimetns required to plot Figure 1
python run_experiments.py --figure fig1
```

Individual experiment can be run by
```bash
# run experiment `mnist_fcnn_lin`
python run_experiments.py --experiments mnist_fcnn_lin

# run experiments `mnist_fcnn_lin` and `cifar10_cnn_inf`
python run_experiments.py --experiments mnist_fcnn_lin cifar10_cnn_inf
```