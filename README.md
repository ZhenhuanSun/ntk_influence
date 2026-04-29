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
python run_all_experiments.py
```
4. Plot figures by running the jupyter notebooks in `./plots/`


Experiments corresponding to each figure can be run by
```bash
# Figure 1
python experiments/exp_mnist_fcnn_lin.py --overwrite_results -delete_cache
python experiments/exp_cifar10_cnn_lin.py --overwrite_results -delete_cache

# Figure 2
python experiments/exp_lambda.py

# Figure 3
python experiments/exp_mnist_fcnn.py --overwrite_results -delete_cache

# Figure 4
python experiments/exp_mnist_fcnn_inf.py
python experiments/exp_cifar10_cnn_inf.py
```
