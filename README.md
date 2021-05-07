# Robust Federated Learning by Mixture of Experts (MoE-FL)

This repository contains implementation of "[**Robust Federated Learning by Mixture of Experts**](https://arxiv.org/abs/2104.11700)". This study presents a novel weighted average model based on the mixture of experts (MoE) concept to provide robustness in Federated learning (FL) against the poisoned/corrupted/outdated local models.

# Getting Started
We have tested and run the code using Ubuntu 20.04. You can follow the instruction below on Ubuntu 20.04, or you may install appropriate libraries considering your distribution.

## Requirements 
Before running the experiments, make sure you have installed the following packages system-wide:
```
sudo apt install \
    python3-dev build-essential libsrtp2-dev libavformat-dev \
    libavdevice-dev python3-wheel python3-venv
```
After installing the required tools, you might want to create a Python virtual environment for easier management of python packages. Therefore:
```
# Create Python virtual environment
python3 -m venv moe-fl-vene

# Activate virtual environment
source moe-fl-venv/bin/activate

# Do not forget to upgrade the pip and install wheel/setuptools
pip install --upgrade pip 
pip install wheel setuptools
```
Install python packages using:
```
pip install -r requirements.txt
```
## Configurations
You can change the configuration by modifying `configs/defaults.yml`. Some of the available parameters and their default values that you can change are:
```yml
runtime:
    epochs: 1
    rounds: 400
    lr: 0.01
    momentum: 0.5
    batch_size: 20
    random_seed: 12345
    weight_decay: 0.0
    test_batch_size: 20
    use_cuda: False # We have run all experiments using CPU only, not tested CUDA

attack:
    attackers_num: 50
    attack_type: 1

server:
    data_fraction: 0.15 # Fraction of data that will be shared with the server

mnist:
    load_fraction: 1
    shards_num: 200 # With 200 shards, there would be 300 samples per each shards
    shards_per_worker_num: 2
    selected_users_num: 30 
    total_users_num: 100 # Total number of users to partion data among them
```

## Experiments
You have two options to run the experiments, either with IID or Non-IID datasets. To do so, you can run `run-study-iid-mnist.py` or `run-study-niid-mnist.py`, respectively.

```bash
run-study-iid-mnist.py 
        (--avg | --opt) \ # average mode or optimized (moe-fl) mode
        [--epochs=NUM] \
        [--rounds=NUM] \
        [--attack-type=ID] \
        [--attackers-num=num] \[--selected-workers=NUM] \
        [--log] \ # Enable logging
        [--nep-log] \ # Enable neptune logging
        [--output-prefix=NAME] 
```

> :warning: **Due to a bug in PySyft version < 0.2.9, you cannot run a long experiment because of memory leakage. As of submitting this study, there was no solution to this problem. Hence, we have to manually break the experiments and save the states of each run and continue again. Therefore, there is an argument `--start-round` in the Non-IID experiment.**


```bash
run-study-niid-mnist.py 
        (--avg | --opt) \ # average mode or optimized (moe-fl) mode
        --start-round=NUM \ # start from previously run experiment (see warning above)
        --rounds=NUM \
        [--not-pure] \ # force the experiment use not pure dataset
        [--attackers-num=num] \
        [--log] \ # Enable logging
        [--nep-log] \ # Enable neptune logging
        [--output-prefix=NAME] 

```

# Results and Output
Each execution of the code will generate the following files:

- **accuracy**: Accuracy for each round
- **all_users**: List of all workers names
- **attackers**: List of attackers out of all workers
- **configs**: Saved configuration file used when experiment started
- **mapped_dataset**: Binary file, used to handle memory leakage problem of PySyft. We have to manage the database given to each worker and maintain the same allocation in all subsequent running.
- **models**: Folder containing saved models of workers and the server
- **seeds**: Seeds used for the experiment
- **selected_workers**: List of selected workers out of all workers
- **server_pub_dataset**: Binary file, used for saving the dataset of server
- **train_loss**: Loss of each round

