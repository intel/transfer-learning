# Distributed Training with PyTorch and Intel® Transfer Learning Tool

## Multinode setup

### Create and activate a Python3 virtual environment

We encourage you to use a python virtual environment (virtualenv or conda) for consistent package management. Make sure to follow only the chosen method on all the nodes. Mixing those configurations is not supported. 

There are two ways to do this:

a. Using `virtualenv`:

1. Login to one of the participating nodes.

2. Create and activate a new python3 virtualenv

```
virtualenv -p python3 tlt_dev_venv
source tlt_dev_venv/bin/activate
```

3. Install Intel® Transfer Learning Tool (see main [README](/README.md))
```
pip install --editable .
```

4. Install multinode dependencies from the requirements text file. You can also compile `torch_ccl` manually from [here](https://github.com/intel/torch-ccl)
```
pip install -r tlt/distributed/pytorch/requirements.txt
```

b. Or `conda`:

1. Login to one of the participating nodes.

2. Create and activate a new conda environment
```
conda create -n tlt_dev_venv python=3.8 --yes
conda activate tlt_dev_venv
```

3. Install Intel® Transfer Learning Tool (see main [README](/README.md))
```
pip install --editable .
```

4. Install dependencies from the shell script
```
bash tlt/distributed/pytorch/run_install.sh
```

## Verify multinode setup

Create a `hostfile` with a list of IP addresses of the participating nodes and type the following command. You should see a list of hostnames of the nodes.
```
mpiexec.hydra -ppn 1 -f hostfile hostname
```
**Note:** If the above command errors out as `'mpiexec.hydra' command not found`, activate the oneAPI environment:
```
source /opt/intel/oneapi/setvars.sh
```

## Launch a distributed training job with TLT CLI

**Step 1:** Create a `hostfile` with a list of IP addresses of the participating nodes. Make sure 
the first IP address to be of the current node.

**Step 2:** Launch a distributed training job with TLT CLI using the appropriate flags.
```
tlt train \
    -f pytorch \
    --model_name resnet50 \
    --dataset_name CIFAR10 \
    --output_dir $OUTPUT_DIR \
    --dataset_dir $DATASET_DIR \
    --distributed \
    --hostfile hostfile \
    --nnodes 2 \
    --nproc_per_node 2
```

## Troubleshooting

- "Port already in use" - Might happen when you keyboard interrupt training.

**Fix:** Release the port from the terminal (or) log out and log in again to free the port.

- "HTTP Connection error" - Might happen if there are several attempts to train text classification model
as it uses Hugging Face API to make calls to get dataset, model, tokenizer.

**Fix:** Wait for about few seconds and try again.
