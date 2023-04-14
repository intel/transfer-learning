# Distributed Training with PyTorch and Intel® Transfer Learning Tool

## Multinode setup

1. Login to one of the participating nodes.

2. Create a new conda environment called `multi-node`
```
conda create -n multi-node python=3.8 --yes

conda activate multi-node
```

3. Install dependencies from the shell script
```
sh run_install.sh
```

4. Install Intel® Transfer Learning Tool excluding framework dependencies (see main [README](/README.md))
```
EXCLUDE_FRAMEWORK=True pip install --editable .
```

## Verify multinode setup

Create a `hostfile` with a list of IP addresses of the participating nodes and type the following command. You should see a list of hostnames of the nodes.
```
mpiexec.hydra -ppn 1 -f hostfile hostname
```

## Launch a distributed training job with TLT CLI

**Step 1:** Create a `hostfile` with a list of IP addresses of the participating nodes. Make sure 
the first IP address to be of the current node. For testing, you can use the nodes present in this [hostfile](hostfile)

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