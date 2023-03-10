# Distributed Training with PyTorch and Intel® Transfer Learning Tool

## Prerequisites

- Participating nodes should have Intel® oneAPI Base Toolkit installed. Verify the files under `/opt/intel/oneapi`
- Participating nodes should have passwordless SSH setup. Instructions to setup are given below.

### Passwordless SSH setup

- Use an existing (or create an) SSH key pair.

    - Check under `~/.ssh` and see if they exist. If present, make sure they have default names `(id_rsa.pub id_rsa)` and they don't have any passphrase.

    - To remove passphrase, type `ssh-keygen -p [-P old_passphrase] [-N new_passphrase] [-f keyfile]` by replacing `new_passphrase` with a blank space.

- How to create SSH key pair:

    - Get to your .ssh directory `cd ~/.ssh` (if this gives you an error, change the permissions: `chmod u+x ~/.ssh`)

    - Run the command: `ssh-keygen -t rsa`

    - The first prompt will ask you what you want to call your key files `(id_rsa.pub id_rsa)`. Press `<enter>` to use the default key names.

    - The second prompt will ask for passphrase. Do not enter any passphrase, just press `<enter>`.

- Locate the two ssh key pair files in your `.ssh` directory (`id_rsa.pub`, `id_rsa`):

    - Open the Public Key in an editor like vi/vim/nano/pico (this is the `.pub` file)

    - The ending of the public key may say `<your_idsid>@<hostname.domain>`, edit this file to omit the `"@<hostname.domain>"` at the end. The result will be your `idsid` only.

    - Create a file in your .ssh directory called `authorized_keys`

    - Paste your entire public key into this file

    - Make sure your new ssh key pair files AND `authorized_keys` files are read-write only for yourself with no permissions for anyone else `(chmod 600 file1 file2 file3)`

- Test the SSH `ssh <ip_or_hostname.domain>`

IMPORTANT NOTE: You have to make sure the `authorized_keys` file exists on all of the target systems that will participate in running the workload (in your local home dir in your `.ssh` directory) with contents of public key inside as well.

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

**Step 2:** Launch a distributed training job with Intel Transfer Learning Tool CLI using the appropriate flags.
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

**Fix:** Kill the port from the terminal (or) log out and log in again to free the port.

- "HTTP Connection error" - Might happen if there are several attempts to train text classification model
as it uses Hugging Face API to make calls to get dataset, model, tokenizer.

**Fix:** Wait for about 2 min and try again.
