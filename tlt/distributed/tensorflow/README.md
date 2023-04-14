# Distributed Training with TensorFlow and Intel® Transfer Learning Tool

## Prerequisites

Prior to going over setup, ensure you 2 or more Servers/VMs with following specs:

- Each server has passwordless SSH access to the other servers
- Python 3.8 or newer
- Python Pip
- Python Virtualenv
- Python Development tools
- GCC/G++ 8 or newer
- CMake 3 or newer
- OpenMPI(or MPICH)

## Multinode setup

You can choose to install multinode dependencies in your existing TLT virtualenv (or) you can create a new virtualenv and install TLT and multinode dependencies.

Step 1: Create a virtualenv and activate it (or) activate your existing TLT virtualenv

```
virtualenv -p python3 tlt_tf_multinode
source tlt_tf_multinode/bin/activate
```

Step 2: Install TLT from the `setup.py` script (You can skip this step if you already have TLT installed)

```
pip install --editable .
```

Step 3: Install multinode dependencies

```
bash tf_hvd_setup.sh
```

**Note:** Repeat the steps (or) copy over the virtualenv for **all** the participating nodes and make sure all the nodes have the virtualenv in the **same location**.

## Verify multinode setup

Run any of the following commands on a head node by providing required env variables and IP addresses. You should see a list of hostnames of the nodes.

### Using `mpirun`
```
source tlt_tf_multinode/bin/activate && \
mpirun --allow-run-as-root -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 \
    -np 4 \
    -H node_01:2,node_02:2 \
    hostname
```

### Using `horovodrun`
```
source tlt_tf_multinode/bin/activate && \
horovodrun \
    -np 4 \
    -H node_01:2,node_02:2 \
    hostname
```

## Launch a distributed training job with TLT CLI

**Step 1:** Create a hostfile with a list of IP addresses of the participating nodes. Make sure the first IP address to be of the current node. The IP addresses can be in any of the following varieties.
```
"127.0.0.1"
"127.0.0.1 slots=2"
"127.0.0.1:2"
"hostname-example.com"
"hostname-example.com slots=2"
"hostname-example.com:2"
```
**Step 2:** Launch a distributed training job with TLT CLI using the appropriate flags.

```
tlt train \
    -f tensorflow \
    --dataset-dir $DATASET_DIR \
    --output-dir $OUTPUT_DIR \
    --dataset-name cifar10 \
    --model-name efficientnet_b0 \
    --distributed \
    --hostfile hostfile \
    --nnodes 2 \
    --nproc_per_node 2
```

(Optional): Use the `--use_horovod` flag to run using horovodrun instead of default mpirun.