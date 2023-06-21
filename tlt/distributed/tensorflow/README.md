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
virtualenv -p python3 tlt_dev_venv
source tlt_dev_venv/bin/activate
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
source tlt_dev_venv/bin/activate && \
mpirun --allow-run-as-root -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 \
    -np 4 \
    -H node_01:2,node_02:2 \
    hostname
```

### Using `horovodrun`
```
source tlt_dev_venv/bin/activate && \
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

## Launch a distributed training job with `horovodrun`

You can also use the `run_train_tf.py` script alone with `horovodrun` to do distributed training for TensorFlow hub/Huggingface models on TensorFlow Datasets. 

```
Distributed training with TensorFlow.

optional arguments:
  -h, --help            show this help message and exit
  --use-case {image_classification,text_classification}, --use_case {image_classification,text_classification}
                        Use case (image_classification|text_classification)
  --epochs EPOCHS       Total epochs to train the model
  --batch_size BATCH_SIZE
                        Global batch size to distribute data (default: 128)
  --batch_denom BATCH_DENOM
                        Batch denominator to be used to divide global batch size (default: 1)
  --shuffle             Shuffle dataset while training
  --scaling {weak,strong}
                        Weak or Strong scaling. For weak scaling, lr is scaled by a factor of sqrt(batch_size/batch_denom) and uses global batch size for
                        all the processes. For strong scaling, lr is scaled by world size and divides global batch size by world size (default: weak)
  --tlt_saved_objects_dir TLT_SAVED_OBJECTS_DIR
                        Path to TLT saved distributed objects. The path must be accessible to all the nodes. For example: mounted NFS drive. This arg is
                        helpful when using TLT API/CLI. See DistributedTF.load_saved_objects() for more information.
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length that the model will be used with
  --hf_bert_tokenizer HF_BERT_TOKENIZER
                        Name of the Hugging Face BertTokenizer to use to prepare the data.
  --dataset-dir DATASET_DIR, --dataset_dir DATASET_DIR
                        Path to dataset directory to save/load tfds dataset. This arg is helpful if you plan to use this as a stand-alone script. Custom
                        dataset is not supported yet!
  --output-dir OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to save the trained model and store logs. This arg is helpful if you plan to use this as a stand-alone script
  --dataset-name DATASET_NAME, --dataset_name DATASET_NAME
                        Dataset name to load from tfds. This arg is helpful if you plan to use this as a stand-alone script. Custom dataset is not
                        supported yet!
  --model-name MODEL_NAME, --model_name MODEL_NAME
                        TensorFlow image classification model url/ feature vector url from TensorFlow Hub (or) Huggingface hub name for text
                        classification models. This arg is helpful if you plan to use this as a stand-alone script.
  --image-size IMAGE_SIZE, --image_size IMAGE_SIZE
                        Input image size to the given model, for which input shape is determined as (image_size, image_size, 3). This arg is helpful if
                        you plan to use this as a stand-alone script.
```

Here are some examples:

**For image classification:**

```
horovodrun \
    -np 10 \
    -H server_1:6,server_2:4 \
    python tlt/distributed/tensorflow/run_train_tf.py \
    --use-case image_classification \
    --model-name https://tfhub.dev/google/efficientnet/b1/feature-vector/1 \
    --dataset-name cifar10
 
 ```

 **For text classification**:

 ```
 horovodrun \
    -np 10 \
    -H server_1:6,server_2:4 \
    python tlt/distributed/tensorflow/run_train_tf.py \
    --use-case text_classification \
    --model-name bert-base-uncased \
    --dataset-name imdb_reviews
 ```
 