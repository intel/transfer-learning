# Helm chart values

The tables below list and describe the variables used in the Helm chart. The default parameters listed below reflect
what is set in the [`chart/values.yaml`](chart/values.yaml) file. The values set in other values files for specific use
case will vary. The values for these parameters can be modified to run your own workload.

## PyTorch Job parameters

This example use the [Kubeflow PyTorchJob training operator](https://www.kubeflow.org/docs/components/training/pytorch/)
in order to launch a distributed fine tuning job. These parameters specify the metadata for the job.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `metadata.name` | string | `distributed-llama2` | Prefix used to name the Kubernetes resources. For example, the PyTorchJob will be named distributed-llama2-pytorchjob, the PVC will be name distributed-llama2-pvc, etc.  |
| `metadata.namespace` | string | `kubeflow` | The name of the Kubernetes namespace where the job will be deployed. |

## Secret parameters

A Kubernetes secret is to store your Hugging Face token.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `secret.encodedToken` | string | None | Hugging Face token encoded using base64. |

## Docker image

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `image.name` | string | `intel/ai-workflow` | Name of the image to use for the PyTorch job. The container should include the fine tuning script and all the dependencies required to run the job. |
| `image.tag` | string | `torch-2.0.1-huggingface-multinode-py3.9` | The image tag for the container that will be used to run the PyTorch job. The container should include the fine tuning script and all the dependencies required to run the job. |
| `image.pullPolicy` | string | `IfNotPresent` | Determines when the kubelet will pull the image to the worker nodes. Choose from: `IfNotPresent`, `Always`, or `Never`. If updates to the image have been made, use `Always` to ensure the newest image is used. |

## Elastic policy

These parameters are used by the Kubeflow PyTorchJob training operator. For more information on the available
parameters, see the [Kubeflow Elastic Policy documentation](https://github.com/kubeflow/training-operator/blob/master/sdk/python/docs/KubeflowOrgV1ElasticPolicy.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `elasticPolicy.rdzvBackend` | string | `c10d` | The rendezvous backend type (c10d, etcd, or etcd-v2). |
| `elasticPolicy.minReplicas` | integer | `1` | The lower limit for the number of replicas to which the job can scale down. |
| `elasticPolicy.maxReplicas` | integer | `4` | The upper limit for the number of pods that can be set by the autoscaler. Cannot be smaller than `elasticPolicy.minReplicas` or `distributed.workers`. |
| `elasticPolicy.maxRestarts` | integer | `10` | The maximum number of restart times for pods in elastic mode. |

## Distributed job parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distributed.workers` | integer | `4` | The number of worker pods to deploy. |
| `distributed.script` | string | `/workspace/scripts/finetune.py` | The script that will be executed using `torch.distributed.launch`. |
| `distributed.modelNameOrPath` | string | `meta-llama/Llama-2-7b-hf` | The name or path of the pretrained model to pass to the Hugging Face transformers training arguments.|
| `distributed.logLevel` | string | `info` | The Hugging Face Transformers logging level (`debug`, `info`, `warning`, `error`, and `critical`. |
| `distributed.doTrain` | bool | `True` | If set to True, training will be run using the Hugging Face Transformers library. |
| `distributed.doEval` | bool | `True` | If set to True, evaluation will be run with the validation split of the dataset, using the Hugging Face Transformers library. |
| `distributed.doBenchmark` | bool | `False` | If set to True, the Intel Neural Compressor will be used to benchmark the trained model. If the model is being quantized, the quantized model will also be benchmarked. |
| `distributed.doQuantize` | bool | `False` | If set to True, the Intel Neural Compressor will be used to quantize the trained model. |

### Training parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distributed.train.datasetName` | string | None | Name of a Hugging Face dataset to use. If no dataset name is provided, the dataFile path will be used instead. |
| `distributed.train.dataFile` | string | None | Path to a Llama2 formatted data file to use, if no dataset name is provided. |
| `distributed.train.datasetConcatenation` | bool | `True` | Whether to concatenate the sentence for more efficient training. |
| `distributed.train.perDeviceBatchSize` | integer | `8` | The batch size per device. |
| `distributed.train.epochs` | integer | `3` | Number of training epochs to perform. |
| `distributed.train.maxSteps` | integer | `-1` | If set to a positive number, the total number of training steps to perform. Overrides the number of training epochs. |
| `distributed.train.gradientAccumulationSteps` | integer | `1` | Number of updates steps to accumulate before performing a backward/update pass. |
| `distributed.train.learningRate` | float | `2e-4` | The initial learning rate. |
| `distributed.train.ddpFindUnusedParameters` | bool | `False` | The `find_used_parameters` flag to pass to DistributedDataParallel. |
| `distributed.train.ddpBackend` | string | `ccl` | The backend to be used for distributed training. It is recommended to use `ccl` with the Intel Extension for PyTorch. |
| `distributed.train.useFastTokenizer` | bool | `False` | Whether to use one of the fast tokenizer (backed by the tokenizers library) or not. |
| `distributed.train.outputDir` | string | `/tmp/pvc-mount/output/saved_model` | The output directory where the model predictions and checkpoints will be written. |
| `distributed.train.loggingSteps` | integer | `10` | Log every X updates steps. Should be an integer or a float in range `[0,1]`. If smaller than 1, will be interpreted as ratio of total training steps. |
| `distributed.train.saveTotalLimit` | integer | `2` | If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`. |
| `distributed.train.saveStrategy` | string | `epoch` | The checkpoint save strategy to use (`no`, `steps`, or `epoch`). |
| `distributed.train.useLora` | boolean | `True` | Whether or not to use LoRA. |
| `distributed.train.loraRank` | integer | `8` | Rank parameter in the LoRA method. |
| `distributed.train.loraAlpha` | integer | `32` | Alpha parameter in the LoRA method. |
| `distributed.train.loraDropout` | float | `0.05` | Dropout parameter in the LoRA method. |
| `distributed.train.loraTargetModules` | string | `q_proj v_proj` | Target modules for the LoRA method. |
| `distributed.train.noCuda` | bool | `True` | Use CPU when set to True. |
| `distributed.train.overwriteOutputDir` | bool | `True` | Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory. |
| `distributed.train.bf16` | bool | `True` | Whether to use bf16 (mixed) precision instead of 32-bit. Requires hardware that supports bfloat16. |
| `distributed.train.useIpex` | bool | `True` | Use Intel extension for PyTorch when it is available. |

### Evaluation parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distributed.eval.perDeviceBatchSize` | integer | `8` | Batch size to use for evaluation for each device. |
| `distributed.eval.validationSplitPercentage` | float | `0.20` | The percentage of the train set used as validation set in case there's no validation split. Set to 0.20 for a 20% validation split. |

### Benchmark parameters

If benchmarking is enabled, the [Intel Neural Compressor](https://github.com/intel/neural-compressor)
is used to [benchmark](https://github.com/intel/neural-compressor/blob/master/docs/source/benchmark.md) the trained
model. If quantization is also enabled, the quantized model will also get benchmarked so that the performance of the
full precision model can be compared with the quantized model.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distributed.benchmark.warmup` | integer | `10` | When benchmarking is enabled, the number of iterations to warmup before running performance tests. |
| `distributed.benchmark.iterations` | integer | `100` | When benchmarking is enabled, the number of iterations to run performance tests. |
| `distributed.benchmark.coresPerInstance` | integer | `-1` | When benchmarking is enabled, the number of CPU cores to use per instance |
| `distributed.benchmark.numInstances` | integer | `1` | When benchmarking is enabled, the number of instances to use for performance testing. |

### Quantization parameters

If quantization is enabled, the [Intel Neural Compressor](https://github.com/intel/neural-compressor) is used to
quantize the trained model as a way of reducing the number of bits required to improve inference speed. In this case,
[weight only quantization (WOQ](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md)
is used, as it is recommended for use with large language models to provide a good balance between performance and
accuracy.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `distributed.quantize.peftModelDir` | string | `/tmp/pvc-mount/output/saved_model` | When quantization is enabled, the path the Peft model to load. If Peft was not used during training, leave this field blank and the `distributed.modelNameOrPath` will be loaded for quanitization. |
| `distributed.quantize.outputDir` | string | `/tmp/pvc-mount/output/quantized_model` | Location to save the quantized model, when quantization is enabled. |
| `distributed.quantize.woqBits` | integer | `8` | Bits for weight only quantization, 1-8 bits. |
| `distributed.quantize.woqGroupSize` | integer | `-1` | Bits for weight only quantization, 1-8 bits. |
| `distributed.quantize.woqScheme` | string | `sym` | Scheme for weight only quantization. Choose from 'sym' and 'asym'. |
| `distributed.quantize.woqAlgo` | integer | `-1` | Algorithm for weight only quantization. Choose from: RTN, AWQ, GPTQ, or TEQ. |

### Environment variables

The table below lists environment variables that will get set in the worker pods.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| envVars.ldPreload | string | `/opt/conda/lib/libiomp5.so:/opt/conda/lib/libtcmalloc.so` | Paths set to the LD_PRELOAD environment variable. |
| envVars.logLevel | string | `INFO` | Value set to the LOG_LEVEL environment variable. |
| envVars.transformersCache | string | `/tmp/pvc-mount/transformers_cache` | Location for the Transformers cache (using the TRANSFORMERS_CACHE environment variable). |
| envVars.hfDatasetsCache | string | `/tmp/pvc-mount/dataset_cache` | Path to a directory used to cache Hugging Face datasets using the HF_DATASETS_CACHE environment variable. |
| envVars.cclWorkerCount | string | `1` | Value for the CCL_WORKER_COUNT environment variable. Must be >1 to use the CCL DDP backend. |
| envVars.httpProxy | string | None | Set the http_proxy environment variable. |
| envVars.httpsProxy | string | None | Set the https_proxy environment variable. |
| envVars.noProxy | string | None | Set the no_proxy environment variable. |
| envVars.ftpProxy | string | None | Set the ftp_proxy environment variable. |
| envVars.socksProxy | string | None | Set the socks_proxy environment variable. |

### Resource parameters

The resource parameters define which type of nodes the worker pods will run on and the amount of CPU and memory
resources that will be allocated toward the pods.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| resources.cpuRequest | string | None | Optionally specify the amount of CPU resources requested for each worker, where 1 CPU unit is equivalent to 1 pysical CPU core or 1 virtual core. For more information see the [Kubernetes documentation on CPU resource units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu). |
| resources.cpuLimit | string | None | Optionally specify the maximum amount of CPU resources for each worker, where 1 CPU unit is equivalent to 1 pysical CPU core or 1 virtual core. For more information see the [Kubernetes documentation on CPU resource units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu). |
| resources.memoryRequest | string | None | Optionally specify the amount of memory resources requested for each worker. For more information see the [Kubernetes documentation on memory resource units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory). |
| resources.memoryLimit | string | None | Optionally specify the maximum amount of memory resources for each worker. For more information see the [Kubernetes documentation on memory resource units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory). |
| resources.nodeSelectorLabel | string | None | Optionally specify a label for the type of node that will be used for the PyTorch job workers. |
| resources.nodeSelectorValue | string | None | If `resources.nodeSelectorLabel` is set, specify the value for the node selector label. |

### Storage parameters

The storage parameters defined the [persistent volume claim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
that will be created and used with the PyTorch job. The PVC is used to store artifacts from the job such as checkpoints,
saved model files, etc.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| stroage.storageClassName | string | `nfs-client` | Name of the storage class to use for the persistent volume claim. To list the available storage classes use: `kubectl get storageclass` |
| storage.resources | string | `50Gi` | Specify the [capacity](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#capacity) for the persistent volume claim. |
| storage.pvcMountPath | string | `/tmp/pvc-mount` | The location where the persistent volume claim will be mounted in the worker pods. |
