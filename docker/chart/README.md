# TLT TF Distributed Training

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| batchDenom | int | `1` | Batch denominator to be used to divide global batch size |
| batchSize | int | `128` | Global batch size to distributed data |
| datasetName | string | `"cifar10"` | Dataset name to load from tfds |
| epochs | int | `1` | Total epochs to train the model |
| imageName | string | `"intel/ai-tools"` |  |
| imageTag | string | `"0.5.0-dist-devel"` |  |
| metadata.name | string | `"tlt-distributed"` |  |
| metadata.namespace | string | `"kubeflow"` |  |
| modelName | string | `"https://tfhub.dev/google/efficientnet/b1/feature-vector/1"` | TF Hub or HuggingFace model URL |
| pvcName | string | `"tlt"` |  |
| pvcResources.data | string | `"2Gi"` | Amount of Storage for Dataset |
| pvcResources.output | string | `"1Gi"` | Amount of Storage for Output Directory |
| pvcScn | string | `"nil"` | PVC `StorageClassName` |
| resources.cpu | int | `2` | Number of Compute for Launcher |
| resources.memory | string | `"4Gi"` | Amount of Memory for Launcher |
| scaling | string | `"strong"` | For `weak` scaling, `lr` is scaled by a factor of `sqrt(batch_size/batch_denom)` and uses global batch size for all the processes. For `strong` scaling, lr is scaled by world size and divides global batch size by world size |
| slotsPerWorker | int | `1` | Number of Processes Per Worker |
| useCase | string | `"image_classification"` | Use case (`image_classification`|`text_classification`) |
| workerResources.cpu | int | `4` | Number of Compute per Worker |
| workerResources.memory | string | `"8Gi"` | Amount of Memory per Worker |
| workers | int | `4` | Number of Workers |
