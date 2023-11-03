# Docker and Kubernetes

This directory has examples that run transfer learning workloads using Docker containers and Kubernetes clusters.

| Example | Domain: Use Case | Framework| Description |
|---------|------------------|----------|-------------|
| [Distributed training using the Intel® Transfer Learning Tool](tlt_tf_k8s) | Image and text classification | TensorFlow | This example uses the [KubeFlow* MPI Training operator](https://www.kubeflow.org/docs/components/training/mpi/) to run a distributed TensorFlow training job with the Intel Transfer Learning Tool. [Values in the Helm chart](tlt_tf_k8s/chart/README.md) can be modified to specify parameters such as the number of workers, model name, dataset name, batch size, etc. |
| [Fine tuning Llama 2 with multiple nodes on a Kubernetes cluster](hf_k8s) | NLP: Text generation | PyTorch | Demonstrates distributed fine tuning of Llama 2 models from Hugging Face using the [Kubeflow PyTorch Training operator](https://www.kubeflow.org/docs/components/training/pytorch/) on a Kubernetes cluster. |
