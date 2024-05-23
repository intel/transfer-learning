# LLM fine tuning with Kubernetes and Intel® Gaudi® AI Accelerators

## Docker Image

Prior to deploying the fine tuning job to Kubernetes, a Docker image needs to be built and pushed to a container repo,
or copied to the Gaudi nodes on your Kubernetes cluster. The [`Dockerfile.gaudi`](Dockerfile.gaudi) used to run the
fine tuning job uses `vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest` as it's
base and then adds on library installations like [optimum-habana](https://github.com/huggingface/optimum-habana). The
container also includes example scripts from optimum-habana and [Llama 2 fine tuning script](scripts/finetune.py) from
this workflow directory.

### Container Build

The [`Dockerfile.gaudi`](Dockerfile.gaudi) has build arguments for the following versions:

| Argument | Default Value | Description |
|----------|---------------|-------------|
| GAUDI_SW_VER | `intel/intel-optimized-pytorch` | SynapseAI / Gaudi driver version  |
| OS | `ubuntu22.04` | Base image tag |
| TORCH_VER | `2.2.0` | Torch version |
| OPTIMUM_HABANA_VER | `1.11.1` | Hugging Face Optimum Habana version |

The container can be built with the default package versions using the following command:
```
docker build -t <image name>:<tag> .
```

Alternatively, build arguments can be passed to the build command to use different versions:
```
export GAUDI_SW_VER=<GAUDI_SW_VER>
export OS=<OS>
export TORCH_VER=<TORCH_VER>
export OPTIMUM_HABANA_VER=<OPTIMUM_HABANA_VER>

docker build \
 --build-arg GAUDI_SW_VER=${GAUDI_SW_VER} \
 --build-arg OS=${OS} \
 --build-arg TORCH_VER=${TORCH_VER} \
 --build-arg OPTIMUM_HABANA_VER=${OPTIMUM_HABANA_VER} \
 -t <image name>:<tag> .
```

### Container Push

The container needs to be pushed for the Kubernetes cluster to have access to the image. If you have a Docker container
registry (such as [DockerHub](https://hub.docker.com)), you can push the container to that registry. Otherwise, we have
alternative instructions for getting the container distributed to the cluster nodes by saving the image and copying it
to the nodes.

Use one of these options to push the container:

a. First, ensure that you are logged in with your container registry account using
   [`docker login`](https://docs.docker.com/engine/reference/commandline/login/). Next,
   [re-tag your image](https://docs.docker.com/engine/reference/commandline/tag/) and then
   [push the image](https://docs.docker.com/engine/reference/commandline/push/) to the registry.
   ```
   # Retag the image by providing the source image and destination image
   docker tag <source image>:<tag> <destination image name>:<tag>

   # Push the image to the registry
   docker push <image name>:<tag>
   ```
b. If you don't have a container registry, use the commands below to save the container, copy it to the nodes on the
   Kubernetes cluster, and then load it into Docker.
   ```
   # Save the image to a tar.gz file
   docker save <image name>:<tag> | gzip > hf_k8s.tar.gz

   # Copy the tar file to every Kubernetes node that could be used to run the fine tuning job
   scp hf_k8s.tar.gx <user>@<host>:/tmp/hf_k8s.tar.gz

   # SSH to each of the Kubernetes nodes and load the image to Docker
   docker load --input /tmp/hf_k8s.tar.gz
   ```

> Note: The `<image name>:<tag>` that was pushed needs to be specified in the Helm chart values file.

## Setting up the Gaudi Device Plugin
With a Gaudi device deployed in the Kubernetes cluster, this plugin will enable the registration of that device for use. The daemonset can be deployed using the following .yaml file from the [Intel Gaudi Docs](https://docs.habana.ai/en/latest/Orchestration/Gaudi_Kubernetes/Device_Plugin_for_Kubernetes.html). Be sure to refer to the Intel Gaudi Docs for more details if need be.

Deployment

```
kubectl create -f https://vault.habana.ai/artifactory/docker-k8s-device-plugin/habana-k8s-device-plugin.yaml
```

Checking Deployment
```
kubectl get pods -n habana-system
```

Sample Output:
```
NAME                                      READY    STATUS             RESTARTS           AGE
habanalabs-device-plugin-daemonset-#xxxx   1/1     Running            0                  1s
...
```
Once this is running, the Kubernetes job will know to look for a Gaudi device for usage in the job.

## Running the fine tuning job on the Kubernetes cluster

There are two Helm values files that are setup to run LLM fine tuning with Gaudi:

| Value file name | Description |
|-----------------|-------------|
| [`gaudi_values.yaml`](chart/gaudi_values.yaml) | Uses a single Gaudi card to fine tune `meta-llama/Llama-2-7b-chat-hf` using a subset of the [Financial alpaca dataset](https://huggingface.co/datasets/gbharti/finance-alpaca)  |
| [`gaudi_multicard_values.yaml`](chart/gaudi_multicard_values.yaml) | Uses 8 Gaudi cards to fine tune `meta-llama/Llama-2-7b-hf` using the [Medical Meadow flashcards dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) |

Pick one of the value files to use depending on your desired use case, make modifications to customize your fine tuning
job, and then use the instructions below to deploy the job to your cluster.

> Before running the fine tuning job on the cluster, the Docker image must be built and pushed to a container
> registry or loaded into Docker on the cluster nodes. See the [container build](#container-build) and
> [container push](#container-push) sections for instructions.

1. If you are using a gated model, get a [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) with read
   access and use your terminal to get the base64 encoding for your token using a terminal using
   `echo <your token> | base64`. If you are not using a gated model, you can skip this step.

   For example:
   ```
   $ echo hf_ABCDEFG | base64
   aGZfQUJDREVGRwo=
   ```

   Copy and paste the encoded token value into your values yaml file `encodedToken` field in the `secret` section.
   For example:
   ```
   secret:
     name: hf-token-secret
     encodedToken: aGZfQUJDREVGRwo=
   ```

2. Edit your values file based on the parameters that you would like to use and your cluster. Key parameters to look
   at and edit are:
   * `image.name` should be set to the name of your docker image
   * `image.tag` should be set to the tag of your docker image
   * `resources.hpu` specifies the number of Gaudi cards to use.
   * `resources.memoryRequest` and `resources.memoryLimit` values should be updated based on the amount of memory
     available on the nodes in your cluster
   * `resources.hugePages2Mi` to specify the hugepages-2Mi request/limit based on your Gaudi node.
   * `storage.storageClassName` should be set to your Kubernetes NFS storage class name (use `kubectl get storageclass`
     to see a list of storage classes on your cluster)

   In the same values file, edit the security context parameters to have the containers run with a non-root user:
   * `securityContext.runAsUser` should be set to your user ID (UID)
   * `securityContext.runAsGroup` should be set to your group ID
   * `securityContext.fsGroup` should be set to your file system group ID

   See a complete list and descriptions of the available parameters in the [Helm chart values documentation](values.md).

3. Deploy the helm chart to the cluster using the `kubeflow` namespace:
   ```
   # Navigate to the directory that contains the Hugging Face Kubernetes example
   cd docker/hf_k8s

   # Deploy the job using the helm chart, specifying the values file with the -f parameter
   helm install --namespace kubeflow -f chart/<values file>.yaml gaudi-llm ./chart
   ```

4. (Optional) If a custom dataset is being used, the file needs to be uploaded to the persistent volume claim (PVC), so
   that it can be accessed by the worker pods. If your values yaml file is using a Hugging Face dataset (such as
   `medalpaca/medical_meadow_medical_flashcards`), you can skip this step.

   The dataset can be uploaded to the PVC using the [`kubectl cp` command](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#cp).
   The destination path for the dataset needs to match the `train.dataFile` path in your values yaml file.  Note that the worker pods would keep failing and restarting until you upload your dataset.
   ```
   # Copies a local "dataset" folder to the PVC at /tmp/pvc-mount/dataset
   kubectl cp dataset <dataaccess pod name>:/tmp/pvc-mount/dataset

   # Verify that the data file is at the expected path
   kubectl exec <dataaccess pod name> -- ls -l /tmp/pvc-mount/dataset
   ```

   For example:

   The [`gaudi_values.yaml`](chart/gaudi_values.yaml) file requires this step for uploading the custom dataset to the cluster. Run the [`download_financial_dataset.sh`](scripts/download_financial_dataset.sh) script to create a custom dataset and copy it to the PVC, as mentioned below.

   ```
   # Set a location for the dataset to download
   export DATASET_DIR=/tmp/dataset

   # Run the download shell script
   bash scripts/download_financial_dataset.sh

   # Copy the local "dataset" folder to the PVC at /tmp/pvc-mount/dataset
   kubectl cp ${DATASET_DIR} llama2-gaudi-finetuning-dataaccess:/tmp/pvc-mount/dataset
   ```

5. The training job can be monitored using by checking the status of the PyTorchJob using:
   * `kubectl get pytorchjob -n kubeflow`: Lists the PyTorch jobs that have been deployed to the cluster along with
     their status.
   * `kubectl describe pytorchjob <job name> -n kubeflow`: Lists the details of a particular PyTorch job, including
     information about events related to the job, such as pods getting created for each worker.
   The worker pods can be monitored using:
   * `kubectl get pods -n kubeflow`: To see the pods in the `kubeflow` namespace and their status. Also, adding
     `-o wide` to the command will additionally list out which node each pod is running on.
   * `kubectl logs <pod name> -n kubeflow`: Dumps the log for the specified pod. Add `-f` to the command to
     stream/follow the logs as the pod is running.

6. After the job completes, files can be copied from the persistent volume claim to your local system using the
   [`kubectl cp` command](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands#cp) using the
   data access pod. The path to the trained model is in the values file field called `distributed.train.outputDir` and
   if quantization was also done, the quanted model path is in the `distributed.quantize.outputDir` field.

   As an example, the trained model from the Medical Meadows use case can be copied from the
   `/tmp/pvc-mount/output/saved_model` path to the local system using the following command:
   ```
   kubectl cp --namespace kubeflow <dataaccess pod name>:/tmp/pvc-mount/output/saved_model .
   ```
7. Finally, the resources can be deleted from the cluster using the
   [`helm uninstall`](https://helm.sh/docs/helm/helm_uninstall/) command. For example:
   ```
   helm uninstall --namespace kubeflow gaudi-llm
   ```
   A list of all the deployed helm releases can be seen using `helm list`.

## Citations

```
@misc{touvron2023llama,
      title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
      author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov and Soumya Batra and Prajjwal Bhargava and Shruti Bhosale and Dan Bikel and Lukas Blecher and Cristian Canton Ferrer and Moya Chen and Guillem Cucurull and David Esiobu and Jude Fernandes and Jeremy Fu and Wenyin Fu and Brian Fuller and Cynthia Gao and Vedanuj Goswami and Naman Goyal and Anthony Hartshorn and Saghar Hosseini and Rui Hou and Hakan Inan and Marcin Kardas and Viktor Kerkez and Madian Khabsa and Isabel Kloumann and Artem Korenev and Punit Singh Koura and Marie-Anne Lachaux and Thibaut Lavril and Jenya Lee and Diana Liskovich and Yinghai Lu and Yuning Mao and Xavier Martinet and Todor Mihaylov and Pushkar Mishra and Igor Molybog and Yixin Nie and Andrew Poulton and Jeremy Reizenstein and Rashi Rungta and Kalyan Saladi and Alan Schelten and Ruan Silva and Eric Michael Smith and Ranjan Subramanian and Xiaoqing Ellen Tan and Binh Tang and Ross Taylor and Adina Williams and Jian Xiang Kuan and Puxin Xu and Zheng Yan and Iliyan Zarov and Yuchen Zhang and Angela Fan and Melanie Kambadur and Sharan Narang and Aurelien Rodriguez and Robert Stojnic and Sergey Edunov and Thomas Scialom},
      year={2023},
      eprint={2307.09288},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{han2023medalpaca,
  title={MedAlpaca--An Open-Source Collection of Medical Conversational AI Models and Training Data},
  author={Han, Tianyu and Adams, Lisa C and Papaioannou, Jens-Michalis and Grundmann, Paul and Oberhauser, Tom and L{\"o}ser, Alexander and Truhn, Daniel and Bressem, Keno K},
  journal={arXiv preprint arXiv:2304.08247},
  year={2023}
}
```
