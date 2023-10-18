# Image Classification Finetuning and Inference

## Solution Technical Overview

The vision fine-tuning (transfer learning) and inference workflow demonstrates Image Classification workflows/pipelines using Intel® Transfer Learning Tool to be run along with Intel-optimized software represented using toolkits, domain kits, packages, frameworks and other libraries for effective use of Intel hardware leveraging Intel's AI instructions for fast processing and increased performance. The workflows can be easily used by applications or reference kits showcasing usage.

The workflow supports:
```
Image Classification Finetuning
Image Classification Inference
```
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Validated Hardware Details](#validated-hardware-details)
- [Software Requirements](#software-requirements)
- [How it Works?](#how-it-works)
- [Get Started](#get-started)
- [Run Using Docker](#run-using-docker)
- [Run Using Bare Metal](#run-using-bare-metal) 
- [Expected Output](#expected-output)
- [Learn More](#learn-more)
- [Support](#support)

## Overview 

The vision workflow aims to train an image classifier that takes in contrast-enhanced spectral mammography (CESM) images. The pipeline creates prediction for the diagnosis of breast cancer. The goal is to minimize an expert’s involvement in categorizing samples as normal, benign, or malignant, by developing and optimizing a decision support system that automatically categorizes the CESM with the help of radiologist

## Dataset
The dataset is a collection of 2,006 high-resolution contrast-enhanced spectral mammography (CESM) images (1003 low energy images and 1003 subtracted CESM images) with annotations of 326 female patients. See Figure-1. Each patient has 8 images, 4 representing each side with two views (Top Down looking and Angled Top View) consisting of low energy and subtracted CESM images. Medical reports, written by radiologists, are provided for each case along with manual segmentation annotation for the abnormal findings in each image. As a preprocessing step, we segment the images based on the manual segmentation to get the region of interest and group annotation notes based on the subject and breast side. 

For more details of the dataset, visit the wikipage of the [CESM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8) and read [Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research](https://www.nature.com/articles/s41597-022-01238-0).

## Validated Hardware Details
There are workflow-specific hardware and software setup requirements depending on how the workflow is run. Bare metal development system and Docker image running locally have the same system requirements. 

| Recommended Hardware         | Precision  |
| ---------------------------- | ---------- |
| Intel® 4th Gen Xeon® Scalable Performance processors| BF16 |
| Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processors| FP32 |

To execute the reference solution presented here, use CPU for fine tuning.

## Software Requirements 
Linux OS (Ubuntu 22.04) is used to validate this reference solution. Make sure the following dependencies are installed.

1. `sudo apt-get update`
2. `sudo apt-get install -y build-essential gcc git libgl1-mesa-glx libglib2.0-0 python3-dev`
3. `sudo apt-get install python3.9 python3-pip`
4.  `virtualenv` through python3-venv or conda
5. `pip install dataset-librarian`

## How It Works?

The Vision reference Implementation component uses [Intel Transfer Learning Toolkit based vision workload](https://github.com/IntelAI/transfer-learning), which is optimized for image fine-tuning and inference. This workload uses Tensorflowhub's ResNet-50 model to fine-tune a new convolutional neural network model with subtracted CESM image dataset. The images are preprocessed by using domain expert-defined segmented regions to reduce redundancies during training.

## Get Started

### Download the repository

git clone https://github.com/IntelAI/transfer-learning.git vision_workflow
cd vision_workflow/workflows/disease_prediction


### Create a new python environment
```shell
conda create -n <env name> python=3.9
conda activate <env name>
```
### Download and Preprocess the Datasets
Use the links below to download the image datasets. Or skip to the [Docker](#run-using-docker) section to download the dataset using a container.

- [High-resolution Contrast-enhanced spectral mammography (CESM) images](https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/260?passcode=5335d2514638afdaf03237780dcdfec29edf4238#)

Once you have downloaded the image files and placed them into the data directory, proceed by executing the following command. This command will initiate the download of segmentation and annotation data, followed by the application of segmentation and preprocessing operations.

Command-line Interface:
- -d : Directory location where the raw dataset will be saved on your system. It's also where the preprocessed dataset files will be written. If not set, a directory with the dataset name will be created.
- --split_ratio: Split ratio of the test data, the default value is 0.1.

More details of the dataset_librarian can be found [here](https://pypi.org/project/dataset-librarian/).


```
python -m dataset_librarian.dataset -n brca --download --preprocess -d data/ --split_ratio 0.1
```

**Note:** See this dataset's applicable license for terms and conditions. Intel Corporation does not own the rights to this dataset and does not confer any rights to it.


## Run Using Bare Metal 

### Install package for running vision-finetuning-inference--workflows
```shell
pip install -r requirements.txt
```

Note: Configure the right configurations in the config.yaml

The 'config.yaml' file includes the following parameters:

- args:
  - dataset_dir: contains the path for dataset_dir
  - finetune_output: saves results of finetuning in a yaml file
  - inference_output : saves the results of the model on test data in the yaml file
  - model: Pretrained model name (default resnetv150)
  - finetune: runs vision fine-tuning
  - inference: runs inference only if set to true , if false finetunes the model before inference
  - saved_model_dir: Directory where trained model gets saved
- training_args:
  - batch_size: Batch size for training ( default 32)
  - bf16: Enable BF16 by default
  - epochs: Number of epochs for training
  - output_dir: Output of training model
```shell
python src/run.py --config_file config/config.yaml 
```

## Run Using Docker

### 1. Set Up Docker Engine And Docker Compose
You'll need to install Docker Engine on your development system. Note that while **Docker Engine** is free to use, **Docker Desktop** may require you to purchase a license. See the [Docker Engine Server installation instructions](https://docs.docker.com/engine/install/#server) for details.


To build and run this workload inside a Docker Container, ensure you have Docker Compose installed on your machine. If you don't have this tool installed, consult the official [Docker Compose installation documentation](https://docs.docker.com/compose/install/linux/#install-the-plugin-manually).


```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.7.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
docker compose version
```

### 2. Set Up Docker Image
Pull the provided docker image.


```bash
docker pull intel/ai-workflows:pa-vision-tlt-disease-prediction
```

### 3. Run With Docker Compose

```bash
cd docker
export CONFIG=<config_file_name_without_.yaml>
docker compose run dev
```

| Environment Variable Name | Default Value | Description |
| --- | --- | --- |
| CONFIG | n/a | Config file name |

### 4. Clean Up Docker Container
Stop container created by docker compose and remove it.

```bash
docker compose down
```
## Expected Output

Successful run should dump the output in a yaml file . The ouput would look like this
```
label:
- Benign
- Malignant
- Normal
label_id:
- 0
- 1
- 2
metric:
  acc: 0.7163363099098206
  loss: 0.6603914499282837
results:
  P100_R_CM_CC.jpg:
  - label: Normal
    pred: Normal
    pred_prob:
    - 0.3331606984138489
    - 0.28302037715911865
    - 0.38381892442703247
  P100_R_CM_MLO.jpg:
  - label: Normal
    pred: Benign
    pred_prob:
    - 0.38328817486763
    - 0.2962200343608856
    - 0.320491760969162
```
### How to customize this use case
Tunable configurations and parameters are exposed using yaml config files allowing users to change model training hyperparameters, datatypes, paths, and dataset settings without having to modify or search through the code.

#### Adopt to your dataset
To deploy this reference use case on a different or customized dataset, you can easily modify the config.yaml file. For instance, if you have a new dataset, simply update the paths of finetune_output and inference_output and adjust the dataset features in the config.yaml file, as demonstrated below.

```
    dataset_dir: --> updated Dataset path
```    

#### Adopt to your model

To implement this reference use case on a different or customized pre-training model, modifications to the config.yaml file are straightforward. For instance, to use an alternate model, one can update the path of the model by modifying the 'model' fields in the config.yaml file structure. The following example illustrates this process:

```
   model --> new_model ( eg: efficienetnetb0)
```
## Learn More

For more information or to read about other relevant workflow examples, see these guides and software resources:
- [Intel® AI Analytics Toolkit (AI Kit)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html)
- [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
- [Intel® Transfer Learning Tool](https://github.com/IntelAI/transfer-learning/tree/v0.5.0)

## Support
If you have any questions with this workflow, want help with troubleshooting, want to report a bug or submit enhancement requests, please submit a GitHub issue.
