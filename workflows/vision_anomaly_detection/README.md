# Vision-based Anomaly Detection workflow
Deep-learning based visual anomaly detection workload. The inputs are images of the part of MVTEC Anomaly Detection dataset (https://www.mvtec.com/company/research/datasets/mvtec-ad). The workload is executed in 4 steps - 
1) It extracts the features of images using Deep Neural Network based models (i.e. ResNet50) pretrained on Imagenet dataset. 
2) Leverages PCA to reduce these features to smaller dimensions while retanining maximum variance (99%). 
3) Convert PCA components to the original feature space using PCA kernel learned in step 2.
4) Compute the information loss between original image and rgenerated image through PCA. If the loss is below a thresold, the input image is a 'good' image otherwise an anomaly.

This workflow demonstrates Anomaly Detection workflows/pipelines using tlt toolkit to be run along with Intel optimized software represented using toolkits, domainkits, packages, frameworks and other libraries for effective use of Intel hardware leveraging Intel's AI instructions for fast processing and increased performance.The workflows can be easily used by applications or reference kits showcasing usage.

The workflow supports:
- Fine-tuning and inference on custom dataset
- Three feature extractors
  - Pre-trained model (without fine-tuning)
  - Fine-tune model (Based on Simsiam)
  - Fine-tune model (Based on CutPaste self-supervised techniques)

# Getting started
## Deploy the test environment

### Create a new python environment
```shell
conda create -n <env name> python=3.9
conda activate <env name>
```

### Install package for running vision-based-anomaly-detection-workflow
```shell
pip install -r requirements.txt
```

## Running 

```shell
python src/vision_anomaly_wrapper.py --config_file config/config.yaml
```
Note: Configure the right configurations in the config.yaml

## Build Container

```bash
docker build \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    ../../ \
     -f ./Dockerfile \
     -t intel:tlt-anomaly
```

## Run Container

```bash
docker run --rm \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -v /path/to/mvtec:/workspace/workflows/vision_anomaly_detection/datasets/mvtec \
    -v /path/to/simsiam/checkpoint_0099.pth.tar:/workspace/workflows/vision_anomaly_detection/simsiam/checkpoint_0099.pth.tar \
    -v $PWD/output:/workspace/workflows/vision_anomaly_detection/output \
    --shm-size=8GB \
    intel:tlt-anomaly \
    python ./src/vision_anomaly_wrapper.py --config_file ./config/config.yaml
```
