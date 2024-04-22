# Transfer Learning for PyTorch Image Anomaly Classification using the Intel® Transfer Learning Tool API

This notebook demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for
image anomaly detection using PyTorch.

The notebook performs the following steps:
1. Import dependencies and setup parameters
1. Get the model
1. Get the dataset
1. Prepare the dataset
1. Predict using the original model
1. Finetuning / Feature Extraction
1. Predict
1. Export
1. Post-Training Quantization

## Running the notebook

To run the notebook, follow the instructions to setup the [PyTorch notebook environment](/notebooks/setup.md).

To use Gaudi for training and inference, install required software for Intel Gaudi: 
1.  Temporarily uninstall torch
```
# Torch will later be re-installed below
pip uninstall torch
```
2.  Install the Gaudi Intel SW Stack
```
wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.15.0/habanalabs-installer.sh
chmod +x habanalabs-installer.sh
sudo apt-get update
```
```
# Note: This may not be required depending on what is already installed on your machine
./habanalabs-installer.sh install --type base
```
3.	Install the Gaudi Intel Pytorch environment
```
# Note: This step may not be required depending on what is already installed on your machine
./habanalabs-installer.sh install -t dependencies
```
```
./habanalabs-installer.sh install --type pytorch –venv
```

See [Habana Docs](https://docs.habana.ai/en/latest/Installation_Guide/SW_Verification.html) for detailed installation instructions

## Dataset Citations
```
Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: International Journal of Computer Vision 129(4):1038-1059, 2021, DOI: 10.1007/s11263-020-01400-4.
```
```
Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection; in: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019, DOI: 10.1109/CVPR.2019.00982.
```