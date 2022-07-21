# Environment Setup and Running the Notebooks

Use the instructions below to install the dependencies required to run the
[PyTorch](#pytorch-environment) or [TensorFlow](#tensorflow-environment) notebooks.

System Requirements:
* Ubuntu 20.04
* Python 3.7, 3.8, 3.9, or 3.10
* git

## PyTorch Environment

1. Get a clone of this Transfer Learning repository from GitHub:
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git transfer_learning
   export TLK_REPO=$(pwd)/transfer_learning
   ```
2. Create a Python3 virtual environment and install required packages.

   You can use virtualenv:
   ```
   python3 -m venv intel-pyt-venv
   source intel-pyt-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-pyt python=3.8
   conda activate intel-pyt
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   pip install --upgrade pip
   pip install -r ${TLK_REPO}/notebooks/pytorch_requirements.txt
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the dataset to
   the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   If the `DATASET_DIR` and `OUTPUT_DIR` variables are not defined, the notebooks will
   default to use `~/dataset` and `~/output`.
   ```
   export DATASET_DIR=<directory to download the dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the Transfer Learning repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${TLK_REPO}/notebooks
   PYTHONPATH=${TLK_REPO} jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run any of the
   PyTorch notebooks.

## TensorFlow Environment

1. Get a clone of this Transfer Learning repository from GitHub:
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git transfer_learning
   export TLK_REPO=$(pwd)/transfer_learning
   ```
2. Create a Python3 virtual environment and install required packages.

   You can use virtualenv:
   ```
   python3 -m venv intel-tf-venv
   source intel-tf-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-tf python=3.8
   conda activate intel-tf
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   pip install --upgrade pip
   pip install -r ${TLK_REPO}/notebooks/tensorflow_requirements.txt
   pip install --no-deps tensorflow-text==2.8.2
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the dataset to
   the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   If the `DATASET_DIR` and `OUTPUT_DIR` variables are not defined, the notebooks will
   default to use `~/dataset` and `~/output`.
   ```
   export DATASET_DIR=<directory to download the dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the Transfer Learning repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${TLK_REPO}/notebooks
   PYTHONPATH=${TLK_REPO} jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run any of the
   TensorFlow notebooks.
