# Environment Setup and Running the Notebooks

Use the instructions below to install the dependencies required to run the notebooks.

System Requirements:
1. Ubuntu 20.04
2. Python3 (3.8, 3.9, or 3.10), Pip/Conda and Virtualenv
3. git

## Install Intel® Transfer Learning Tool
This is required for the TLT tutorial notebooks and E2E notebooks. Follow the instructions in the
main [README](/README.md#build-and-install). You can skip this step if you are only running
the native framework notebooks.

## Notebook Environment

1. Get a clone of this Transfer Learning repository from GitHub:
   ```
   git clone https://github.com/IntelAI/transfer-learning.git transfer_learning
   export TLT_REPO=$(pwd)/transfer_learning
   ```
2. Activate the Intel Transfer Learning Tool environment or create a new Python3 virtual environment and install required packages.

   You can use virtualenv:
   ```
   virtualenv -p python3 tlt-notebook-venv
   source tlt-notebook-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n tlt-notebook-venv python=3.8
   conda activate tlt-notebook-venv
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   pip install --upgrade pip
   pip install -r ${TLT_REPO}/notebooks/requirements.txt
   ```
   This is only required for TensorFlow text classification notebooks that use TF Hub:
   ```
   pip install tensorflow-text==2.11.0
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
   cd ${TLT_REPO}/notebooks
   PYTHONPATH=${TLT_REPO} jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run the notebooks.
