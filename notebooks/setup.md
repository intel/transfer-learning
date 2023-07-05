# Environment Setup and Running the Notebooks

Use the instructions below to install the dependencies required to run the notebooks.

Software Requirements:
1. Linux* system (validated on Ubuntu* 20.04/22.04 LTS)
2. Python3 (3.8, 3.9, or 3.10), Pip/Conda and Virtualenv
3. git

## Set Up Notebook Environment

1. Install Intel® Transfer Learning Tool using the Advanced Installation option in the [Get Started](/GetStarted.md) Guide.
   This is required for the Intel Transfer Learning Tool tutorial notebooks, E2E notebooks, and performance comparison. Follow the
   instructions in the [Get Started Guide](/GetStarted.md). You can
   skip this step if you are only running the native framework notebooks.

2. Activate the virtualenv or conda environment used to install Intel Transfer Learning Tool,
   then from inside the activated environment, run these steps:
   ```
   pip install --upgrade pip
   pip install -r notebooks/requirements.txt
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
   cd transfer-learning/notebooks
   jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run the notebooks.

Once you have the environment and dependencies set up, see the list of available
[notebooks](/notebooks/README.md).
