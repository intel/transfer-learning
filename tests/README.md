# Testing Intel® Transfer Learning Tool

To run these tests, first install the [Intel Transfer Learning Tool](/tlt) for TensorFlow and/or PyTorch.
Then install the following dependencies:

```
# Clone this repo, if you don't already have it
git clone https://github.com/IntelAI.transfer-learning.git
cd transfer-learning

# Run tests with make, or skip this step to run individually
make test

# Run only unittests with make
make unittest

# Run only integration tests with make
make integration

# Create a virtual env or conda env for the test environment
conda create --name tlt_dev_venv python=3.9

# Install tlt for TensorFlow and/or PyTorch
pip3 install --editable .

# Install the test requirements
pip3 install -r tests/requirements-test.txt
```

## API Tests
There are unit and integration tests that exercise the API.
Make sure you are in the `transfer-learning/` directory and use the command
below to run all tests:
```
PYTHONPATH=$(pwd)/tests py.test -s
```

### Markers

The following custom markers have been defined in the transfer learning tests:
```
@pytest.mark.tensorflow: test requires tensorflow to be installed

@pytest.mark.pytorch: test requires pytorch to be installed

@pytest.mark.common: test does not require a specific framework to be installed

@pytest.mark.integration: test will run all integration tests
```

### Sample test run commands using markers

To run only the TensorFlow tests run:
```
PYTHONPATH=$(pwd)/tests py.test -s -m tensorflow
```

To run the TensorFlow tests and the common tests:
```
PYTHONPATH=$(pwd)/tests py.test -s -m "tensorflow or common"
```

To run only the PyTorch tests run:
```
PYTHONPATH=$(pwd)/tests py.test -s -m pytorch
```

> Note: After the tests have run, there will be downloaded data in `/tmp/data`
that has not been cleaned up. Currently, the developer has to manage this, but
we should create fixtures that take care of it.

There are some executable examples in module docstrings. To run them as tests, follow
the steps in the [docs README.md](/docs/README.md).

## Jupyter Notebook Tests
There are Makefile targets and a bash script that will automatically run the Jupyter notebooks.
There are a few different ways to use them. All of the ways require that you are in the `transfer-learning/` directory
and that you have set dataset and output directories:

```
export DATASET_DIR=<directory to download the datasets>
export OUTPUT_DIR=<output directory for the saved models>
```

To run the <b>Intel Transfer Learning Tool tutorial notebooks</b> using custom datasets:
```
make test_notebook_custom
```

To run the <b>Intel Transfer Learning Tool tutorial notebooks</b> using datasets from public catalogs:
```
make test_notebook_catalog
```

To run all the <b>native PyTorch notebooks</b> using a test environment for PyTorch without Intel Transfer Learning Tool:
```
make test_pyt_notebook
```

To run all the <b>native TensorFlow notebooks</b> using a test environment for TensorFlow without Intel Transfer Learning Tool:
```
make test_tf_notebook
```

To use the virtual environment of your choice and run a single notebook or multiple notebooks in the same directory:
```
source <env>/bin/activate
bash run_notebooks.sh <directory or file path>
```

Optional: to run a notebook with certain cells omitted, send in the metadata tag as a second argument. For example:
```
source <env>/bin/activate
bash run_notebooks.sh notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb remove_for_tf_dataset
```

