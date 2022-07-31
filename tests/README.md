# Testing TLK

To run these tests, first install [tlk](/tlk) for TensorFlow and/or PyTorch.
Then install the following dependencies:

```
# Clone this repo, if you don't already have it
git clone git@github.com:intel-innersource/frameworks.ai.transfer-learning.git
cd frameworks.ai.transfer-learning

# Run tests with make, or skip this step to run individually
make test

# Create a virtual env or conda env for the test environment
conda create --name tlk_tests python=3.9

# Install tlk for TensorFlow and/or PyTorch
pip3 install --editable .[tensorflow,pytorch]

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
