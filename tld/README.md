# Transfer Learn and Deploy Tool

## Build and Install

Requirements:
* Linux system (or WSL2 on Windows)
* git
* python3

1. Clone this repo and navigate to the repo directory:
   ```
   git clone git@github.com:intel-innersource/frameworks.ai.transfer-learning.git
   cd frameworks.ai.transfer-learning
   ```

1. Create and activate a Python3 virtual environment using `virtualenv`:
   ```
   python3 -m virtualenv tld_env
   source activate tld_env/bin/activate
   ```

   Or `conda`:
   ```
   conda create --name tld_env python=3.8
   conda activate tld_env
   ```

1. Install the tool by either building and installing the wheel:

   ```
   python setup.py bdist_wheel --universal
   pip install dist/tld-0.0.1-py2.py3-none-any.whl
   ```
   Or for developers, do an editable install:
   ```
   pip install --editable .
   ```

   > Note that the tool can be installed for a specific framework by specifying
   > the framework name as an option like: `pip install --editable .[tensorflow]` or
   > `pip install --editable .[pytorch]`
