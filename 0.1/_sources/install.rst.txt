Installation
============

Requirements
------------

* Linux system (or WSL2 on Windows)
* git
* python3

Install Steps
-------------

1. Clone this repo and navigate to the repo directory::

    git clone git@github.com:intel-innersource/frameworks.ai.transfer-learning.git
    cd frameworks.ai.transfer-learning

2. Create and activate a Python3 virtual environment using `virtualenv`::

    python3 -m virtualenv tlk_env
    source tlk_env/bin/activate

   Or `conda`::

    conda create --name tlk_env python=3.8
    conda activate tlk_env

3. Install the tool with the `tensorflow` and/or `pytorch` option by either building and installing the wheel::

    python setup.py bdist_wheel --universal
    pip install dist/tlk-0.0.1-py2.py3-none-any.whl[tensorflow]

   Or for developers, do an editable install::

    pip install --editable .[tensorflow]
