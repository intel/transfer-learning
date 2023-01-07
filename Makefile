#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

# Note: These are just placeholders for future additions to Makefile.
# You can remove these comments later.
ACTIVATE_TLT = "tlt_env/bin/activate"
ACTIVATE_TF = "intel_tf/bin/activate"
ACTIVATE_PYT = "intel_pyt/bin/activate"
ACTIVATE_TEST = "tlt_tests/bin/activate"
ACTIVATE_DOCS = $(ACTIVATE_TEST)
ACTIVATE_NOTEBOOK = $(ACTIVATE_TEST)

# Customize sample test run commands
# PY_TEST_EXTRA_ARGS="'-vvv -k test_platform_util_with_no_args'" make test
# PY_TEST_EXTRA_ARGS="'--collect-only'" make test
PY_TEST_EXTRA_ARGS ?= ""

venv_test: $(CURDIR)/tests/requirements-test.txt
	@echo "Creating a virtualenv tlt_tests..."
	@test -d tlt_tests || virtualenv -p python3 tlt_tests

	@echo "Building the TLT API in tlt_tests env..."
	@. $(ACTIVATE_TEST) && pip install --editable .[tensorflow,pytorch]
	@echo "Required for TensorFlow text classification..."
	@. $(ACTIVATE_TEST) && pip install tensorflow-text==2.10.0

	@echo "Installing test dependencies..."
	@. $(ACTIVATE_TEST) && pip install -r $(CURDIR)/tests/requirements-test.txt

venv_intel_tf: $(CURDIR)/notebooks/tensorflow_requirements.txt
	@echo "Creating a virtualenv intel_tf..."
	@test -d intel_tf || virtualenv -p python3 intel_tf

	@echo "Installing TF notebook dependencies..."
	@. $(ACTIVATE_TF) && pip install -r $(CURDIR)/notebooks/tensorflow_requirements.txt && \
	pip install tensorflow-text==2.10.0

venv_intel_pyt: $(CURDIR)/notebooks/pytorch_requirements.txt
	@echo "Creating a virtualenv intel_pyt..."
	@test -d intel_pyt || virtualenv -p python3 intel_pyt

	@echo "Installing PYT notebook dependencies..."
	@. $(ACTIVATE_PYT) && pip install -r $(CURDIR)/notebooks/pytorch_requirements.txt

test: venv_test
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/tests" py.test $(PY_TEST_EXTRA_ARGS) -s --cov --cov-fail-under=85

lint: venv_test
	@echo "Style checks..."
	@. $(ACTIVATE_TEST) && flake8 tlt
	@. $(ACTIVATE_TEST) && flake8 tests

clean:
	rm -rf tlt_tests

venv_docs: venv_test $(CURDIR)/docs/requirements-docs.txt
	@echo "Installing docs dependencies..."
	@. $(ACTIVATE_DOCS) && pip install -r $(CURDIR)/docs/requirements-docs.txt

html: venv_docs
	@echo "Building Sphinx documentation..."
	@. $(ACTIVATE_DOCS) && $(MAKE) -C docs clean html

test_docs: html
	@echo "Testing Sphinx documentation..."
	@. $(ACTIVATE_DOCS) && $(MAKE) -C docs doctest

venv_notebook: venv_test
	@echo "Installing notebook dependencies..."
	@. $(ACTIVATE_NOTEBOOK) && pip install -r $(CURDIR)/notebooks/tensorflow_requirements.txt

test_notebook: venv_notebook
	@echo "Testing Jupyter notebooks..."
	@. $(ACTIVATE_NOTEBOOK) && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb remove_for_tf_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_pyt_image_classification/TLT_PyTorch_Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_pyt_image_classification/TLT_PyTorch_Image_Classification_Transfer_Learning.ipynb remove_for_tv_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_tf_text_classification/TLT_TF_Text_Classification.ipynb remove_for_custom_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_tf_text_classification/TLT_TF_Text_Classification.ipynb remove_for_tf_dataset

test_tf_notebook: venv_intel_tf
	@. $(ACTIVATE_TF) && bash run_notebooks.sh tensorflow

test_pyt_notebook: venv_intel_pyt
	@. $(ACTIVATE_PYT) && bash run_notebooks.sh pytorch

dist: venv_docs
	@echo "Create binary wheel..."
	@. $(ACTIVATE_DOCS) && python setup.py bdist_wheel

check_dist: dist
	@echo "Testing the wheel..."
	@. $(ACTIVATE_DOCS) && \
	pip install twine && \
	python setup.py bdist_wheel && \
	twine check dist/*
