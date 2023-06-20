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
# SPDX-License-Identifier: Apache-2.0
#

# Note: These are just placeholders for future additions to Makefile.
# You can remove these comments later.
ACTIVATE_TLT_VENV = "tlt_dev_venv/bin/activate"
ACTIVATE_NOTEBOOK_VENV = "tlt_notebook_venv/bin/activate"
ACTIVATE_TEST_VENV = "tlt_test_venv/bin/activate"
ACTIVATE_DOCS_VENV = $(ACTIVATE_TEST_VENV)

# Customize sample test run commands
# PY_TEST_EXTRA_ARGS="'-vvv -k test_platform_util_with_no_args'" make test
# PY_TEST_EXTRA_ARGS="'--collect-only'" make test
PY_TEST_EXTRA_ARGS ?= "--durations=0"

tlt_test_venv: $(CURDIR)/tests/requirements-test.txt
	@echo "Creating a virtualenv tlt_test_venv..."
	@test -d tlt_test_venv || virtualenv -p python3 tlt_test_venv

	@echo "Building the TLT API in tlt_test_venv env..."
	@. $(ACTIVATE_TEST_VENV) && pip install --editable .

	@echo "Installing test dependencies..."
	@. $(ACTIVATE_TEST_VENV) && pip install -r $(CURDIR)/tests/requirements-test.txt

tlt_notebook_venv: $(CURDIR)/notebooks/requirements.txt
	@echo "Creating a virtualenv tlt_notebook_venv..."
	@test -d tlt_notebook_venv || virtualenv -p python3 tlt_notebook_venv

	@echo "Installing TF & PYT notebook dependencies..."
	@. $(ACTIVATE_NOTEBOOK_VENV) && pip install -r $(CURDIR)/notebooks/requirements.txt

test: unittest integration

unittest: tlt_test_venv
	@echo "Testing unit test API..."
	@. $(ACTIVATE_TEST_VENV) && PYTHONPATH=$(CURDIR)/tests py.test -vvv -s $(PY_TEST_EXTRA_ARGS) "-k not integration and not skip"

integration: tlt_test_venv
	@echo "Testing integration test API..."
	@. $(ACTIVATE_TEST_VENV) && PYTHONPATH=$(CURDIR)/tests py.test -vvv -s $(PY_TEST_EXTRA_ARGS) "-k integration and not skip"

lint: tlt_test_venv
	@echo "Style checks..."
	@. $(ACTIVATE_TEST_VENV) && flake8 tlt tests downloader

clean:
	rm -rf tlt_test_venv

tlt_docs_venv: tlt_test_venv $(CURDIR)/docs/requirements-docs.txt
	@echo "Installing docs dependencies..."
	@. $(ACTIVATE_DOCS_VENV) && pip install -r $(CURDIR)/docs/requirements-docs.txt

html: tlt_docs_venv
	@echo "Building Sphinx documentation..."
	@. $(ACTIVATE_DOCS_VENV) && $(MAKE) -C docs clean html

test_docs: html
	@echo "Testing Sphinx documentation..."
	@. $(ACTIVATE_DOCS_VENV) && $(MAKE) -C docs doctest

tlt_notebook_venv: tlt_test_venv
	@echo "Installing notebook dependencies..."
	@. $(ACTIVATE_TEST_VENV) && pip install -r $(CURDIR)/notebooks/requirements.txt

test_notebook_custom: tlt_notebook_venv
	@echo "Testing Jupyter notebooks with custom datasets..."
	@. $(ACTIVATE_TEST_VENV) && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_pyt_image_classification/TLT_PyTorch_Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_tf_text_classification/TLT_TF_Text_Classification.ipynb remove_for_custom_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_pyt_text_classification/TLT_PYT_Text_Classification.ipynb remove_for_custom_dataset

test_notebook_catalog: tlt_notebook_venv
	@echo "Testing Jupyter notebooks with public catalog datasets..."
	@. $(ACTIVATE_TEST_VENV) && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb remove_for_tf_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_pyt_image_classification/TLT_PyTorch_Image_Classification_Transfer_Learning.ipynb remove_for_tv_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_tf_text_classification/TLT_TF_Text_Classification.ipynb remove_for_tf_dataset && \
	bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_pyt_text_classification/TLT_PYT_Text_Classification.ipynb remove_for_hf_dataset

test_tf_notebook: tlt_notebook_venv
	@. $(ACTIVATE_TEST_VENV) && bash run_notebooks.sh tensorflow

test_pyt_notebook: tlt_notebook_venv
	@. $(ACTIVATE_TEST_VENV) && bash run_notebooks.sh pytorch

dist: tlt_docs_venv
	@echo "Create binary wheel..."
	@. $(ACTIVATE_DOCS_VENV) && python setup.py bdist_wheel

check_dist: dist
	@echo "Testing the wheel..."
	@. $(ACTIVATE_DOCS_VENV) && \
	pip install twine && \
	python setup.py bdist_wheel && \
	twine check dist/*
