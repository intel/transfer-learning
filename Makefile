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
DOCS_VENV = ".venv/docs"
NOTEBOOKS_VENV = ".venv/notebooks"
PYTORCH_VENV = ".venv/pytorch"
TENSORFLOW_VENV = ".venv/tensorflow"
TEST_VENV = ".venv/test"
RELEASE_VENV = ".venv/releases"
LATEST_RELEASE=$(RELEASE_VENV)/$(shell date +"%Y-%m-%d-%H-%M-%S")

ACTIVATE_DOCS_VENV = $(DOCS_VENV)/bin/activate
ACTIVATE_NOTEBOOKS_VENV = $(NOTEBOOKS_VENV)/bin/activate
ACTIVATE_PYTORCH_VENV = $(PYTORCH_VENV)/bin/activate
ACTIVATE_TENSORFLOW_VENV = ${TENSORFLOW_VENV}/bin/activate
ACTIVATE_TEST_VENV = "$(TEST_VENV)/bin/activate"
ACTIVATE_RELEASE_VENV = "$(RELEASE_VENV)/bin/activate"

# Customize sample test run commands
# PY_TEST_EXTRA_ARGS="'-vvv -k test_platform_util_with_no_args'" make test
# PY_TEST_EXTRA_ARGS="'--collect-only'" make test
PY_TEST_EXTRA_ARGS ?= "--durations=0"

test_venv:
	@echo "Creating a virtualenv for testing..."
	@test -d $(TEST_VENV) || virtualenv -p python3 $(TEST_VENV)
	@echo "Installing test dependencies..."
	@. $(ACTIVATE_TEST_VENV) && pip install -r $(CURDIR)/tests/requirements-test.txt

pytorch_venv:
	@echo "Creating a virtualenv for testing pytorch..."
	@test -d $(PYTORCH_VENV) || virtualenv -p python3 $(PYTORCH_VENV)
	@echo "Installing test dependencies..."
	@. $(ACTIVATE_PYTORCH_VENV) && \
		pip install -r $(CURDIR)/tests/requirements-test.txt && \
		echo "Building the TLT API in pytorch env..." && \
		pip install --extra-index-url https://download.pytorch.org/whl/cpu .[pytorch]

tensorflow_venv: test_venv
	@echo "Creating a virtualenv for testing tensorflow..."
	@test -d $(TENSORFLOW_VENV) || virtualenv -p python3 $(TENSORFLOW_VENV)
	@echo "Installing test dependencies..." && \
		. $(ACTIVATE_TENSORFLOW_VENV) && \
		pip install -r $(CURDIR)/tests/requirements-test.txt && \
		echo "Building the TLT API in tensorflow env..." && \
		pip install --extra-index-url https://download.pytorch.org/whl/cpu .[tensorflow]

notebooks_venv:
	@echo "Creating a virtualenv for notebook testing..."
	@test -d $(NOTEBOOKS_VENV) || virtualenv -p python3 $(NOTEBOOKS_VENV)
	@echo "Installing TF & PYT notebook dependencies..." && \
		. $(ACTIVATE_NOTEBOOKS_VENV) && \
		pip install --extra-index-url https://download.pytorch.org/whl/cpu -r $(CURDIR)/notebooks/requirements.txt

lint: test_venv
	@echo "Style checks..."
	@. $(ACTIVATE_TEST_VENV) && \
		flake8 tlt tests downloader docker && \
		flake8 notebooks && \
		flake8 docs

pytorch_unittest: pytorch_venv
	@echo "Testing pytorch unit test API..." && \
		. $(ACTIVATE_PYTORCH_VENV) && \
		py.test --ignore=tests/tensorflow_tests -vvv -s $(PY_TEST_EXTRA_ARGS) "-k not integration and not skip and not tensorflow"

tensorflow_unittest: tensorflow_venv
	@echo "Testing tensorflow unit test API..." && \
		. $(ACTIVATE_TENSORFLOW_VENV) && \
		py.test --ignore=tests/pytorch_tests -vvv -s $(PY_TEST_EXTRA_ARGS) "-k not integration and not skip and not pytorch"

unittest: pytorch_unittest tensorflow_unittest

pytorch_integration: pytorch_venv
	@echo "Testing pytorch  unit test API..." && \
		. $(ACTIVATE_PYTORCH_VENV) && \
		py.test --ignore=tests/tensorflow_tests -vvv -s $(PY_TEST_EXTRA_ARGS) "-k integration and not skip and not tensorflow"

tensorflow_integration: tensorflow_venv
	@echo "Testing tensorflow  unit test API..." && \
		. $(ACTIVATE_TENSORFLOW_VENV) && \
		py.test --ignore=tests/pytorch_tests -vvv -s $(PY_TEST_EXTRA_ARGS) "-k integration and not skip and not pytorch"

integration: pytorch_integration tensorflow_integration

pytorch_test: pytorch_unittest pytorch_integration

tensorflow_test: tensorflow_unittest tensorflow_integration

test: pytorch_test tensorflow_test

clean_pytorch:
	rm -rf $(PYTORCH_VENV)

clean_tensorflow:
	rm -rf $(TENSORFLOW_VENV)

clean_notebooks:
	rm -rf $(NOTEBOOKS_VENV)

clean_docs:
	rm -rf $(DOCS_VENV)

clean: clean_docs clean_notebooks clean_pytorch clean_tensorflow

docs_venv:
	@echo "Creating a virtualenv for build and test docs..."
	@test -d $(DOCS_VENV) || virtualenv -p python3 $(DOCS_VENV)
	@echo "Installing docs dependencies..." && \
	. $(ACTIVATE_DOCS_VENV) && \
	pip install -r $(CURDIR)/docs/requirements-docs.txt && \
	echo "Installing all TLT dependencies" && \
	pip install --extra-index-url https://download.pytorch.org/whl/cpu .[pytorch,tensorflow]

html: docs_venv
	@echo "Building Sphinx documentation..."
	@. $(ACTIVATE_DOCS_VENV) && $(MAKE) -C docs clean html

test_docs: html
	@echo "Testing Sphinx documentation..."
	@. $(ACTIVATE_DOCS_VENV) && $(MAKE) -C docs doctest

test_pytorch_notebooks: notebooks_venv
	@. $(ACTIVATE_NOTEBOOKS_VENV) && \
		pip install --extra-index-url https://download.pytorch.org/whl/cpu .[pytorch] && \
		bash run_notebooks.sh pytorch

test_tensorflow_notebooks: notebooks_venv
	@. $(ACTIVATE_NOTEBOOKS_VENV) && \
		pip install --extra-index-url https://download.pytorch.org/whl/cpu .[tensorflow] && \
		bash run_notebooks.sh tensorflow


test_custom_notebooks: notebooks_venv
	@echo "Testing Jupyter notebooks with custom datasets..."
	@. $(ACTIVATE_NOTEBOOKS_VENV) && \
		pip install --extra-index-url https://download.pytorch.org/whl/cpu .[pytorch,tensorflow] && \
		bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_pyt_image_classification/TLT_PyTorch_Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
		bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_pyt_text_classification/TLT_PYT_Text_Classification.ipynb remove_for_custom_dataset && \
		bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb remove_for_custom_dataset && \
		bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_tf_text_classification/TLT_TF_Text_Classification.ipynb remove_for_custom_dataset

test_catalog_notebooks: notebooks_venv
	@echo "Testing Jupyter notebooks with public catalog datasets..."
	@. $(ACTIVATE_NOTEBOOKS_VENV) && \
		pip install --extra-index-url https://download.pytorch.org/whl/cpu .[pytorch,tensorflow] && \
		bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_pyt_image_classification/TLT_PyTorch_Image_Classification_Transfer_Learning.ipynb remove_for_tv_dataset && \
		bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_pyt_text_classification/TLT_PYT_Text_Classification.ipynb remove_for_hf_dataset && \
		bash run_notebooks.sh $(CURDIR)/notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb remove_for_tf_dataset && \
		bash run_notebooks.sh $(CURDIR)/notebooks/text_classification/tlt_api_tf_text_classification/TLT_TF_Text_Classification.ipynb remove_for_tf_dataset

release_venv:
	@echo "Creating a virtualenv for to create the installer..."
	@test -d $(RELEASE_VENV) || virtualenv -p python3 $(RELEASE_VENV)
	@. $(ACTIVATE_RELEASE_VENV) && \
		python setup.py bdist_wheel && \
		mkdir $(LATEST_RELEASE) && \
		mv build dist *.egg-info $(LATEST_RELEASE) && \
		cp -rp $(LATEST_RELEASE) $(RELEASE_VENV)/latest

check_release: release_venv
	@echo "Create and smoke test PyPi wheel..." && \
		. $(ACTIVATE_RELEASE_VENV) && \
		pip install twine && \
		twine check $(RELEASE_VENV)/latest/dist/intel_transfer_learning_tool*.whl

# Only run this if absolutely sure to publish the wheel on PyPi
# release:
# 	@echo "Publish PyPi wheel..." && \
# 		. $(ACTIVATE_RELEASE_VENV) && \
# 		twine upload --verbose --repository-url https://upload.pypi.org/legacy/ $(RELEASE_VENV)/<RELEASE_DIR>/dist/<WHEEL_NAME_HERE>
