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

venv_test: $(CURDIR)/tests/requirements-test.txt
	@echo "Creating a virtualenv tlt_tests..."
	@test -d tlt_tests || virtualenv -p python tlt_tests

	@echo "Building the TLT API in tlt_tests env..."
	@. $(ACTIVATE_TEST) && pip install --editable .[tensorflow,pytorch]

	@echo "Required for TensorFlow text classification..."
	@. $(ACTIVATE_TEST) && pip install tensorflow-text==2.9.0

	@echo "Installing test dependencies..."
	@. $(ACTIVATE_TEST) && pip install -r $(CURDIR)/tests/requirements-test.txt

test: venv_test
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/tests" py.test -s --cov --cov-fail-under=85

lint: venv_test
	@echo "Style checks..."
	@. $(ACTIVATE_TEST) && flake8 tlt/tools

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

dist: venv_docs
	@echo "Create binary wheel..."
	@. $(ACTIVATE_DOCS) && python setup.py bdist_wheel

check_dist: dist
	@echo "Testing the wheel..."
	@. $(ACTIVATE_DOCS) && \
	pip install twine && \
	python setup.py bdist_wheel && \
	twine check dist/*
