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

	@echo "Installing test dependencies..."
	@. $(ACTIVATE_TEST) && pip install -r $(CURDIR)/tests/requirements-test.txt

test: venv_test
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/tests" py.test -s

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

dist:
	@echo "Create binary wheel..."
	@. $(ACTIVATE_DOCS) && python setup.py bdist_wheel

check_dist: dist
	@echo "Testing the wheel..."
	@. $(ACTIVATE_DOCS) && \
	pip install twine && \
	python setup.py bdist_wheel && \
	twine check dist/*
