# Note: These are just placeholders for future additions to Makefile.
# You can remove these comments later.
ACTIVATE_TLT = "tlt_env/bin/activate"
ACTIVATE_TF = "intel_tf/bin/activate"
ACTIVATE_PYT = "intel_pyt/bin/activate"
ACTIVATE_TEST = "tlt_tests/bin/activate"

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
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/tests" py.test -s

clean:
	rm -rf tlt_tests
	
