# Note: These are just placeholders for future additions to Makefile.
# You can remove these comments later.
ACTIVATE_TLK = "tlk_env/bin/activate"
ACTIVATE_TF = "intel_tf/bin/activate"
ACTIVATE_PYT = "intel_pyt/bin/activate"
ACTIVATE_TEST = "tlk_tests/bin/activate"

venv_test: $(CURDIR)/tests/requirements-test.txt
	@echo "Creating a virtualenv tlk_tests..."
	@test -d tlk_tests || virtualenv -p python tlk_tests

	@echo "Building the TLK API in tlk_tests env..."
	@. $(ACTIVATE_TEST) && pip install --editable .[tensorflow,pytorch]

	@echo "Required for TensorFlow text classification..."
	@. $(ACTIVATE_TEST) && pip install tensorflow-text==2.9.0

	@echo "Installing test dependencies..."
	@. $(ACTIVATE_TEST) && pip install -r $(CURDIR)/tests/requirements-test.txt

test: venv_test
	@echo "Testing the API..."
	@. $(ACTIVATE_TEST) && PYTHONPATH="$(CURDIR)/tests" py.test -s

clean:
	rm -rf tlk_tests
	
