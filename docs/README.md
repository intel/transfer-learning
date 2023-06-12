# Building Documentation

## Sphinx Documentation

Install `tlt` and its dependencies for developers as described the [Get Started](/GetStarted) guide.
```bash
# Run these commands from root of the project
python3 -m virtualenv tlt_dev_venv
source tlt_dev_venv/bin/activate
python -m pip install --editable .
```

Install Pandoc, Sphinx and a few other tools required to build docs
```bash
sudo apt-get install pandoc
pip install -r docs/requirements-docs.txt
```

Navigate to the `docs` directory and run the doctests to ensure all tests pass:
```bash
# run this command from within docs directory
make doctest
```

This should produce output similiar to:
```bash
Doctest summary
===============
    6 tests
    0 failures in tests
    0 failures in setup code
    0 failures in cleanup code
build succeeded.
```

Finally generate the html docs (from within `docs` directory):
```bash
make clean html
```

The output HTML files will be located in `transfer-learning/docs/_build/html`.

To start a local HTTP server and view the docs locally, try:
```bash
make serve
Serving HTTP on 127.0.1.1 port 9999 (http://127.0.1.1:9999/) ...
```

If you need to view the docs from another machine, please try either port forwarding or
provide appropriate values for `LISTEN_IP/LISTEN_PORT` arguments.
For example:
```bash
LISTEN_IP=0.0.0.0 make serve
Serving HTTP on 0.0.0.0 port 9999 (http://0.0.0.0:9999/) ...
```

runs the docs server on the host while listening to all hosts.
Now you can navigate to `HOSTNAME:9999` to view the docs.
