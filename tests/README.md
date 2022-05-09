# Testing TLK

To run these tests, first install [tlk](/tlk). Then install the following dependencies:

```
pip3 install -r requirements-test.txt

```

## API Tests
There are unit and integration tests that exercise the API. 
Make sure you are in the `transfer-learning/tests` directory and run:

```
py.test -s
```

> Note: After the tests have run, there will be downloaded data in `/tmp/data` 
that has not been cleaned up. Currently, the developer has to manage this, but
we should create fixtures that take care of it.

There are some executable examples in module docstrings. To run them as tests, follow
the steps in the [docs README.md](/docs/README.md).
