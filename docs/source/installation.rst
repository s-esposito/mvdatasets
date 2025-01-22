Installation
============

How to install the library depends on your use case.

If you want to install the package for *regular usage* without needing further modifications, install the module from source using ``pip install .``.

To perform an *editable installation* (also known as a development mode installation), install the module from source using ``pip install -e .``.
Changes made to the code in the local directory will take effect immediately when you import the package in Python.

To run tests, install the module with the ``tests`` extra (e.g. ``pip install ".[tests]"``).
To compile the documentation, install the module with the ``docs`` extra (e.g. ``pip install ".[docs]"``).

The library is tested with ``Python 3.10``; other versions (``>=3.8``) may work, but are not officially supported.