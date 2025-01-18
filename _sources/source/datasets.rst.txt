Datasets
========

Currently supported datasets in the library.

Download
--------

Download each dataset by running scripts in the `scripts/download` directory. The scripts will download the data and save it to the `data` directory.

Configuration
-------------

Each dataset has a individual configuration `dataclass` that extends more general configuration options that are shared across all datasets.
The configuration file contains the default settings for the dataset.
Configuration can be overridden by command line arguments or by modifying the configuration file.