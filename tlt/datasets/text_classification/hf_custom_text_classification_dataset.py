import os
from typing import List, Optional, Union, Any

import pandas as pd

from datasets import Dataset

from tlt.datasets.hf_dataset import HFDataset
from tlt.datasets.text_classification.text_classification_dataset import TextClassificationDataset


class HFCustomTextClassificationDataset(TextClassificationDataset, HFDataset):
    def __init__(
        self,
        dataset_dir,
        dataset_name: str,
        file_name: str,
        class_names: List[str],
        column_names: List[str] = None,
        label_map_func: callable = Any,
        header: Optional[Union[int, List[int], None]] = "infer",
        delimiter: Optional[str] = ",",
        shuffle_files: Optional[bool] = False,
        num_workers: Optional[int] = 0,
        usecols: List[int] = None
    ):
        """
        Constructor method used to load a custom text classification dataset from given directory for
        HuggingFace models. The dataset file can be:
            - Comma separated
            - Tab separated


        Args:
            dataset_dir (str): Directory containing the dataset(s)
            dataset_name (str): Name of the dataset. If not given, the file name is used as dataset name
            file_name (str): Name of the file to load from the dataset directory
            class_names (list(str)): List of class label names for the dataset
            column_names (list(str)): List of column names to include in the dataset, if any
            label_map_func (callable): Callable function to map label name to a number
            header (int, list of int, None): Row number(s) to use as the column names, and the start of the data.
            Defualts to "infer"
            delimiter (str): String character that separates the text in eaxh row. Defaults to ",'
            shuffle_files (bool): Boolean to specify whether to shuffle the dataset.
            num_workers (int): Number of workers required when creating a data loader.
            usecols (list(int)): List of integer column indexes to include in the dataset

        Raises:
            FileNotFoundError if the given file_name is not found in the dataset directory
            TypeError if types of class_names (or) label_map_func mismatch
            ValueError if class_names is empty

        """
        # Sanity checks
        self._verify_dataset_file(dataset_dir, file_name)

        # Get the dataset file with the extension
        dataset_file = None
        for f in os.listdir(dataset_dir):
            if os.path.splitext(f)[0] == file_name:
                dataset_file = os.path.join(dataset_dir, f)

        if not isinstance(class_names, list):
            raise TypeError("The class_names is expected to be a list, but found a {}", type(class_names))
        if len(class_names) == 0:
            raise ValueError("The class_names list cannot be empty.")

        if label_map_func and not callable(label_map_func):
            raise TypeError("The label_map_func is expected to be a function, but found a {}", type(label_map_func))

        # The dataset name is only used for informational purposes. Default to use the file name without extension.
        if not dataset_name:
            dataset_name = file_name

        self._class_names = class_names
        self._validation_type = 'custom'
        self._preprocessed = {}
        self._shuffle = shuffle_files
        self._num_workers = num_workers

        TextClassificationDataset.__init__(self, dataset_dir, dataset_name, dataset_catalog=None)

        if column_names:
            dataset_df = pd.read_csv(dataset_file, delimiter=delimiter, header=header, names=column_names,
                                     usecols=usecols, encoding='utf-8', dtype=str)
        else:
            dataset_df = pd.read_csv(dataset_file, delimiter=delimiter, header=header,
                                     usecols=usecols, encoding='utf-8', dtype=str)

        dataset_df['label'] = dataset_df['label'].apply(label_map_func)
        self._dataset = Dataset.from_pandas(dataset_df)

        self._info = {
            'name': dataset_name,
            'class_names': class_names
        }

    def _verify_dataset_file(self, dataset_dir, file_name):
        for dataset_file in os.listdir(dataset_dir):
            if os.path.splitext(dataset_file)[0] == file_name and \
                    os.path.isfile(os.path.join(dataset_dir, dataset_file)):
                return True
        raise FileNotFoundError("The dataset file ({}) does not exist".format(os.path.join(dataset_dir, file_name)))

    @property
    def dataset(self):
        return self._dataset

    @property
    def class_names(self):
        return self._class_names

    @property
    def info(self):
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}
