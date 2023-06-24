#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import os
import math
import pytest
import shutil
import tempfile
import pandas as pd
from numpy.testing import assert_array_equal
from PIL import Image
import random
import string

from tlt.datasets.dataset_factory import get_dataset, load_dataset

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.datasets.image_classification.torchvision_image_classification_dataset import TorchvisionImageClassificationDataset  # noqa: E501
except ModuleNotFoundError:
    print("Unable to import TorchvisionImageClassificationDataset. Torch may not be installed")

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlt.datasets.image_classification.pytorch_custom_image_classification_dataset import PyTorchCustomImageClassificationDataset  # noqa: E501
except ModuleNotFoundError:
    print("Unable to import PyTorchCustomImageClassificationDataset. Torch may not be installed")

try:
    from tlt.datasets.image_anomaly_detection.pytorch_custom_image_anomaly_detection_dataset import PyTorchCustomImageAnomalyDetectionDataset  # noqa: E501
except ModuleNotFoundError:
    print("Unable to import PyTorchCustomImageAnomalyDetectionDataset. Torch may not be installed")

try:
    from tlt.datasets.text_classification.hf_text_classification_dataset import HFTextClassificationDataset
except ModuleNotFoundError:
    print("Unable to import HFTextClassificationDataset. Hugging Face's 'tranformers' API may not be installed \
            in the current env")

try:
    from tlt.datasets.text_classification.hf_custom_text_classification_dataset import HFCustomTextClassificationDataset
except ModuleNotFoundError:
    print("Unable to import HFCustomTextClassificationDataset. Hugging Face's 'tranformers' API may not be \
            installed in the current env")


@pytest.mark.pytorch
def test_torchvision_subset():
    """
    Checks that a torchvision test subset can be loaded
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision', split=["test"])
    assert type(data) == TorchvisionImageClassificationDataset
    assert len(data.dataset) > 0


@pytest.mark.pytorch
def test_defined_split():
    """
    Checks that dataset can be loaded into train and test subsets based on torchvision splits and then
    re-partitioned with shuffle-split
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10',
                       'torchvision', split=['train', 'test'])

    dataset_size = len(data.dataset)
    assert dataset_size > 0
    assert len(data.train_subset) <= dataset_size
    assert len(data.test_subset) <= len(data.train_subset)
    assert data.validation_subset is None
    assert data._validation_type == 'defined_split'

    # Apply shuffle split and verify new subset sizes
    data.shuffle_split(.6, .2, .2, seed=10)
    assert len(data.train_subset) == dataset_size * .6
    assert len(data.validation_subset) == dataset_size * .2
    assert len(data.test_subset) == dataset_size * .2
    assert data._validation_type == 'shuffle_split'


@pytest.mark.pytorch
def test_shuffle_split():
    """
    Checks that dataset can be split into train, validation, and test subsets
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision')
    data.shuffle_split(seed=10)
    assert len(data.train_subset) == 37500
    assert len(data.validation_subset) == 12500
    assert data.test_subset is None
    assert data._validation_type == 'shuffle_split'


@pytest.mark.integration
@pytest.mark.pytorch
def test_shuffle_split_deterministic_tv():
    """
    Checks that dataset can be split into train, validation, and test subsets in a way that is reproducible
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'DTD', 'torchvision', split=['test'])
    data.preprocess(224, 128)
    data.shuffle_split(seed=10)

    data2 = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'DTD', 'torchvision', split=['test'])
    data2.preprocess(224, 128)
    data2.shuffle_split(seed=10)

    for i in range(3):
        image_1, label_1 = data.get_batch()
        image_2, label_2 = data2.get_batch()
        assert_array_equal(image_1, image_2)
        assert_array_equal(label_1, label_2)


@pytest.mark.pytorch
def test_shuffle_split_deterministic_custom():
    """
    Checks that custom datasets can be split into train, validation, and test subsets in a way that is reproducible
    """
    dataset_dir = '/tmp/data'
    class_names = ['foo', 'bar']
    seed = 10
    image_size = 224
    batch_size = 1
    ic_dataset1 = None
    ic_dataset2 = None
    try:
        ic_dataset1 = ImageDatasetForTest(dataset_dir, None, None, class_names)
        tlt_dataset1 = ic_dataset1.tlt_dataset
        tlt_dataset1.preprocess(image_size, batch_size)
        tlt_dataset1.shuffle_split(seed=seed)

        ic_dataset2 = ImageDatasetForTest(dataset_dir, None, None, class_names)
        tlt_dataset2 = ic_dataset2.tlt_dataset
        tlt_dataset2.preprocess(image_size, batch_size)
        tlt_dataset2.shuffle_split(seed=seed)

        for i in range(10):
            image_1, label_1 = tlt_dataset1.get_batch()
            image_2, label_2 = tlt_dataset2.get_batch()
            assert_array_equal(image_1, image_2)
            assert_array_equal(label_1, label_2)
    finally:
        if ic_dataset1:
            ic_dataset1.cleanup()
        if ic_dataset2:
            ic_dataset2.cleanup()


@pytest.mark.integration
@pytest.mark.pytorch
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names,batch_size',
                         [['/tmp/data', 'DTD', 'torchvision', None, 32],
                          ['/tmp/data', 'DTD', 'torchvision', None, 1],
                          ['/tmp/data', None, None, ['foo', 'bar'], 8],
                          ['/tmp/data', None, None, ['foo', 'bar'], 1]])
def test_batching(dataset_dir, dataset_name, dataset_catalog, class_names, batch_size):
    """
    Checks that dataset can be batched with valid positive integer values
    """
    ic_dataset = ImageDatasetForTest(dataset_dir, dataset_name, dataset_catalog, class_names)

    try:
        tlt_dataset = ic_dataset.tlt_dataset

        tlt_dataset.preprocess(224, batch_size)
        assert len(tlt_dataset.get_batch()[0]) == batch_size
    finally:
        ic_dataset.cleanup()


@pytest.mark.integration
@pytest.mark.pytorch
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names',
                         [['/tmp/data', 'DTD', 'torchvision', None],
                          ['/tmp/data', None, None, ['foo', 'bar']]])
def test_batching_error(dataset_dir, dataset_name, dataset_catalog, class_names):
    """
    Checks that preprocessing cannot be run twice
    """
    ic_dataset = ImageDatasetForTest(dataset_dir, dataset_name, dataset_catalog, class_names)

    try:
        tlt_dataset = ic_dataset.tlt_dataset
        tlt_dataset.preprocess(224, 1)
        with pytest.raises(Exception) as e:
            tlt_dataset.preprocess(256, 32)
        assert 'Data has already been preprocessed: {}'.\
            format(tlt_dataset._preprocessed) == str(e.value)
    finally:
        ic_dataset.cleanup()


class ImageDatasetForTest:
    def __init__(self, dataset_dir, dataset_name=None, dataset_catalog=None, class_names=None,
                 splits=None, use_case=None):
        """
        This class wraps initialization for image classification datasets (either from torchvision or custom).

        For a custom dataset, provide a dataset dir and class names, with or without splits such as ['train',
        'validation', 'test']. A temporary directory will be created with dummy folders for the specified split
        subfolders and class names and 50 images in each folder. The dataset factory will be used to load the custom
        dataset from the dataset directory.

        For an image classification dataset from a catalog, provide the dataset_dir, dataset_name, and dataset_catalog.
        The dataset factory will be used to load the specified dataset.
        """
        use_case = 'image_classification' if use_case is None else use_case
        framework = 'pytorch'

        def make_n_files(file_dir, n):
            os.makedirs(file_dir)
            for i in range(n):
                img = Image.new(mode='RGB', size=(24, 24))
                img.save(os.path.join(file_dir, 'img_{}.jpg'.format(i)))

        if dataset_name and dataset_catalog:
            self._dataset_catalog = dataset_catalog
            self._tlt_dataset = get_dataset(dataset_dir, use_case, framework, dataset_name, dataset_catalog)
        elif class_names:
            self._dataset_catalog = "custom"
            dataset_dir = tempfile.mkdtemp(dir=dataset_dir)
            if not isinstance(class_names, list):
                raise TypeError("class_names needs to be a list")

            if isinstance(splits, list):
                for folder in splits:
                    for dir_name in class_names:
                        make_n_files(os.path.join(dataset_dir, folder, dir_name), 50)
            elif splits is None:
                for dir_name in class_names:
                    make_n_files(os.path.join(dataset_dir, dir_name), 50)
            else:
                raise ValueError("Splits must be None or a list of strings, got {}".format(splits))

            self._tlt_dataset = load_dataset(dataset_dir, use_case, framework)

        self._dataset_dir = dataset_dir

    @property
    def tlt_dataset(self):
        """
        Returns the tlt dataset object
        """
        return self._tlt_dataset

    def cleanup(self):
        """
        Clean up - remove temp files that were created for custom datasets
        """
        if self._dataset_catalog == "custom":
            print("Deleting temp directory:", self._dataset_dir)
            shutil.rmtree(self._dataset_dir)
            # TODO: Should we delete Torchvision directories too?


# Metadata about Torchvision datasets
torchvision_metadata = {
    'CIFAR10': {
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'size': 50000
    }
}

# Dataset parameters used to define datasets that will be initialized and tested using TestImageClassificationDataset
# The parameters are: dataset_dir, dataset_name, dataset_catalog, and class_names, which map to the constructor
# parameters for ImageDatasetForTest, which initializes the datasets using the dataset factory.
dataset_params = [("/tmp/data", "CIFAR10", "torchvision", None, None),
                  ("/tmp/data", None, None, ["a", "b", "c"], None),
                  ("/tmp/data", None, None, ["a", "b", "c"], ['train', 'test'])]


@pytest.fixture(scope="class", params=dataset_params)
def image_classification_data(request):
    params = request.param

    ic_dataset = ImageDatasetForTest(*params)

    dataset_dir, dataset_name, dataset_catalog, dataset_classes, splits = params

    def cleanup():
        ic_dataset.cleanup()

    request.addfinalizer(cleanup)

    # Return the tlt dataset along with metadata that tests might need
    return (ic_dataset.tlt_dataset, dataset_name, dataset_classes, splits)


@pytest.mark.pytorch
class TestImageClassificationDataset:
    """
    This class contains image classification dataset tests that only require the dataset to be initialized once. These
    tests will be run once for each of the dataset defined in the dataset_params list.
    """

    def test_class_names_and_size(self, image_classification_data):
        """
        Verify the class type, dataset class names, and dataset length after initializaion
        """
        tlt_dataset, dataset_name, dataset_classes, splits = image_classification_data

        if dataset_name is None:
            assert type(tlt_dataset) == PyTorchCustomImageClassificationDataset
            assert len(tlt_dataset.class_names) == len(dataset_classes)
            if splits is None:
                assert len(tlt_dataset.dataset) == len(dataset_classes) * 50
            else:
                assert len(tlt_dataset.dataset) == len(dataset_classes) * len(splits) * 50
        else:
            assert type(tlt_dataset) == TorchvisionImageClassificationDataset
            assert len(tlt_dataset.class_names) == len(torchvision_metadata[dataset_name]['class_names'])
            assert len(tlt_dataset.dataset) == torchvision_metadata[dataset_name]['size']

    @pytest.mark.parametrize('batch_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_batch_sizes(self, batch_size, image_classification_data):
        """
        Ensures that a ValueError is raised when an invalid batch size is passed
        """
        tlt_dataset, dataset_name, dataset_classes, splits = image_classification_data
        with pytest.raises(ValueError):
            tlt_dataset.preprocess(224, batch_size)

    @pytest.mark.parametrize('image_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_image_size(self, image_size, image_classification_data):
        """
        Ensures that a ValueError is raised when an invalid image size is passed
        """
        tlt_dataset, dataset_name, dataset_classes, splits = image_classification_data
        with pytest.raises(ValueError):
            tlt_dataset.preprocess(image_size, batch_size=8)

    def test_preprocessing(self, image_classification_data):
        """
        Checks that dataset can be preprocessed only once
        """
        tlt_dataset, dataset_name, dataset_classes, splits = image_classification_data
        tlt_dataset.preprocess(224, 8)
        preprocessing_inputs = {'image_size': 224, 'batch_size': 8}
        assert tlt_dataset._preprocessed == preprocessing_inputs
        # Trying to preprocess again should throw an exception
        with pytest.raises(Exception) as e:
            tlt_dataset.preprocess(324, 32)
        assert 'Data has already been preprocessed: {}'.format(preprocessing_inputs) == str(e.value)
        print(tlt_dataset.info)

    def test_shuffle_split_errors(self, image_classification_data):
        """
        Checks that splitting into train, validation, and test subsets will error if inputs are wrong
        """
        tlt_dataset, dataset_name, dataset_classes, splits = image_classification_data

        with pytest.raises(Exception) as e:
            tlt_dataset.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)
        assert 'Sum of percentage arguments must be less than or equal to 1.' == str(e.value)
        with pytest.raises(Exception) as e:
            tlt_dataset.shuffle_split(train_pct=1, val_pct=0)
        assert 'Percentage arguments must be floats.' == str(e.value)

    def test_shuffle_split(self, image_classification_data):
        """
        Checks that dataset can be split into train, validation, and test subsets
        """
        tlt_dataset, dataset_name, dataset_classes, splits = image_classification_data

        # Before the shuffle split, validation type should be recall
        if splits is None:
            assert tlt_dataset._validation_type is None
        else:
            assert 'defined_split' == tlt_dataset._validation_type

        # Perform shuffle split with default percentages
        tlt_dataset.shuffle_split(seed=10)
        default_train_pct = 0.75
        default_val_pct = 0.25

        # Get the full dataset size
        len_splits = 1 if splits is None else len(splits)
        ds_size = torchvision_metadata[dataset_name]['size'] if dataset_name else len(dataset_classes) * len_splits * 50

        # Divide by the batch size that was used to preprocess earlier
        ds_size = ds_size / tlt_dataset.info['preprocessing_info']['batch_size']

        # The PyTorch loaders are what gets batched and they can be off by 1 from the floor value
        assert math.floor(
            ds_size * default_train_pct) <= len(tlt_dataset.train_loader) <= math.ceil(ds_size * default_train_pct)
        assert math.floor(
            ds_size * default_val_pct) <= len(tlt_dataset.validation_loader) <= math.ceil(ds_size * default_val_pct)
        assert tlt_dataset.test_loader is None
        assert tlt_dataset._validation_type == 'shuffle_split'


# Tests for Image Anomaly Detection datasets
@pytest.mark.pytorch
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names,use_case',
                         [["/tmp/data", "CIFAR10", "torchvision", [], 'anomaly_detection'],
                          ["/tmp/data", None, None, ["a", "b", "c"], 'image_anomaly_detection']])
def test_bad_anomaly_dataset(dataset_dir, dataset_name, dataset_catalog, class_names, use_case):
    """
    Checks that torchvision datasets are not implemented and that a nonexistent 'good' folder will throw an error
    """
    try:
        get_dataset(dataset_dir, use_case, 'pytorch', dataset_name, dataset_catalog)
        assert False
    except NotImplementedError:
        assert True
    try:
        load_dataset(dataset_dir, use_case, framework="pytorch")
        assert False
    except FileNotFoundError as e:
        assert "Couldn't find 'good' folder" in str(e)


anomaly_dataset_params = [("/tmp/data", None, None, ["good", "bad"], None, 'anomaly_detection'),
                          ("/tmp/data", None, None, ["good", "foo", "bar"], None, 'image_anomaly_detection')]


@pytest.fixture(scope="class", params=anomaly_dataset_params)
def anomaly_detection_data(request):
    params = request.param

    ad_dataset = ImageDatasetForTest(*params)

    dataset_dir, dataset_name, dataset_catalog, dataset_classes, splits, use_case = params

    def cleanup():
        ad_dataset.cleanup()

    request.addfinalizer(cleanup)

    # Return the tlt dataset along with metadata that tests might need
    return (ad_dataset.tlt_dataset, dataset_name, dataset_classes, use_case)


# Tests for Image Anomaly Detection use case
@pytest.mark.pytorch
class TestImageAnomalyDetectionDataset:
    """
    This class contains image anomaly detection dataset tests that only require the dataset to be initialized once.
    These tests will be run once for each of the dataset defined in the anomaly_dataset_params list.
    """

    def test_classes_defects_and_size(self, anomaly_detection_data):
        """
        Verify the class type, dataset class names, defect_names, and dataset length after initializaion
        """
        tlt_dataset, dataset_name, dataset_classes, use_case = anomaly_detection_data

        assert type(tlt_dataset) == PyTorchCustomImageAnomalyDetectionDataset
        assert len(tlt_dataset.class_names) == 2  # Always 2 for anomaly detection
        assert len(tlt_dataset.defect_names) == len(dataset_classes) - 1  # Subtract 1 for the "good" class
        assert len(tlt_dataset.dataset) == len(dataset_classes) * 50

    def test_preprocessing(self, anomaly_detection_data):
        """
        Checks that dataset can be preprocessed only once
        """
        tlt_dataset, dataset_name, dataset_classes, use_case = anomaly_detection_data
        tlt_dataset.preprocess(224, 8)
        preprocessing_inputs = {'image_size': 224, 'batch_size': 8}
        assert tlt_dataset._preprocessed == preprocessing_inputs
        # Trying to preprocess again should throw an exception
        with pytest.raises(Exception) as e:
            tlt_dataset.preprocess(324, 32)
        assert 'Data has already been preprocessed: {}'.format(preprocessing_inputs) == str(e.value)
        print(tlt_dataset.info)

    def test_shuffle_split_errors(self, anomaly_detection_data):
        """
        Checks that splitting into train, validation, and test subsets will error if inputs are wrong
        """
        tlt_dataset, dataset_name, dataset_classes, use_case = anomaly_detection_data

        with pytest.raises(Exception) as e:
            tlt_dataset.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)
        assert 'Sum of percentage arguments must be less than or equal to 1.' == str(e.value)
        with pytest.raises(Exception) as e:
            tlt_dataset.shuffle_split(train_pct=1, val_pct=0)
        assert 'Percentage arguments must be floats.' == str(e.value)

    def test_shuffle_split(self, anomaly_detection_data):
        """
        Checks that dataset can be split into train, validation, and test subsets
        """
        tlt_dataset, dataset_name, dataset_classes, use_case = anomaly_detection_data

        # Before the shuffle split, validation type should be None
        assert tlt_dataset._validation_type is None

        # Perform shuffle split with default percentages
        tlt_dataset.shuffle_split(seed=10)
        default_train_pct = 0.75
        default_val_pct = 0.25

        # Get the full dataset size
        ds_size = torchvision_metadata[dataset_name]['size'] if dataset_name else len(dataset_classes) * 50

        # Divide by the batch size that was used to preprocess earlier
        ds_size = ds_size / tlt_dataset.info['preprocessing_info']['batch_size']
        good_size = 50 / tlt_dataset.info['preprocessing_info']['batch_size']

        # The PyTorch loaders are what gets batched and they can be off by 1 from the floor value
        assert math.floor(
            good_size * default_train_pct) <= len(tlt_dataset.train_loader) <= math.ceil(good_size * default_train_pct)
        assert math.floor(good_size * default_val_pct) + (ds_size - good_size) <= \
            len(tlt_dataset.validation_loader) <= math.ceil(good_size * default_val_pct) + (ds_size - good_size)
        assert tlt_dataset.test_loader is None
        assert tlt_dataset._validation_type == 'shuffle_split'


# =======================================================================================

# Testing for Text classification use case


hf_metadata = {
    'imdb': {
        'class_names': ['neg', 'pos'],
        'size': 25000
    }
}


class TextClassificationDatasetForTest:
    def __init__(self, dataset_dir, dataset_name=None, dataset_catalog=None, class_names=None):
        """
        This class wraps initialization for text classification datasets from Hugging Face.

        For a text classification dataset from Hugging Face catalog, provide the dataset_dir, dataset_name, and \
        dataset_catalog. The dataset factory will be used to load the specified dataset.
        """
        use_case = 'text_classification'
        framework = 'pytorch'
        dataset_dir = tempfile.mkdtemp(dir=dataset_dir)

        if dataset_name and dataset_catalog:
            self._dataset_catalog = dataset_catalog
            self._tlt_dataset = get_dataset(dataset_dir, use_case, framework, dataset_name, dataset_catalog)
        elif class_names:
            self._dataset_catalog = 'custom'
            if not isinstance(class_names, list):
                raise TypeError("class_names needs to be a list")

            df = self._create_dataset(n_rows=50, class_names=class_names)
            df.to_csv(os.path.join(dataset_dir, 'random_text_dataset'), index=False)

            self._tlt_dataset = load_dataset(dataset_dir, use_case, framework,
                                             csv_file_name='random_text_dataset', header=True)

        self._dataset_dir = dataset_dir

    @property
    def tlt_dataset(self):
        """
        Returns the tlt dataset object
        """
        return self._tlt_dataset

    def _create_dataset(self, n_rows, class_names):
        n_sentences = list(range(3, 10))

        def get_random_word(n_chars):
            return ''.join(random.choices(string.ascii_letters, k=n_chars))

        def get_random_sentence(n_words):
            sentence = ''
            for _ in range(n_words):
                sentence += '{} '.format(get_random_word(random.choice(list(range(2, 10)))))
            return sentence.rstrip()

        dataset = []
        for row in range(n_rows):
            dataset.append([class_names[row % len(class_names)], get_random_sentence(random.choice(n_sentences))])

        return pd.DataFrame(dataset, columns=['label', 'text'])

    def cleanup(self):
        """
        Clean up - remove temp files that were created
        """
        print("Deleting temp directory:", self._dataset_dir)
        shutil.rmtree(self._dataset_dir)


# Dataset parameters used to define datasets that will be initialized and tested using TestTextClassificationDataset
# The parameters are: dataset_dir, dataset_name, dataset_catalog, dataset_classes which map to the constructor
# parameters for TextClassificationDatasetForTest, which initializes the dataset using the dataset factory.
dataset_params = [("/tmp/data", "imdb", "huggingface", ['neg', 'pos']),
                  ("/tmp/data", None, None, ["a", "b", "c"])]


@pytest.fixture(scope="class", params=dataset_params)
def text_classification_data(request):
    params = request.param

    tc_dataset = TextClassificationDatasetForTest(*params)

    dataset_dir, dataset_name, dataset_catalog, class_names = params

    def cleanup():
        tc_dataset.cleanup()

    request.addfinalizer(cleanup)

    # Return the tlt dataset along with metadata that tests might need
    return (tc_dataset.tlt_dataset, dataset_dir, dataset_name, dataset_catalog, class_names)


@pytest.mark.integration
@pytest.mark.pytorch
class TestTextClassificationDataset:
    """
    This class contains text classification dataset tests that only require the dataset to be initialized once. These
    tests will be run once for each of the datasets defined in the dataset_params list.
    """

    def test_tlt_dataset(self, text_classification_data):
        """
        Tests whether a matching Intel Transfer Learning Tool dataset object is returned
        """
        tlt_dataset, _, dataset_name, _, _ = text_classification_data
        if dataset_name is None:
            assert type(tlt_dataset) == HFCustomTextClassificationDataset
        else:
            assert type(tlt_dataset) == HFTextClassificationDataset

    def test_class_names_and_size(self, text_classification_data):
        """
        Verify the class type, dataset class names, and dataset length after initializaion
        """
        tlt_dataset, _, dataset_name, _, class_names = text_classification_data
        if dataset_name is None:
            assert len(tlt_dataset.class_names) == len(class_names)
            assert len(tlt_dataset.dataset) == 50
        else:
            assert tlt_dataset.class_names == class_names
            assert len(tlt_dataset.dataset) == hf_metadata[dataset_name]['size']

    @pytest.mark.parametrize('batch_size',
                             ['foo',  # A string
                              -17,  # A negative int
                              20.5,  # A float
                              ])
    def test_invalid_batch_size_type(self, batch_size, text_classification_data):
        """
        Ensures that a ValueError is raised when an invalid batch size type is passed
        """
        tlt_dataset, _, _, _, _ = text_classification_data
        with pytest.raises(ValueError):
            tlt_dataset.preprocess('', batch_size)

    def test_shuffle_split_errors(self, text_classification_data):
        """
        Checks that splitting into train, validation, and test subsets will error if inputs are wrong
        """
        tlt_dataset, _, _, _, _ = text_classification_data
        with pytest.raises(ValueError) as sum_err_message:
            tlt_dataset.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)

        with pytest.raises(ValueError) as float_err_message:
            tlt_dataset.shuffle_split(train_pct=1, val_pct=0)

        assert 'Sum of percentage arguments must be less than or equal to 1.' == str(sum_err_message.value)
        assert 'Percentage arguments must be floats.' == str(float_err_message.value)
