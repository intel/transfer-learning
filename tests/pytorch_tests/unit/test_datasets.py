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
# SPDX-License-Identifier: EPL-2.0
#

import os
import math
import pytest
import shutil
import tempfile
from numpy.testing import assert_array_equal
from PIL import Image

from tlk.datasets.dataset_factory import get_dataset, load_dataset

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlk.datasets.image_classification.torchvision_image_classification_dataset import TorchvisionImageClassificationDataset
except ModuleNotFoundError as e:
    print("WARNING: Unable to import TorchvisionImageClassificationDataset. Torch may not be installed")

try:
    # Do torch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlk.datasets.image_classification.pytorch_custom_image_classification_dataset import PyTorchCustomImageClassificationDataset
except ModuleNotFoundError as e:
    print("WARNING: Unable to import PyTorchCustomImageClassificationDataset. Torch may not be installed")


@pytest.mark.pytorch
def test_torchvision_subset():
    """
    Checks that a torchvision test subset can be loaded
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10', 'torchvision', split=["test"])
    assert type(data) == TorchvisionImageClassificationDataset
    assert len(data.dataset) < 50000

@pytest.mark.pytorch
def test_defined_split():
    """
    Checks that dataset can be loaded into train and test subsets based on torchvision splits and then
    re-partitioned with shuffle-split
    """
    data = get_dataset('/tmp/data', 'image_classification', 'pytorch', 'CIFAR10',
                       'torchvision', split=['train', 'test'])
    assert len(data.dataset) == 60000
    assert len(data.train_subset) == 50000
    assert len(data.test_subset) == 10000
    assert data.validation_subset is None
    assert data._train_indices == range(50000)
    assert data._test_indices == range(50000, 60000)
    assert data._validation_type == 'defined_split'

    # Apply shuffle split and verify new subset sizes
    data.shuffle_split(.6, .2, .2, seed=10)
    assert len(data.train_subset) == 36000
    assert len(data.validation_subset) == 12000
    assert len(data.test_subset) == 12000
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
        ic_dataset1 = ImageClassificationDatasetForTest(dataset_dir, None, None, class_names)
        tlk_dataset1 = ic_dataset1.tlk_dataset
        tlk_dataset1.preprocess(image_size, batch_size)
        tlk_dataset1.shuffle_split(seed=seed)

        ic_dataset2 = ImageClassificationDatasetForTest(dataset_dir, None, None, class_names)
        tlk_dataset2 = ic_dataset2.tlk_dataset
        tlk_dataset2.preprocess(image_size, batch_size)
        tlk_dataset2.shuffle_split(seed=seed)

        for i in range(10):
            image_1, label_1 = tlk_dataset1.get_batch()
            image_2, label_2 = tlk_dataset2.get_batch()
            assert_array_equal(image_1, image_2)
            assert_array_equal(label_1, label_2)
    finally:
        if ic_dataset1:
            ic_dataset1.cleanup()
        if ic_dataset2:
            ic_dataset2.cleanup()


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
    ic_dataset = ImageClassificationDatasetForTest(dataset_dir, dataset_name, dataset_catalog, class_names)

    try:
        tlk_dataset = ic_dataset.tlk_dataset

        tlk_dataset.preprocess(224, batch_size)
        assert len(tlk_dataset.get_batch()[0]) == batch_size
    finally:
        ic_dataset.cleanup()


@pytest.mark.pytorch
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names',
                         [['/tmp/data', 'DTD', 'torchvision', None],
                          ['/tmp/data', None, None, ['foo', 'bar']]])
def test_batching_error(dataset_dir, dataset_name, dataset_catalog, class_names):
    """
    Checks that preprocessing cannot be run twice
    """
    ic_dataset = ImageClassificationDatasetForTest(dataset_dir, dataset_name, dataset_catalog, class_names)

    try:
        tlk_dataset = ic_dataset.tlk_dataset
        tlk_dataset.preprocess(224, 1)
        with pytest.raises(Exception) as e:
            tlk_dataset.preprocess(256, 32)
        assert 'Data has already been preprocessed: {}'.\
                   format(tlk_dataset._preprocessed) == str(e.value)
    finally:
        ic_dataset.cleanup()


class ImageClassificationDatasetForTest:
    def __init__(self, dataset_dir, dataset_name=None, dataset_catalog=None, class_names=None):
        """
        This class wraps initialization for image classification datasets (either from torchvision or custom).

        For a custom dataset, provide a dataset dir and class names. A temporary directory will be created with
        dummy folders for the specified class names and 50 images in each folder. The dataset factory will be used to
        load the custom dataset from the dataset directory.

        For an image classification dataset from a catalog, provide the dataset_dir, dataset_name, and dataset_catalog.
        The dataset factory will be used to load the specified dataset.
        """
        use_case = 'image_classification'
        framework = 'pytorch'

        if dataset_name and dataset_catalog:
            self._dataset_catalog = dataset_catalog
            self._tlk_dataset = get_dataset(dataset_dir, use_case, framework, dataset_name, dataset_catalog)
        elif class_names:
            self._dataset_catalog = "custom"
            dataset_dir = tempfile.mkdtemp(dir=dataset_dir)
            if not isinstance(class_names, list):
                raise TypeError("class_names needs to be a list")

            for dir_name in class_names:
                image_class_dir = os.path.join(dataset_dir, dir_name)
                os.makedirs(image_class_dir)
                for n in range(50):
                    img = Image.new(mode='RGB', size=(24, 24))
                    img.save(os.path.join(image_class_dir, 'img_{}.jpg'.format(n)))

            self._tlk_dataset = load_dataset(dataset_dir, use_case, framework)

        self._dataset_dir = dataset_dir

    @property
    def tlk_dataset(self):
        """
        Returns the tlk dataset object
        """
        return self._tlk_dataset

    def cleanup(self):
        """
        Clean up - remove temp files that were created for custom datasets
        """
        if self._dataset_catalog == "custom":
            print("Deleting temp directory:", self._dataset_dir)
            shutil.rmtree(self._dataset_dir)
            # TODO: Should we delete torchvision directories too?

# Metadata about torchvision datasets
torchvision_metadata = {
    'CIFAR10': {
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'size': 50000
    }
}

# Dataset parameters used to define datasets that will be initialized and tested using TestImageClassificationDataset
# The parameters are: dataset_dir, dataset_name, dataset_catalog, and class_names, which map to the constructor
# parameters for ImageClassificationDatasetForTest, which initializes the datasets using the dataset factory.
dataset_params = [("/tmp/data", "CIFAR10", "torchvision", None),
                  ("/tmp/data", None, None, ["a", "b", "c"])]


@pytest.fixture(scope="class", params=dataset_params)
def image_classification_data(request):
    params = request.param

    ic_dataset = ImageClassificationDatasetForTest(*params)

    dataset_dir, dataset_name, dataset_catalog, dataset_classes = params

    def cleanup():
        ic_dataset.cleanup()

    request.addfinalizer(cleanup)

    # Return the tlk dataset along with metadata that tests might need
    return (ic_dataset.tlk_dataset, dataset_name, dataset_classes)


@pytest.mark.pytorch
class TestImageClassificationDataset:
    """
    This class contains image classification dataset tests that only require the dataset to be initialized once. These
    tests will be run once for each of the dataset defined in the dataset_params list.
    """

    @pytest.mark.pytorch
    def test_class_names_and_size(self, image_classification_data):
        """
        Verify the TLK class type, dataset class names, and dataset length after initializaion
        """
        tlk_dataset, dataset_name, dataset_classes = image_classification_data

        if dataset_name is None:
            assert type(tlk_dataset) == PyTorchCustomImageClassificationDataset
            assert len(tlk_dataset.class_names) == len(dataset_classes)
            assert len(tlk_dataset.dataset) == len(dataset_classes) * 50
        else:
            assert type(tlk_dataset) == TorchvisionImageClassificationDataset
            assert len(tlk_dataset.class_names) == len(torchvision_metadata[dataset_name]['class_names'])
            assert len(tlk_dataset.dataset) == torchvision_metadata[dataset_name]['size']

    @pytest.mark.pytorch
    @pytest.mark.parametrize('batch_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_batch_sizes(self, batch_size, image_classification_data):
        """
        Ensures that a ValueError is raised when an invalid batch size is passed
        """
        tlk_dataset, dataset_name, dataset_classes = image_classification_data
        with pytest.raises(ValueError):
            tlk_dataset.preprocess(224, batch_size)

    @pytest.mark.pytorch
    @pytest.mark.parametrize('image_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_image_size(self, image_size, image_classification_data):
        """
        Ensures that a ValueError is raised when an invalid image size is passed
        """
        tlk_dataset, dataset_name, dataset_classes = image_classification_data
        with pytest.raises(ValueError):
            tlk_dataset.preprocess(image_size, batch_size=8)

    @pytest.mark.pytorch
    def test_preprocessing(self, image_classification_data):
        """
        Checks that dataset can be preprocessed only once
        """
        tlk_dataset, dataset_name, dataset_classes = image_classification_data
        tlk_dataset.preprocess(224, 8)
        preprocessing_inputs = {'image_size': 224, 'batch_size': 8}
        assert tlk_dataset._preprocessed == preprocessing_inputs
        # Trying to preprocess again should throw an exception
        with pytest.raises(Exception) as e:
            tlk_dataset.preprocess(324, 32)
        assert 'Data has already been preprocessed: {}'.format(preprocessing_inputs) == str(e.value)
        print(tlk_dataset.info)

    @pytest.mark.pytorch
    def test_shuffle_split_errors(self, image_classification_data):
        """
        Checks that splitting into train, validation, and test subsets will error if inputs are wrong
        """
        tlk_dataset, dataset_name, dataset_classes = image_classification_data

        with pytest.raises(Exception) as e:
            tlk_dataset.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)
        assert 'Sum of percentage arguments must be less than or equal to 1.' == str(e.value)
        with pytest.raises(Exception) as e:
            tlk_dataset.shuffle_split(train_pct=1, val_pct=0)
        assert 'Percentage arguments must be floats.' == str(e.value)

    @pytest.mark.pytorch
    def test_shuffle_split(self, image_classification_data):
        """
        Checks that dataset can be split into train, validation, and test subsets
        """
        tlk_dataset, dataset_name, dataset_classes = image_classification_data

        # Before the shuffle split, validation type should be recall
        assert 'recall' == tlk_dataset._validation_type

        # Perform shuffle split with default percentages
        tlk_dataset.shuffle_split(seed=10)
        default_train_pct = 0.75
        default_val_pct = 0.25

        # Get the full dataset size
        dataset_size = torchvision_metadata[dataset_name]['size'] if dataset_name else len(dataset_classes) * 50

        # Divide by the batch size that was used to preprocess earlier
        dataset_size = dataset_size / tlk_dataset.info['preprocessing_info']['batch_size']

        # The PyTorch loaders are what gets batched and they can be off by 1 from the floor value
        assert math.floor(dataset_size * default_train_pct) <= len(tlk_dataset.train_loader) <= math.ceil(dataset_size * default_train_pct)
        assert math.floor(dataset_size * default_val_pct) <= len(tlk_dataset.validation_loader) <= math.ceil(dataset_size * default_val_pct)
        assert tlk_dataset.test_loader is None
        assert tlk_dataset._validation_type == 'shuffle_split'
