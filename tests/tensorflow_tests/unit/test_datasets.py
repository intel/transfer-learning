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

import math
import os
import pytest
import shutil
import tempfile
from numpy.testing import assert_array_equal
from PIL import Image

from tlk.datasets.dataset_factory import get_dataset, load_dataset

try:
    # Do TF specific imports in a try/except to prevent pytest test loading from failing when running in a PyTorch env
    from tlk.datasets.image_classification.tf_image_classification_dataset import TFImageClassificationDataset
except ModuleNotFoundError as e:
    print("WARNING: Unable to import TFImageClassificationDataset. TensorFlow may not be installed")

try:
    # Do TF specific imports in a try/except to prevent pytest test loading from failing when running in a PyTorch env
    from tlk.datasets.image_classification.tf_custom_image_classification_dataset import TFCustomImageClassificationDataset
except ModuleNotFoundError as e:
    print("WARNING: Unable to import TFCustomImageClassificationDataset. TensorFlow may not be installed")

@pytest.mark.tensorflow
def test_tf_flowers_10pct():
    """
    Checks that a 10% tf_flowers subset can be loaded
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                                          'tf_datasets', split=["train[:10%]"])
    assert type(flowers) == TFImageClassificationDataset
    assert len(flowers.dataset) < 3670


@pytest.mark.tensorflow
def test_defined_split():
    """
    Checks that dataset can be loaded into train, validation, and test subsets based on TFDS splits and then
    re-partitioned with shuffle-split
    """
    beans = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'beans',
                          'tf_datasets', split=["train", "validation"])
    assert len(beans.dataset) == 1167
    assert len(beans.train_subset) == 1034
    assert len(beans.validation_subset) == 133
    assert beans.test_subset is None
    assert beans._validation_type == 'defined_split'

    # Apply shuffle split and verify new subset sizes
    beans.shuffle_split(.6, .2, .2, seed=10)
    assert len(beans.train_subset) == 700
    assert len(beans.validation_subset) == 233
    assert len(beans.test_subset) == 234
    assert beans._validation_type == 'shuffle_split'


@pytest.mark.tensorflow
def test_shuffle_split():
    """
    Checks that dataset can be split into train, validation, and test subsets
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers',
                          'tf_datasets', split=["train[:30%]"])
    flowers.shuffle_split(seed=10)
    assert len(flowers.train_subset) == 825
    assert len(flowers.validation_subset) == 275
    assert flowers.test_subset is None
    assert flowers._validation_type == 'shuffle_split'


@pytest.mark.tensorflow
def test_shuffle_split_deterministic_tfds():
    """
    Checks that tfds datasets can be split into train, validation, and test subsets in a way that is reproducible
    """
    flowers = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    flowers.preprocess(224, 1)
    flowers.shuffle_split(seed=10)

    flowers2 = get_dataset('/tmp/data', 'image_classification', 'tensorflow', 'tf_flowers', 'tf_datasets')
    flowers2.preprocess(224, 1)
    flowers2.shuffle_split(seed=10)

    for i in range(10):
        image_1, label_1 = flowers.get_batch()
        image_2, label_2 = flowers2.get_batch()
        assert_array_equal(image_1, image_2)
        assert_array_equal(label_1, label_2)


@pytest.mark.tensorflow
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


@pytest.mark.tensorflow
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names,batch_size',
                         [['/tmp/data', 'tf_flowers', 'tf_datasets', None, 32],
                          ['/tmp/data', 'tf_flowers', 'tf_datasets', None, 1],
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


@pytest.mark.tensorflow
@pytest.mark.parametrize('dataset_dir,dataset_name,dataset_catalog,class_names',
                         [['/tmp/data', 'tf_flowers', 'tf_datasets', None],
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
        assert 'Data has already been preprocessed: {}'.format(tlk_dataset._preprocessed) == str(e.value)
    finally:
        ic_dataset.cleanup()


class ImageClassificationDatasetForTest:
    def __init__ (self, dataset_dir, dataset_name=None, dataset_catalog=None, class_names=None):
        """
        This class wraps initialization for image classification datasets (either from TFDS or custom).
        
        For a custom dataset, provide a dataset dir and class names. A temporary directory will be created with
        dummy folders for the specified class names and 50 images in each folder. The dataset factory will be used to
        load the custom dataset from the dataset directory.
        
        For an image classification datsaet from a catalog, provide the dataset_dir, dataset_name, and dataset_catalog.
        The dataset factory will be used to load the specified dataset.
        """
        use_case = 'image_classification'
        framework = 'tensorflow'

        if dataset_name and dataset_catalog:
            self._dataset_catalog = dataset_catalog
            self._tlk_dataset = get_dataset(dataset_dir, use_case,  framework, dataset_name, dataset_catalog)
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

            self._tlk_dataset = load_dataset(dataset_dir, use_case, framework, seed=10)

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
        # TODO: Should we delete tfds directories too?

# Metadata about tfds datasets
tfds_metadata = {
    'tf_flowers': {
        'class_names': ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses'],
        'size': 3670
    }
}

# Dataset parameters used to define datasets that will be initialized and tested using TestImageClassificationDataset
# The parameters are: dataset_dir, dataset_name, dataset_catalog, and class_names, which map to the constructor
# parameters for ImageClassificationDatasetForTest, which initializes the datasets using the dataset factory.
dataset_params = [("/tmp/data", "tf_flowers", "tf_datasets", None),
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


@pytest.mark.tensorflow
class TestImageClassificationDataset:
    """
    This class contains image classification dataset tests that only require the dataset to be initialized once. These
    tests will be run once for each of the dataset defined in the dataset_params list.
    """

    @pytest.mark.tensorflow
    def test_class_names_and_size(self, image_classification_data):
        """
        Verify the TLK class type, dataset class names, and dataset length after initializaion
        """
        tlk_dataset, dataset_name, dataset_classes = image_classification_data

        if dataset_name is None:
            assert type(tlk_dataset) == TFCustomImageClassificationDataset
            assert len(tlk_dataset.class_names) == len(dataset_classes)
            assert len(tlk_dataset.dataset) == len(dataset_classes) * 50
        else:
            assert type(tlk_dataset) == TFImageClassificationDataset
            assert len(tlk_dataset.class_names) == len(tfds_metadata[dataset_name]['class_names'])
            assert len(tlk_dataset.dataset) == tfds_metadata[dataset_name]['size']

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
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

    @pytest.mark.tensorflow
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
        dataset_size = tfds_metadata[dataset_name]['size'] if dataset_name else len(dataset_classes) * 50

        # Divide by the batch size that was used to preprocess earlier
        dataset_size = dataset_size / tlk_dataset.info['preprocessing_info']['batch_size']

        assert len(tlk_dataset.train_subset) == math.floor(dataset_size * default_train_pct)
        assert len(tlk_dataset.validation_subset) == math.floor(dataset_size * default_val_pct)
        assert tlk_dataset.test_subset is None
        assert tlk_dataset._validation_type == 'shuffle_split'
