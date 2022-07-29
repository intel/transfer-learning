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
    from tlk.datasets.text_classification.tfds_text_classification_dataset import TFDSTextClassificationDataset
except ModuleNotFoundError as e:
    print("WARNING: Unable to import TFDSTextClassificationDataset. TensorFlow may not be installed")

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
@pytest.mark.parametrize('dataset_name,use_case,train_split,val_split,test_split,train_len,val_len,test_len',
                         [['beans', 'image_classification', 'train', 'validation', None, 1034, 133, 0],
                          ['glue/cola', 'text_classification', 'train', 'validation', 'test', 8551, 1043, 1063]])
def test_defined_split(dataset_name, use_case, train_split, val_split, test_split, train_len, val_len, test_len):
    """
    Checks that dataset can be loaded into train, validation, and test subsets based on TFDS splits and then
    re-partitioned with shuffle-split
    """
    splits = [train_split, val_split, test_split]
    splits = [s for s in splits if s]  # Filter out ones that are None
    data = get_dataset('/tmp/data', use_case, 'tensorflow', dataset_name, 'tf_datasets', split=splits)

    total_len = train_len + val_len + test_len
    assert len(data.dataset) == total_len

    if train_len:
        assert len(data.train_subset) == train_len
    else:
        assert data.train_subset is None

    if val_len:
        assert len(data.validation_subset) == val_len
    else:
        assert data.validation_subset is None

    if test_len:
        assert len(data.test_subset) == test_len
    else:
        assert data.test_subset is None

    assert data._validation_type == 'defined_split'

    # Apply shuffle split and verify new subset sizes
    train_percent = .6
    val_percent = .2
    test_percent = .2
    data.shuffle_split(train_percent, val_percent, test_percent, seed=10)
    assert len(data.train_subset) == int(total_len * train_percent)
    assert len(data.validation_subset) == int(total_len * val_percent)
    assert len(data.test_subset) == total_len - len(data.train_subset) - len(data.validation_subset)
    assert data._validation_type == 'shuffle_split'


@pytest.mark.tensorflow
@pytest.mark.parametrize('dataset_name,use_case,train_split,train_len,val_len',
                         [['tf_flowers', 'image_classification', 'train[:30%]', 825, 275],
                          ['glue/cola', 'text_classification', 'train[:10%]', 641, 213]])
def test_shuffle_split(dataset_name, use_case, train_split, train_len, val_len):
    """
    Checks that dataset can be split into train, validation, and test subsets. The expected train subset length is
    75% of the specified train_split. The expected validation length is 25% of the specified train split.
    """
    flowers = get_dataset('/tmp/data', use_case, 'tensorflow', dataset_name, 'tf_datasets', split=[train_split])
    flowers.shuffle_split(seed=10)
    assert len(flowers.train_subset) == train_len
    assert len(flowers.validation_subset) == val_len
    assert flowers.test_subset is None
    assert flowers._validation_type == 'shuffle_split'


@pytest.mark.tensorflow
@pytest.mark.parametrize('dataset_name,use_case,image_size',
                         [['tf_flowers', 'image_classification', 224],
                          ['glue/cola', 'text_classification', None]])
def test_shuffle_split_deterministic_tfds(dataset_name, use_case, image_size):
    """
    Checks that tfds datasets can be split into train, validation, and test subsets in a way that is reproducible
    """
    seed = 10

    data1 = get_dataset('/tmp/data', use_case, 'tensorflow', dataset_name, 'tf_datasets', shuffle_files=False)
    if image_size:
        data1.preprocess(image_size, batch_size=1)
    else:
        data1.preprocess(batch_size=1)

    data1.shuffle_split(seed=seed)

    data2 = get_dataset('/tmp/data', use_case, 'tensorflow', dataset_name, 'tf_datasets', shuffle_files=False)

    if image_size:
        data2.preprocess(image_size, batch_size=1)
    else:
        data2.preprocess(batch_size=1)

    data2.shuffle_split(seed=seed)

    for i in range(10):
        sample_1, label_1 = data1.get_batch()
        sample_2, label_2 = data2.get_batch()
        assert_array_equal(sample_1, sample_2)
        assert_array_equal(label_1, label_2)


@pytest.mark.tensorflow
def test_shuffle_split_deterministic_custom():
    """
    Checks that custom datasets can be split into train, validation, and test subsets in a way that is reproducible
    """
    dataset_dir = '/tmp/data'
    use_case = 'image_classification'
    class_names = ['foo', 'bar']
    seed = 10
    image_size = 224
    batch_size = 1
    ic_dataset1 = None
    ic_dataset2 = None
    try:
        ic_dataset1 = DatasetForTest(dataset_dir, use_case, None, None, class_names)
        tlk_dataset1 = ic_dataset1.tlk_dataset
        tlk_dataset1.preprocess(image_size, batch_size)
        tlk_dataset1.shuffle_split(seed=seed)

        ic_dataset2 = DatasetForTest(dataset_dir, use_case, None, None, class_names)
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
@pytest.mark.parametrize('dataset_dir,use_case,dataset_name,dataset_catalog,class_names,batch_size',
                         [['/tmp/data', 'image_classification', 'tf_flowers', 'tf_datasets', None, 32],
                          ['/tmp/data', 'image_classification', 'tf_flowers', 'tf_datasets', None, 1],
                          ['/tmp/data', 'image_classification',  None, None, ['foo', 'bar'], 8],
                          ['/tmp/data', 'image_classification', None, None, ['foo', 'bar'], 1],
                          ['/tmp/data', 'text_classification', 'glue/cola', 'tf_datasets', None, 1],
                          ['/tmp/data', 'text_classification', 'glue/cola', 'tf_datasets', None, 32]])
def test_batching(dataset_dir, use_case, dataset_name, dataset_catalog, class_names, batch_size):
    """
    Checks that dataset can be batched with valid positive integer values
    """
    ic_dataset = DatasetForTest(dataset_dir, use_case, dataset_name, dataset_catalog, class_names)

    try:
        tlk_dataset = ic_dataset.tlk_dataset
        if use_case == 'image_classification':
            tlk_dataset.preprocess(224, batch_size)  # image classification needs an image size
        else:
            tlk_dataset.preprocess(batch_size=batch_size)

        assert len(tlk_dataset.get_batch()[0]) == batch_size
    finally:
        ic_dataset.cleanup()


@pytest.mark.tensorflow
@pytest.mark.parametrize('dataset_dir,use_case,dataset_name,dataset_catalog,class_names',
                         [['/tmp/data', 'image_classification', 'tf_flowers', 'tf_datasets', None],
                          ['/tmp/data', 'image_classification', None, None, ['foo', 'bar']],
                          ['/tmp/data', 'text_classification', 'glue/cola', 'tf_datasets', None]])
def test_batching_error(dataset_dir, use_case, dataset_name, dataset_catalog, class_names):
    """
    Checks that preprocessing cannot be run twice
    """
    ic_dataset = DatasetForTest(dataset_dir, use_case, dataset_name, dataset_catalog, class_names)

    try:
        tlk_dataset = ic_dataset.tlk_dataset

        if use_case == 'image_classification':
            tlk_dataset.preprocess(224, 1)  # image classification needs an image size
        else:
            tlk_dataset.preprocess(batch_size=1)

        with pytest.raises(Exception) as e:
            if use_case == 'image_classification':
                tlk_dataset.preprocess(256, 32)
            else:
                tlk_dataset.preprocess(batch_size=32)

        assert 'Data has already been preprocessed: {}'.format(tlk_dataset._preprocessed) == str(e.value)
    finally:
        ic_dataset.cleanup()


@pytest.mark.tensorflow
@pytest.mark.parametrize('dataset_name,use_case,expected_class_names',
                         [['glue/cola', 'text_classification', ['unacceptable', 'acceptable']],
                          ['glue/sst2', 'text_classification', ['negative', 'positive']],
                          ['imdb_reviews', 'text_classification', ['neg', 'pos']]])
def test_supported_tfds_datasets(dataset_name, use_case, expected_class_names):
    """
    Verifies that we are able to load supported datasets and get class names
    """
    dataset = get_dataset('/tmp/data', use_case, 'tensorflow', dataset_name, 'tf_datasets', split=["train[:10%]"])

    assert dataset.class_names == expected_class_names


@pytest.mark.tensorflow
@pytest.mark.parametrize('dataset_name,use_case',
                         [['glue', 'text_classification'],
                          ['sst2', 'text_classification'],
                          ['taco', 'text_classification']])
def test_unsupported_tfds_datasets(dataset_name, use_case):
    """
    Verifies that unsupported datasets get the proper error
    """

    with pytest.raises(ValueError) as e:
        get_dataset('/tmp/data', use_case, 'tensorflow', dataset_name, 'tf_datasets', split=["train[:10%]"])

    assert "Dataset name is not supported" in str(e)


class DatasetForTest:
    def __init__ (self, dataset_dir, use_case, dataset_name=None, dataset_catalog=None, class_names=None):
        """
        This class wraps initialization for datasets (either from TFDS or custom).
        
        For a custom dataset, provide a dataset dir and class names. A temporary directory will be created with
        dummy folders for the specified class names and 50 images in each folder. The dataset factory will be used to
        load the custom dataset from the dataset directory.
        
        For a dataset from a catalog, provide the dataset_dir, dataset_name, and dataset_catalog.
        The dataset factory will be used to load the specified dataset.
        """
        framework = 'tensorflow'

        if dataset_name and dataset_catalog:
            self._dataset_catalog = dataset_catalog
            self._tlk_dataset = get_dataset(dataset_dir, use_case,  framework, dataset_name, dataset_catalog)
        elif class_names:
            self._dataset_catalog = "custom"
            dataset_dir = tempfile.mkdtemp(dir=dataset_dir)
            if not isinstance(class_names, list):
                raise TypeError("class_names needs to be a list")

            if use_case == 'image_classification':
                for dir_name in class_names:
                    image_class_dir = os.path.join(dataset_dir, dir_name)
                    os.makedirs(image_class_dir)
                    for n in range(50):
                        img = Image.new(mode='RGB', size=(24, 24))
                        img.save(os.path.join(image_class_dir, 'img_{}.jpg'.format(n)))
            else:
                raise NotImplementedError("The custom dataset option has only been implemented for images")

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
    },
    'glue/cola': {
        'class_names': ['unacceptable', 'acceptable'],
        'size': 8551
    }
}

# Dataset parameters used to define datasets that will be initialized and tested using DatasetForTest class.
# The parameters are: dataset_dir, use_case, dataset_name, dataset_catalog, and class_names, which map to the
# constructor parameters for DatasetForTest, which initializes the datasets using the dataset factory.
dataset_params = [("/tmp/data", 'image_classification', "tf_flowers", "tf_datasets", None),
                  ("/tmp/data", 'image_classification', None, None, ["a", "b", "c"]),
                  ("/tmp/data", 'text_classification', "glue/cola", "tf_datasets", None)]


@pytest.fixture(scope="class", params=dataset_params)
def test_data(request):
    params = request.param

    ic_dataset = DatasetForTest(*params)

    dataset_dir, use_case, dataset_name, dataset_catalog, dataset_classes = params

    def cleanup():
        ic_dataset.cleanup()

    request.addfinalizer(cleanup)

    # Return the tlk dataset along with metadata that tests might need
    return (ic_dataset.tlk_dataset, dataset_name, dataset_classes, use_case)


@pytest.mark.tensorflow
class TestImageClassificationDataset:
    """
    This class contains image classification dataset tests that only require the dataset to be initialized once. These
    tests will be run once for each of the dataset defined in the dataset_params list.
    """

    @pytest.mark.tensorflow
    def test_class_names_and_size(self, test_data):
        """
        Verify the TLK class type, dataset class names, and dataset length after initialization
        """
        tlk_dataset, dataset_name, dataset_classes, use_case = test_data

        if dataset_name is None:
            assert type(tlk_dataset) == TFCustomImageClassificationDataset
            assert len(tlk_dataset.class_names) == len(dataset_classes)
            assert len(tlk_dataset.dataset) == len(dataset_classes) * 50
        else:
            if use_case == 'image_classification':
                assert type(tlk_dataset) == TFImageClassificationDataset
            elif use_case == 'text_classification':
                assert type(tlk_dataset) == TFDSTextClassificationDataset
                
            assert len(tlk_dataset.class_names) == len(tfds_metadata[dataset_name]['class_names'])
            assert len(tlk_dataset.dataset) == tfds_metadata[dataset_name]['size']

    @pytest.mark.tensorflow
    @pytest.mark.parametrize('batch_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_batch_sizes(self, batch_size, test_data):
        """
        Ensures that a ValueError is raised when an invalid batch size is passed
        """
        tlk_dataset, dataset_name, dataset_classes, use_case = test_data
        with pytest.raises(ValueError):
            if use_case == 'image_classification':
                tlk_dataset.preprocess(224, batch_size)
            else:
                tlk_dataset.preprocess(batch_size=batch_size)

    @pytest.mark.tensorflow
    @pytest.mark.parametrize('image_size',
                             ['foo',
                              -17,
                              20.5])
    def test_invalid_image_size(self, image_size, test_data):
        """
        Ensures that a ValueError is raised when an invalid image size is passed. This test only applies to
        image dataset.
        """
        tlk_dataset, dataset_name, dataset_classes, use_case = test_data

        if use_case == 'image_classification':
            with pytest.raises(ValueError):
                tlk_dataset.preprocess(image_size, batch_size=8)

    @pytest.mark.tensorflow
    def test_preprocessing(self, test_data):
        """
        Checks that dataset can be preprocessed only once
        """
        tlk_dataset, dataset_name, dataset_classes, use_case = test_data

        if use_case == 'image_classification':
            tlk_dataset.preprocess(224, 8)
            preprocessing_inputs = {'image_size': 224, 'batch_size': 8}
        else:
            tlk_dataset.preprocess(batch_size=8)
            preprocessing_inputs = {'batch_size': 8}

        assert tlk_dataset._preprocessed == preprocessing_inputs

        # Trying to preprocess again should throw an exception
        with pytest.raises(Exception) as e:
            if use_case == 'image_classification':
                tlk_dataset.preprocess(324, 32)
            else:
                tlk_dataset.preprocess(batch_size=32)
        assert 'Data has already been preprocessed: {}'.format(preprocessing_inputs) == str(e.value)
        print(tlk_dataset.info)

    @pytest.mark.tensorflow
    def test_shuffle_split_errors(self, test_data):
        """
        Checks that splitting into train, validation, and test subsets will error if inputs are wrong
        """
        tlk_dataset, dataset_name, dataset_classes, use_case = test_data

        with pytest.raises(Exception) as e:
            tlk_dataset.shuffle_split(train_pct=.5, val_pct=.5, test_pct=.2)
        assert 'Sum of percentage arguments must be less than or equal to 1.' == str(e.value)
        with pytest.raises(Exception) as e:
            tlk_dataset.shuffle_split(train_pct=1, val_pct=0)
        assert 'Percentage arguments must be floats.' == str(e.value)

    @pytest.mark.tensorflow
    def test_shuffle_split(self, test_data):
        """
        Checks that dataset can be split into train, validation, and test subsets
        """
        tlk_dataset, dataset_name, dataset_classes, use_case = test_data

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
