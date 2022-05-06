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
import tensorflow as tf
import tensorflow_datasets as tfds

from tlk.datasets.image_classification.image_classification_dataset import ImageClassificationDataset


class TFImageClassificationDataset(ImageClassificationDataset):
    """
    Base class for an image classification dataset from the TensorFlow datasets catalog
    """
    def __init__(self, dataset_dir, dataset_name, split=["train[:75%]"],
                 as_supervised=True, shuffle_files=True):
        ImageClassificationDataset.__init__(self, dataset_dir, dataset_name)
        self._shuffle_files = shuffle_files

        tf.get_logger().setLevel('ERROR')

        os.environ['NO_GCE_CHECK'] = 'true'
        [self._dataset], self._info = tfds.load(
            dataset_name,
            data_dir=dataset_dir,
            split=split,
            as_supervised=as_supervised,
            shuffle_files=shuffle_files,
            with_info=True
        )

    @property
    def class_names(self):
        return self.info.features["label"].names

    @property
    def info(self):
        return self._info

    @property
    def dataset(self):
        return self._dataset

    def get_batch(self):
        """Get a single batch of images and labels from the dataset.

            Returns:
                 (images, labels)

            Raises:
                    ValueError if the dataset is not defined yet
        """
        if self._dataset:
            return next(iter(self._dataset))
        else:
            raise ValueError("Unable to return a batch, because the dataset hasn't been defined.")

    def preprocess(self, image_size, batch_size):
        """Preprocess the images to convert them to float32 and resize the images

            Args:
                image_size (int): desired square image size
                batch_size (int): desired batch size

            Raises:
                ValueError if the dataset is not defined yet
        """
        # NOTE: Should this be part of init? If we get image_size and batch size during init,
        # then we don't need a separate call to preprocess.
        def preprocess_image(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_with_pad(image, image_size, image_size)
            return (image, label)

        if self._dataset:
            self._dataset = self._dataset.map(preprocess_image)

            self._dataset = self._dataset.cache()

            if self._shuffle_files:
                split_key = next(iter(self.info.splits.keys()))
                self._dataset = self._dataset.shuffle(self.info.splits[split_key].num_examples)

            self._dataset = self._dataset.batch(batch_size)
            self._dataset = self._dataset.prefetch(tf.data.AUTOTUNE)
        else:
            raise ValueError("Unable to preprocess, because the dataset hasn't been defined.")
