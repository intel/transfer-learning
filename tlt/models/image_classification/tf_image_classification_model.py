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

import inspect
import os
import numpy as np
import tensorflow as tf

from tlt.models.tf_model import TFModel
from tlt.models.image_classification.image_classification_model import ImageClassificationModel
from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlt.datasets.image_classification.tfds_image_classification_dataset import TFDSImageClassificationDataset
from tlt.datasets.image_classification.tf_custom_image_classification_dataset import TFCustomImageClassificationDataset
from tlt.utils.file_utils import verify_directory, validate_model_name
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.distributed import TLT_DISTRIBUTED_DIR


class TFImageClassificationModel(ImageClassificationModel, TFModel):
    """
    Class to represent a TF custom pretrained model for image classification
    """

    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):
        """
        Class constructor
        """
        self._image_size = None

        # Store the dataset type that this model type can use for Intel Neural Compressor
        self._inc_compatible_dataset = (TFCustomImageClassificationDataset, TFDSImageClassificationDataset)

        # extra properties that will become configurable in the future
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._generate_checkpoints = True

        # placeholder for model definition
        self._num_classes = None

        TFModel.__init__(self, model_name, FrameworkType.TENSORFLOW, UseCaseType.IMAGE_CLASSIFICATION)
        ImageClassificationModel.__init__(self, self._image_size, self._do_fine_tuning, self._dropout_layer_rate,
                                          self._model_name, self._framework, self._use_case)

        # set up the configurable optimizer and loss functions
        self._check_optimizer_loss(optimizer, loss)
        config = {'from_logits': True}
        config.update(kwargs)
        self._optimizer_class = optimizer if optimizer else tf.keras.optimizers.Adam
        self._opt_args = {k: v for k, v in config.items() if k in inspect.getfullargspec(self._optimizer_class).args}
        self._optimizer = None  # This gets initialized later
        self._loss_class = loss if loss else tf.keras.losses.SparseCategoricalCrossentropy
        self._loss_args = {k: v for k, v in config.items() if k in inspect.getfullargspec(self._loss_class).args}
        self._loss = self._loss_class(**self._loss_args)

        if model is None:
            self._model = None
        elif isinstance(model, str):
            self.load_from_directory(model)
            self._num_classes = self._model.output.shape[-1]
            self._image_size = self._model.input.shape[1]
        elif isinstance(model, tf.keras.Model):
            self._model = model
            self._num_classes = self._model.output.shape[-1]
            self._image_size = self._model.input.shape[1]
        else:
            raise TypeError("The model input must be a keras Model, string, or None but found a {}".format(type(model)))

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    def _get_train_callbacks(self, dataset, output_dir, initial_checkpoints, do_eval, early_stopping,
                             lr_decay):
        self._optimizer = self._optimizer_class(learning_rate=self._learning_rate, **self._opt_args)
        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=['acc'])

        if initial_checkpoints:
            if os.path.isdir(initial_checkpoints):
                initial_checkpoints = tf.train.latest_checkpoint(initial_checkpoints)

            self._model.load_weights(initial_checkpoints)

        class CollectBatchStats(tf.keras.callbacks.Callback):
            def __init__(self):
                self.batch_losses = []
                self.batch_acc = []

            def on_epoch_begin(self, epoch, logs=None):
                self.batch_losses = []
                self.batch_acc = []

            def on_train_batch_begin(self, batch, logs=None):
                self.model.reset_metrics()

            def on_train_batch_end(self, batch, logs=None):
                self.batch_losses.append(logs['loss'])
                self.batch_acc.append(logs['acc'])

            def on_epoch_end(self, epoch, logs=None):
                # Using the average over all batches is also common instead of just the last batch
                logs['loss'] = self.batch_losses[-1]  # np.mean(self.batch_losses)
                logs['acc'] = self.batch_acc[-1]  # np.mean(self.batch_acc)

        batch_stats_callback = CollectBatchStats()

        callbacks = [batch_stats_callback]

        if early_stopping:
            stop_early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            callbacks.append(stop_early_callback)

        # Create a callback for generating checkpoints
        if self._generate_checkpoints:
            valid_model_name = validate_model_name(self.model_name)
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(valid_model_name))
            verify_directory(checkpoint_dir)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, valid_model_name), save_weights_only=True)
            print("Checkpoint directory:", checkpoint_dir)
            callbacks.append(checkpoint_callback)

        # Create a callback for learning rate decay
        if do_eval and lr_decay:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                verbose=2,
                mode='auto',
                cooldown=1,
                min_lr=0.0000000001))

        train_dataset = dataset.train_subset if dataset.train_subset else dataset.dataset

        validation_data = dataset.validation_subset if do_eval else None

        return callbacks, train_dataset, validation_data

    def _fit_distributed(self, saved_objects_dir, epochs, shuffle, hostfile, nnodes, nproc_per_node, use_horovod):
        import subprocess  # nosec: B404
        distributed_vision_script = os.path.join(TLT_DISTRIBUTED_DIR, 'tensorflow', 'run_train_tf.py')

        if use_horovod:
            run_cmd = 'horovodrun'
        else:
            run_cmd = 'mpirun'

            # mpirun requires these flags to be set
            run_cmd += ' --allow-run-as-root -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0'  # noqa: E501

        hostfile_cmd = ''
        np_cmd = ''
        if os.path.isfile(hostfile):

            hostfile_info = self._parse_hostfile(hostfile)
            node_count = 0
            if sum(hostfile_info['slots']) == 0:
                for ip_addr in hostfile_info['ip_addresses']:
                    hostfile_cmd += '{}:{},'.format(ip_addr, nproc_per_node)
                    node_count += 1
                    if node_count == nnodes:
                        break

            elif sum(hostfile_info['slots']) == nnodes * nproc_per_node:
                for ip_addr, slots in zip(hostfile_info['ip_addresses'], hostfile_info['slots']):
                    hostfile_cmd += '{}:{},'.format(ip_addr, slots)
            else:
                print("WARNING: nproc_per_node and slots in hostfile do not add up. Making equal slots for all nodes")
                for ip_addr in hostfile_info['ip_addresses']:
                    hostfile_cmd += '{}:{},'.format(ip_addr, nproc_per_node)

            hostfile_cmd = hostfile_cmd[:-1]  # Remove trailing comma

            # Final check to correct the `-np` flag's value
            nprocs = nnodes * nproc_per_node
            np_cmd = str(nprocs)
        else:
            raise ValueError("Error: Invalid file \'{}\'".format(hostfile))
        script_cmd = 'python ' + distributed_vision_script
        script_cmd += ' --use_case {}'.format('image_classification')
        script_cmd += ' --epochs {}'.format(epochs)
        script_cmd += ' --tlt_saved_objects_dir {}'.format(saved_objects_dir)
        if shuffle:
            script_cmd += ' --shuffle'

        bash_command = run_cmd.split(' ') + ['-np', np_cmd, '-H', hostfile_cmd] + script_cmd.split(' ')
        print(' '.join(str(e) for e in bash_command))
        subprocess.run(bash_command)

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, enable_auto_mixed_precision=None,
              shuffle_files=True, seed=None, distributed=False, hostfile=None, nnodes=1, nproc_per_node=1,
              callbacks=None, **kwargs):
        """
        Trains the model using the specified image classification dataset. The model is compiled and trained for
        the specified number of epochs. If a path to initial checkpoints is provided, those weights are loaded before
        training.

        Args:
            dataset (ImageClassificationDataset): Dataset to use when training the model
            output_dir (str): Path to a writeable directory for checkpoint files
            epochs (int): Number of epochs to train the model (default: 1)
            initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                latest checkpoint will be used.
            do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch.
            early_stopping (bool): Enable early stopping if convergence is reached while training
            lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
            enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                It is recommended to enable auto mixed precision training when running on platforms that support
                bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                does not support bfloat16, it can be detrimental to the training performance. If
                enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                running with Intel fourth generation Xeon processors, and disabled for other platforms.
            shuffle_files (bool): Boolean specifying whether to shuffle the training data before each epoch.
            seed (int): Optionally set a seed for reproducibility.
            callbacks (list): List of keras.callbacks.Callback instances to apply during training.

        Returns:
            History object from the model.fit() call

        Raises:
           FileExistsError: if the output directory is a file
           TypeError: if the dataset specified is not an ImageClassificationDataset
           TypeError: if the output_dir parameter is not a string
           TypeError: if the epochs parameter is not a integer
           TypeError: if the initial_checkpoints parameter is not a string
           RuntimeError: if the number of model classes is different from the number of dataset classes
        """
        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints)

        dataset_num_classes = len(dataset.class_names)

        # Check that the number of classes matches the model outputs
        if dataset_num_classes != self.num_classes:
            raise RuntimeError("The number of model outputs ({}) differs from the number of dataset classes ({})".
                               format(self.num_classes, dataset_num_classes))

        if callbacks and not isinstance(callbacks, list):
            callbacks = list(callbacks) if isinstance(callbacks, tuple) else [callbacks]

        if callbacks and not all(isinstance(callback, tf.keras.callbacks.Callback) for callback in callbacks):
            raise TypeError('Callbacks must be tf.keras.callbacks.Callback instances')

        self._set_seed(seed)

        # Set auto mixed precision
        self.set_auto_mixed_precision(enable_auto_mixed_precision)

        train_callbacks, train_data, val_data = self._get_train_callbacks(dataset, output_dir, initial_checkpoints,
                                                                          do_eval, early_stopping, lr_decay)
        if callbacks:
            train_callbacks += callbacks

        if distributed:
            try:
                saved_objects_dir = self.export_for_distributed(
                    export_dir=os.path.join(output_dir, "tlt_saved_objects"),
                    train_data=train_data,
                    val_data=val_data
                )
                self._fit_distributed(saved_objects_dir, epochs, shuffle_files, hostfile, nnodes, nproc_per_node,
                                      kwargs.get('use_horovod'))
            except Exception as err:
                print("Error: \'{}\' occured while distributed training".format(err))
            finally:
                self.cleanup_saved_objects_for_distributed()
        else:
            history = self._model.fit(train_data, epochs=epochs, shuffle=shuffle_files, callbacks=train_callbacks,
                                      validation_data=val_data)
            self._history = history.history
            return self._history

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False, callbacks=None):
        """
        Evaluate the accuracy of the model on a dataset.

        If there is a validation subset, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, the entire non-partitioned dataset will be
        used for evaluation.
        """
        if use_test_set:
            if dataset.test_subset:
                eval_dataset = dataset.test_subset
            else:
                raise ValueError("No test subset is defined")
        elif dataset.validation_subset:
            eval_dataset = dataset.validation_subset
        else:
            eval_dataset = dataset.dataset

        if callbacks and not isinstance(callbacks, list):
            callbacks = list(callbacks) if isinstance(callbacks, tuple) else [callbacks]

        if callbacks and not all(isinstance(callback, tf.keras.callbacks.Callback) for callback in callbacks):
            raise TypeError('Callbacks must be tf.keras.callbacks.Callback instances')

        return self._model.evaluate(eval_dataset, callbacks=callbacks)

    def predict(self, input_samples, return_type='class', callbacks=None):
        """
        Perform feed-forward inference and predict the classes of the input_samples.

        Args:
            input_samples (tensor): Input tensor with one or more samples to perform inference on
            return_type (str): Using 'class' will return the highest scoring class (default), using 'scores' will
                               return the raw output/logits of the last layer of the network, using 'probabilities' will
                               return the output vector after applying a softmax function (so results sum to 1)
            callbacks (list): List of keras.callbacks.Callback instances to apply during predict

        Returns:
            List of classes, probability vectors, or raw score vectors

        Raises:
            ValueError: if the return_type is not one of 'class', 'probabilities', or 'scores'
        """
        return_types = ['class', 'probabilities', 'scores']
        if not isinstance(return_type, str) or return_type not in return_types:
            raise ValueError('Invalid return_type ({}). Expected one of {}.'.format(return_type, return_types))

        if callbacks and not isinstance(callbacks, list):
            callbacks = list(callbacks) if isinstance(callbacks, tuple) else [callbacks]

        if callbacks and not all(isinstance(callback, tf.keras.callbacks.Callback) for callback in callbacks):
            raise TypeError('Callbacks must be tf.keras.callbacks.Callback instances')

        predictions = self._model.predict(input_samples, callbacks=callbacks)
        if return_type == 'class':
            return np.argmax(predictions, axis=-1)
        elif return_type == 'probabilities':
            return tf.nn.softmax(predictions)
        else:
            return predictions
