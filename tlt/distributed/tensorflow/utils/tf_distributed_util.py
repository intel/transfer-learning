#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
import dill  # nosec: B403
import time

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

from transformers import TFBertModel, BertConfig

from pydoc import locate
from tlt.utils.dataset_utils import prepare_huggingface_input_data
from tlt.models.model_factory import get_model_info


# This needs to be imported last to avoid "free(): invalid pointer" error
import horovod.tensorflow.keras as hvd


class DistributedTrainingArguments:

    def __init__(self, use_case, train_data, model, optimizer, loss, test_data=None, val_data=None,
                 epochs=1, global_batch_size=128, shuffle=True, scaling='weak', **kwargs) -> None:

        self.use_case = use_case

        # Model related arguments
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        # Data related arguments
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.num_classes = kwargs.get('num_classes', None)

        # Training related arguments
        self.epochs = epochs
        self.scaling = scaling
        self.global_batch_size = global_batch_size
        self.batch_denom = kwargs.get('batch_denom', 1)
        self.shuffle = shuffle

        # Use case related arguments
        # For image classification
        self.image_size = kwargs.get('image_size', None)
        self.image_shape = kwargs.get('image_shape', None)
        # For text classification
        self.max_seq_length = kwargs.get('max_seq_length', None)
        self.padding = kwargs.get('padding', None)
        self.truncation = kwargs.get('truncation', None)
        self.hf_bert_tokenizer = kwargs.get('hf_bert_tokenizer', None)


class DistributedTF:

    def __init__(self) -> None:
        hvd.init()

    def prepare_dataset(self, dataset, use_case, global_batch_size, scaling, **kwargs):
        if scaling.lower() == 'weak':
            batch_size = global_batch_size
        elif scaling.lower() == 'strong':
            batch_size = global_batch_size // hvd.size()

        if use_case == 'image_classification':
            dataset = dataset.shard(num_shards=hvd.size(), index=hvd.rank())
            dataset = dataset.cache()
            if 'map_func' in kwargs:
                dataset = dataset.map(map_func=kwargs.get('map_func'), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        elif use_case == 'text_classification':
            max_seq_length = kwargs.get('max_seq_length')
            bert_tokenizer = kwargs.get('hf_bert_tokenizer')

            input_ids_shape = (len(dataset), max_seq_length)
            attention_mask_shape = (len(dataset), max_seq_length)

            input_ids = tf.zeros(input_ids_shape, dtype=tf.int32)
            attention_mask = tf.zeros(attention_mask_shape, dtype=tf.int32)
            labels = tf.ones(len(dataset), dtype=tf.int32)

            # Preprocessing text could be done only on one worker and the tensors are synced later among workers
            if hvd.rank() == 0:
                dataset = [(sentence.numpy().decode(), label.numpy()) for sentence, label in dataset]

                sentences = [x[0] for x in dataset]
                labels = [x[1] for x in dataset]

                print('Tokenizing the dataset...')
                tokenized_dataset = bert_tokenizer(sentences, padding='max_length', max_length=max_seq_length,
                                                   truncation=True, return_tensors='tf')

                input_ids = tokenized_dataset['input_ids']
                attention_mask = tokenized_dataset['attention_mask']
                labels = tf.convert_to_tensor(labels, dtype=tf.int32)

            input_ids = hvd.allreduce(input_ids, average=False, name='barrier1')
            attention_mask = hvd.allreduce(attention_mask, average=False, name='barrier2')
            labels = hvd.allreduce(labels, average=False, name='labels')

            dataset = ({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }, labels)

            dataset = tf.data.Dataset.from_tensor_slices(dataset)
            dataset = dataset.shard(hvd.size(), hvd.rank())
            dataset = dataset.cache()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def prepare_model(self, model_name, use_case, input_shape, num_classes, **kwargs):
        # Try to get model url from TLT supported models
        model_info = get_model_info(model_name, 'tensorflow', use_case)
        if model_info != {}:
            fw_enum = list(model_info.keys())[0]
            model_name = model_info[fw_enum]['tensorflow']['feature_vector']
        if use_case == 'image_classification':
            model = tf.keras.models.Sequential([
                hub.KerasLayer(model_name, input_shape=input_shape),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        elif use_case == 'text_classification':
            bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
            bert_model = TFBertModel.from_pretrained(model_name, config=bert_config, from_pt=True)

            dense_layer_dims = 1 if num_classes == 2 else num_classes

            input_ids = tf.keras.layers.Input(input_shape, dtype=tf.int32, name='input_ids')
            attention_mask = tf.keras.layers.Input(input_shape, dtype=tf.int32, name='attention_mask')
            bert_output = bert_model.bert(input_ids, attention_mask=attention_mask)[1]
            classifier = tf.keras.layers.Dense(dense_layer_dims, activation=None, name='classifier')(bert_output)

            model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=classifier)

        return model

    def launch_distributed_job(self, training_args: DistributedTrainingArguments):
        model = training_args.model
        optimizer = training_args.optimizer
        loss = training_args.loss

        # This is required if using intel-tensorflow==2.12.0
        optimizer = self._get_legacy_optimizer(optimizer)

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        if training_args.scaling.lower() == 'weak':
            multiplier = np.sqrt(training_args.global_batch_size // training_args.batch_denom)
            optimizer.lr = optimizer.lr * multiplier
            batch_size = training_args.global_batch_size
        elif training_args.scaling.lower() == 'strong':
            optimizer.lr = optimizer.lr * hvd.size()
            batch_size = training_args.global_batch_size // hvd.size()

        if training_args.use_case == 'image_classification':
            hvd_optimizer = hvd.DistributedOptimizer(
                optimizer, backward_passes_per_step=5, average_aggregated_gradients=True)
        elif training_args.use_case == 'text_classification':
            hvd_optimizer = hvd.DistributedOptimizer(
                optimizer, backward_passes_per_step=1, average_aggregated_gradients=True)

        model.compile(
            loss=loss,
            optimizer=hvd_optimizer,
            metrics=['acc'],
            experimental_run_tf_function=False
        )

        warmup = 3
        if hvd.size() == 1:
            warmup = 1

        callbacks = []
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        # Horovod: average metrics among workers at the end of every epoch.
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final accuracy.
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
            initial_lr=optimizer.lr, warmup_epochs=warmup, verbose=1))

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
                os.environ['HOME'], 'model_checkpoints'), save_weights_only=False, monitor='val_acc',
                mode='max', save_best_only=True)
            callbacks.append(model_checkpoint_callback)

        # Horovod: write logs on worker 0.
        verbose = 1 if hvd.rank() == 0 else 0

        x_input_data = training_args.train_data
        y_target_data = None
        val_data = training_args.val_data

        # Prepare dataset for Hugging Face text classification
        if training_args.hf_bert_tokenizer:
            bert_tokenizer_name = training_args.hf_bert_tokenizer
            max_seq_length = training_args.max_seq_length
            tokenized_data, labels = prepare_huggingface_input_data(x_input_data, bert_tokenizer_name, max_seq_length)
            x_input_data = [tokenized_data['input_ids'], tokenized_data['attention_mask']]
            y_target_data = tf.convert_to_tensor(labels)

            if training_args.val_data:
                tokenized_val_data, val_labels = prepare_huggingface_input_data(training_args.val_data,
                                                                                bert_tokenizer_name, max_seq_length)
                val_data = ([tokenized_val_data['input_ids'], tokenized_val_data['attention_mask']],
                            tf.convert_to_tensor(val_labels))

        start = time.time()
        steps_per_epoch_per_worker = len(training_args.train_data) // batch_size
        steps_per_epoch_per_worker = steps_per_epoch_per_worker // hvd.size()
        if hvd.size() > 2:
            steps_per_epoch_per_worker += 1
        self.history = model.fit(
            x=x_input_data,
            y=y_target_data,
            validation_data=val_data,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch_per_worker,
            epochs=training_args.epochs,
            initial_epoch=0,
            verbose=verbose
        )
        end = time.time()
        if hvd.rank() == 0:
            print("Total elapsed time in seconds = ", end - start)
            print("Total elapsed time in minutes = ", ((end - start) / 60))
            print("Total epochs = ", len(self.history.history['loss']))
            print("Time per epoch in seconds = ", ((end - start) / len(self.history.history['loss'])))
            print("Maximum validation accuracy = ", np.max(self.history.history['val_acc']))

    def _get_legacy_optimizer(self, optimizer):
        optimizer_config = optimizer.get_config()
        optimizer_name = optimizer_config['name']

        legacy_optimizer_class = locate('tensorflow.keras.optimizers.legacy.{}'.format(optimizer_name))

        if legacy_optimizer_class is None:
            # No matching legacy optimizer is found.
            return optimizer

        legacy_optimizer_config = legacy_optimizer_class().get_config()
        legacy_optimizer = legacy_optimizer_class.from_config(
            {k: v for k, v in optimizer_config.items() if k in legacy_optimizer_config}
        )

        return legacy_optimizer

    def load_saved_objects(self, saved_objects_dir):
        # Load the saved_model.pb
        model = tf.keras.models.load_model(filepath=saved_objects_dir, compile=False)

        # Load the optimizer and restore its state
        checkpoint = tf.train.Checkpoint(optimizer=tf.optimizers.Adam())
        checkpoint.restore(os.path.join(saved_objects_dir, 'saved_optimizer-1'))

        # Load the saved loss class name and instatiate the loss
        with open(os.path.join(saved_objects_dir, 'saved_loss'), 'rb') as f:
            loss_class, loss_args = dill.load(f)  # nosec: B301

        # load the dataset(s)
        train_data = tf.data.Dataset.load(os.path.join(saved_objects_dir, 'train_data'))
        try:
            val_data = tf.data.Dataset.load(os.path.join(saved_objects_dir, 'val_data'))
        except FileNotFoundError:
            val_data = None

        if loss_class is None:
            dataset = train_data.unbatch()
            dataset = list(dataset.as_numpy_iterator())
            labels = list()
            for _, label in dataset:
                labels.append(label)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) if len(set(labels)) == 2 else \
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = loss_class(**loss_args)

        return (model, checkpoint.optimizer, loss, train_data, val_data)
