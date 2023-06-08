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
import tempfile
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

from tlt.distributed.tensorflow.utils.tf_distributed_util import (
    DistributedTF,
    DistributedTrainingArguments
)


if __name__ == '__main__':

    default_data_dir = os.path.join(tempfile.gettempdir(), 'data')
    default_output_dir = os.path.join(tempfile.gettempdir(), 'output')

    for d in [default_data_dir, default_output_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    def directory_path(path):
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError("'{}' is not a valid directory path.".format(path))

    print("******Distributed Training*****")

    description = 'Distributed training with TensorFlow.'

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--use-case', '--use_case', type=str, required=True, choices=['image_classification',
                        'text_classification'], help='Use case (image_classification|text_classification)')
    parser.add_argument('--epochs', type=int, required=False, default=1, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, required=False, default=128,
                        help='Global batch size to distribute data (default: 128)')
    parser.add_argument("--batch_denom", type=int, required=False, default=1,
                        help="Batch denominator to be used to divide global batch size (default: 1)")
    parser.add_argument('--shuffle', action='store_true', required=False, help="Shuffle dataset while training")
    parser.add_argument('--scaling', type=str, required=False, default='weak', choices=['weak', 'strong'],
                        help='Weak or Strong scaling. For weak scaling, lr is scaled by a factor of '
                        'sqrt(batch_size/batch_denom) and uses global batch size for all the processes. For '
                        'strong scaling, lr is scaled by world size and divides global batch size by world size '
                        '(default: weak)')
    parser.add_argument('--tlt_saved_objects_dir', type=directory_path, required=False, help='Path to TLT saved '
                        'distributed objects. The path must be accessible to all the nodes. For example: mounted '
                        'NFS drive. This arg is helpful when using TLT API/CLI. See DistributedTF.load_saved_objects()'
                        ' for more information.')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length that the model will be used with')
    parser.add_argument('--dataset-dir', '--dataset_dir', type=directory_path, default=default_data_dir,
                        help="Path to dataset directory to save/load tfds dataset. This arg is helpful if you "
                        "plan to use this as a stand-alone script. Custom dataset is not supported yet!")
    parser.add_argument('--output-dir', '--output_dir', type=directory_path, default=default_output_dir,
                        help="Path to save the trained model and store logs. This arg is helpful if you "
                        "plan to use this as a stand-alone script")
    parser.add_argument('--dataset-name', '--dataset_name', type=str, default=None,
                        help="Dataset name to load from tfds. This arg is helpful if you "
                        "plan to use this as a stand-alone script. Custom dataset is not supported yet!")
    parser.add_argument('--model-name', '--model_name', type=str, default=None,
                        help="TensorFlow image classification model url/ feature vector url from TensorFlow Hub "
                        "(or) Huggingface hub name for text classification models. This arg is helpful if you "
                        "plan to use this as a stand-alone script.")
    parser.add_argument('--image-size', '--image_size', type=int, default=None,
                        help="Input image size to the given model, for which input shape is determined as "
                        "(image_size, image_size, 3). This arg is helpful if you "
                        "plan to use this as a stand-alone script.")

    args = parser.parse_args()

    dtf = DistributedTF()

    model = None
    optimizer, loss = None, None
    train_data, train_labels = None, None
    val_data, val_labels = None, None

    if args.tlt_saved_objects_dir is not None:
        model, optimizer, loss, train_data, val_data = dtf.load_saved_objects(args.tlt_saved_objects_dir)
    else:
        if args.dataset_name is None:
            raise argparse.ArgumentError(args.dataset_name, "Please provide a dataset name to load from tfds "
                                         "using --dataset-name")
        if args.model_name is None:
            raise argparse.ArgumentError(args.model_name, "Please provide TensorFlow Hub's model url/feature "
                                         "vector url (or) Huggingface hub name using --model-name")

        train_data, data_info = tfds.load(args.dataset_name, data_dir=args.dataset_dir, split='train',
                                          as_supervised=True, with_info=True)
        val_data = tfds.load(args.dataset_name, data_dir=args.dataset_dir, split='test', as_supervised=True)
        num_classes = data_info.features['label'].num_classes

        if args.use_case == 'image_classification':
            if args.image_size is not None:
                input_shape = (args.image_size, args.image_size, 3)
            else:
                try:
                    input_shape = data_info.features['image'].shape
                except (KeyError, AttributeError):
                    raise argparse.ArgumentError(args.image_size, "Unable to determine input_shape, please "
                                                 "provide --image-size/--image_size")

            train_data = dtf.prepare_dataset(train_data, args.use_case, args.batch_size, args.scaling)
            val_data = dtf.prepare_dataset(val_data, args.use_case, args.batch_size, args.scaling)

            model = dtf.prepare_model(args.model_name, args.use_case, input_shape, num_classes)

        elif args.use_case == 'text_classification':
            input_shape = (args.max_seq_length,)
            from transformers import BertTokenizer
            hf_bert_tokenizer = BertTokenizer.from_pretrained(args.model_name)

            train_data = dtf.prepare_dataset(train_data, args.use_case, args.batch_size, args.scaling,
                                             max_seq_length=args.max_seq_length, hf_bert_tokenizer=hf_bert_tokenizer)
            val_data = dtf.prepare_dataset(val_data, args.use_case, args.batch_size, args.scaling,
                                           max_seq_length=args.max_seq_length, hf_bert_tokenizer=hf_bert_tokenizer)
            model = dtf.prepare_model(args.model_name, args.use_case, input_shape, num_classes)

        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) if num_classes == 2 else \
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    training_args = DistributedTrainingArguments(
        use_case=args.use_case,
        model=model,
        optimizer=optimizer,
        loss=loss,
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        scaling=args.scaling,
        batch_size=args.batch_size,
        batch_denom=args.batch_denom,
        shuffle=args.shuffle,
        max_seq_length=args.max_seq_length,
        hf_bert_tokenizer=args.model_name if args.tlt_saved_objects_dir is not None and
        args.use_case == 'text_classification' else None
    )

    dtf.launch_distributed_job(training_args)
