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
import argparse
import tempfile

from transformers import AutoTokenizer

from filelock import FileLock

from downloader.datasets import DataDownloader
from downloader.models import ModelDownloader

from tlt.distributed.pytorch.utils.pyt_distributed_utils import (
    DistributedTorch,
    DistributedTrainingArguments,
    HorovodTrainer
)


if __name__ == "__main__":

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

    description = 'Distributed training with PyTorch.'

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--master_addr', type=str, required=False, help="Master node to run this script")
    parser.add_argument('--master_port', type=str, required=False, default='29500', help='Master port')
    parser.add_argument('--backend', type=str, required=False, default='ccl', help='Type of backend to use '
                        '(default: ccl)')
    parser.add_argument('--use-case', '--use_case', type=str, required=True,
                        help='Use case (image_classification|text_classification)')
    parser.add_argument('--epochs', type=int, required=False, default=1, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, required=False, default=128,
                        help='Global batch size to distribute data (default: 128)')
    parser.add_argument('--disable_ipex', action='store_true', required=False, help="Disables IPEX optimization to "
                        "the model. No effect when given --use-horovod as horovod with IPEX isn't supported.")
    parser.add_argument('--tlt_saved_objects_dir', type=directory_path, required=False, help='Path to TLT saved '
                        'distributed objects. The path must be accessible to all the nodes. For example: mounted '
                        'NFS drive. This arg is helpful when using TLT API/CLI. '
                        'See DistributedTorch.load_saved_objects() for more information.')
    parser.add_argument('--use-horovod', '--use_horovod', action='store_true', help='Use horovod for distributed '
                        'training.')
    parser.add_argument('--cuda', action='store_true', help='Use cuda device for distributed training')
    parser.add_argument('--dataset-dir', '--dataset_dir', type=directory_path, default=default_data_dir,
                        help="Path to dataset directory to save/load tfds dataset. This arg is helpful if you "
                        "plan to use this as a stand-alone script. Custom dataset is not supported yet!")
    parser.add_argument('--output-dir', '--output_dir', type=directory_path, default=default_output_dir,
                        help="Path to save the trained model and store logs. This arg is helpful if you "
                        "plan to use this as a stand-alone script")
    parser.add_argument('--dataset-name', '--dataset_name', type=str, default=None,
                        help="Dataset name to load from torchvision/Huggingface. This arg is helpful if you "
                        "plan to use this as a stand-alone script. Custom dataset is not supported yet!")
    parser.add_argument('--model-name', '--model_name', type=str, default=None,
                        help="Torchvision image classification model name "
                        "(or) Huggingface hub name for text classification models. This arg is helpful if you "
                        "plan to use this as a stand-alone script.")
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length that the model will be used with for text classification')

    args = parser.parse_args()

    train_data = None
    model = None
    optimizer, loss = None, None
    data_kwargs = {}

    if args.tlt_saved_objects_dir is not None:
        # Load the saved dataset and model objects
        loaded_objects = DistributedTorch.load_saved_objects(args.tlt_saved_objects_dir)

        train_data = loaded_objects.get('train_data')
        model = loaded_objects['model']
        loss = loaded_objects['loss']
        optimizer = loaded_objects['optimizer']
        data_kwargs['is_preprocessed'] = True
    else:
        if args.dataset_name is None:
            raise argparse.ArgumentError(args.dataset_name, "Please provide a dataset name to load from torchvision "
                                         "(or) datasets using --dataset-name")
        if args.model_name is None:
            raise argparse.ArgumentError(args.model_name, "Please provide torchvision model name (or) "
                                         "Huggingface hub name using --model-name")

        catalog = 'torchvision' if args.use_case == 'image_classification' else 'hugging_face'
        with FileLock(os.path.expanduser('~/.horovod_lock')):
            train_data = DataDownloader(args.dataset_name, args.dataset_dir, catalog).download(split='train')
            model = ModelDownloader(args.model_name, catalog, args.output_dir).download()
            if args.use_case == 'text_classification':
                data_kwargs['hf_tokenizer'] = AutoTokenizer.from_pretrained(args.model_name)
                data_kwargs['max_seq_length'] = args.max_seq_length
                data_kwargs['text_column_names'] = [c for c in train_data.column_names if c != 'label']
        data_kwargs['is_preprocessed'] = False

    if args.use_horovod:
        hvd_trainer = HorovodTrainer(args.cuda)

        train_loader, train_sampler = hvd_trainer.prepare_data(train_data, args.use_case, args.batch_size,
                                                               **data_kwargs)
        hvd_trainer.prepare_model(model, args.use_case, optimizer, loss)

        hvd_trainer.fit(train_loader, train_sampler, args.use_case, args.epochs)
    else:
        # Launch distributed job
        training_args = DistributedTrainingArguments(
            dataset=train_data,
            model=model,
            criterion=loss,
            optimizer=optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            disable_ipex=args.disable_ipex
        )

        dt = DistributedTorch(use_case=args.use_case)
        dt.launch_distributed_job(training_args, args.master_addr, args.master_port, args.backend)
