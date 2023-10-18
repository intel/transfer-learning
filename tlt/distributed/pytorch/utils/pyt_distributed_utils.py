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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T

from tqdm import tqdm
from random import Random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import oneccl_bindings_for_pytorch  # noqa # pylint: disable=unused-import
import intel_extension_for_pytorch as ipex

import horovod.torch as hvd


class HorovodTrainer:
    def __init__(self, cuda=False) -> None:
        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)

        dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (dataloader_kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            dataloader_kwargs['multi_processing_context'] = 'forkserver'

        self.dataloader_kwargs = dataloader_kwargs

        # Init horovod
        hvd.init()

    def prepare_data(self, dataset, use_case, batch_size=128, **kwargs):
        if not kwargs.get('is_preprocessed'):
            if use_case == 'image_classification':
                dataset.transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                pass
            elif use_case == 'text_classification':
                hf_tokenizer = kwargs.get('hf_tokenizer')
                max_seq_length = kwargs.get('max_seq_length')
                text_column_names = kwargs.get('text_column_names')

                def tokenize_func(sample):
                    args = (sample[c] for c in text_column_names)
                    result = hf_tokenizer(*args, padding='max_length', max_length=max_seq_length,
                                          truncation=True)
                    return result
                dataset = dataset.map(tokenize_func)
                dataset.set_format('torch')
        data_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler,
                                **self.dataloader_kwargs)
        return dataloader, data_sampler

    def prepare_model(self, model, use_case, optimizer=None, loss=None, scale_lr=True):
        if optimizer is None:
            if use_case == 'image_classification':
                optimizer = torch.optim.Adam(model.parameters())
            elif use_case == 'text_classification':
                optimizer = torch.optim.AdamW(model.parameters())

        if loss is None:
            loss = torch.nn.CrossEntropyLoss()
        # Horovod: scale learning rate by lr_scaler.
        if scale_lr:
            scaled_lr = optimizer.param_groups[0]['lr'] * hvd.size()
            optimizer.param_groups[0]['lr'] = scaled_lr

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=hvd.Compression.none,
            op=hvd.Average
        )

        self.model = model
        self.optimizer = optimizer
        self.criterion = loss

    def fit(self, dataloader, data_sampler, use_case, epochs=1, log_interval=10):
        if use_case == 'image_classification':
            for epoch in range(1, epochs + 1):
                self.model.train()

                # Horovod: set epoch to sampler for shuffling
                data_sampler.set_epoch(epoch)
                for batch_idx, (data, target) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    if batch_idx % log_interval == 0:
                        # Horovod: use train_sampler to determine the number of examples in
                        # this worker's partition.
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(data_sampler),
                            100. * batch_idx / len(dataloader), loss.item()))
        elif use_case == 'text_classification':
            for epoch in range(1, epochs + 1):
                self.model.train()

                data_sampler.set_epoch(epoch)
                for batch_idx, data in enumerate(dataloader):
                    inputs = {k: v for k, v in data.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                    labels = data['label']
                    outputs = self.model(**inputs)
                    loss = self.criterion(outputs.logits, labels)

                    loss.backward()
                    self.optimizer.step()

                    if batch_idx % log_interval == 0:
                        # Horovod: use train_sampler to determine the number of examples in
                        # this worker's partition.
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(data_sampler),
                            100. * batch_idx / len(dataloader), loss.item()))


""" Dataset partitioning helper classes and methods """


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []

        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, batch_size):
    world_size = dist.get_world_size()
    bsz = int(batch_size / world_size)
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_loader = DataLoader(partition, batch_size=bsz, shuffle=True)

    return train_loader, bsz


""" Distributed Torch helper classes """


class DistributedTrainingArguments:
    def __init__(self, **kwargs) -> None:
        self.__dict__ = dict(kwargs)


class DistributedTorch:

    def __init__(self, use_case: str) -> None:
        self.use_case = use_case

    def launch_distributed_job(
            self,
            training_args: DistributedTrainingArguments,
            master_addr: str,
            master_port: str,
            backend: str = 'ccl'
    ):
        DistributedTorch.setup_ddp(master_addr, master_port, backend)

        self._fit(training_args)

        DistributedTorch.cleanup_ddp()

    def _fit(self, training_args: DistributedTrainingArguments):
        self._model = training_args.model
        self._optimizer = training_args.optimizer
        self._criterion = training_args.criterion

        if not training_args.disable_ipex:
            self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)

        self._ddp_model = DDP(self._model)

        dataset = training_args.dataset
        batch_size = training_args.batch_size
        epochs = training_args.epochs

        dataloader, bsz = partition_dataset(dataset, batch_size)
        epoch_accuracies, epoch_losses = [], []

        # Since we are loading the model from disk, we have to set 'requires_grad'
        # to True for the optimizer to update the model parameters.
        for param in self._ddp_model.parameters():
            param.requires_grad = True

        if self.use_case == 'text_classification':
            for epoch in range(epochs):
                print(f'Epoch {epoch+1}/{epochs}')
                print('-' * 10)

                # Training phase
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data_batch in tqdm(dataloader):
                    inputs = {k: v for k, v in data_batch.items()
                              if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                    labels = data_batch['label']

                    # zero the parameter gradients
                    self._optimizer.zero_grad()

                    # Forward pass
                    outputs = self._ddp_model(**inputs)
                    loss = self._criterion(outputs.logits, labels)

                    # Backward pass
                    loss.backward()
                    self.average_gradients()
                    self._optimizer.step()

                    # Statistics
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    running_loss += torch.as_tensor(loss.item() * data_batch['input_ids'].size(0))
                    running_corrects += torch.sum(predictions == labels)

                dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(running_corrects, op=dist.ReduceOp.SUM)
                epoch_loss = running_loss / len(dataset)
                epoch_acc = running_corrects / len(dataset)
                epoch_accuracies.append(epoch_acc)
                epoch_losses.append(epoch_loss)

                print("Loss: {}".format(epoch_loss))
                print("Acc: {}".format(epoch_acc))

            training_loss = epoch_losses[-1]
            training_acc = epoch_accuracies[-1]

            if dist.get_rank() == 0:
                print("Training loss:", training_loss)
                print("Training accuracy:", training_acc)
        elif self.use_case == 'image_classification':
            for epoch in range(epochs):
                print('Epoch {}/{}'.format(epoch + 1, epochs))

                running_loss = 0
                running_corrects = 0
                for data, target in tqdm(dataloader):
                    self._optimizer.zero_grad()
                    out = self._ddp_model(data)
                    loss = self._criterion(out, target)
                    loss.backward()
                    self.average_gradients()
                    self._optimizer.step()

                    # Statistics
                    preds = torch.argmax(out, dim=1)
                    running_loss += torch.as_tensor(loss.item() * data.size(0))
                    running_corrects += torch.sum(preds == target)

                # Collect all the running_loss and running_corrects tensors
                dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(running_corrects, op=dist.ReduceOp.SUM)
                epoch_loss = running_loss / len(dataset)
                epoch_acc = running_corrects / len(dataset)
                epoch_accuracies.append(epoch_acc)
                epoch_losses.append(epoch_loss)

                print("Loss: {}".format(epoch_loss))
                print("Acc: {}".format(epoch_acc))

            training_loss = epoch_losses[-1]
            training_acc = epoch_accuracies[-1]

            if dist.get_rank() == 0:
                print("Training loss:", training_loss)
                print("Training accuracy:", training_acc)
        else:
            raise ValueError("PyTorch Distributed Training for {} is not implemeted yet"
                             .format(self.use_case))

    def average_gradients(self):
        size = float(dist.get_world_size())
        for param in self._ddp_model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

    @classmethod
    def setup_ddp(cls, master_addr: str, master_port: str, backend: str = 'ccl'):
        if dist.is_initialized():
            print("Process Group already initialized")
        else:
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            os.environ['RANK'] = os.environ.get('PMI_RANK', '0')
            os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')

            if backend == 'ccl':
                dist.init_process_group(
                    backend=backend,
                    init_method='env://'
                )

    @classmethod
    def cleanup_ddp(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    @classmethod
    def load_saved_objects(cls, saved_objects_dir):
        """
        Helper function to load saved dataset and model objects

        Args:
            use_case (str): Use case of the saved datasets and models.

        Returns:
            dict with loaded dataset and model objects
        """
        saved_objects_file = 'torch_saved_objects.obj'

        return torch.load(os.path.join(saved_objects_dir, saved_objects_file))
