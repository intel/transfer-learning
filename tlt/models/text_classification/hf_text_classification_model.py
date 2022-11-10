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
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler
)

import intel_extension_for_pytorch as ipex

from datasets import load_metric
from datasets.arrow_dataset import Dataset

from torch.utils.data import DataLoader

from tlt import TLT_BASE_DIR
from tlt.utils.file_utils import read_json_file
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.models.hf_model import HFModel
from tlt.models.text_classification.text_classification_model import TextClassificationModel
from tlt.datasets.text_classification.hf_text_classification_dataset import HFTextClassificationDataset


MODEL_CONFIG_DIR = os.path.join(TLT_BASE_DIR, "models/configs")


class HFTextClassificationModel(TextClassificationModel, HFModel):
    def __init__(self, model_name: str, model=None):

        # extra properties that will become configurable in the future
        self._model_name = model_name
        self._dropout_layer_rate = 0.1
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._device = 'cpu'
        self._optimizer_class = torch.optim.AdamW  # Just the class, it needs to be initialized with the model object
        self._optimizer = None
        self._loss = torch.nn.CrossEntropyLoss()
        self._lr_scheduler = None
        self._generate_checkpoints = True
        self._tokenizer = None

        TextClassificationModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_CLASSIFICATION,
                                         self._dropout_layer_rate)
        HFModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_CLASSIFICATION)

        # model definition
        config_dict = read_json_file(os.path.join(MODEL_CONFIG_DIR, "hf_text_classification_models.json"))

        self._model = AutoModelForSequenceClassification.from_pretrained(config_dict[model_name]["hub_name"])
        self._num_classes = None

        if not self._model:
            if model is None:
                self._model = None
            elif isinstance(model, str):
                self.load_from_directory(model)
                layers = list(self._model.children())
                self._num_classes = layers[-1].out_features
            elif isinstance(model, torch.nn.Module):
                self._model = model
                layers = list(self._model.children())
                self._num_classes = layers[-1].out_features
            else:
                raise TypeError("The model input must be a torch.nn.Module, string or",
                                "None but found a {}". format(type(model)))

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    def _fit(self, dataset, output_dir, epochs, do_eval, ipex_optimize):

        train_data_loader = None
        validation_data_loader = None

        if isinstance(dataset, HFTextClassificationDataset):
            if not dataset._preprocessed:
                raise ValueError("dataset is not preprocessed yet")
            self._tokenizer = dataset._tokenizer

            # Get the data loader objects
            train_data_loader = dataset.train_loader
            validation_data_loader = dataset.validation_loader
            train_data_length = len(train_data_loader)
        elif isinstance(dataset, Dataset):
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

            # Create new data loader objects
            train_data_loader = DataLoader(dataset, batch_size=16)
            validation_data_loader = DataLoader(dataset, batch_size=16)
            train_data_length = len(train_data_loader)
        else:
            raise ValueError("Invalid dataset type: {}".format(type(dataset)))

        num_training_steps = epochs * train_data_length
        lr_scheduler = get_scheduler(
            name="linear", optimizer=self._optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        if ipex_optimize:
            self._model = ipex.optimize(self._model)
        # Training loop
        self._model.to(self._device)
        self._model.train()
        self._history = {}
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)

            # Training phase
            running_loss = 0.0
            accuracy_metric = load_metric("accuracy")

            # Iterate over data.
            for data_batch in tqdm(train_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                data_batch = {k: v.to(self._device) for k, v in data_batch.items()}

                # Forward pass
                outputs = self._model(**data_batch)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                self._optimizer.step()
                lr_scheduler.step()
                self._optimizer.zero_grad()

                # Statistics
                labels = data_batch['labels']
                predictions = torch.argmax(outputs.logits, dim=-1)

                running_loss += loss.item() * len(labels)
                accuracy_metric.add_batch(predictions=predictions, references=labels)

            # At the epoch end
            train_epoch_loss = running_loss / train_data_length
            train_epoch_acc = accuracy_metric.compute()['accuracy']

            self._update_history('Loss', train_epoch_loss)
            self._update_history('Acc', train_epoch_acc)

            loss_acc_output = f'Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_acc:.4f}'

            if do_eval and validation_data_loader is not None:
                eval_epoch_loss, eval_epoch_acc = self.evaluate(validation_data_loader)

                self._update_history('Val Loss', eval_epoch_loss)
                self._update_history('Val Acc', eval_epoch_acc)

                loss_acc_output += f' - Val Loss: {eval_epoch_loss:.4f} - Val Acc: {eval_epoch_acc:.4f}'

    def train(
        self,
        dataset,
        output_dir: str,
        epochs: int = 1,
        learning_rate: float = 1e-5,
        initial_checkpoints=None,
        do_eval: bool = True,
        device: str = "cpu",
        ipex_optimize: bool = True
    ):

        self._device = device
        self.train_data_loader = None
        self.validation_data_loader = None

        # Initialize the optimizer class and create a learning rate scheduler
        self._optimizer = self._optimizer_class(self._model.parameters(), lr=learning_rate)

        # Call the _fit method to train the model with native PyTorch API
        self._fit(dataset, output_dir, epochs, do_eval, ipex_optimize)

        return self._history

    def evaluate(self, dataset_or_dataloader):
        if isinstance(dataset_or_dataloader, Dataset):
            dataloader = DataLoader(dataset_or_dataloader, batch_size=16)
        elif isinstance(dataset_or_dataloader, DataLoader):
            dataloader = dataset_or_dataloader
        elif isinstance(dataset_or_dataloader, HFTextClassificationDataset):
            dataloader = dataset_or_dataloader.validation_loader

        self._model.eval()
        running_loss = 0.0
        accuracy_metric = load_metric("accuracy")
        for data_batch in tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
            data_batch = {k: v.to(self._device) for k, v in data_batch.items()}

            outputs = self._model(**data_batch)
            loss = outputs.loss

            # Statistics
            labels = data_batch['labels']
            predictions = torch.argmax(outputs.logits, dim=-1)

            running_loss += loss.item() * len(labels)
            accuracy_metric.add_batch(predictions=predictions, references=labels)

        validation_loss = running_loss / len(dataloader)
        validation_accuracy = accuracy_metric.compute()['accuracy']

        return (validation_loss, validation_accuracy)

    def predict(self, input_samples):
        encoded_input = None
        if isinstance(input_samples, str):
            raw_input_text = list(input_samples)
            encoded_input = self._tokenizer(raw_input_text, padding=True, return_tensors='pt')
        elif isinstance(input_samples, list):
            encoded_input = self._tokenizer(input_samples, padding=True, return_tensors='pt')
        elif isinstance(input_samples, HFTextClassificationDataset):
            if input_samples._preprocessed:
                encoded_input = {
                    'input_ids': input_samples['input_ids'],
                    'attention_mask': input_samples['attention_mask']
                }
        elif isinstance(input_samples, DataLoader):
            raise ValueError("Prediction using Dataloader hasn't been implmented yet. \
                                Use raw text or Dataset as input!")

        output = self._model(**encoded_input)
        _, predictions = torch.max(output.logits, dim=1)
        return predictions

    def export(self, output_dir: str):
        dir_name_to_save = self._model_name
        path_to_dir_name = os.path.join(output_dir, dir_name_to_save)

        if not os.path.exists(path_to_dir_name):
            os.makedirs(path_to_dir_name)

        output_model_file_path = os.path.join(path_to_dir_name, 'pytorch_model.bin')

        # If we have a distributed model, save only the encapsulated model
        # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = self._model.module if hasattr(self._model, 'module') else self._model

        torch.save(model_to_save.state_dict(), output_model_file_path)

        print('Model saved at {}'.format(path_to_dir_name))

    def load_from_directory(self, model_dir: str):
        """
        Loads a saved pytorch model from the given model_dir directory

        Args:
            model_dir(str): Path to the saved model directory
        """

        saved_model_file_path = os.path.join(model_dir, 'pytorch_model.bin')

        state_dict = torch.load(saved_model_file_path)
        self._model.load_state_dict(state_dict)
