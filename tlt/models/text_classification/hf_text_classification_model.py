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

import inspect
import os
import torch
import numpy as np
import intel_extension_for_pytorch as ipex
from tqdm import tqdm
from torch.utils.data import DataLoader

# Hugging Face imports
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    Trainer,
    get_scheduler
)

from datasets.arrow_dataset import Dataset

from tlt import TLT_BASE_DIR
from tlt.utils.file_utils import read_json_file, validate_model_name
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.models.hf_model import HFModel
from tlt.models.text_classification.text_classification_model import TextClassificationModel
from tlt.datasets.text_classification.hf_text_classification_dataset import HFTextClassificationDataset
from tlt.datasets.text_classification.hf_custom_text_classification_dataset import HFCustomTextClassificationDataset


MODEL_CONFIG_DIR = os.path.join(TLT_BASE_DIR, "models/configs")


class HFTextClassificationModel(TextClassificationModel, HFModel):
    """
    Class to represent a Hugging Face pretrained model that can be used for multi-class text classification
    fine tuning.
    """
    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):

        # extra properties that will become configurable in the future
        self._model_name = model_name
        self._dropout_layer_rate = 0.1
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._device = 'cpu'
        self._lr_scheduler = None
        self._generate_checkpoints = True
        self._tokenizer = None

        TextClassificationModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_CLASSIFICATION,
                                         self._dropout_layer_rate)
        HFModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_CLASSIFICATION)

        # set up the configurable optimizer and loss functions
        self._check_optimizer_loss(optimizer, loss)
        self._optimizer_class = optimizer if optimizer else torch.optim.AdamW
        self._opt_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._optimizer_class).args}
        self._optimizer = None  # This gets initialized later
        self._loss_class = loss if loss else torch.nn.CrossEntropyLoss
        self._loss_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._loss_class).args}
        self._loss = self._loss_class(**self._loss_args)

        # model definition
        config_dict = read_json_file(os.path.join(MODEL_CONFIG_DIR, "hf_text_classification_models.json"))
        self.hub_name = config_dict[self._model_name]["hub_name"]
        self._model = None
        self._num_classes = None
        self._trainer = None
        self._history = None

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    def _fit(self, dataset, epochs, do_eval, ipex_optimize):
        train_data_loader = None
        validation_data_loader = None

        if isinstance(dataset, HFTextClassificationDataset) or \
                isinstance(dataset, HFCustomTextClassificationDataset):
            if not dataset._preprocessed:
                raise ValueError("dataset is not preprocessed yet")
            self._tokenizer = dataset._tokenizer

            # Get the data loader objects
            train_data_loader = dataset.train_loader
            validation_data_loader = dataset.validation_loader
            train_data_length = len(dataset.train_subset)
        elif isinstance(dataset, Dataset):
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

            # Create new data loader objects
            train_data_loader = DataLoader(dataset, batch_size=16)
            validation_data_loader = DataLoader(dataset, batch_size=16)
            train_data_length = len(dataset)
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
            running_corrects = 0

            # Iterate over data.
            for data_batch in tqdm(train_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = {k: v.to(self._device) for k, v in data_batch.items() if k != 'labels'}
                labels = data_batch['labels']

                # zero the parameter gradients
                self._optimizer.zero_grad()

                # Forward pass
                outputs = self._model(**inputs)
                loss = self._loss(outputs.logits, labels)

                # Backward pass
                loss.backward()
                self._optimizer.step()
                lr_scheduler.step()

                # Statistics
                predictions = torch.argmax(outputs.logits, dim=-1)

                running_loss += loss.item()
                running_corrects += torch.sum(predictions == labels).item()

            # At the epoch end
            train_epoch_loss = running_loss / train_data_length
            train_epoch_acc = running_corrects / train_data_length

            self._update_history('Loss', train_epoch_loss)
            self._update_history('Acc', train_epoch_acc)

            loss_acc_output = f'Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_acc:.4f}'

            if do_eval and validation_data_loader is not None:
                eval_epoch_loss, eval_epoch_acc = self.evaluate(validation_data_loader)

                self._update_history('Val Loss', eval_epoch_loss)
                self._update_history('Val Acc', eval_epoch_acc)

                loss_acc_output += f' - Val Loss: {eval_epoch_loss:.4f} - Val Acc: {eval_epoch_acc:.4f}'

                # Put the model back to train mode
                self._model.train()

            print(loss_acc_output)

    def train(
        self,
        dataset,
        output_dir: str,
        epochs: int = 1,
        learning_rate: float = 1e-5,
        do_eval: bool = True,
        device: str = "cpu",
        ipex_optimize: bool = True,
        use_trainer: bool = False
    ):

        """
        Trains the model using the specified text classification dataset.

        Args:
            dataset (TextClassificationDataset/datasets.arrow_dataset.Dataset): The dataset to use for training.
                If a train subset has been defined, that subset will be used to fit the model. Otherwise, the
                entire non-partitioned dataset will be used.
            output_dir (str): A writeable output directory to write checkpoint files during training
            epochs (int): The number of training epochs [default: 1]
            do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                at the end of each epoch.
            device (str): Device to train the model [default: "cpu"]
            ipex_optimize (bool): Optimize the model using Intel® Extension for PyTorch
            use_trainer (bool): If use_trainer is True, then the model training is done using the Hugging Face Trainer
                and if use_trainer is False, the model training is done using native PyTorch training loop

        Returns:
            Dictionary containing the model training history

        Raises:
            TypeError if the dataset specified is not a TextClassificationDataset/datasets.arrow_dataset.Dataset
            ValueError if the given dataset has not been preprocessed yet

        """

        if not self._model:
            self._num_classes = len(dataset.class_names)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.hub_name,
                                                                             num_labels=self._num_classes,
                                                                             force_download=False)
        self._device = device
        self.train_data_loader = None
        self.validation_data_loader = None

        # Initialize the optimizer class and create a learning rate scheduler
        self._optimizer = self._optimizer_class(self._model.parameters(), lr=learning_rate, **self._opt_args)

        if use_trainer:
            training_args = TrainingArguments(
                output_dir=output_dir,
                do_eval=do_eval,
                do_train=True,
                no_cuda=True,
                overwrite_output_dir=True,
                per_device_train_batch_size=dataset.info['preprocessing_info']['batch_size'],
                evaluation_strategy="epoch",
                num_train_epochs=epochs,
                max_steps=75,
            )

            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = np.argmax(preds, axis=1)
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

            # Initialize our Trainer
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=dataset.train_subset,
                eval_dataset=dataset.validation_subset,
                compute_metrics=compute_metrics,
                tokenizer=self._tokenizer
            )

            self._trainer.train()
            if do_eval:
                self._history = self._trainer.evaluate()
                print("Val Acc: {:.5f}".format(self._history.get("eval_accuracy")))
        else:
            self._trainer = None
            # Call the _fit method to train the model with native PyTorch API
            self._fit(dataset, epochs, do_eval, ipex_optimize)

        return self._history

    def evaluate(self, dataset_or_dataloader=None):
        """
           Evaulates the model on the given dataset (or) dataloader. If Hugging Face Trainer object was used to
           train the model, it evaluates on the 'eval_dataset' given in the Trainer arguments

           Args:
               dataset_or_dataloader (datasets.arrow_dataset.Dataset/DataLoader/TextClassificationDataset): The
                    dataset/dataloader to use for evaluation.

           Returns:
               Tuple with loss and accuracy metrics

           Raises:
               TypeError if the dataset specified is not a datasets.arrow_dataset.Dataset (or) a
                    TextClassificationDataset (or) a DataLoader
        """
        if self._trainer:
            results = self._trainer.evaluate()
            validation_loss = None
            validation_accuracy = results.get("eval_accuracy")
            print("Val Acc: {:.5f}".format(validation_accuracy))
        else:
            if isinstance(dataset_or_dataloader, Dataset):
                dataloader = DataLoader(dataset_or_dataloader, batch_size=16)
                validation_data_length = len(dataset_or_dataloader)
            elif isinstance(dataset_or_dataloader, DataLoader):
                dataloader = dataset_or_dataloader
                validation_data_length = len(dataloader) * dataloader.batch_size
            elif isinstance(dataset_or_dataloader, HFTextClassificationDataset) or \
                    isinstance(dataset_or_dataloader, HFCustomTextClassificationDataset):
                dataloader = dataset_or_dataloader.validation_loader
                validation_data_length = len(dataset_or_dataloader)
            else:
                raise TypeError("Invalid dataset/dataloader: {}".format(dataset_or_dataloader))

            if not self._model:
                num_classes = len(dataset_or_dataloader.class_names)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.hub_name,
                                                                                 num_labels=num_classes)

            self._model.eval()
            running_loss = 0.0
            running_corrects = 0
            for data_batch in tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = {k: v.to(self._device) for k, v in data_batch.items() if k != 'labels'}
                labels = data_batch['labels']

                outputs = self._model(**inputs)
                loss = self._loss(outputs.logits, labels)

                # Statistics
                predictions = torch.argmax(outputs.logits, dim=-1)

                running_loss += loss.item()
                running_corrects += torch.sum(predictions == labels).item()

            validation_loss = running_loss / validation_data_length
            validation_accuracy = running_corrects / validation_data_length

        return (validation_loss, validation_accuracy)

    def predict(self, input_samples):
        """
           Generates predictions for the specified input samples.

           Args:
               input_samples (str, list, encoded dict, TextClassificationDataset):
                    Input samples to use to predict.

           Returns:
               Numpy array of scores

           Raises:
               NotImplementedError if the given input_samples is of type DataLoader
        """
        encoded_input = None

        # If 'input_samples' is a single text string or a list of text strings
        if isinstance(input_samples, str) or isinstance(input_samples, list):
            encoded_input = self._tokenizer(input_samples, padding=True, return_tensors='pt')
        # If 'input_samples' is an encoded input dict
        elif isinstance(input_samples, dict):
            encoded_input = input_samples
        # If 'input_samples' is of type HFTextClassificationDataset
        elif isinstance(input_samples, HFTextClassificationDataset):
            if input_samples._preprocessed:
                encoded_input = {
                    'input_ids': input_samples['input_ids'],
                    'attention_mask': input_samples['attention_mask'],
                    'token_type_ids': input_samples['token_type_ids']
                }
        # if 'input_samples' is a DataLoader object
        elif isinstance(input_samples, DataLoader):
            raise NotImplementedError("Prediction using Dataloader hasn't been implmented yet. \
                                Use raw text or Dataset as input!")

        output = self._model(**encoded_input)
        _, predictions = torch.max(output.logits, dim=1)
        return predictions

    def export(self, output_dir: str):
        """
        Saves the model to the given output_dir directory.

        Args:
            output_dir (str): Path to save the model.
        """
        dir_name_to_save = validate_model_name(self._model_name)
        path_to_dir_name = os.path.join(output_dir, dir_name_to_save)

        if not os.path.exists(path_to_dir_name):
            os.makedirs(path_to_dir_name)

        output_model_file_path = os.path.join(path_to_dir_name, 'pytorch_model.bin')

        # If we have a distributed model, save only the encapsulated model
        # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = self._model.module if hasattr(self._model, 'module') else self._model

        torch.save(model_to_save.state_dict(), output_model_file_path)

        print('Model saved at {}'.format(path_to_dir_name))

    def load_from_directory(self, model_dir: str, num_classes: int):
        """
        Loads a saved pytorch model from the given model_dir directory

        Args:
            model_dir(str): Path to the saved model directory
            num_classes(int): Number of class labels
        """

        saved_model_file_path = os.path.join(model_dir, 'pytorch_model.bin')

        state_dict = torch.load(saved_model_file_path)

        if not self._model:
            self._model = AutoModelForSequenceClassification.from_pretrained(self.hub_name,
                                                                             num_labels=num_classes)
        self._model.load_state_dict(state_dict)
