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
import time
from tqdm import tqdm
import dill

import torch
import intel_extension_for_pytorch as ipex

from tlt import TLT_BASE_DIR
from tlt.models.pytorch_model import PyTorchModel
from tlt.models.image_classification.image_classification_model import ImageClassificationModel
from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlt.utils.file_utils import read_json_file, verify_directory
from tlt.utils.types import FrameworkType, UseCaseType


class PyTorchImageClassificationModel(ImageClassificationModel, PyTorchModel):
    """
    Class used to represent a PyTorch model for image classification
    """

    def __init__(self, model_name: str, model=None):
        """
        Class constructor
        """
        # PyTorch models generally do not enforce a fixed input shape
        self._image_size = 'variable'

        # extra properties that will become configurable in the future
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._device = 'cpu'
        self._optimizer_class = torch.optim.Adam  # Just the class, it needs to be initialized with the model object
        self._optimizer = None
        self._loss = torch.nn.CrossEntropyLoss()
        self._lr_scheduler = None
        self._generate_checkpoints = True

        # placeholder for model definition
        self._model = None
        self._num_classes = None

        PyTorchModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.IMAGE_CLASSIFICATION)
        ImageClassificationModel.__init__(self, self._image_size, self._do_fine_tuning, self._dropout_layer_rate,
                                          self._model_name, self._framework, self._use_case)

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
            raise TypeError("The model input must be a torch.nn.Module, string, or None but found a {}".format(type(model)))

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    def _fit(self, output_dir, dataset, epochs, do_eval, lr_decay):
        """Main PyTorch training loop"""
        since = time.time()

        device = torch.device(self._device)
        self._model = self._model.to(device)

        if dataset.train_subset:
            train_data_loader = dataset.train_loader
            data_length = len(dataset.train_subset)
        else:
            train_data_loader = dataset.data_loader
            data_length = len(dataset.dataset)

        if do_eval and dataset.validation_subset:
            validation_data_loader = dataset.validation_loader
            validation_data_length = len(dataset.validation_subset)
        else:
            validation_data_loader = None
            validation_data_length = 0

        if lr_decay:
            self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=0.2, patience=5,
                                                                            cooldown=1, min_lr=0.0000000001)

        self._history = {}
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs - 1}')
            print('-' * 10)

            # Training phase
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(train_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Forward and backward pass
                with torch.set_grad_enabled(True):
                    outputs = self._model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self._loss(outputs, labels)
                    loss.backward()
                    self._optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_epoch_loss = running_loss / data_length
            train_epoch_acc = float(running_corrects) / data_length
            self._update_history('Loss', train_epoch_loss)
            self._update_history('Acc', train_epoch_acc)

            loss_acc_output = f'Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_acc:.4f}'

            if do_eval and validation_data_loader is not None:
                self._model.eval()
                running_loss = 0.0
                running_corrects = 0

                with torch.no_grad():
                    for inputs, labels in validation_data_loader:
                        outputs = self._model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self._loss(outputs, labels)

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                eval_epoch_loss = running_loss / validation_data_length
                eval_epoch_acc = float(running_corrects) / validation_data_length
                self._update_history('Val Loss', eval_epoch_loss)
                self._update_history('Val Acc', eval_epoch_acc)

                loss_acc_output += f' - Val Loss: {eval_epoch_loss:.4f} - Val Acc: {eval_epoch_acc:.4f}'

                if lr_decay:
                    lr = self._lr_scheduler.optimizer.param_groups[0]['lr']
                    self._update_history('LR', lr)
                    loss_acc_output += f' - LR: {lr:.4f}'
                    self._lr_scheduler.step(eval_epoch_loss)

            print(loss_acc_output)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if self._generate_checkpoints:
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(self.model_name))
            verify_directory(checkpoint_dir)
            torch.save({
                'epoch': epochs,
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'loss': train_epoch_loss,
            }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, lr_decay=True, seed=None):
        """
            Trains the model using the specified image classification dataset. The first time training is called, it
            will get the model from torchvision and add on a fully-connected dense layer with linear activation
            based on the number of classes in the specified dataset. The model and optimizer are defined and trained
            for the specified number of epochs.

            Args:
                dataset (ImageClassificationDataset): Dataset to use when training the model
                output_dir (str): Path to a writeable directory for output files
                epochs (int): Number of epochs to train the model (default: 1)
                initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                    latest checkpoint will be used.
                do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                    at the end of each epoch.
                lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
                seed (int): Optionally set a seed for reproducibility.

            Returns:
                Trained PyTorch model object
        """
        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints)

        dataset_num_classes = len(dataset.class_names)

        # Check that the number of classes matches the model outputs
        if dataset_num_classes != self.num_classes:
            raise RuntimeError("The number of model outputs ({}) differs from the number of dataset classes ({})".
                               format(self.num_classes, dataset_num_classes))

        self._set_seed(seed)

        self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Call ipex.optimize
        self._model.train()
        self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)

        self._fit(output_dir, dataset, epochs, do_eval, lr_decay)

        return self._history

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False):
        """
        Evaluate the accuracy of the model on a dataset.

        If there is a validation set, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, the entire non-partitioned dataset will be
        used for evaluation.
        """
        if use_test_set:
            if dataset.test_subset:
                eval_loader = dataset.test_loader
                data_length = len(dataset.test_subset)
            else:
                raise ValueError("No test subset is defined")
        elif dataset.validation_subset:
            eval_loader = dataset.validation_loader
            data_length = len(dataset.validation_subset)
        else:
            eval_loader = dataset.data_loader
            data_length = len(dataset.dataset)

        model = self._model
        optimizer = self._optimizer

        # Do the evaluation
        device = torch.device(self._device)
        model = model.to(device)

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(eval_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self._loss(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_length
        epoch_acc = float(running_corrects) / data_length

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return [epoch_loss, epoch_acc]

    def predict(self, input_samples):
        """
        Perform feed-forward inference and predict the classes of the input_samples
        """
        self._model.eval()
        predictions = self._model(input_samples)
        _, predicted_ids = torch.max(predictions, 1)
        return predicted_ids

    def export(self, output_dir):
        """
        Save a serialized version of the model to the output_dir path
        """
        if self._model:
            # Save the model in a format that can be re-loaded for inference
            verify_directory(output_dir)
            saved_model_dir = os.path.join(output_dir, self.model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")
            
            verify_directory(saved_model_dir)
            model_copy = dill.dumps(self._model)
            torch.save(model_copy, os.path.join(saved_model_dir, 'model.pt'))
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")
