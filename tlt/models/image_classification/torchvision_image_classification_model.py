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
from pydoc import locate
from tqdm import tqdm

import torch
import intel_extension_for_pytorch as ipex

from tlt import TLT_BASE_DIR
from tlt.models.image_classification.pytorch_image_classification_model import PyTorchImageClassificationModel
from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlt.utils.file_utils import read_json_file


class TorchvisionImageClassificationModel(PyTorchImageClassificationModel):
    """
    Class used to represent a Torchvision pretrained model for image classification
    """

    def __init__(self, model_name: str):
        """
        Class constructor
        """
        torchvision_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/torchvision_image_classification_models.json"))
        if model_name not in torchvision_model_map.keys():
            raise ValueError("The specified Torchvision image classification model ({}) "
                             "is not supported.".format(model_name))

        PyTorchImageClassificationModel.__init__(self, model_name)

        self._classification_layer = torchvision_model_map[model_name]["classification_layer"]
        self._image_size = torchvision_model_map[model_name]["image_size"]

        # placeholder for model definition
        self._model = None
        self._num_classes = None

    def _get_hub_model(self, num_classes, ipex_optimize=True, extra_layers=None):
        if not self._model:
            pretrained_model_class = locate('torchvision.models.{}'.format(self._model_name))
            self._model = pretrained_model_class(pretrained=True)

            if not self._do_fine_tuning:
                for param in self._model.parameters():
                    param.requires_grad = False

            if len(self._classification_layer) == 2:
                base_model = getattr(self._model, self._classification_layer[0])
                classifier = getattr(self._model, self._classification_layer[0])[self._classification_layer[1]]
                self._model.classifier = base_model[0: self._classification_layer[1]]
                num_features = classifier.in_features
                if extra_layers:
                    for layer in extra_layers:
                        self._model.classifier.append(torch.nn.Linear(num_features, layer))
                        self._model.classifier.append(torch.nn.ReLU(inplace=True))
                        num_features = layer
                self._model.classifier.append(torch.nn.Linear(num_features, num_classes))
            else:
                classifier = getattr(self._model, self._classification_layer[0])
                if self._classification_layer[0] == "heads":
                    num_features = classifier.head.in_features
                else:
                    num_features = classifier.in_features

                if extra_layers:
                    # assuming its always just the output layer that exists here.
                    setattr(self._model, self._classification_layer[0], torch.nn.Sequential())
                    classifier = getattr(self._model, self._classification_layer[0])
                    for layer in extra_layers:
                        classifier.append(torch.nn.Linear(num_features, layer))
                        classifier.append(torch.nn.ReLU(inplace=True))
                        num_features = layer
                    classifier.append(torch.nn.Linear(num_features, num_classes))
                else:
                    setattr(self._model, self._classification_layer[0], torch.nn.Linear(num_features, num_classes))

            self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

            if ipex_optimize:
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)
        self._num_classes = num_classes
        return self._model, self._optimizer

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, lr_decay=True, seed=None, extra_layers=None):
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
                extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                    layer. This can help increase accuracy when fine-tuning a Pytorch model.
                    The input should be a list of integers representing the number and size of the layers,
                    for example [1024, 512] will insert two dense layers, the first with 1024 neurons and the
                    second with 512 neurons.

            Returns:
                Trained PyTorch model object
        """
        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints)

        if extra_layers:
            if not isinstance(extra_layers, list):
                raise TypeError("The extra_layers parameter must be a list of ints but found {}".format(
                    type(extra_layers)))
            else:
                for layer in extra_layers:
                    if not isinstance(layer, int):
                        raise TypeError("The extra_layers parameter must be a list of ints but found a list "
                                        "containing {}".format(type(layer)))

        dataset_num_classes = len(dataset.class_names)

        # If the number of classes doesn't match what was used before, clear out the previous model
        if dataset_num_classes != self.num_classes:
            self._model = None

        # If are loading weights, the state dicts need to be loaded before calling ipex.optimize, so get the model
        # from torchvision, but hold off on the ipex optimize call.
        ipex_optimize = False if initial_checkpoints else True

        self._set_seed(seed)

        self._model, self._optimizer = self._get_hub_model(dataset_num_classes, ipex_optimize=ipex_optimize,
                                                           extra_layers=extra_layers)

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Call ipex.optimize now, since we didn't call it from _get_hub_model()
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

        if self._model is None:
            # The model hasn't been trained yet, use the original ImageNet trained model
            print("The model has not been trained yet, so evaluation is being done using the original model ",
                  "and its classes")
            pretrained_model_class = locate('torchvision.models.{}'.format(self._model_name))
            model = pretrained_model_class(pretrained=True)
            optimizer = self._optimizer_class(model.parameters(), lr=self._learning_rate)
            # We shouldn't need ipex.optimize() for evaluation
        else:
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
        if self._model is None:
            print("The model has not been trained yet, so predictions are being done using the original model")
            pretrained_model_class = locate('torchvision.models.{}'.format(self.model_name))
            model = pretrained_model_class(pretrained=True)
            predictions = model(input_samples)
        else:
            self._model.eval()
            predictions = self._model(input_samples)
        _, predicted_ids = torch.max(predictions, 1)
        return predicted_ids
