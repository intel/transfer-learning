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
import time
import dill  # nosec: B403
import tempfile
import shutil

from tqdm import tqdm

import torch

from tlt.distributed import TLT_DISTRIBUTED_DIR
from tlt.models.pytorch_model import PyTorchModel
from tlt.models.image_classification.image_classification_model import ImageClassificationModel
from tlt.datasets.image_classification.image_classification_dataset import ImageClassificationDataset
from tlt.datasets.image_classification.pytorch_custom_image_classification_dataset \
    import PyTorchCustomImageClassificationDataset
from tlt.datasets.image_classification.torchvision_image_classification_dataset \
    import TorchvisionImageClassificationDataset
from tlt.utils.file_utils import verify_directory, validate_model_name
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.utils.platform_util import PlatformUtil

try:
    habana_import_error = None
    import habana_frameworks.torch.core as htcore
    is_hpu_available = True
except Exception as e:
    is_hpu_available = False
    habana_import_error = str(e)


class PyTorchImageClassificationModel(ImageClassificationModel, PyTorchModel):
    """
    Class to represent a PyTorch model for image classification
    """

    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):
        """
        Class constructor
        """
        # PyTorch models generally do not enforce a fixed input shape
        self._image_size = 'variable'

        # Store the dataset type that this model type can use for Intel Neural Compressor
        self._inc_compatible_dataset = (PyTorchCustomImageClassificationDataset, TorchvisionImageClassificationDataset)

        # extra properties that will become configurable in the future
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._device = kwargs.get("device", "cpu")
        self._lr_scheduler = None
        self._generate_checkpoints = True

        # placeholder for model definition
        self._model = None
        self._num_classes = None
        self._enable_auto_mixed_precision = False

        use_case = kwargs.get('use_case', UseCaseType.IMAGE_CLASSIFICATION)

        PyTorchModel.__init__(self, model_name, FrameworkType.PYTORCH, use_case)
        ImageClassificationModel.__init__(self, self._image_size, self._do_fine_tuning, self._dropout_layer_rate,
                                          self._model_name, self._framework, self._use_case)

        # set up the configurable optimizer and loss functions
        self._check_optimizer_loss(optimizer, loss)
        self._optimizer_class = optimizer if optimizer else torch.optim.Adam
        self._opt_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._optimizer_class).args}
        self._optimizer = None  # This gets initialized later
        self._loss_class = loss if loss else torch.nn.CrossEntropyLoss
        self._loss_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._loss_class).args}
        self._loss = self._loss_class(**self._loss_args)

        if model is None:
            self._model = None
        elif isinstance(model, str):
            self.load_from_directory(model)
            layers = list(self._model.children())
            if isinstance(layers[-1], torch.nn.Sequential):
                self._num_classes = layers[-1][-1].out_features
            else:
                self._num_classes = layers[-1].out_features
        elif isinstance(model, torch.nn.Module):
            self._model = model
            layers = list(self._model.children())
            if isinstance(layers[-1], torch.nn.Sequential):
                self._num_classes = layers[-1][-1].out_features
            else:
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

    def _fit(self, output_dir, dataset, epochs, do_eval, early_stopping, lr_decay, enable_auto_mixed_precision):
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

        # For early stopping, if enabled
        patience = 10
        trigger_time = 0
        last_loss = 1.0

        if lr_decay:
            self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=0.2, patience=5,
                                                                            cooldown=1, min_lr=0.0000000001)

        self._history = {}
        self._model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
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
                    if enable_auto_mixed_precision:
                        # Call model using the torch automatic mixed precision context when mixed precision is enabled
                        with torch.autocast(device_type=self._device, dtype=torch.bfloat16):
                            outputs = self._model(inputs)
                    else:
                        outputs = self._model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self._loss(outputs, labels)
                    loss.backward()
                    if is_hpu_available and self._device == "hpu":
                        htcore.mark_step()
                        self._optimizer.step()
                        htcore.mark_step()
                    else:
                        self._optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if data_length == 0:
                train_epoch_loss = 0
                train_epoch_acc = 0
            else:
                train_epoch_loss = running_loss / data_length
                train_epoch_acc = float(running_corrects) / data_length
            self._update_history('Loss', train_epoch_loss)
            self._update_history('Acc', train_epoch_acc)

            loss_acc_output = f'Loss: {train_epoch_loss:.4f} - Acc: {train_epoch_acc:.4f}'

            if do_eval and validation_data_loader is not None:
                self._model = self._model.to(device)
                self._model.eval()
                running_loss = 0.0
                running_corrects = 0

                with torch.no_grad():
                    print("Performing Evaluation")
                    for inputs, labels in tqdm(validation_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = self._model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self._loss(outputs, labels)

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                        if self._device == "hpu" and is_hpu_available:
                            htcore.mark_step()

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

            if early_stopping:
                if eval_epoch_loss >= last_loss:
                    trigger_time += 1

                    if trigger_time >= patience:
                        # Stop Early
                        print("Early stopping has been triggered after " + str(epoch) + " epochs.")
                        break
                else:
                    trigger_time = 0

                last_loss = eval_epoch_loss

            print(loss_acc_output)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        if self._generate_checkpoints:
            valid_model_name = validate_model_name(self.model_name)
            checkpoint_dir = os.path.join(output_dir, "{}_checkpoints".format(valid_model_name))
            verify_directory(checkpoint_dir)
            try:
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': train_epoch_loss,
                }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
            except KeyError:
                # Calling state_dict() on an IPEX optimizer calls into the torch optimizer's __setstate__ method
                # which in PyTorch 1.12 assumes that the first state value will always have a 'step' key
                state_values = list(self._optimizer.state.values())
                if 'step' not in state_values[0].keys():
                    state_values[0]['step'] = torch.tensor([])
                torch.save({
                    'epoch': epochs,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'loss': train_epoch_loss,
                }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

    def _fit_distributed(self, saved_objects_dir, hostfile, nnodes, nproc_per_node, epochs, batch_size, ipex_optimize,
                         use_horovod, hvd_start_timeout):
        import subprocess  # nosec: B404
        distributed_vision_script = os.path.join(TLT_DISTRIBUTED_DIR, "pytorch", "run_train_pyt.py")

        if use_horovod:
            with open(hostfile, 'r') as f:
                ip_addresses = [line.rstrip() for line in f]
            bash_cmd = 'horovodrun'
            bash_cmd += ' -np {}'.format(nnodes * nproc_per_node)
            bash_cmd += ' -H '
            for node in range(nnodes):
                bash_cmd += '{}:{},'.format(ip_addresses[node], nproc_per_node)
            bash_cmd = bash_cmd[:-1]  # Remove trailing comma
            bash_cmd += ' --start-timeout {}'.format(hvd_start_timeout)
            bash_cmd += ' python {}'.format(distributed_vision_script)
            bash_cmd += ' --use_horovod'
            bash_cmd += ' --use_case {}'.format('image_classification')
            bash_cmd += ' --tlt_saved_objects_dir {}'.format(saved_objects_dir)
            bash_cmd += ' --epochs {}'.format(epochs)
            bash_cmd += ' --batch_size {}'.format(batch_size)
            if not ipex_optimize:
                bash_cmd += ' --disable_ipex'
            print(bash_cmd)
            subprocess.run(bash_cmd.split(' '))
        else:

            default_port = '29500'
            default_master_addr = '127.0.0.1'

            addresses = []

            if hostfile is not None:
                if os.path.isfile(hostfile):
                    # if addresses are given as line separated IP addresses
                    with open(hostfile) as hf:
                        addresses = hf.readlines()
                    addresses = [a.strip('\n') for a in addresses]
                else:
                    # if addresses are given as a comma separated IP addresses
                    addresses = hostfile.split(',')

                default_master_addr = addresses[0]

                # If port is given in the format of "0.0.0.0:9999"
                if ':' in default_master_addr:
                    colon_index = default_master_addr.index(':')
                    default_port = default_master_addr[colon_index + 1:]
                    default_master_addr = default_master_addr[:colon_index]

                    # We create/rewrite the hostfile to contain only IP addresses
                    with open('hostfile', 'w') as hf:
                        for addr in addresses:
                            if ':' in addr:
                                addr = addr[:addr.index(':')]
                            hf.write(addr + '\n')
                    hostfile = 'hostfile'

            bash_command = 'python -m intel_extension_for_pytorch.cpu.launch --distributed'
            bash_command += ' --hostfile {}'.format(hostfile)
            bash_command += ' --nnodes {}'.format(nnodes)
            bash_command += ' --nproc_per_node {}'.format(nproc_per_node)
            bash_command += ' {}'.format(distributed_vision_script)
            bash_command += ' --master_addr {}'.format(default_master_addr)
            bash_command += ' --master_port {}'.format(default_port)
            bash_command += ' --backend {}'.format('ccl')
            bash_command += ' --tlt_saved_objects_dir {}'.format(saved_objects_dir)
            bash_command += ' --use_case {}'.format('image_classification')
            bash_command += ' --epochs {}'.format(epochs)
            bash_command += ' --batch_size {}'.format(batch_size)
            if not ipex_optimize:
                bash_command += ' --disable_ipex'

            print(bash_command)
            subprocess.run(bash_command.split(' '))

    def train(self, dataset: ImageClassificationDataset, output_dir, epochs=1, initial_checkpoints=None,
              do_eval=True, early_stopping=False, lr_decay=True, seed=None, ipex_optimize=True, distributed=False,
              hostfile=None, nnodes=1, nproc_per_node=1, use_horovod=False, hvd_start_timeout=30,
              enable_auto_mixed_precision=None, device=None):
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
                early_stopping (bool): Enable early stopping if convergence is reached while training
                    at the end of each epoch.
                lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                    is applied at the end of each epoch.
                seed (int): Optionally set a seed for reproducibility.
                ipex_optimize (bool): Use Intel Extension for PyTorch (IPEX). Defaults to True.
                distributed (bool): Boolean flag to use distributed training. Defaults to False.
                hostfile (str): Name of the hostfile for distributed training. Defaults to None.
                nnodes (int): Number of nodes to use for distributed training. Defaults to 1.
                nproc_per_node (int): Number of processes to spawn per node to use for distributed training. Defaults
                    to 1.
                enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    It is recommended to enable auto mixed precision training when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the training performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms. Defaults to
                    None.
                device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")

            Returns:
                Trained PyTorch model object
        """
        self._check_train_inputs(output_dir, dataset, ImageClassificationDataset, epochs, initial_checkpoints,
                                 distributed, hostfile)

        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
            # Gaudi is not compatible with IPEX
            if ipex_optimize:
                print("Note: IPEX is not compatible with Gaudi, setting ipex_optimize=False")
                ipex_optimize = False
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif self._device == "hpu" and is_hpu_available:
            if ipex_optimize:
                print("Note: IPEX is not compatible with Gaudi, setting ipex_optimize=False")
                ipex_optimize = False

        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision training will be disabled.".format(str(e)))

        self._enable_auto_mixed_precision = enable_auto_mixed_precision

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

        if distributed:
            try:
                saved_objects_dir = self.export_for_distributed(
                    export_dir=os.path.join(output_dir, 'tlt_saved_objects'),
                    train_data=dataset.train_subset,
                    val_data=dataset.validation_subset
                )
                batch_size = dataset._preprocessed['batch_size']
                self._fit_distributed(saved_objects_dir, hostfile, nnodes, nproc_per_node, epochs, batch_size,
                                      ipex_optimize, use_horovod, hvd_start_timeout)
            except Exception as err:
                print("Error: \'{}\' occured while distributed training".format(err))
            finally:
                self.cleanup_saved_objects_for_distributed()

        else:
            # Call ipex.optimize
            if ipex_optimize:
                import intel_extension_for_pytorch as ipex
                ipex_dtype = torch.bfloat16 if self._enable_auto_mixed_precision else None
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer, dtype=ipex_dtype)
            self._fit(output_dir, dataset, epochs, do_eval, early_stopping, lr_decay, enable_auto_mixed_precision)

        return self._history

    def evaluate(self, dataset: ImageClassificationDataset, use_test_set=False, enable_auto_mixed_precision=None,
                 device=None):
        """
        Evaluate the accuracy of the model on a dataset.
        If there is a validation set, evaluation will be done on it (by default) or on the test set
        (by setting use_test_set=True). Otherwise, the entire non-partitioned dataset will be
        used for evaluation.
        """
        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision training will be disabled.".format(str(e)))
        self._enable_auto_mixed_precision = enable_auto_mixed_precision

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

        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"

        device = torch.device(self._device)
        model = model.to(device)

        # Do the evaluation
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

            if self._device == "hpu" and is_hpu_available:
                htcore.mark_step()

        epoch_loss = running_loss / data_length
        epoch_acc = float(running_corrects) / data_length

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return [epoch_loss, epoch_acc]

    def predict(self, input_samples, return_type='class', enable_auto_mixed_precision=None, device=None):
        """
        Perform feed-forward inference and predict the classes of the input_samples.

        Args:
            input_samples (tensor): Input tensor with one or more samples to perform inference on
            return_type (str): Using 'class' will return the highest scoring class (default), using 'scores' will
                               return the raw output/logits of the last layer of the network, using 'probabilities' will
                               return the output vector after applying a softmax function (so results sum to 1)
            device (str): Enter "cpu" or "hpu" to specify which hardware device to run training on.
                    If device="hpu" is specified, but no HPU hardware or installs are detected,
                    CPU will be used. (default: "cpu")

        Returns:
            List of classes, probability vectors, or raw score vectors

        Raises:
            ValueError: if the return_type is not one of 'class', 'probabilities', or 'scores'
        """
        return_types = ['class', 'probabilities', 'scores']
        if not isinstance(return_type, str) or return_type not in return_types:
            raise ValueError('Invalid return_type ({}). Expected one of {}.'.format(return_type, return_types))

        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision training will be disabled.".format(str(e)))
        self._enable_auto_mixed_precision = enable_auto_mixed_precision

        # Only change the device if one is passed in
        if device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"
        elif device == "hpu" and is_hpu_available:
            self._device = device
        elif device == "cpu":
            self._device = device

        # If No device is passed in, but model was initialized with hpu, must check if hpu is available
        if self._device == "hpu" and not is_hpu_available:
            print("No Gaudi HPUs were found or required device drivers are not installed. Running on CPUs")
            print(habana_import_error)
            self._device = "cpu"

        self._model.eval()
        with torch.no_grad():
            self._model = self._model.to(self._device)
            predictions = self._model(input_samples)
        if return_type == 'class':
            _, predicted_ids = torch.max(predictions, 1)
            return predicted_ids
        elif return_type == 'probabilities':
            # logic from torch.nn.functional _get_softmax_dim()
            dim = input_samples.dim()
            if dim == 0 or dim == 1 or dim == 3:
                dim = 0
            else:
                dim = 1
            return torch.nn.functional.softmax(predictions, dim=dim)
        else:
            return predictions

    def export(self, output_dir):
        """
        Save a serialized version of the model to the output_dir path
        """
        if self._model:
            # Save the model in a format that can be re-loaded for inference
            verify_directory(output_dir)
            valid_model_name = validate_model_name(self.model_name)
            saved_model_dir = os.path.join(output_dir, valid_model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")
            verify_directory(saved_model_dir)
            model_copy = dill.dumps(self._model, recurse=True)
            torch.save(model_copy, os.path.join(saved_model_dir, 'model.pt'))
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")

    def export_for_distributed(self, export_dir=None, train_data=None, val_data=None):
        """
        Exports the model, optimizer, loss, train data and validation data to the export_dir for distributed
        script to access. Note that the export_dir must be accessible to all the nodes. For example: NFS shared
        systems. Note that the export_dir is created using mkdtemp which reults in a unique dir name. For
        example: "<export_dir_Am83Iw". If the export_dir is None, the default name is "saved_objects"

        Args:
            export_dir (str): Directory name to export the model, optimizer, loss, train data and validation
                data. export_dir must be accessible to all the nodes. For example: NFS shared systems. export_dir
                is created using mkdtemp which reults in a unique dir name. For example: "<export_dir_Am83Iw".
                If the export_dir is None, the default name is "saved_objects"
            train_data (PyTorchDataset): Train dataset
            val_data (PyTorchDataset): Validation dataset
        """

        temp_dir_prefix = os.path.join(os.environ['HOME'], "saved_objects_") if export_dir is None else export_dir + "_"
        self._temp_dir = tempfile.mkdtemp(prefix=temp_dir_prefix)

        objects_to_save = {
            "train_data": train_data,
            "model": self._model,
            "optimizer": self._optimizer,
            "loss": self._loss
        }
        torch.save(objects_to_save, os.path.join(self._temp_dir, "torch_saved_objects.obj"))
        return self._temp_dir

    def cleanup_saved_objects_for_distributed(self):
        try:
            print('Cleaning saved objects...')
            shutil.rmtree(self._temp_dir)
        except OSError as ose:
            print('Error while cleaning the saved objects: {}'.format(ose))
