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
import tempfile
import shutil
import dill  # nosec: B403
import torch
import numpy as np
import intel_extension_for_pytorch as ipex
from requests.adapters import ProxyError
from tqdm import tqdm
from torch.utils.data import DataLoader

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    TrainingArguments,
    Trainer,
    get_scheduler,
    set_seed
)

from datasets.arrow_dataset import Dataset

from downloader.models import ModelDownloader
from tlt import TLT_BASE_DIR
from tlt.distributed import TLT_DISTRIBUTED_DIR
from tlt.utils.file_utils import read_json_file, validate_model_name, verify_directory
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.models.hf_model import HFModel
from tlt.models.text_classification.text_classification_model import TextClassificationModel
from tlt.datasets.text_classification.text_classification_dataset import TextClassificationDataset
from tlt.datasets.text_classification.hf_text_classification_dataset import HFTextClassificationDataset
from tlt.datasets.text_classification.hf_custom_text_classification_dataset import HFCustomTextClassificationDataset


MODEL_CONFIG_DIR = os.path.join(TLT_BASE_DIR, "models/configs")


class PyTorchHFTextClassificationModel(TextClassificationModel, HFModel):
    """
    Class to represent a PyTorch Hugging Face pretrained model that can be used for multi-class text classification
    fine tuning.
    """

    def __init__(self, model_name: str, model=None, optimizer=None, loss=None, **kwargs):

        hf_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/pytorch_hf_text_classification_models.json"))

        # extra properties that will become configurable in the future
        self._model_name = model_name
        self._dropout_layer_rate = 0.1
        self._do_fine_tuning = False
        self._dropout_layer_rate = None
        self._device = 'cpu'
        self._lr_scheduler = None
        self._generate_checkpoints = True
        self._tokenizer = None
        self._classification_layer = hf_model_map[model_name]["classification_layer"]

        TextClassificationModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_CLASSIFICATION,
                                         self._dropout_layer_rate)
        HFModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_CLASSIFICATION)

        # Store the dataset type that this model type can use for Intel Neural Compressor
        self._inc_compatible_dataset = (HFCustomTextClassificationDataset, HFTextClassificationDataset)

        # set up the configurable optimizer and loss functions
        self._check_optimizer_loss(optimizer, loss)
        self._optimizer_class = optimizer if optimizer else torch.optim.AdamW
        self._opt_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._optimizer_class).args}
        self._optimizer = None  # This gets initialized later
        self._loss_class = loss if loss else torch.nn.CrossEntropyLoss
        self._loss_args = {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(self._loss_class).args}
        self._loss = self._loss_class(**self._loss_args)

        # model definition
        config_dict = read_json_file(os.path.join(MODEL_CONFIG_DIR, "pytorch_hf_text_classification_models.json"))
        self.hub_name = config_dict[self._model_name]["hub_name"]
        self._model = None
        self._num_classes = None
        self._trainer = None
        self._history = None

    def export_for_distributed(self, export_dir, train_data=None, val_data=None):
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

    @property
    def num_classes(self):
        """
        The number of output neurons in the model; equal to the number of classes in the dataset
        """
        return self._num_classes

    def _fit(self, output_dir, dataset, epochs, do_eval, early_stopping, lr_decay):
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

        # For early stopping, if enabled
        patience = 10
        trigger_time = 0
        last_loss = 1.0

        num_training_steps = epochs * train_data_length
        lr_scheduler = get_scheduler(
            name="linear", optimizer=self._optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # Training loop
        since = time.time()
        self._model.to(self._device)
        self._history = {}
        self._model.train()
        # Training loop
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)

            # Training phase
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data_batch in tqdm(train_data_loader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = {k: v.to(self._device) for k, v in data_batch.items()
                          if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                labels = data_batch['label']

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
                eval_metrics = self.evaluate(validation_data_loader)
                eval_epoch_loss = eval_metrics['eval_loss']
                eval_epoch_acc = eval_metrics['eval_accuracy']
                self._update_history('Val Loss', eval_epoch_loss)
                self._update_history('Val Acc', eval_epoch_acc)

                loss_acc_output += f' - Val Loss: {eval_epoch_loss:.4f} - Val Acc: {eval_epoch_acc:.4f}'

                if lr_decay:
                    lr = lr_scheduler.optimizer.param_groups[0]['lr']
                    self._update_history('LR', lr)
                    loss_acc_output += f' - LR: {lr:.4f}'
                    lr_scheduler.step(eval_epoch_loss)

                # Put the model back to train mode
                self._model.train()

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

        self._update_history('train_runtime', round(time_elapsed, 4))
        self._update_history('train_samples_per_second', round(train_data_length * (epoch + 1) / time_elapsed, 3))

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

    def _fit_distributed(self, saved_objects_dir, hostfile, nnodes, nproc_per_node, epochs, batch_size, ipex_optimize):
        import subprocess  # nosec: B404

        distributed_text_script = os.path.join(TLT_DISTRIBUTED_DIR, "pytorch", "run_train_pyt.py")

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
        bash_command += ' {}'.format(distributed_text_script)
        bash_command += ' --master_addr {}'.format(default_master_addr)
        bash_command += ' --master_port {}'.format(default_port)
        bash_command += ' --backend {}'.format('ccl')
        bash_command += ' --tlt_saved_objects_dir {}'.format(saved_objects_dir)
        bash_command += ' --use_case {}'.format('text_classification')
        bash_command += ' --epochs {}'.format(epochs)
        bash_command += ' --batch_size {}'.format(batch_size)
        if not ipex_optimize:
            bash_command += ' --disable_ipex'

        print(bash_command)
        subprocess.run(bash_command.split(' '))

    def _get_hub_model(self, model_name, num_classes, force_download=False):
        downloader = ModelDownloader(model_name, model_dir=None, hub='hugging_face',
                                     num_labels=num_classes, force_download=force_download)
        try:
            model = downloader.download()
        except ProxyError:
            print('Max retries reached. Sleeping for 10 sec...')
            time.sleep(10)
            model = downloader.download()

        return model

    def train(
        self,
        dataset,
        output_dir: str,
        epochs: int = 1,
        initial_checkpoints=None,
        learning_rate: float = 1e-5,
        do_eval: bool = True,
        early_stopping: bool = False,
        lr_decay: bool = True,
        seed: int = None,
        extra_layers: list = None,
        device: str = "cpu",
        ipex_optimize: bool = True,
        use_trainer: bool = False,
        force_download: bool = False,
        distributed: bool = False,
        hostfile: str = None,
        nnodes: int = 1,
        nproc_per_node: int = 1,
        **kwargs
    ):
        """
        Trains the model using the specified text classification dataset.

        Args:
            dataset (TextClassificationDataset/datasets.arrow_dataset.Dataset): The dataset to use for training.
                If a train subset has been defined, that subset will be used to fit the model. Otherwise, the
                entire non-partitioned dataset will be used.
            output_dir (str): A writeable output directory to write checkpoint files during training
            epochs (int): The number of training epochs [default: 1]
            initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                latest checkpoint will be used.
            learning_rate (float): Learning rate for the model to train. Defaults to 1e-5
            do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                at the end of each epoch. If the dataset does not have a validation split, the test subset will be used.
            early_stopping (bool): Enable early stopping if convergence is reached while training
                at the end of each epoch.
            lr_decay (bool): If lr_decay is True and do_eval is True, learning rate decay on the validation loss
                is applied at the end of each epoch.
            seed (int): Optionally set a seed for reproducibility.
            extra_layers (list[int]): Optionally insert additional dense layers between the base model and output
                layer. This can help increase accuracy when fine-tuning a PyTorch model.
                The input should be a list of integers representing the number and size of the layers,
                for example [1024, 512] will insert two dense layers, the first with 1024 neurons and the
                second with 512 neurons.
            device (str): Device to train the model. Defaults to "cpu"
            ipex_optimize (bool): Optimize the model using Intel® Extension for PyTorch. Defaults to True
            use_trainer (bool): If use_trainer is True, then the model training is done using the Hugging Face Trainer
                and if use_trainer is False, the model training is done using native PyTorch training loop
            force_download (bool): Downloads the model with default parameters. Defaults to False.
            distributed (bool): Boolean flag to use distributed training. Defaults to False.
            hostfile (str): Name of the hostfile for distributed training. Defaults to None.
            nnodes (int): Number of nodes to use for distributed training. Defaults to 1.
            nproc_per_node (int): Number of processes to spawn per node to use for distributed training. Defaults
                to 1.

        Returns:
            If use_trainer=True, a Hugging Face TrainOutput object is returned.
            If use_trainer=False, a dictionary containing the model training history is returned.

        Raises:
            TypeError: if the dataset specified is not a TextClassificationDataset/datasets.arrow_dataset.Dataset
            ValueError: if the given dataset has not been preprocessed yet

        """
        self._check_train_inputs(output_dir, dataset, TextClassificationDataset,
                                 extra_layers, epochs, distributed, hostfile)

        if not self._model:
            self._num_classes = len(dataset.class_names)
            self._model = self._get_hub_model(model_name=self.hub_name, num_classes=self._num_classes,
                                              force_download=force_download)

        if not self._optimizer:
            self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

        self._device = device
        self.train_data_loader = None
        self.validation_data_loader = None

        if initial_checkpoints:
            checkpoint = torch.load(initial_checkpoints)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if extra_layers:
            classifier = getattr(self._model, self._classification_layer[0])
            num_features = classifier.in_features
            setattr(self._model, self._classification_layer[0], torch.nn.Sequential())
            classifier = getattr(self._model, self._classification_layer[0])
            for layer in extra_layers:
                classifier.append(torch.nn.Linear(num_features, layer))
                classifier.append(torch.nn.ReLU(inplace=True))
                num_features = layer
            classifier.append(torch.nn.Linear(num_features, self._num_classes))

        # Initialize the optimizer class and create a learning rate scheduler
        self._optimizer = self._optimizer_class(self._model.parameters(), lr=learning_rate, **self._opt_args)

        if seed is not None:
            set_seed(seed)

        if use_trainer:
            if distributed:
                raise ValueError("Distributed training with Trainer is not implemented yet")

            # Get the eval_dataset. We always have to do this, because it seems like even it do_eval=False, the
            # Trainer will still require an eval_dataset.
            eval_dataset = None
            try:
                eval_dataset = dataset.validation_subset
            except ValueError:
                try:
                    eval_dataset = dataset.test_subset
                except ValueError:
                    if do_eval:
                        print("Warning: The dataset provided does not have a validation or test subset.")

            training_args = TrainingArguments(
                output_dir=output_dir,
                do_eval=do_eval,
                do_train=True,
                no_cuda=True,
                overwrite_output_dir=True,
                per_device_train_batch_size=dataset.info['preprocessing_info']['batch_size'],
                per_device_eval_batch_size=dataset.info['preprocessing_info']['batch_size'],
                evaluation_strategy="epoch",
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                data_seed=seed,
                use_ipex=ipex_optimize
            )

            if seed is not None:
                training_args.seed = seed

            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = np.argmax(preds, axis=1)
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

            # Initialize our Trainer
            self._tokenizer = dataset._tokenizer
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=dataset.train_subset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=self._tokenizer
            )

            self._history = self._trainer.train()
        elif distributed:
            try:
                saved_objects_dir = self.export_for_distributed(
                    export_dir=os.path.join(output_dir, 'tlt_saved_objects'),
                    train_data=dataset.train_subset,
                    val_data=dataset.validation_subset
                )
                self._fit_distributed(saved_objects_dir, hostfile, nnodes, nproc_per_node, epochs,
                                      dataset._preprocessed["batch_size"], ipex_optimize)
            except Exception as err:
                print("Error: \'{}\' occured while distributed training".format(err))
            finally:
                self.cleanup_saved_objects_for_distributed()
        else:
            self._trainer = None
            self._model.train()
            if ipex_optimize:
                self._model, self._optimizer = ipex.optimize(self._model, optimizer=self._optimizer)
            # Call the _fit method to train the model with native PyTorch API
            self._fit(output_dir, dataset, epochs, do_eval, early_stopping, lr_decay)

        return self._history

    def evaluate(self, dataset_or_dataloader=None):
        """
           Evaulates the model on the given dataset (or) dataloader. If Hugging Face Trainer object was used to
           train the model, it evaluates on the 'eval_dataset' given in the Trainer arguments

           Args:
               dataset_or_dataloader (datasets.arrow_dataset.Dataset/DataLoader/TextClassificationDataset): The
                    dataset/dataloader to use for evaluation.

           Returns:
               Dictionary with loss, accuracy, runtime, and samples per second metrics

           Raises:
               TypeError: if the dataset specified is not a datasets.arrow_dataset.Dataset (or) a
                    TextClassificationDataset (or) a DataLoader
        """
        if self._trainer:
            results = self._trainer.evaluate()
            print("Val Acc: {:.5f}".format(results.get("eval_accuracy")))
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
                # The model hasn't been trained yet, use the original transformers model
                self._num_classes = len(dataset_or_dataloader.class_names)
                self._model = self._get_hub_model(self.hub_name, self._num_classes)

            # Do the evaluation
            device = torch.device(self._device)
            self._model = self._model.to(device)

            self._model.eval()
            running_loss = 0.0
            running_corrects = 0

            start = time.time()

            for data_batch in tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
                inputs = {k: v.to(device) for k, v in data_batch.items()
                          if k in ['input_ids', 'token_type_ids', 'attention_mask']}
                labels = data_batch['label'].to(device)

                outputs = self._model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                loss = self._loss(outputs.logits, labels)

                # Statistics
                running_loss += loss.item()
                running_corrects += torch.sum(predictions == labels).item()

            time_elapsed = time.time() - start
            samples_per_second = validation_data_length / time_elapsed

            if validation_data_length == 0:
                validation_loss, validation_accuracy = 0.0, 0.0
            else:
                validation_loss = running_loss / validation_data_length
                validation_accuracy = running_corrects / validation_data_length

            results = {
                'eval_loss': validation_loss,
                'eval_accuracy': validation_accuracy,
                'eval_runtime': round(time_elapsed, 4),
                'eval_samples_per_second': round(samples_per_second, 3)
            }

        return results

    def predict(self, input_samples, return_raw=False):
        """
           Generates predictions for the specified input samples.

           Args:
               input_samples (str, list, encoded dict, TextClassificationDataset):
                    Input samples to use to predict.
               return_raw (Bool):
                    Option to return the HF SequenceClassifierOutput object containing the
                    logits Torch Tensor, if set to True.

           Returns:
               Torch Tensor of scores or HF SequenceClassifierOutput if return_raw is set to True.

           Raises:
               NotImplementedError: if the given input_samples is of type DataLoader
        """
        encoded_input = None

        # If 'input_samples' is a single text string or a list of text strings
        if isinstance(input_samples, str) or isinstance(input_samples, list):
            encoded_input = self._tokenizer(input_samples, padding=True, return_tensors='pt')
        # If 'input_samples' is an encoded input dict
        elif isinstance(input_samples, dict):
            # Requires at least 'input_ids' key and any other mentioned below
            required_keys = ['input_ids', 'attention_mask', 'token_type_ids']
            encoded_input = {k: v for k, v in input_samples.items() if k in required_keys}
        # If 'input_samples' is of type HFTextClassificationDataset
        elif isinstance(input_samples, HFTextClassificationDataset) or\
                isinstance(input_samples, HFCustomTextClassificationDataset):
            if input_samples._preprocessed:
                encoded_input = {
                    'input_ids': input_samples['input_ids'],
                    'attention_mask': input_samples['attention_mask'],
                    'token_type_ids': input_samples['token_type_ids']
                }
        # If the 'input_samples' are already pre-processed, then it will be a Dataset object
        elif isinstance(input_samples, Dataset):
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
        if return_raw:
            return output

        _, predictions = torch.max(output.logits, dim=1)
        return predictions

    def export(self, output_dir: str):
        """
        Saves the model to the given output_dir directory.

        Args:
            output_dir (str): Path to save the model.
        """
        if self._model:
            verify_directory(output_dir)
            valid_model_name = validate_model_name(self.model_name)
            saved_model_dir = os.path.join(output_dir, valid_model_name)
            if os.path.exists(saved_model_dir) and len(os.listdir(saved_model_dir)):
                saved_model_dir = os.path.join(saved_model_dir, "{}".format(len(os.listdir(saved_model_dir)) + 1))
            else:
                saved_model_dir = os.path.join(saved_model_dir, "1")
            verify_directory(saved_model_dir)
            # If we have a distributed model, save only the encapsulated model
            # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
            model_copy = dill.dumps(self._model.module if hasattr(self._model, 'module') else self._model)  # noqa: E501, nosec: B301
            torch.save(model_copy, os.path.join(saved_model_dir, 'model.pt'))
            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")

    def load_from_directory(self, model_dir: str):
        """
        Loads a saved pytorch model from the given model_dir directory

        Args:
            model_dir(str): Path to the saved model directory
        """

        verify_directory(model_dir, require_directory_exists=True)
        model_copy = torch.load(os.path.join(model_dir, 'model.pt'))
        self._model = dill.loads(model_copy)  # nosec: B301
        self._optimizer = self._optimizer_class(self._model.parameters(), lr=self._learning_rate)

    def list_layers(self, verbose=False):
        """
        Lists all of the named modules (e.g. features, avgpool, classifier) and layers
        (ReLU, MaxPool2d, Dropout, Linear, etc) in a given PyTorch model

        Args:
            verbose (bool): True/False option set by default to be False, displays only high-level modules
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be summarized.')

        # Display a high-level list of the modules e.g. features, avgpool, classifier
        print("\nModel Layers\n============")
        for (name, module) in self._model.named_children():
            if not verbose or not list(module.named_children()):
                print('{}: {}/{} parameters are trainable'.format(
                    name, sum(p.numel() for p in module.parameters() if p.requires_grad),
                    sum(p.numel() for p in module.parameters())))
            else:
                print('{}:'.format(name))
                for (layer_name, layer) in module.named_children():
                    print('  {}: {}/{} parameters are trainable'.format(
                        layer_name, sum(p.numel() for p in layer.parameters() if p.requires_grad),
                        sum(p.numel() for p in layer.parameters())))

        trainable_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print('\nTotal Trainable Parameters: {}/{}'.format(
            trainable_parameters,
            sum(p.numel() for p in self._model.parameters())))

        return trainable_parameters

    def freeze_layer(self, layer_name):
        """
        Freezes the model's layer using a layer name
        Args:
            layer_name (string): The layer name that will be frozen in the model
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be frozen.')

        # Freeze everything in the layer
        for (name, module) in self._model.named_children():
            if name == layer_name:
                for param in module.parameters():
                    param.requires_grad = False

        return

    def unfreeze_layer(self, layer_name):
        """
        Unfreezes the model's layer using a layer name
        Args:
            layer_name (string): The layer name that will be frozen in the model
        """

        if self._model is None:
            raise RuntimeError('The model must be trained at least one epoch before its layers can be unfrozen.')

        # Unfreeze everything in the layer
        for (name, module) in self._model.named_children():
            if name == layer_name:
                for param in module.parameters():
                    param.requires_grad = True

        return
