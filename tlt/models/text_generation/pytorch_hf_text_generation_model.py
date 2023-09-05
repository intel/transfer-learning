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

import math
import os
import time
import torch
from requests.adapters import ProxyError

# Hugging Face imports
from peft import LoraConfig, TaskType, get_peft_model, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    GenerationConfig,
    Trainer
)

from downloader.models import ModelDownloader
from tlt import TLT_BASE_DIR
from tlt.utils.file_utils import read_json_file, validate_model_name, verify_directory
from tlt.utils.platform_util import PlatformUtil
from tlt.utils.types import FrameworkType, UseCaseType
from tlt.models.hf_model import HFModel
from tlt.models.pytorch_model import PyTorchModel
from tlt.models.text_generation.text_generation_model import TextGenerationModel
from tlt.datasets.text_generation.text_generation_dataset import TextGenerationDataset
from tlt.datasets.text_generation.hf_custom_text_generation_dataset import HFCustomTextGenerationDataset


MODEL_CONFIG_DIR = os.path.join(TLT_BASE_DIR, "models/configs")


class PyTorchHFTextGenerationModel(TextGenerationModel, HFModel, PyTorchModel):
    """
    Class to represent a PyTorch Hugging Face pretrained model that can be used for text generation
    fine tuning.
    """

    def __init__(self, model_name: str, model=None, **kwargs):

        hf_model_map = read_json_file(os.path.join(
            TLT_BASE_DIR, "models/configs/pytorch_hf_text_generation_models.json"))

        # extra properties that will become configurable in the future
        self._model_name = model_name
        self._generate_checkpoints = True
        self._device = 'cpu'
        self._tokenizer = None
        self._enable_auto_mixed_precision = False

        TextGenerationModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_GENERATION)
        HFModel.__init__(self, model_name, FrameworkType.PYTORCH, UseCaseType.TEXT_GENERATION)
        PyTorchModel.__init__(self, model_name, framework=FrameworkType.PYTORCH, use_case=UseCaseType.TEXT_GENERATION)

        # Store the dataset type that this model type can use for Intel Neural Compressor
        self._inc_compatible_dataset = (HFCustomTextGenerationDataset)

        # model definition
        self.hub_name = hf_model_map[model_name]["hub_name"]
        self._model = None
        self._trainer = None
        self._history = None

        if model and isinstance(model, str):
            self.load_from_directory(model)

    def _get_hub_model(self, model_name, force_download=False):
        downloader = ModelDownloader(model_name, model_dir=None, hub='hugging_face',
                                     hf_model_class='AutoModelForCausalLM', force_download=force_download)
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
        temperature=1.0,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.05,
        max_train_samples=None,
        do_eval: bool = True,
        device: str = "cpu",
        ipex_optimize: bool = True,
        use_trainer: bool = True,
        force_download: bool = False,
        enable_auto_mixed_precision: bool = None,
        **kwargs
    ):
        """
        Trains the model using the specified text generation dataset.

        Args:
            dataset (TextGenerationDataset): The dataset to use for training. If a train subset has been defined, that
                                             subset will be used to fit the model. Otherwise, the entire
                                             non-partitioned dataset will be used.
            output_dir (str): A writeable output directory to write checkpoint files during training
            epochs (int): The number of training epochs [default: 1]
            initial_checkpoints (str): Path to checkpoint weights to load. If the path provided is a directory, the
                                       latest checkpoint will be used.
            temperature (float): The value used to modulate the next token probabilities [default: 1.0]
            lora_rank (int): LoRA rank parameter [default: 8]
            lora_alpha (int): LoRA alpha parameter [default: 32]
            lora_dropout (float): LoRA dropout parameter [default: 0.05]
            max_train_samples (int or None): Use this to truncate the training set to a maximum number of samples
                                             for quick testing [default: None]
            do_eval (bool): If do_eval is True and the dataset has a validation subset, the model will be evaluated
                            at the end of each epoch. If the dataset does not have a validation split, the test subset
                            will be used.
            device (str): Device to train the model. Defaults to "cpu"
            ipex_optimize (bool): Optimize the model using Intel® Extension for PyTorch. Defaults to True
            use_trainer (bool): Placeholder argument, model training is done using the Hugging Face Trainer
                                and a native PyTorch training loop is not yet implemented.
            force_download (bool): Downloads the model with default parameters. Defaults to False.
            enable_auto_mixed_precision (bool or None): Enable auto mixed precision for training. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make training run faster and use less memory.
                    It is recommended to enable auto mixed precision training when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the training performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms. Defaults to
                    None.

        Returns:
            Hugging Face TrainOutput object

        Raises:
            TypeError: if the dataset specified is not a TextGenerationDataset
            ValueError: if the given dataset has not been preprocessed yet

        """
        self._check_train_inputs(output_dir, dataset, TextGenerationDataset, None, epochs, False, None,
                                 enable_auto_mixed_precision)

        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                enable_auto_mixed_precision = False
                print("Unable to determine the CPU type: {}. Mixed precision training will be disabled.".format(str(e)))

        self._enable_auto_mixed_precision = enable_auto_mixed_precision

        if not self._model:
            self._model = self._get_hub_model(model_name=self.hub_name, force_download=force_download)

        self._model.train()
        self._device = device
        self.train_data_loader = None
        self.validation_data_loader = None

        # Get the eval_dataset
        eval_dataset = None
        try:
            eval_dataset = dataset.validation_subset
        except ValueError:
            try:
                eval_dataset = dataset.test_subset
            except ValueError:
                if do_eval:
                    print("Warning: The dataset provided does not have a validation or test subset.")

        # Truncate the train dataset if desired
        train_dataset = dataset.train_subset
        if max_train_samples is not None:
            print("Truncating training dataset to size {}".format(max_train_samples))
            train_dataset = train_dataset.select(range(max_train_samples))

        # Initialize tokenizer
        if self._tokenizer is None:
            self._tokenizer = dataset._tokenizer
        self._tokenizer.pad_token_id = (0)
        self._tokenizer.padding_side = "left"

        print('Using Low-Rank Adaptation (LoRA) for {}'.format(self.model_name))

        # PEFT settings
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self._model = get_peft_model(self._model, peft_config)
        self._model.print_trainable_parameters()
        self._model.train()

        if use_trainer:
            # Randomly mask the tokens
            data_collator = DataCollatorForSeq2Seq(self._tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                   padding=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                do_eval=do_eval,
                do_train=True,
                no_cuda=True,
                overwrite_output_dir=True,
                per_device_train_batch_size=dataset.info['preprocessing_info']['batch_size'],
                per_device_eval_batch_size=dataset.info['preprocessing_info']['batch_size'],
                use_ipex=ipex_optimize,
                bf16=enable_auto_mixed_precision
            )

            # Initialize our Trainer
            self._trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self._tokenizer,
                data_collator=data_collator
            )

            self._history = self._trainer.train(resume_from_checkpoint=initial_checkpoints)
        else:
            raise ValueError("Training without the Hugging Face Trainer is not implemented yet")

        return self._history

    def evaluate(self, dataset=None, enable_auto_mixed_precision=None):
        """
        Evaluates the model on the 'eval_dataset' given in the Trainer arguments

        Args:
            dataset (TextGenerationDataset): The dataset to use for evaluation. If Hugging Face Trainer object was used
                                             to train the model, it evaluates on the 'eval_dataset' given in the Trainer
                                             arguments
            enable_auto_mixed_precision (bool or None): Enable auto mixed precision for evaluation. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make evaluation run faster and use less memory.
                    It is recommended to enable auto mixed precision when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the evaluation performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms. Defaults to
                    None.
        Returns:
            Perplexity metric

        Raises:
            RuntimeError: if the model has not been trained yet and does not have an associated Trainer
        """
        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                enable_auto_mixed_precision = False
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision will be disabled for evaluation.".format(str(e)))

        self._enable_auto_mixed_precision = enable_auto_mixed_precision
        self._model.eval()

        if self._trainer:
            eval_results = self._trainer.evaluate()
        else:
            if not isinstance(dataset, TextGenerationDataset):
                raise ValueError("Expected a dataset of type TextGenerationDataset and got {}".format(type(dataset)))
            train_dataset = dataset.train_subset
            eval_dataset = dataset.validation_subset
            batch_size = dataset.info['preprocessing_info']['batch_size']
            tokenizer = dataset._tokenizer
            tokenizer.pad_token_id = (0)
            tokenizer.padding_side = "left"
            data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

            training_args = TrainingArguments(
                output_dir='/tmp/output',
                do_eval=True,
                do_train=False,
                no_cuda=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                bf16=enable_auto_mixed_precision
            )

            # Initialize Trainer
            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator
            )

            eval_results = trainer.evaluate()

        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        return eval_results

    def generate(self, input_samples, temperature=1.0, top_p=0.75, top_k=40, repetition_penalty=1.0, num_beams=4,
                 max_new_tokens=128, decode=True, enable_auto_mixed_precision=None):
        """
        Generates text completions for the specified input samples.

        Args:
            input_samples (dict, encoded dict): Input sample to use to generate text completion.
            temperature (float): The value used to modulate the next token probabilities [default: 1.0]
            top_p (float): If set to float < 1, only the smallest set of most probable tokens with probabilities that
                           add up to top_p or higher are kept for generation [default: 0.75]
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering [default: 40]
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty. [default: 1.0]
            num_beams (int): Number of beams for beam search. 1 means no beam search. [default: 4]
            max_new_tokens (int): The maximum number of new tokens generated [default: 128]
            decode (bool): Set to False if the tokenized output is desired, otherwise if True, the decoded response
                           will be returned [default: True]
            enable_auto_mixed_precision (bool or None): Enable auto mixed precision for evaluation. Mixed precision
                    uses both 16-bit and 32-bit floating point types to make evaluation run faster and use less memory.
                    It is recommended to enable auto mixed precision when running on platforms that support
                    bfloat16 (Intel third or fourth generation Xeon processors). If it is enabled on a platform that
                    does not support bfloat16, it can be detrimental to the evaluation performance. If
                    enable_auto_mixed_precision is set to None, auto mixed precision will be automatically enabled when
                    running with Intel fourth generation Xeon processors, and disabled for other platforms. Defaults to
                    None.

        Returns:
            List of strings

        Raises:
            NotImplementedError: if the given input_samples is of type DataLoader
        """
        if enable_auto_mixed_precision is None:
            try:
                # Only automatically enable auto mixed precision for SPR
                enable_auto_mixed_precision = PlatformUtil().cpu_type == 'SPR'
            except Exception as e:
                enable_auto_mixed_precision = False
                print("Unable to determine the CPU type: {}.\n"
                      "Mixed precision will be disabled for generation.".format(str(e)))

        self._enable_auto_mixed_precision = enable_auto_mixed_precision

        if self._model is None:
            print("The model has not been fine-tuned yet, so generation is being done using the original model")
            self._model = self._get_hub_model(model_name=self.hub_name)

        self._model.eval()

        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.hub_name)
            except ProxyError:
                print("Max retries reached. Sleeping for 10 sec...")
                time.sleep(10)
                self._tokenizer = AutoTokenizer.from_pretrained(self.hub_name)
            self._tokenizer.pad_token_id = (0)
            self._tokenizer.padding_side = "left"

        # If 'input_samples' is a single text string or a list of text strings
        if isinstance(input_samples, str) or isinstance(input_samples, list):
            encoded_input = self._tokenizer(input_samples, padding=True, return_tensors='pt')
        # If 'input_samples' is an encoded input dict
        elif isinstance(input_samples, dict) and 'input_ids' in input_samples.keys():
            # Requires at least the keys below
            required_keys = ['input_ids', 'attention_mask', 'labels']
            encoded_input = {k: v for k, v in input_samples.items() if k in required_keys}
        # If 'input_samples' is a single unencoded dict
        elif isinstance(input_samples, dict):
            encoded_input = self._tokenizer(input_samples, padding=True, return_tensors='pt')
        # if 'input_samples' is any other kind of object
        else:
            raise NotImplementedError("Generation using a List, Dataset, or Dataloader hasn't been implemented yet. "
                                      "Use an unencoded or encoded dictionary.")

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams
        )

        if self._enable_auto_mixed_precision:
            with torch.no_grad():
                with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                    output = self._model.generate(input_ids=encoded_input['input_ids'],
                                                  generation_config=generation_config,
                                                  max_new_tokens=max_new_tokens)
        else:
            with torch.no_grad():
                output = self._model.generate(input_ids=encoded_input['input_ids'],
                                              generation_config=generation_config,
                                              max_new_tokens=max_new_tokens)

        if not decode:
            return output
        else:
            return self._tokenizer.batch_decode(output)

    def export(self, output_dir: str):
        """
        Saves the model and tokenizer to the given output_dir directory.

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

            self._trainer.save_model(saved_model_dir)

            print("Saved model directory:", saved_model_dir)

            return saved_model_dir
        else:
            raise ValueError("Unable to export the model, because it hasn't been trained yet")

    def load_from_directory(self, model_dir: str):
        """
        Loads a saved pytorch model from the given model_dir directory. Requires a 'config.json' and
        'pytorch_model.bin' file corresponding to a transformers model in the model_dir.

        Args:
            model_dir(str): Path to the transformers model directory
        """
        verify_directory(model_dir, require_directory_exists=True)

        try:
            model = AutoModelForCausalLM.from_pretrained(self.hub_name)
            self._model = PeftModelForCausalLM.from_pretrained(model, model_dir)
        except Exception:
            raise ValueError("Unable to load model from {}".format(model_dir))
