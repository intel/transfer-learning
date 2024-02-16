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

from click.testing import CliRunner

import os
import pytest
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from downloader.datasets import DataDownloader
from tlt.tools.cli.commands.train import train
from tlt.tools.cli.commands.generate import generate


@pytest.mark.integration
@pytest.mark.pytorch
class TestTextGenerationCustomDataset:
    """
    Tests for PyTorch text generation using a custom dataset
    """
    @classmethod
    def setup_class(cls):
        temp_dir = tempfile.mkdtemp(dir='/tmp/data')
        download_url = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_2k.json"
        data_downloader = DataDownloader("code_alpaca_2k", temp_dir, url=download_url)
        data_downloader.download()

        os.makedirs('/tmp/output', exist_ok=True)
        cls._output_dir = tempfile.mkdtemp(dir='/tmp/output')
        os.environ["TORCH_HOME"] = cls._output_dir
        cls._temp_dir = temp_dir
        cls._dataset_dir = temp_dir

    @classmethod
    def teardown_class(cls):
        # remove directories
        for dir in [cls._output_dir, cls._temp_dir]:
            if os.path.exists(dir):
                print("Deleting test directory:", dir)
                shutil.rmtree(dir)

    @patch("tlt.models.text_generation.pytorch_hf_text_generation_model.PyTorchHFTextGenerationModel.generate")
    @pytest.mark.parametrize('model_name,prompt',
                             [['distilgpt2', 'The size of an apple is'],
                              ['distilgpt2', 'A large fruit is'],
                              ['distilgpt2',
                               'The input describes a task.\\n\\nInstruction:\nWrite a song.\\n\\n### Response:\n']])
    def test_base_generation(self, mock_generate, model_name, prompt):
        """
        Tests the CLI generate command for PYT text generation using a HF pretrained model
        """
        runner = CliRunner()

        # Define a dummy response
        mock_generate.return_value = [prompt + ' so good.']

        # Generate a text completion
        result = runner.invoke(generate,
                               ["--model-name", model_name, "--prompt", prompt])

        # Verify that the TLT generate method was called with a properly formatted prompt string
        assert len(mock_generate.call_args_list) == 1
        prompt_arg = mock_generate.call_args_list[0][0]
        assert "\\n" not in prompt_arg

        # Verify that we didn't get any errors
        assert result is not None
        assert result.exit_code == 0

    @patch("tlt.models.model_factory.get_model")
    @patch("tlt.datasets.dataset_factory.load_dataset")
    @pytest.mark.parametrize('model_name,prompt',
                             [['distilgpt2', 'Write a function to print hello world']])
    def test_pretrained_generation(self, mock_load_dataset, mock_get_model, model_name, prompt):
        """
        Tests the full workflow for PYT text generation using a custom dataset
        """
        runner = CliRunner()

        framework = "pytorch"
        dataset_file = "code_alpaca_2k.json"

        model_mock = MagicMock()
        data_mock = MagicMock()

        model_mock.framework = framework
        model_mock.use_case = "text_generation"

        mock_get_model.return_value = model_mock
        mock_load_dataset.return_value = data_mock

        train_result = runner.invoke(train,
                                     ["--framework", framework, "--dataset_dir", self._dataset_dir, "--output_dir",
                                      self._output_dir, "--dataset-file", dataset_file, "--model-name", model_name,
                                      "--instruction-key", "instruction", "--context-key", "input", "--response-key",
                                      "output", "--prompt-with-context", "Below is an instruction that describes a \
                                       task,  Write a response that appropriately completes the text."])

        # Verify that the expected calls were made
        mock_load_dataset.assert_called_once_with(self._dataset_dir, model_mock.use_case, model_mock.framework)

        assert model_mock.train.called

        # Verify that the train command exit code is successful
        assert train_result.exit_code == 0
        saved_model_dir = os.path.join(self._output_dir, model_name, '1')
        os.makedirs(os.path.join(saved_model_dir, 'training_args.bin'), exist_ok=True)

        # Generate a text completion
        result = runner.invoke(generate,
                               ["--model-dir", saved_model_dir, "--prompt", prompt])
        assert result is not None
        assert result.exit_code == 0
