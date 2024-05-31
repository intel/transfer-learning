#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
import pytest
import sys
import tempfile
import yaml

from downloader.datasets import DataDownloader
from shutil import rmtree
from tlt import TLT_BASE_DIR
from unittest.mock import patch


@pytest.mark.pytorch
def test_no_hf_token():
    """
    Verifies that none of our values yaml files have tokens (the value should be blank)
    """
    helm_chart_dir = os.path.join(TLT_BASE_DIR, "../docker/hf_k8s/chart")

    for values_file in [d for d in os.listdir(helm_chart_dir) if "values" in d]:
        file_path = os.path.join(helm_chart_dir, values_file)
        with open(file_path, 'r') as f:
            values_yaml = yaml.safe_load(f)
            assert "secret" in values_yaml
            assert "encodedToken" in values_yaml["secret"]
            assert values_yaml["secret"]["encodedToken"] is None, "encodedToken value found in {}".format(values_file)


@pytest.mark.integration
@pytest.mark.pytorch
def test_llm_finetune_script():
    """
    This is a basic test that runs the LLM fine tuning using distilgpt2 with the code_alpaca_2k with a
    limited number of steps.
    """
    sys.path.append(os.path.join(TLT_BASE_DIR, "../docker/hf_k8s/scripts"))
    from finetune import BenchmarkArguments, DataArguments, FinetuneArguments, main, ModelArguments, \
        QuantizationArguments, TrainingArguments

    # Define the dataset directory and download a test dataset
    dataset_dir = os.getenv('DATASET_DIR', tempfile.mkdtemp(dir='/tmp/data'))
    dataset_path = os.path.join(dataset_dir, 'code_alpaca_2k.json')
    if not os.path.exists(dataset_path):
        download_url = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_2k.json"
        data_downloader = DataDownloader("code_alpaca_2k", dataset_dir, url=download_url)
        data_downloader.download()
    assert os.path.exists(dataset_path)

    # Define the output directory
    output_dir = os.getenv('OUTPUT_DIR', '/tmp/output')
    os.makedirs(output_dir, exist_ok=True)
    output_dir = tempfile.mkdtemp(dir=output_dir)

    try:
        with patch('transformers.HfArgumentParser.parse_args_into_dataclasses') as mock_parser:
            model_args = ModelArguments()
            model_args.model_name_or_path = "distilbert/distilgpt2"

            data_args = DataArguments(train_file=dataset_path, validation_split_percentage=0.2, max_eval_samples=5)
            finetune_args = FinetuneArguments(use_lora=False)
            training_args = TrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, max_steps=5)
            benchmark_args = BenchmarkArguments(do_benchmark=False)
            quant_args = QuantizationArguments(do_quantize=False)

            mock_parser.return_value = model_args, data_args, finetune_args, quant_args, \
                benchmark_args, training_args, {}
            main()
            assert len(os.listdir(output_dir)) > 0
    finally:
        for d in [dataset_dir, output_dir]:
            rmtree(d, ignore_errors=True)


@pytest.mark.pytorch
def test_optimum_habana_unavailable():
    """
    This test checks that the is_optimum_habana_available() method returns False in our test environment, which does
    not have optimum-habana installed.
    """
    sys.path.append(os.path.join(TLT_BASE_DIR, "../docker/hf_k8s/scripts"))
    from finetune import is_optimum_habana_available

    # In our normal test environment, optimum-habana is unavailable
    assert not is_optimum_habana_available()
