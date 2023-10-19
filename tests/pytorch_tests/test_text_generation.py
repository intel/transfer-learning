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
import pytest
import shutil
import tempfile

from numpy.testing import assert_almost_equal
from downloader.datasets import DataDownloader
from tlt.datasets import dataset_factory
from tlt.models import model_factory


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

    @pytest.mark.parametrize('model_name,batch_size,ipex_optimize,test_inc',
                             [['distilgpt2', 4, True, False],
                              ['distilgpt2', 8, False, False]])
    def test_custom_dataset_workflow(self, model_name, batch_size, ipex_optimize, test_inc):
        """
        Tests the full workflow for PYT text generation using a custom dataset
        """
        framework = 'pytorch'
        use_case = 'text_generation'

        # Get the dataset
        dataset = dataset_factory.load_dataset(self._dataset_dir, use_case=use_case, framework=framework,
                                               dataset_file='code_alpaca_2k.json', shuffle_files=False)

        # Get the model
        model = model_factory.get_model(model_name, framework)

        prompt_dict = {"prompt_with_context": "Below is an instruction",
                       "prompt_without_context": "Below is a different instruction"}
        dataset_schema = {
            "instruction_key": "instruction",
            "context_key": "input",
            "response_key": "output"
        }

        # Preprocess the dataset and split to get small subsets for training and validation
        dataset.preprocess(model_name=model.hub_name, batch_size=batch_size, dataset_schema=dataset_schema,
                           prompt_dict=prompt_dict)
        dataset.shuffle_split(train_pct=0.1, val_pct=0.1, seed=10)

        # Train for 1 epoch
        model.train(dataset, output_dir=self._output_dir, epochs=1, do_eval=False, seed=10, ipex_optimize=ipex_optimize)

        # Evaluate
        metrics = model.evaluate(dataset)

        # Generate a text completion
        completion = model.generate("Test input")
        assert completion is not None

        # export the saved model
        saved_model_dir = model.export(self._output_dir)
        assert os.path.isdir(saved_model_dir)
        assert os.path.isfile(os.path.join(saved_model_dir, "adapter_model.bin"))

        # Reload the saved model
        reload_model = model_factory.get_model(model_name, framework)
        reload_model.load_from_directory(saved_model_dir)

        # Evaluate
        reload_metrics = reload_model.evaluate(dataset)
        assert_almost_equal(reload_metrics['eval_loss'], metrics['eval_loss'], decimal=2)

        # Placeholder for testing benchmarking and quantization
        if test_inc:
            inc_output_dir = os.path.join(self._output_dir, "quantized", model_name)
            os.makedirs(inc_output_dir, exist_ok=True)
            model.quantize(inc_output_dir, dataset)
            assert os.path.exists(os.path.join(inc_output_dir, "model.pt"))
            model.benchmark(saved_model_dir=inc_output_dir, dataset=dataset)

    @pytest.mark.pytorch
    def test_bad_json(self):
        """
        Tests that HuggingFace load dataset methodology handles bad JSON reads
        """
        json_path = os.path.join(self._dataset_dir, 'code_alpaca_2k.json')
        output_path = os.path.join(self._output_dir, "bad_json.json")

        shutil.copyfile(json_path, output_path)

        with open(output_path, "a", encoding="utf-8") as file:
            file.write("}")

        import gzip
        with pytest.raises(gzip.BadGzipFile):
            dataset_factory.load_dataset(self._output_dir, use_case="text_generation", framework="pytorch",
                                         dataset_file="bad_json.json", shuffle_files=False)
