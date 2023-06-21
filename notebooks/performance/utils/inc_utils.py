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

import numpy as np


def performance(saved_model_dir, batch_size, image_size, dataset_dir, framework, warmup=10, iteration=100,
                cores_per_instance=None, num_of_instance=None, inter_num_of_threads=None, intra_num_of_threads=None):
    """
    Uses the Intel Neural Compressor to get performance metrics for the specified model.
    
    :param saved_model_dir: Model to load
    :param batch_size: Batch size
    :param image_size: Image input size
    :param dataset_dir: Dataset directory (for a custom image classification dataset)
    :param framework: Framework (i.e. tensorflow)
    :param warmup: Number of warmup iterations before running performance tests
    :param iteration: The number of iterations to run for the performance test
    :param cores_per_instance: Number of CPU cores to use per instance
    :param num_of_instance: Number of instances to use for performance testing
    :param inter_num_of_threads: Number of threads to use for inter-thread operations
    :param intra_num_of_threads: Number of threads to use for intra-thread operations
    :return: accuracy, batch_size, result_list
    """

    from neural_compressor.benchmark import fit
    from neural_compressor.config import BenchmarkConfig
    from neural_compressor.utils.create_obj_from_config import create_dataloader

    dataloader_args = {
        'batch_size': batch_size,
        'dataset': {'ImageFolder': {'root': dataset_dir}},
        'transform': {'PaddedCenterCrop': {'size': image_size, 'crop_padding': 32},
                      'Resize': {'size': image_size, 'interpolation': 'bicubic'},
                      'Rescale': {}
                      },
        'filter': None
    }

    eval_dataloader = create_dataloader(framework, dataloader_args)

    conf = BenchmarkConfig(warmup=warmup, iteration=iteration)
    try:
        return fit(model=saved_model_dir, config=conf, b_dataloader=eval_dataloader)
    except Exception:
        # Retry a second time due to the known ZQMError when running from Jupyter
        print("Retrying benchmarking a second time")
        return fit(model=saved_model_dir, config=conf, b_dataloader=eval_dataloader)


def calculate_latency_and_throughput(results):
    """
    Parses the results from the benchmarking function and returns the latency (ms) and throughput (samples/sec)
    
    :param results: Return value from calling the performance util function
    :param batch_size: batch size
    :return: latency (ms) and throughput (images/sec)
    """
    _, batch_size, result_list = results['performance']
    latency = np.array(result_list).mean() / batch_size
    latency_ms = latency * 1000
    throughput = 1. / latency
    return latency_ms, throughput
