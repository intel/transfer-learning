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
import pytest
import shutil
import uuid
import tempfile

from pathlib import Path
from unittest.mock import patch

from tlk.models import model_factory

try:
    # Do PyTorch specific imports in a try/except to prevent pytest test loading from failing when running in a TF env
    from tlk.models.image_classification.torchvision_image_classification_model import TorchvisionImageClassificationModel
except ModuleNotFoundError as e:
    print("WARNING: Unable to import TorchvisionImageClassificationModel. PyTorch or torchvision may not be installed")


@pytest.mark.pytorch
def test_torchvision_image_classification_optimize_graph_not_implemented():
    """
    Verifies the error that gets raise if graph optimization is attempted with a PyTorch model
    """
    try:
        output_dir = tempfile.mkdtemp()
        saved_model_dir = tempfile.mkdtemp()
        dummy_config_file = os.path.join(output_dir, "config.yaml")
        Path(dummy_config_file).touch()
        model = model_factory.get_model('resnet50', 'pytorch')
        # Graph optimization is not enabled for PyTorch, so this should fail 
        with patch('neural_compressor.experimental.Graph_Optimization') as mock_o:
            with pytest.raises(NotImplementedError):
                model.optimize_graph(saved_model_dir, output_dir)
        
        # Verify that the installed version of INC throws a SystemError
        from neural_compressor.experimental import Graph_Optimization, common
        from neural_compressor.utils.utility import set_backend
        set_backend('pytorch')
        graph_optimizer = Graph_Optimization()
        with pytest.raises(SystemExit):
            graph_optimizer.model = common.Model(model._model)
    
    finally:    
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)

