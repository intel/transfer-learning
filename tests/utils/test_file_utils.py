#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-2022 Intel Corporation
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

import io
import mock
import os
import pytest
import shutil
import tempfile
from unittest.mock import MagicMock

from tlt.utils.file_utils import validate_model_name, download_file


@pytest.mark.common
@pytest.mark.parametrize('model_name,valid_model_name',
                         [['my/model', 'my_model'],
                          ['m.o/d~e!@#l', 'm_o_d_e___l'],
                          ['mod\tel', 'mod_el'],
                          ['m        odel', 'm_odel'],
                          ['-.,!@#$%^&*()', '-____________']])
def test_validate_model_name(model_name, valid_model_name):
    """
    Verifies that the model name passed as a string into the
    validate_model_name() function gives us the proper value based on
    the output string provided.
    """
    val = validate_model_name(model_name)
    assert val == valid_model_name


@pytest.mark.common
def test_download():
    output_dir = tempfile.mkdtemp()
    expected = os.path.join(output_dir, 'example.txt')

    # Set up mock return value
    mock_file_contents = MagicMock()
    mock_file_contents.read.return_value = b''

    mock_file = MagicMock(spec=io.BytesIO)
    mock_file.__enter__.return_value = mock_file
    mock_file.raw = mock_file_contents

    # Patch requests.get
    with mock.patch('requests.get', return_value=mock_file) as mock_get:
        result = download_file('https://example-files.online-convert.com/document/txt/example.txt', output_dir)
        assert result == expected

        # Check that the mock was called as expected
        mock_get.assert_called_with('https://example-files.online-convert.com/document/txt/example.txt',
                                    stream=True, timeout=30)
        mock_file_contents.read.assert_called_once()

    # Delete the temp output directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
