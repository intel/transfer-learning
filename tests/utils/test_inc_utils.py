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

import pytest

from tlt.utils.inc_utils import get_inc_config


@pytest.mark.common
@pytest.mark.parametrize('accuracy_criterion,valid',
                         [[0.1, True],
                          [-1, False],
                          [0.01, True],
                          [1.434, False],
                          ['foo', False]])
def test_inc_config_accuracy_criterion(accuracy_criterion, valid):
    """
    Tests an INC config with good and bad accuracy_criterion_relative values
    """
    if not valid:
        with pytest.raises(ValueError):
            get_inc_config(accuracy_criterion_relative=accuracy_criterion)
    else:
        config = get_inc_config(accuracy_criterion_relative=accuracy_criterion)
        assert config.accuracy_criterion.relative == accuracy_criterion


@pytest.mark.common
@pytest.mark.parametrize('timeout,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, True],
                          [60, True],
                          ['foo', False]])
def test_inc_config_timeout(timeout, valid):
    """
    Tests an INC config with good and bad exit_policy_timeout values
    """
    if not valid:
        with pytest.raises(ValueError):
            get_inc_config(exit_policy_timeout=timeout)
    else:
        config = get_inc_config(exit_policy_timeout=timeout)
        assert config.timeout == timeout


@pytest.mark.common
@pytest.mark.parametrize('max_trials,valid',
                         [[0.1, False],
                          [-1, False],
                          [0, False],
                          [1, True],
                          [60, True],
                          ['foo', False]])
def test_inc_config_max_trials(max_trials, valid):
    """
    Tests an INC config with good and bad exit_policy_max_trials values
    """
    if not valid:
        with pytest.raises(ValueError):
            get_inc_config(exit_policy_max_trials=max_trials)
    else:
        config = get_inc_config(exit_policy_max_trials=max_trials)
        assert config.max_trials == max_trials


@pytest.mark.common
@pytest.mark.parametrize('approach,valid',
                         [['foo', False],
                          [-1, False],
                          [0, False],
                          ['static', True],
                          ['dynamic', True],
                          [True, False],
                          [False, False]])
def test_inc_config_approach(approach, valid):
    """
    Tests an INC config with good and bad approach values
    """
    if not valid:
        with pytest.raises(ValueError):
            get_inc_config(approach=approach)
    else:
        config = get_inc_config(approach=approach)
        assert config.approach == 'post_training_{}_quant'.format(approach)
