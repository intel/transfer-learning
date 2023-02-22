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

import click

from tlt.tools.cli.commands.benchmark import benchmark
from tlt.tools.cli.commands.eval import eval
from tlt.tools.cli.commands.list import list_group
from tlt.tools.cli.commands.optimize import optimize
from tlt.tools.cli.commands.quantize import quantize
from tlt.tools.cli.commands.train import train


@click.group('cli')
def cli_group():
    pass


# Add top level commands
cli_group.add_command(list_group)
cli_group.add_command(train)
cli_group.add_command(eval)
cli_group.add_command(quantize)
cli_group.add_command(benchmark)
cli_group.add_command(optimize)

if __name__ == '__main__':
    cli_group()
