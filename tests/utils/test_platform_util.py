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

import json
import pytest
import os
from mock import mock_open, patch

from tlt.utils.platform_util import PlatformUtil, OptimizedPlatformUtil, CPUInfo
from test_utils import platform_config


def setup_mock_values(mock_platform, mock_os, mock_subprocess):
    platform_config.set_mock_system_type(mock_platform)
    platform_config.set_mock_os_access(mock_os)
    platform_config.set_mock_lscpu_subprocess_values(mock_subprocess)


@pytest.fixture
def os_mock(patch):
    return patch("os.access")


@pytest.fixture
def subprocess_mock(patch):
    return patch("subprocess.check_output")


@pytest.fixture
def subprocess_popen_mock(patch):
    return patch("subprocess.Popen")


@pytest.fixture
def platform_mock(patch):
    return patch("system_platform.system")


@pytest.fixture
def read_mock(patch):
    return patch("read")


@pytest.mark.common
@patch("tlt.utils.platform_util.PlatformUtil._get_cpuset")
def test_platform_util_lscpu_parsing(get_cpuset_mock, platform_mock, subprocess_mock, os_mock):
    """
    Verifies that platform_utils gives us the proper values that we expect
    based on the lscpu_output string provided.
    """
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    os_mock.return_value = True
    get_cpuset_mock.return_value = "0-111"
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    platform_util = PlatformUtil(verbose=True)
    platform_util.linux_init()
    assert platform_util.num_cpu_sockets == 2
    assert platform_util.num_cores_per_socket == 28
    assert platform_util.num_threads_per_core == 2
    assert platform_util.num_logical_cpus == 112
    assert platform_util.num_numa_nodes == 2
    assert platform_util.cpu_family == '6'
    assert platform_util.cpu_model == '143'
    assert platform_util.cpu_type == 'SPR'


@pytest.mark.common
@pytest.mark.parametrize('cpu_model,expected_type',
                         [['85', 'SKX, CPX, CLX'],
                          ['143', 'SPR'],
                          ['123', '']])
@patch("tlt.utils.platform_util.PlatformUtil._get_cpuset")
def test_platform_util_known_cpu_types(get_cpuset_mock, platform_mock, subprocess_mock, os_mock,
                                       cpu_model, expected_type):
    """
    Verifies the are getting the proper CPU type for cpu family/models that we know. For ones that we
    don't know, it should return an empty string.
    """
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    os_mock.return_value = True
    get_cpuset_mock.return_value = "0-111"

    # get the lscpu sample output, but replace in the parameterized cpu model id
    lscpu_value = platform_config.LSCPU_OUTPUT
    original_model_value = "Model:                 143\n"  # model test value from the test platform config
    new_model_value = "Model:                 {}\n".format(cpu_model)
    lscpu_value = lscpu_value.replace(original_model_value, new_model_value)

    subprocess_mock.return_value = lscpu_value
    platform_util = PlatformUtil(verbose=True)
    platform_util.linux_init()
    assert platform_util.cpu_type == expected_type


@pytest.mark.common
def test_platform_util_unsupported_os(platform_mock, subprocess_mock, os_mock):
    """
    Verifies that platform_utils gives us the proper values that we expect
    based on the lscpu_output string provided.
    """
    os_mock.return_value = True
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    # Mac is not supported yet
    platform_mock.return_value = "Mac"
    with pytest.raises(NotImplementedError) as e:
        PlatformUtil(verbose=True)
    assert "Mac Support not yet implemented" in str(e)


@pytest.mark.common
def test_cpu_info_binding_information(subprocess_mock):
    """
    Verifies that cpu_info binding_information property gives us the proper values
    that we expect based on the lscpu_output string provided.
    """
    subprocess_mock.return_value = (
        '# The following is the parsable format, which can be fed to other\n'
        '# programs. Each different item in every column has an unique ID\n'
        '# starting from zero.\n# CPU,Core,Socket,Node\n0,0,0,0\n1,1,0,0\n'
        '2,2,0,0\n3,3,0,0\n4,4,0,0\n5,5,0,0\n6,6,0,0\n7,7,0,0\n8,8,0,0\n'
        '9,9,0,0\n10,10,0,0\n11,11,0,0\n12,12,0,0\n13,13,0,0\n14,14,0,0\n'
        '15,15,0,0\n16,16,0,0\n17,17,0,0\n18,18,0,0\n19,19,0,0\n20,20,0,0\n'
        '21,21,0,0\n22,22,0,0\n23,23,0,0\n24,24,0,0\n25,25,0,0\n26,26,0,0\n'
        '27,27,0,0\n28,28,1,1\n29,29,1,1\n30,30,1,1\n31,31,1,1\n32,32,1,1\n'
        '33,33,1,1\n34,34,1,1\n35,35,1,1\n36,36,1,1\n37,37,1,1\n38,38,1,1\n'
        '39,39,1,1\n40,40,1,1\n41,41,1,1\n42,42,1,1\n43,43,1,1\n44,44,1,1\n'
        '45,45,1,1\n46,46,1,1\n47,47,1,1\n48,48,1,1\n49,49,1,1\n50,50,1,1\n'
        '51,51,1,1\n52,52,1,1\n53,53,1,1\n54,54,1,1\n55,55,1,1\n56,0,0,0\n'
        '57,1,0,0\n58,2,0,0\n59,3,0,0\n60,4,0,0\n61,5,0,0\n62,6,0,0\n63,7,0,0\n'
        '64,8,0,0\n65,9,0,0\n66,10,0,0\n67,11,0,0\n68,12,0,0\n69,13,0,0\n'
        '70,14,0,0\n71,15,0,0\n72,16,0,0\n73,17,0,0\n74,18,0,0\n75,19,0,0\n'
        '76,20,0,0\n77,21,0,0\n78,22,0,0\n79,23,0,0\n80,24,0,0\n81,25,0,0\n'
        '82,26,0,0\n83,27,0,0\n84,28,1,1\n85,29,1,1\n86,30,1,1\n87,31,1,1\n'
        '88,32,1,1\n89,33,1,1\n90,34,1,1\n91,35,1,1\n92,36,1,1\n93,37,1,1\n'
        '94,38,1,1\n95,39,1,1\n96,40,1,1\n97,41,1,1\n98,42,1,1\n99,43,1,1\n'
        '100,44,1,1\n101,45,1,1\n102,46,1,1\n103,47,1,1\n104,48,1,1\n105,49,1,1\n'
        '106,50,1,1\n107,51,1,1\n108,52,1,1\n109,53,1,1\n110,54,1,1\n111,55,1,1\n')
    generated_value = CPUInfo().binding_information
    tests_data_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(tests_data_dir, "files", "sorted_membind_info.json")) as json_data:
        expected_value = json.load(json_data)

    assert generated_value == expected_value


@pytest.mark.common
def test_cpu_info_binding_information_no_numa(subprocess_mock):
    """
    Verifies that cpu_info binding_information property gives us the proper values
    that we expect based on the lscpu_output string provided.
    """
    subprocess_mock.return_value = (
        '# The following is the parsable format, which can be fed to other\n'
        '# programs. Each different item in every column has an unique ID\n'
        '# starting from zero.\n# CPU,Core,Socket,Node\n0,0,0,\n1,1,1,\n'
        '2,2,2,\n3,3,3,\n')
    generated_value = CPUInfo().binding_information
    tests_data_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(tests_data_dir, "files", "sorted_membind_info_no_numa.json")) as json_data:
        expected_value = json.load(json_data)

    assert generated_value == expected_value


@pytest.mark.common
@patch("tlt.utils.platform_util.PlatformUtil._get_cpuset")
def test_numa_cpu_core_list(get_cpuset_mock, subprocess_mock, subprocess_popen_mock, platform_mock, os_mock):
    """ Test the platform utils to ensure that we are getting the proper core lists """
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    subprocess_popen_mock.return_value.stdout.readlines.return_value = platform_config.NUMA_CORES_OUTPUT
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    get_cpuset_mock.return_value = "0-111"
    os_mock.return_value = True
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    platform_util = PlatformUtil(verbose=True, numa_cores_per_instance="socket")

    # ensure there are 2 items in the list since there are 2 sockets
    assert len(platform_util.cpu_core_list) == 2

    # ensure each list of cores has the length of the number of cores per socket
    for core_list in platform_util.cpu_core_list:
        assert len(core_list) == platform_util.num_cores_per_socket


@pytest.mark.common
def test_platform_util_wmic_parsing(platform_mock, subprocess_mock, os_mock):
    """
    Verifies that platform_utils gives us the proper values that we expect
    based on the wmic_output string provided.
    """
    platform_mock.return_value = "Windows"
    os_mock.return_value = True
    subprocess_mock.return_value = platform_config.WMIC_OUTPUT
    platform_util = PlatformUtil(verbose=True)
    platform_util.windows_init()
    assert platform_util.num_cpu_sockets == 2
    assert platform_util.num_cores_per_socket == 28
    assert platform_util.num_threads_per_core == 28
    assert platform_util.num_logical_cpus == 56
    assert platform_util.num_numa_nodes == 0

    # We don't have this info for Windows yet, so we just expect empty strings
    assert platform_util.cpu_family == ''
    assert platform_util.cpu_model == ''
    assert platform_util.cpu_type == ''


@pytest.mark.common
@patch("tlt.utils.platform_util.PlatformUtil._get_cpuset")
@pytest.mark.parametrize('cpuset_range,expected_list',
                         [['0-5', [0, 1, 2, 3, 4, 5]],
                          ['0-3,7,6', [0, 1, 2, 3, 6, 7]],
                          ['2-3,7,9-11,20', [2, 3, 7, 9, 10, 11, 20]],
                          ['0-3,7-6,11,11', [0, 1, 2, 3, 11]],
                          ['7-9,5-10,6,4', [4, 5, 6, 7, 8, 9, 10]],
                          ['0', [0]],
                          ['', []]])
def test_get_list_from_string_ranges(get_cpuset_mock, platform_mock, subprocess_mock, os_mock,
                                     cpuset_range, expected_list,):
    """
    Tests the PlatformUtils _get_list_from_string_ranges function that converts string
    number ranges to an integer list.
    """
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    get_cpuset_mock.return_value = cpuset_range
    os_mock.return_value = True
    platform_util = PlatformUtil()
    result = platform_util._get_list_from_string_ranges(cpuset_range)
    assert result == expected_list


@pytest.mark.common
@pytest.mark.parametrize('cpuset_range,expected_core_list',
                         [["0-7,28-35",
                           [["0", "1", "2", "3", "4", "5", "6", "7"],
                            ["28", "29", "30", "31", "32", "33", "34", "35"]]],
                          ["0,2-5,20,29-32,1",
                           [["0", "1", "2", "3", "4", "5", "20"],
                            ["29", "30", "31", "32"]]]])
@patch("os.path.exists")
def test_numa_cpu_core_list_cpuset(path_exists_mock, subprocess_mock, subprocess_popen_mock,
                                   platform_mock, os_mock, cpuset_range, expected_core_list):
    """ Test the platform utils to ensure that we are getting the proper core lists """
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    subprocess_popen_mock.return_value.stdout.readlines.return_value = platform_config.NUMA_CORES_OUTPUT
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    os_mock.return_value = True
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    path_exists_mock.return_value = True
    cpuset_mock = mock_open(read_data=cpuset_range)
    with patch("builtins.open", cpuset_mock):
        platform_util = PlatformUtil(verbose=True, numa_cores_per_instance=4)

    # ensure there are 2 items in the list since there are 2 sockets
    assert len(platform_util.cpu_core_list) == 2

    # Check that the core list matches the ranges defined for the cpuset file read
    assert platform_util.cpu_core_list == expected_core_list


@pytest.mark.common
@patch("tlt.utils.platform_util.PlatformUtil._get_cpuset")
@pytest.mark.parametrize('cpuset_range,expected_num_sockets',
                         [['0-5', 1],
                          ['0-3,7,6', 1],
                          ['2-3,7,9-11,20', 1],
                          ['0-3,7-6,11,11', 1],
                          ['7-9,5-10,6,4', 1],
                          ['0-111', 2],
                          ['28-32,84-90', 1]])
def test_platform_utils_num_sockets_with_cpuset(get_cpuset_mock, platform_mock, subprocess_mock,
                                                os_mock, cpuset_range, expected_num_sockets):
    """
    Checks that the number of sockets in platform_utils reflects the proper value based on
    the cpuset. If the cores being used by the container in the cpuset are all on one socket,
    then the num_cpu_sockets should be 1, even if the system itself has 2 sockets (since the
    container only has access to 1).
    """
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    os_mock.return_value = True
    get_cpuset_mock.return_value = cpuset_range
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    platform_util = PlatformUtil(verbose=True)
    platform_util.linux_init()
    assert platform_util.num_cpu_sockets == expected_num_sockets


@pytest.mark.common
def test_platform_util_with_no_args(platform_mock, subprocess_mock):
    """
    Verifies that PlatformUtil object can be created with an empty string, as needed
    by the performance Jupyter notebooks.
    """
    platform_mock.return_value = platform_config.SYSTEM_TYPE
    subprocess_mock.return_value = platform_config.LSCPU_OUTPUT
    platform_util = PlatformUtil()
    assert platform_util.num_logical_cpus == 112


@pytest.mark.common
@pytest.mark.parametrize('omp_num_threads,omp_thread_limit,kmp_blocktime,kmp_affinity,'
                         'tf_num_intraop_threads,tf_num_interop_threads,'
                         'tf_enable_mkl_native_format,ld_preload',
                         [[-1, None, None, None, None, None, None, None],
                          [1000, None, None, None, None, None, None, None],
                          [None, 0, None, None, None, None, None, None],
                          [None, 1000, None, None, None, None, None, None],
                          [None, None, -1, None, None, None, None, None],
                          [None, None, None, 'garbage_string', None, None, None, None],
                          [None, None, None, None, -1, None, None, None],
                          [None, None, None, None, None, -1, None, None],
                          [None, None, None, None, None, None, -1, None],
                          [None, None, None, None, None, None, None, 'path/to/non_so_file.txt'],
                          [None, None, None, None, None, None, None, 'path/to/invalid_file.so']])
def test_optimized_platform_util_invalid_args(omp_num_threads, omp_thread_limit, kmp_blocktime,
                                              kmp_affinity, tf_num_intraop_threads, tf_num_interop_threads,
                                              tf_enable_mkl_native_format, ld_preload):
    with pytest.raises((ValueError, FileNotFoundError)):
        OptimizedPlatformUtil(omp_num_threads, omp_thread_limit, kmp_blocktime, kmp_affinity,
                              tf_num_intraop_threads, tf_num_interop_threads,
                              tf_enable_mkl_native_format, ld_preload)


@pytest.mark.common
@pytest.mark.parametrize('omp_num_threads,omp_thread_limit,kmp_blocktime,kmp_affinity,'
                         'tf_num_intraop_threads,tf_num_interop_threads,'
                         'tf_enable_mkl_native_format,ld_preload',
                         [[28, 112, 0, 'granularity=fine', 28, 2, 1, '/tmp/valid_file.so']])
def test_optimized_platform_util_set_env_vars(omp_num_threads, omp_thread_limit, kmp_blocktime,
                                              kmp_affinity, tf_num_intraop_threads, tf_num_interop_threads,
                                              tf_enable_mkl_native_format, ld_preload):
    try:
        with open('/tmp/valid_file.so', 'x'):
            OptimizedPlatformUtil(omp_num_threads, omp_thread_limit, kmp_blocktime, kmp_affinity,
                                  tf_num_intraop_threads, tf_num_interop_threads,
                                  tf_enable_mkl_native_format, ld_preload)

            assert 'OMP_NUM_THREADS' in os.environ and os.environ.get('OMP_NUM_THREADS') == str(omp_num_threads)
            assert 'OMP_THREAD_LIMIT' in os.environ and os.environ.get('OMP_THREAD_LIMIT') == str(omp_thread_limit)
            assert 'KMP_BLOCKTIME' in os.environ and os.environ.get('KMP_BLOCKTIME') == str(kmp_blocktime)
            assert 'KMP_AFFINITY' in os.environ and os.environ.get('KMP_AFFINITY') == kmp_affinity
            assert 'TF_NUM_INTRAOP_THREADS' in os.environ and os.environ.get(
                'TF_NUM_INTRAOP_THREADS') == str(tf_num_intraop_threads)
            assert 'TF_NUM_INTEROP_THREADS' in os.environ and os.environ.get(
                'TF_NUM_INTEROP_THREADS') == str(tf_num_interop_threads)
            assert 'TF_ENABLE_MKL_NATIVE_FORMAT' in os.environ and os.environ.get(
                'TF_ENABLE_MKL_NATIVE_FORMAT') == str(tf_enable_mkl_native_format)
            assert 'LD_PRELOAD' in os.environ and os.environ.get('LD_PRELOAD') == ld_preload
    finally:
        os.remove('/tmp/valid_file.so')
