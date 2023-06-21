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
import tarfile
import requests
import shutil
import zipfile


def download_file(download_url, destination_directory):
    """
    Downloads a file using the specified url to the destination directory. Returns the
    path to the downloaded file.
    """
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)

    destination_file_path = os.path.join(destination_directory, os.path.basename(download_url))

    print("Downloading {} to {}".format(download_url, destination_directory))
    response = requests.get(download_url, stream=True, timeout=30)
    with open(destination_file_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)

    return destination_file_path


def extract_tar_file(tar_file_path, destination_directory):
    """
    Extracts a tar file on the local file system to the destination directory. Returns a list
    of top-level contents (files and folders) of the extracted archive.
    """
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)

    print("Extracting {} to {}".format(tar_file_path, destination_directory))
    with tarfile.open(tar_file_path) as t:
        t.extractall(path=destination_directory)
        contents = {i.split('/')[0] for i in t.getnames()}
        return list(contents)


def extract_zip_file(zip_file_path, destination_directory):
    """
    Extracts a zip file on the local file system to the destination directory. Returns a list
    of top-level contents (files and folders) of the extracted archive.
    """
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)

    print("Extracting {} to {}".format(zip_file_path, destination_directory))
    with zipfile.ZipFile(zip_file_path, "r") as z:
        z.extractall(path=destination_directory)
        contents = {i.split('/')[0] for i in z.namelist()}
        return list(contents)
