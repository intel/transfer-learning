#!/usr/bin/env bash

conda install -y \
  'numpy==1.23.5' \
  'pytorch==1.13.1' \
  'pyyaml==6.0' \
  'scikit-learn==1.2.2' \
  'torchaudio==1.13.1' \
  'torchvision==0.14.1' \
  'tqdm==4.65.0' \
  cmake \
  cpuonly \
  future \
  gperftools \
  intel-openmp \
  ninja \
  pydot \
  setuptools \
  -c pytorch -c intel -c conda-forge

pip install \
  'datasets~=2.12.0' \
  'intel_extension_for_pytorch==1.13.100' \
  'transformers~=4.30.0'

bash deploy/install_torch_ccl.sh
