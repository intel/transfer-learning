#!/usr/bin/env bash

conda install -y \
  pyyaml \
  cmake \
  cpuonly \
  future \
  gperftools \
  intel-openmp \
  ninja \
  numpy \
  pydot \
  'pytorch==1.13.1' \
  scikit-learn \
  setuptools \
  'torchaudio==1.13.1' \
  'torchvision==0.14.1' \
  tqdm \
  -c pytorch -c intel -c conda-forge

pip install \
  datasets \
  'intel_extension_for_pytorch==1.13.0' \
  transformers

bash deploy/install_torch_ccl.sh
