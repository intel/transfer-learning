#!/usr/bin/env bash

conda install -y \
  'numpy==1.24.4' \
  'pytorch==2.1.0' \
  'pyyaml==6.0.1' \
  'scikit-learn==1.2.2' \
  'torchaudio==2.1.0' \
  'torchvision==0.16.0 \
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
  'datasets==2.14.5' \
  'intel_extension_for_pytorch==2.2.0' \
  'transformers[torch]==4.36.0'

bash deploy/install_torch_ccl.sh
