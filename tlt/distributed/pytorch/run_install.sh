#!/usr/bin/env bash

conda install -y \
  'numpy==1.24.4' \
  'pytorch==2.0.1' \
  'pyyaml==6.0.1' \
  'scikit-learn==1.2.2' \
  'torchaudio==2.0.2' \
  'torchvision==0.15.2' \
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
  'accelerate==0.28.0' \
  'datasets==2.14.5' \
  'intel_extension_for_pytorch==2.2.0' \
  'transformers[torch]==4.38.0'

bash deploy/install_torch_ccl.sh
