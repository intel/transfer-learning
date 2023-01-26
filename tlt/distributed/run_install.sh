#!/usr/bin/bash
conda install -y pytorch==1.12.1 torchvision torchaudio cpuonly intel-openmp gperftools ninja setuptools tqdm future cmake numpy pyyaml scikit-learn pydot -c pytorch -c intel -c conda-forge 
pip install transformers==4.21.1 datasets==2.3.2 intel_extension_for_pytorch==1.12.0
bash deploy/install_torch_ccl.sh