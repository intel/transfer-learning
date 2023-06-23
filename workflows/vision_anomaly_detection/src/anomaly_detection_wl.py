# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

import os
import sys
import yaml
import argparse
from tqdm import tqdm
import torch

sys.path.append(os.path.join(sys.path[0],'../../..'))
sys.path.append('frameworks.ai.transfer-learning/')

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.models.image_anomaly_detection.pytorch_image_anomaly_detection_model import extract_features, pca, get_feature_extraction_model

from torchvision.models import resnet18, resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.transforms.functional import InterpolationMode

def get_dataset(img_dir, image_size, batch_size):
    dataset = dataset_factory.load_dataset(img_dir, 
                                    use_case='image_anomaly_detection', 
                                    framework="pytorch")
    dataset.preprocess(image_size, batch_size=batch_size, interpolation=InterpolationMode.LANCZOS)

    return dataset

def get_base_model(model_name):
    return model_factory.get_model(model_name=model_name, framework="pytorch", use_case='anomaly_detection')

def train_simsiam(base_model, dataset, config):
    simsiam_config = config['simsiam']
    simsiam_model = base_model.train_simsiam(dataset, os.path.join(config['output_path']), epochs=int(simsiam_config['epochs']),
                    feature_dim=1000,pred_dim=250, initial_checkpoints=simsiam_config['initial_ckpt'], precision=config['precision'])
    return simsiam_model

def train_cutpaste(base_model, dataset, config):
    cutpaste_config = config['cutpaste']
    cutpaste_model = base_model.train_cutpaste(dataset, os.path.join(config['output_path']), optim=cutpaste_config['optim'], epochs=cutpaste_config['epochs'],
                     freeze_resnet=cutpaste_config['freeze_resnet'], head_layer=cutpaste_config['head_layer'], cutpaste_type=cutpaste_config['cutpaste_type'],
                     precision=config['precision'])
    return cutpaste_model

def get_features(model,dataloader,model_config):
    layer_name = model_config['layer']
    pool = model_config['pool']
    images, labels = next(iter(dataloader))
    model = get_feature_extraction_model(model,layer_name)
    outputs_inner = extract_features(model, images.to('cpu'), layer_name, pooling=['avg', pool])
    data_mats_orig = torch.empty((outputs_inner.shape[1], len(dataloader.dataset))).to('cpu')
    gt = torch.zeros(len(dataloader.dataset))
    count=0
    with torch.no_grad():
        data_idx = 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to('cpu'), labels.to('cpu')
            num_samples = images.shape[0]
            outputs = extract_features(model, images.to('cpu'), layer_name, pooling=['avg', pool])
            oi = torch.squeeze(outputs)
            data_mats_orig[:, data_idx:data_idx + num_samples] = oi.transpose(1, 0)
            data_idx += num_samples
            gt[count:count + num_samples] = labels
        return data_mats_orig,gt

def fit_pca_kernel(data_mats_orig,pca_thresholds):
    pca_kernel = pca(data_mats_orig, pca_thresholds)
    return pca_kernel

def save_torch_model(model, config, path):
    path = os.path.join(config['output_path'],path)
    torch.save({'state_dict': model.state_dict()}, path)
    print("Saved the model at following path : {}".format(path))
    
def train(dataset, config):
    model_config = config['model']
    model = get_base_model(model_config['name'])
    if config['model']['feature_extractor'] == 'simsiam':
        model = train_simsiam(model, dataset, config)
    elif config['model']['feature_extractor'] == 'cutpaste':
        model = train_cutpaste(model, dataset, config)
    else:
        model = model.load_pretrained_model()
        
    save_torch_model(model, config, os.path.join(config['model']['feature_extractor']+'_'+
                                                 config['model']['name']+'_'+config['dataset']['category_type']+'.pth.tar'))
    return model
