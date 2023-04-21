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
import yaml
import argparse

from anomaly_detection_wl import train, get_dataset


def main(config):
    dataset_config = config['dataset']
    
    dataset = get_dataset(os.path.join(dataset_config['root_dir'],dataset_config['category_type']), 
                        dataset_config['image_size'],dataset_config['batch_size'])
    
    model = train(dataset, config)

    return model
        
if __name__ == "__main__":
    """Base function for anomaly detection workload"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
        
    root_dir = config['dataset']['root_dir']
    category = config['dataset']['category_type']
    all_categories = [os.path.join(root_dir, o).split('/')[-1] for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,o))]
    all_categories.sort()
    print("Precision has been set to {}".format(config['precision']))
    if category == 'all':
        for category in all_categories:
            print("\n#### Fine tuning on "+category.upper()+ " dataset started ##########\n")
            config['dataset']['category_type'] = category
            model = main(config)
            print("\n#### Fine tuning on "+category.upper()+ " dataset completed ########\n")
    else:
        print("\n#### Fine tuning on "+category.upper()+ " dataset started ##########\n")
        model = main(config)
        print("\n#### Fine tuning on "+category.upper()+ " dataset completed ########\n")
