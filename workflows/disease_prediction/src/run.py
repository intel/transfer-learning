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

import yaml
import argparse
import os
import sys
from os import path

root_folder = path.dirname(path.abspath(__file__))
sys.path.insert(0, path.join(root_folder, "../../../"))
print(sys.path)

from vision_wl import train_vision_wl, run_inference, collect_class_labels, load_model, run_inference_per_patient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_dir = config['args']['dataset_dir']
    train_dataset_dir = os.path.join(dataset_dir, "train")
    test_dataset_dir = os.path.join(dataset_dir, "test")
    # output_dir = config['args']['output_dir']
    output_dir = config['training_args']['output_dir']
    batch_size = config['training_args']['batch_size']
    epochs = config['training_args']['epochs']
    bf16 = config['training_args']['bf16']
    model_name = config['args']['model']

    # do_predict = config['training_args']['do_predict']
    do_predict = config['args']['inference']
    do_predict_per_patient = config['args']['inference_per_patient']
    # do_train = config['training_args']['do_train']
    do_train = config['args']['finetune']

    saved_model_dir = config['args']['saved_model_dir']

    # output_file_test = config['args']['output_file_test_dir']
    output_file_test = config['args']['inference_output']

    # output_file_train = config['args']['output_file_train_dir']
    output_file_train = config['args']['finetune_output']

    # this is one is used for place holder 
    vision_int8_inference = 'vision_int_8.yaml' # config['inference_args']['int8_inference']
    class_labels = collect_class_labels(train_dataset_dir)
    if (do_train):
        model, history, dict_metrics = train_vision_wl(train_dataset_dir,
                                                       output_dir, model_name,
                                                       batch_size, epochs, bf16=bf16)
        run_inference(train_dataset_dir, saved_model_dir, class_labels,
                      model_name, vision_int8_inference, output_file_train)

    if (do_predict):
        run_inference(test_dataset_dir, saved_model_dir, class_labels,
                      model_name, vision_int8_inference, output_file_test)
    if (do_predict_per_patient):     
        model = load_model(model_name,saved_model_dir)
        # Sample dict
        patient_dict = {'106L':[os.path.join(train_dataset_dir,"Malignant/P106_L_CM_MLO1.jpg")],\
                        '106R':[os.path.join(train_dataset_dir,"Benign/P106_R_CM_CC1.jpg"),\
                                os.path.join(train_dataset_dir,"Benign/P106_R_CM_CC2.jpg")]}
        results = run_inference_per_patient(model, patient_dict,class_labels)

if __name__ == "__main__":
    main()
