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

import os
import shutil
import yaml
import tensorflow as tf
import numpy as np
import time
from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.types import FrameworkType
from PIL import Image


IMAGE_SIZE = 224


def collect_class_labels(dataset_dir):
    dataset = dataset_factory.load_dataset(dataset_dir=dataset_dir,
                                           use_case='image_classification',
                                           framework='tensorflow')
    return dataset.class_names


def quantize_model(output_dir, saved_model_dir, model):
    clean_output_folder(output_dir, 'quantized_models')

    quantization_output_dir = os.path.join(output_dir, 'quantized_models',
                                           "vision",
                                           os.path.basename(saved_model_dir))
    # Create a tuning workspace directory for INC
    root_folder = os.path.dirname(os.path.abspath(__file__))
    inc_config_file = os.path.join(root_folder, "config.yaml")

    # inc_config_file = 'vision/config.yaml'
    model.quantize(quantization_output_dir, inc_config_file)


def clean_output_folder(output_dir, model_name):
    folder_path = os.path.join(output_dir, model_name)
    if os.path.exists(folder_path):
        shutil.rmtree(os.path.join(output_dir, model_name))


def train_vision_wl(dataset_dir, output_dir, model="resnet_v1_50",
                    batch_size=32,
                    epochs=5, save_model=True, quantization=False, bf16=True):
    # Clean the output folder first
    clean_output_folder(output_dir, model)
    dict_metrics = {}
    #  Loading the model
    tstart = time.time()
    model = model_factory.get_model(model_name=model,
                                    framework=FrameworkType.TENSORFLOW)
    tend = time.time()
    print("\nModel Loading time (s): ", tend - tstart)
    # Load the dataset from the custom dataset path
    # Data loading and preprocessing #
    dataset = dataset_factory.load_dataset(dataset_dir=dataset_dir,
                                           use_case='image_classification',
                                           framework='tensorflow',
                                           shuffle_files=True)

    print("Class names:", str(dataset.class_names))
    dataset.preprocess(model.image_size, batch_size=batch_size,
                       add_aug=['hvflip', 'rotate'])
    dataset.shuffle_split(train_pct=.80, val_pct=.20)
    # Finetuning #
    tstart = time.time()
    history = model.train(dataset, output_dir=output_dir, epochs=epochs,
                          seed=10,
                          enable_auto_mixed_precision=bf16,
                          extra_layers=[1024, 512])
    tend = time.time()
    print("\nTotal Vision Finetuning time (s): ", tend - tstart)
    dict_metrics['e2e_training_time'] = tend - tstart

    metrics = model.evaluate(dataset)
    for metric_name, metric_value in zip(model._model.metrics_names, metrics):
        print("{}: {}".format(metric_name, metric_value))
        dict_metrics[metric_name] = metric_value
    print('dict_metrics:', dict_metrics)
    print('Finished Fine-tuning the vision model...')
    if save_model:
        saved_model_dir = model.export(output_dir)
    if quantization:
        print('Quantizing the model')
        quantize_model(output_dir, saved_model_dir, model)

    print("Done finetuning the vision model ............")
    return (model, history, dict_metrics)


def infer_vision_wl(model, image_location):
    image_shape = (model.image_size, model.image_size)
    image = Image.open(image_location).resize(image_shape)
    # Get the image as a np array and call predict while adding a batch
    # dimension (with np.newaxis)
    image = np.array(image)/255.0
    result = model.predict(image[np.newaxis, ...], 'probabilities')[0]
    return result


def infer_int8_vision_wl(model, image_location):
    image_shape = (IMAGE_SIZE, IMAGE_SIZE)
    image = Image.open(image_location).resize(image_shape)
    # Get the image as a np array and call predict while
    # adding a batch dimension (with np.newaxis)
    image = np.array(image)/255.0
    image = image[np.newaxis, ...].astype('float32')
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    # result = model.predict(image[np.newaxis, ...])
    # result=model.predict(image[np.newaxis, ...], 'probabilities')[0]
    output_name = list(infer.structured_outputs.keys())
    result = infer(tf.constant(image))[output_name[0]][0]
    return result

def preprocess_dataset(dataset_dir, image_size, batch_size):
    """
    Load and preprocess dataset
    """
    dataset = dataset_factory.load_dataset(dataset_dir=dataset_dir,
                                           use_case='image_classification',
                                           framework='tensorflow',
                                           shuffle_files=False)
    dataset.preprocess(image_size, batch_size)
    class_dict = reverse_map(dataset.class_names)
    return dataset, class_dict


def reverse_map(class_names):
    class_dict = {}
    i = 0
    for c in class_names:
        class_dict[i] = c
        i = i + 1
    return class_dict

def load_model(model_name, saved_model_dir):
    vision_model = model_factory.load_model(model_name, saved_model_dir,
                                            "tensorflow",
                                            "image_classification")
    return vision_model

def run_inference_per_patient(model, patient_dict,class_names):
    results = {}
    class_dict = reverse_map(class_names)
    for key, value in patient_dict.items():
        print(key, '->', value)
        results[key] = {}
        for image in value:
            pred_prob = infer_vision_wl(model,image).numpy().tolist()
            infer_result_patient = [
                {
                    "label": image.split('/')[-2],
                    "pred": class_dict[np.argmax(pred_prob).tolist()],
                    "pred_prob": pred_prob
                }
            ]
            results[key][image.split('/')[-1]] = infer_result_patient
    print(results)
    return results
    

def run_inference(test_data_dir, saved_model_dir, class_labels,
                  model_name="resnet_v1_50", vision_int8_inference=False,
                  report="output.yaml"):
    # Load the vision model
    tstart = time.time()
    vision_model_dir = saved_model_dir
    test_dir = test_data_dir
    labels = class_labels
    predictions_report_save_file = report
    predictions_report = {}
    predictions_report["metric"] = {}
    predictions_report["results"] = {}
    # Load model
    vision_model = model_factory.load_model(model_name, vision_model_dir,
                                            "tensorflow",
                                            "image_classification")

    if vision_int8_inference:
        vision_int8_model = tf.saved_model.load(vision_model_dir)

    tend = time.time()
    print("\n Vision Model Loading time: ", tend - tstart)
    # Load dataset for metric evaluation
    dataset, class_dict = preprocess_dataset(test_data_dir,
                                             vision_model.image_size, 32)
    metrics = vision_model.evaluate(dataset)
    for metric_name, metric_value in zip(vision_model._model.metrics_names,
                                         metrics):
        print("{}: {}".format(metric_name, metric_value))
        predictions_report["metric"][metric_name] = metric_value
    tstart = time.time()
    for label in os.listdir(test_dir):
        print("Infering data in folder: ", label)
        fns = os.listdir(os.path.join(test_dir, label))
        for fn in fns:
            patient_id = fn
            fn = os.path.join(os.path.join(test_dir, label, fn))
            # ------------------------
            # call inference on vision WL
            # ------------------------
            if vision_int8_inference:
                result_vision = infer_int8_vision_wl(vision_int8_model, fn)
            else:
                result_vision = infer_vision_wl(vision_model, fn)
            pred_prob = result_vision.numpy().tolist()
            infer_result_patient = [
                {
                    "label": label,
                    "pred": class_dict[np.argmax(pred_prob).tolist()],
                    "pred_prob": pred_prob
                }
            ]
            predictions_report["label"] = labels
            predictions_report["label_id"] = list(class_dict.keys())
            predictions_report["results"][patient_id] = infer_result_patient
    with open(predictions_report_save_file, 'w') as file:
        _ = yaml.dump(predictions_report, file, )
    print("Vision inference time: ", time.time() - tstart)
