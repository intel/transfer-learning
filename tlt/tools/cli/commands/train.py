#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import click
import inspect
import sys

from tlt.distributed import TLT_DISTRIBUTED_DIR
from tlt.utils.types import UseCaseType


@click.command()
@click.option("--framework", "-f",
              required=False,
              default="tensorflow",
              type=click.Choice(['tensorflow', 'pytorch']),
              help="Deep learning framework [default: tensorflow]")
@click.option("--model-name", "--model_name",
              required=True,
              type=str,
              help="Name of the model to use")
@click.option("--use-case", "--use_case",
              required=False,
              type=click.Choice(['image_classification', 'anomaly_detection', 'image_anomaly_detection',
                                 'text_classification', 'text_generation']),
              help="Specify the use case if the model is supported by multiple use cases")
@click.option("--output-dir", "--output_dir",
              required=True,
              type=click.Path(dir_okay=True, file_okay=False),
              help="Output directory for saved models, logs, checkpoints, etc")
@click.option("--dataset-dir", "--dataset_dir",
              required=True,
              type=click.Path(dir_okay=True, file_okay=False),
              help="Dataset directory for a custom dataset, or if a dataset name "
                   "and catalog are being provided, the dataset directory is the "
                   "location where the dataset will be downloaded.")
@click.option("--dataset-file", "--dataset_file",
              required=False,
              type=str,
              help="Name of a file in the dataset directory to load. Used for loading a .csv file for text "
                   "classification fine tuning, or a json / txt file for text generation")
@click.option("--delimiter",
              required=False,
              type=str,
              default=",",
              help="Delimiter used when loading a dataset from a csv file. [default: ,]")
@click.option("--class-names", "--class_names",
              required=False,
              type=str,
              help="Comma separated string of class names for a text classification dataset being loaded from .csv")
@click.option("--dataset-name", "--dataset_name",
              required=False,
              type=str,
              help="Name of the dataset to use from a dataset catalog. Not yet supported for Text Generation.")
@click.option("--dataset-catalog", "--dataset_catalog",
              required=False,
              type=click.Choice(['tf_datasets', 'torchvision', 'huggingface']),
              help="Name of a dataset catalog for a named dataset (Options: "
                   "tf_datasets, torchvision, huggingface). If a dataset name is provided "
                   "and no dataset catalog is given, it will default to use tf_datasets for a TensorFlow "
                   "model, torchvision for PyTorch CV models, and huggingface datasets for HuggingFace models. "
                   "Not yet supported for Text Generation.")
@click.option("--epochs",
              default=1,
              type=click.IntRange(min=1),
              help="Number of training epochs [default: 1]")
@click.option("--init-checkpoints", "--init_checkpoints",
              required=False,
              type=click.Path(dir_okay=True),
              help="Optional path to checkpoint weights to load to resume training. If the path provided is a "
                   "directory, the latest checkpoint from the directory will be used.")
@click.option("--add-aug", "--add_aug",
              type=click.Choice(['hvflip', 'hflip', 'vflip', 'rotate', 'zoom']),
              multiple=True,
              default=[],
              help="Choice of data augmentation to be applied during training.")
@click.option("--ipex_optimize", "--ipex-optimize",
              required=False,
              type=click.BOOL,
              is_flag=True,
              help="Boolean option to optimize model with Intel Extension for PyTorch.")
@click.option("--distributed", "-d",
              required=False,
              type=click.BOOL,
              is_flag=True,
              help="Boolean option to trigger a distributed training job.")
@click.option("--nnodes",
              required=False,
              default=1,
              type=click.IntRange(min=1),
              help="Number of nodes to run the training job [default: 1]")
@click.option("--nproc_per_node", "--nproc-per-node",
              required=False,
              default=1,
              type=click.IntRange(min=1),
              help="Number of processes per node for the distributed training job [default: 1]")
@click.option("--hostfile",
              required=False,
              default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="hostfile with a list of nodes to run distributed training.")
@click.option("--early-stopping", "--early_stopping",
              type=click.BOOL,
              default=False,
              is_flag=True,
              help="Enable early stopping if convergence is reached while training (bool)")
@click.option("--lr-decay", "--lr_decay",
              type=click.BOOL,
              default=False,
              is_flag=True,
              help="If lr_decay is True and do_eval is True, learning rate decay on the validation loss is applied at "
              "the end of each epoch.")
@click.option("--use-horovod", "--use_horovod",
              required=False,
              type=click.BOOL,
              is_flag=True,
              help="Use horovod instead of default MPI")
@click.option("--hvd-start-timeout", "--hvd_start_timeout",
              type=click.IntRange(min=1),
              default=30,
              help="Horovodrun has to perform all the checks and start the processes before the specified timeout. "
              "The default value is 30 seconds.  Alternatively, The environment variable HOROVOD_START_TIMEOUT can "
              "also be used to specify the initialization timeout. Currently only supports PyTorch.")
@click.option("--simsiam",
              required=False,
              type=click.BOOL,
              is_flag=True,
              help="Use SimSiam anomaly detection")
@click.option("--cutpaste",
              required=False,
              type=click.BOOL,
              is_flag=True,
              help="Use CutPaste anomaly detection")
@click.option("--instruction-key", "--instruction_key",
              required=False,
              type=str,
              help="The instruction key from the dataset file used to determine the dataset schema")
@click.option("--context-key", "--context_key",
              required=False,
              type=str,
              help="The context key from the dataset file used to determine the dataset schema")
@click.option("--response-key", "--response_key",
              required=False,
              type=str,
              help="The response key from the dataset file used to determine the dataset schema")
@click.option("--prompt-with-context", "--prompt_with_context",
              required=False,
              type=str,
              default="Below is an instruction that describes a task, paired with an input that provides further"
                      " context. Write a response that appropriately completes the request.\n",
              help="Prompt with added context used to build the prompt dictionary")
@click.option("--prompt-without-context", "--prompt_without_context",
              required=False,
              type=str,
              default="Below is an instruction that describes a task. Write a response that appropriately completes"
                      " the request.\n",
              help="Prompt without added context used to build the prompt dictionary")
@click.option("--lora-rank", "--lora_rank",
              required=False,
              type=int,
              default=8,
              help="LoRA rank parameter (default: 8)")
@click.option("--lora-alpha", "--lora_alpha",
              required=False,
              type=int,
              default=32,
              help="LoRA alpha parameter (default: 32)")
@click.option("--lora-dropout", "--lora_dropout",
              required=False,
              type=float,
              default=0.05,
              help="LoRA dropout parameter (default: 0.05)")
def train(framework, model_name, use_case, output_dir, dataset_dir, dataset_file, delimiter, class_names, dataset_name,
          dataset_catalog, epochs, init_checkpoints, add_aug, early_stopping, lr_decay, ipex_optimize, distributed,
          nnodes, nproc_per_node, hostfile, use_horovod, hvd_start_timeout, simsiam, cutpaste, instruction_key,
          context_key, response_key, prompt_with_context, prompt_without_context, lora_rank, lora_alpha, lora_dropout):
    """
    Trains the model
    """
    session_log = {}  # Initialize an empty dictionary to store information about current training session
    session_verbose = ""

    session_log["model_name"] = model_name
    session_log["framework"] = framework
    session_log["epochs"] = epochs
    session_log["dataset_dir"] = dataset_dir
    session_log["output_directory"] = output_dir

    session_verbose += "Model name: {}\n".format(model_name)
    session_verbose += "Framework: {}\n".format(framework)

    if dataset_name:
        session_verbose += "Dataset name: {}\n".format(dataset_name)
        session_log["dataset_name"] = dataset_name
        if dataset_catalog:
            session_verbose += "Dataset catalog: {}\n".format(dataset_catalog)
            session_log["dataset_catalog"] = dataset_catalog
    session_verbose += "Training epochs: {}\n".format(epochs)

    if init_checkpoints:
        session_verbose += "Initial checkpoints: {}\n".format(init_checkpoints)
        session_log["init_checkpoints"] = init_checkpoints

    if add_aug:
        session_log["add_aug"] = add_aug

    session_verbose += "Dataset dir: {}\n".format(dataset_dir)

    if dataset_file:
        session_verbose += "Dataset file: {}\n".format(dataset_file)
        session_log["dataset_file"] = dataset_file
    if class_names:
        class_names = class_names.split(",")
        session_verbose += "Class names: {}\n".format(class_names)
        session_log["class_names"] = class_names
    if early_stopping:
        session_log["early_stopping"] = early_stopping
        session_verbose += "Early Stopping: {}\n".format(early_stopping)
    if lr_decay:
        session_log["lr_decay"] = lr_decay
        session_verbose += "lr_decay: {}\n".format(lr_decay)

    session_verbose += "Output directory: {}\n".format(output_dir)
    if distributed:
        session_verbose += "Distributed: {}\n".format(distributed)
        session_verbose += "Number of nodes: {}\n".format(nnodes)
        session_verbose += "Number of processes per node: {}\n".format(nproc_per_node)
        session_verbose += "hostfile: {}\n".format(hostfile)
        session_log["distibuted"] = distributed
        session_log["nnodes"] = nnodes
        session_log["nproc_per_node"] = nproc_per_node
        session_log["hostfile"] = hostfile

    print(session_verbose, flush=True)

    # Validate distributed inputs, if given
    if distributed:
        if hostfile is None:
            # TODO: Logic to continute distributed training on single (current) node
            sys.exit("Error: Specify the hostfile with \'--hostfile\' flag")

    from tlt.models import model_factory
    from tlt.datasets import dataset_factory
    # Get the model
    try:
        model = model_factory.get_model(model_name, framework, use_case)
    except Exception as e:
        sys.exit("Error while getting the model (model name: {}, framework: {}):\n{}".format(
            model_name, framework, str(e)))

    print("Use case: {}\n".format(model.use_case), flush=True)

    # Get the dataset
    try:
        if not dataset_name and not dataset_catalog:
            if model.use_case == UseCaseType.TEXT_CLASSIFICATION:
                if not dataset_file:
                    raise ValueError("Loading a text classification dataset requires --dataset-file to specify the "
                                     "file name of the .csv file to load from the --dataset-dir.")
                if not class_names:
                    raise ValueError("Loading a text classification dataset requires --class-names to specify a list "
                                     "of the class labels for the dataset.")
                elif len(class_names) == 0:
                    raise ValueError("Loading a text classification dataset requires --class-names to specify a list "
                                     "of the class labels of which the len > 0")
                dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework, dataset_name,
                                                       class_names=class_names, csv_file_name=dataset_file,
                                                       delimiter=delimiter)
            elif model.use_case == UseCaseType.TEXT_GENERATION:
                dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework,
                                                       dataset_file=dataset_file)

            else:
                dataset = dataset_factory.load_dataset(dataset_dir, model.use_case, model.framework)
        else:
            dataset = dataset_factory.get_dataset(dataset_dir, model.use_case, model.framework, dataset_name,
                                                  dataset_catalog)
        # TODO: get extra configs like batch size and maybe this doesn't need to be a separate call
        if framework in ['tensorflow', 'pytorch']:
            if 'image_size' in inspect.getfullargspec(dataset.preprocess).args:  # For Image classification
                dataset.preprocess(image_size=model.image_size, batch_size=32, add_aug=list(add_aug))
            elif 'prompt_dict' in inspect.getfullargspec(dataset.preprocess).args:
                dataset_schema = {"instruction_key": instruction_key,
                                  "context_key": context_key,
                                  "response_key": response_key}
                prompt_dict = {"prompt_with_context": (prompt_with_context + "\n\n"
                                                       "### Instruction:\n{{{instruction_key}}}\n\n### "
                                                       "Context:\n{{{context_key}}}\n\n### "
                                                       "Response:\n{{{response_key}}}".format(**dataset_schema)),
                               "prompt_without_context": (prompt_without_context + "\n\n"
                                                          "### Instruction:\n{{{instruction_key}}}\n\n### "
                                                          "Response:\n{{{response_key}}}".format(**dataset_schema))}
                dataset.preprocess(model.hub_name, batch_size=32, prompt_dict=prompt_dict,
                                   dataset_schema=dataset_schema, concatenate=True)
            elif 'model_name' in inspect.getfullargspec(dataset.preprocess).args:  # For HF Text classification
                dataset.preprocess(model_name=model_name, batch_size=32)

            else:  # For TF Text classification
                dataset.preprocess(batch_size=32)
            dataset.shuffle_split()
    except Exception as e:
        sys.exit("Error while getting the dataset (dataset dir: {}, use case: {}, framework: {}, "
                 "dataset name: {}, dataset_catalog: {}):\n{}".format(dataset_dir, model.use_case, model.framework,
                                                                      dataset_name, dataset_catalog, str(e)))

    if ipex_optimize and framework != 'pytorch':
        sys.exit("ipex_optimize is only supported for pytorch training\n")

    # Train the model using the dataset
    if framework == 'pytorch':
        try:
            if model.use_case == UseCaseType.IMAGE_ANOMALY_DETECTION:
                model.train(dataset, output_dir=output_dir, epochs=epochs, initial_checkpoints=init_checkpoints,
                            simsiam=simsiam, cutpaste=cutpaste)
            elif model.use_case == UseCaseType.TEXT_GENERATION:
                model.train(dataset, output_dir, epochs=epochs, lora_rank=lora_rank, lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout, init_checkpoints=init_checkpoints, ipex_optimize=ipex_optimize)
            else:
                model.train(dataset, output_dir=output_dir, epochs=epochs, initial_checkpoints=init_checkpoints,
                            early_stopping=early_stopping, lr_decay=lr_decay, ipex_optimize=ipex_optimize,
                            distributed=distributed, hostfile=hostfile, nnodes=nnodes, nproc_per_node=nproc_per_node,
                            use_horovod=use_horovod, hvd_start_timeout=hvd_start_timeout)
        except Exception as e:
            sys.exit("There was an error during model training:\n{}".format(str(e)))

    # Test for tensorflow
    else:
        try:
            model.train(dataset, output_dir=output_dir, epochs=epochs, initial_checkpoints=init_checkpoints,
                        early_stopping=early_stopping, lr_decay=lr_decay, distributed=distributed, hostfile=hostfile,
                        nnodes=nnodes, nproc_per_node=nproc_per_node, use_horovod=use_horovod)
        except Exception as e:
            sys.exit("There was an error during model training:\n{}".format(str(e)))

    if distributed:
        # Cleanup the saved objects
        import os
        for file_name in ["torch_saved_objects.obj", "hf_saved_objects.obj"]:
            if file_name in os.listdir(TLT_DISTRIBUTED_DIR):
                os.remove(os.path.join(TLT_DISTRIBUTED_DIR, file_name))

    # Save the trained model
    try:
        log_output = model.export(output_dir)
    except Exception as e:
        sys.exit("There was an error when saving the model:\n{}".format(str(e)))
    # Save the log file
    try:
        import os
        import json
        json_filename = os.path.join(log_output, "session_log.json")
        session_log["log_path"] = log_output
        json_object = json.dumps(session_log, indent=4)
        with open(json_filename, "w") as outfile:
            outfile.write(json_object)
    except Exception as e:
        print("WARNING: There was an error when saving the session log file"
              "file:\n{}".format(str(e)))
    try:
        if str(model.use_case) == "text_generation":
            if not os.path.exists(os.path.join(dataset_dir, "dataset_schema.json")):
                ds_file = os.path.join(dataset_dir, "dataset_schema.json")
                json_object = json.dumps(dataset_schema, indent=4)
                with open(ds_file, "w") as outfile:
                    outfile.write(json_object)
    except Exception as e:
        print("WARNING: Unable to save dataset_schema.json " + str(e))
