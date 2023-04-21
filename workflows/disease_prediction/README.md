# Workflow purpose
The vision fine-tuning (transfer learning) and inference workflow demonstrates Image Classification workflows/pipelines using Intel® Transfer Learning Tool to be run along with Intel-optimized software represented using toolkits, domain kits, packages, frameworks and other libraries for effective use of Intel hardware leveraging Intel's AI instructions for fast processing and increased performance. The workflows can be easily used by applications or reference kits showcasing usage.

The workflow supports:
```
Image Classification Finetuning
Image Classification Inference
```

# Get Started
## Deploy the test environment

### Create a new python environment
```shell
conda create -n <env name> python=3.9
conda activate <env name>
```

### Install package for running hf-finetuning-inference-nlp-workflows
```shell
pip install -r requirements.txt
```

## Running 

```shell
python src/run.py --config_file config/config.yaml
```

Note: Configure the right configurations in the config.yaml

The 'config.yaml' file includes the following parameters:

- args:
  - dataset_dir: contains the path for dataset_dir
  - dtype_ft: Datatype of Finetuning model(default: fp32 , options fp32, bf16)
  - dtype_inf: Datatype of model when running infernce ( default: fp32, options fp32, bf16)
  - finetune_output: saves results of finetuning in a yaml file
  - inference_output : saves the results of the model on test data in the yaml file
  - model: Pretrained model name (default resnetv150)
  - finetune: runs vision fine-tuning
  - inference: runs inference only if set to true , if false finetunes the model before inference
  - saved_model_dir: Directory where trained model gets saved
- training_args:
  - batch_size: Batch size for training ( default 32)
  - bf16: Enable BF16 by default
  - epochs: Number of epochs for training
  - output_dir: Output of training model
