# Text Generation Intel® Transfer Learning Tool CLI Example

## Fine Tuning Using Your Own Dataset

The example below shows how to fine tune a Pytorch text generation model using your own
dataset in the .JSON format.

The `--dataset-dir` argument is the path to the directory where your dataset is located, and the
`--dataset-file` is the name of the .JSON file to load from that directory. The `instruction-key`,
`--context-key`, and `response-key` are the keys in the JSON file that make up the dataset schema.

This example is downloading a subset of the [Code Alpaca](https://github.com/sahil280114/codealpaca)
dataset, where each record of the dataset contains text fields for "instruction", "input", and "output". For this dataset,
"instruction", "input", and "output" would map to `instruction-key`,`--context-key`, and `response-key`, respectively.

```bash
# Create dataset and output directories
export DATASET_DIR=/tmp/data
export OUTPUT_DIR=/tmp/output
mkdir -p ${DATASET_DIR}
mkdir -p ${OUTPUT_DIR}

# Download and extract the dataset
wget -P ${DATASET_DIR} https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_2k.json

# Train distilgpt2 using our dataset file
tlt train \
    --framework pytorch \
    --dataset_dir ${DATASET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --epochs 3 \
    --dataset-file code_alpaca_2k.json \
    --model-name distilgpt2 \
    --instruction-key instruction \
    --context-key input \
    --response-key output \
    --prompt-with-context "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request." \
    --prompt-without-context "Below is an instruction that describes a task. write a response that appropriately completes the request."

# Evaluate the model exported after training
# Note that your --model-dir path may vary, since each training run creates a new directory
tlt eval \
    --dataset_dir ${DATASET_DIR} \
    --dataset-file code_alpaca_2k.json \
    --model-dir ${OUTPUT_DIR}/distilgpt2/1 \
    --prompt-with-context "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request." \
    --prompt-without-context "Below is an instruction that describes a task. write a response that appropriately completes the request."

# Generate text on our trained model
tlt generate \
    --model-dir ${OUTPUT_DIR}/distilgpt2/1 \
    --prompt "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a function that sorts the following list.\n\n### Context:\n[3, 2, 1]\n\n### Response:\n" \
    --repetition-penalty 6.0
```

## Citation

```
@misc{codealpaca,
  author = {Sahil Chaudhary},
  title = {Code Alpaca: An Instruction-following LLaMA model for code generation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sahil280114/codealpaca}},
}
```
