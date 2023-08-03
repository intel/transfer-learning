# Text Generation Instruction Tuning with PyTorch and the Intel® Transfer Learning Tool API

This notebook demonstrates how to use the Intel Transfer Learning Tool API to do instruction fine-tuning for 
text generation with a [large language model from Hugging Face](https://huggingface.co/models). It uses a custom
[Intel domain dataset](https://raw.githubusercontent.com/intel/intel-extension-for-transformers/1.0.1/examples/optimization/pytorch/huggingface/language-modeling/chatbot/intel_domain.json)
loaded from a json file.

The notebook includes options for bfloat16 precision training and
[Intel® Extension for PyTorch\*](https://intel.github.io/intel-extension-for-pytorch) which extends PyTorch
with optimizations for extra performance boost on Intel hardware.

The notebook performs the following steps:
1. Import dependencies and setup parameters
2. Get the model
3. Load a custom dataset
4. Generate a text completion from the pretrained model
5. Transfer learning (instruction tuning)
6. Export the saved model
7. Generate a text completion from the fine-tuned model

## Running the notebook

To run the notebook, follow the instructions to setup the [notebook environment](/notebooks/setup.md).
