# Text Generation Instruction Tuning with PyTorch and the Intel® Transfer Learning Tool API

This notebook demonstrates how to use the Intel Transfer Learning Tool API to do instruction fine-tuning for 
text generation with a [large language model from Hugging Face](https://huggingface.co/models). It uses a subset
of the [Code Alpaca](https://github.com/sahil280114/codealpaca) dataset loaded from a json file.

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

## References

Dataset Citations

<b>[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)</b> - Copyright (2023) Databricks, Inc. This dataset was developed at Databricks (https://www.databricks.com) and its use is subject to the CC BY-SA 3.0 license. Certain categories of material in the dataset include materials from the following sources, licensed under the CC BY-SA 3.0 license: Wikipedia (various pages) - https://www.wikipedia.org/ Copyright © Wikipedia editors and contributors.

```
@software{together2023redpajama,
  author = {Together Computer},
  title = {RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset},
  month = April,
  year = 2023,
  url = {https://github.com/togethercomputer/RedPajama-Data}
}
```

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
