# Transfer Learning Notebooks

This directory has Jupyter notebooks that demonstrate transfer learning with
models from public model repositories using
[Intel-optimized TensorFlow](https://pypi.org/project/intel-tensorflow/)
and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

## Natural Language Processing

| Notebook | Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [BERT SQuAD fine tuning with TF Hub](/notebooks/question_answering/tfhub_question_answering) | Question Answering | TensorFlow | Demonstrates BERT fine tuning using scripts from the [TensorFlow Model Garden](https://github.com/tensorflow/models) and the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The notebook allows for selecting a BERT large or BERT base model from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [BERT Binary Text Classification with TF Hub](/notebooks/text_classification/tfhub_text_classification) | Text Classification | TensorFlow | Demonstrates BERT binary text classification fine tuning using the [IMDb movie review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) from [TensorFlow Datasets](https://www.tensorflow.org/datasets) or a custom dataset. The notebook allows for selecting a BERT encoder (BERT large, BERT base, or small BERT) to use along with a preprocessor from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [Text Classifier fine tuning with PyTorch & Hugging Face](/notebooks/text_classification/pytorch_text_classification) | Text Classification | PyTorch |Demonstrates fine tuning [Hugging Face models](https://huggingface.co/models) to do sentiment analysis using the [IMDb movie review dataset from Hugging Face Datasets](https://huggingface.co/datasets/imdb) or a custom dataset with [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) |

## Computer Vision

| Notebook | Use Case |  Framework | Description |
| ---------| ---------|------------|-------------|
| [Transfer Learning for Image Classification with TF Hub](/notebooks/image_classification/tf_image_classification) | Image Classification | TensorFlow | Demonstrates transfer learning with multiple [TF Hub](https://tfhub.dev) image classifiers, TF datasets, and custom image datasets |
| [Transfer Learning for Image Classification with TF using the TLK API](/notebooks/image_classification/tlk_api_tf_image_classification) | Image Classification | TensorFlow and the TLK API | Demonstrates how to use the TLK API to do transfer learning for image classification using a TensorFlow model. |
| [Transfer Learning for Image Classification with PyTorch & torchvision](/notebooks/image_classification/pytorch_image_classification) | Image Classification | PyTorch | Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) image classification models, torchvision datasets, and custom datasets |
| [Transfer Learning for Image Classification with PyTorch using the TLK API](/notebooks/image_classification/tlk_api_pyt_image_classification) | Image Classification | PyTorch and the TLK API | Demonstrates how to use the TLK API to do transfer learning for image classification using a PyTorch model. |
| [Transfer Learning for Object Detection with PyTorch & torchvision](/notebooks/object_detection/pytorch_object_detection) | Object Detection | PyTorch |Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) object detection models, a public image dataset, and a customized torchvision dataset |

## Environment setup and running the notebooks

Use the instructions below to install the dependencies required to run the
[PyTorch](#pytorch-environment) or [TensorFlow](#tensorflow-environment) notebooks.

System Requirements:
* Ubuntu 18.04
* Python 3.7, 3.8, 3.9, or 3.10
* git

### PyTorch environment

1. Get a clone of this Transfer Learning Kit repository from GitHub:
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git transfer_learning_kit
   export TLK_REPO=$(pwd)/transfer_learning_kit
   ```
2. Create a Python3 virtual environment and install required packages.

   You can use virtualenv:
   ```
   python3 -m venv intel-pyt-venv
   source intel-pyt-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-pyt python=3.8
   conda activate intel-pyt
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   pip install --upgrade pip
   pip install -r ${TLK_REPO}/notebooks/pytorch_requirements.txt
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the dataset to
   the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   If the `DATASET_DIR` and `OUTPUT_DIR` variables are not defined, the notebooks will
   default to use `~/dataset` and `~/output`.
   ```
   export DATASET_DIR=<directory to download the dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${TLK_REPO}/notebooks
   PYTHONPATH=${TLK_REPO} jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run any of the
   PyTorch notebooks.

### TensorFlow environment

1. Get a clone of this Transfer Learning Kit repository from GitHub:
   ```
   git clone https://github.com/intel-innersource/frameworks.ai.transfer-learning.git transfer_learning_kit
   export TLK_REPO=$(pwd)/transfer_learning_kit
   ```
2. Create a Python3 virtual environment and install required packages.

   You can use virtualenv:
   ```
   python3 -m venv intel-tf-venv
   source intel-tf-venv/bin/activate
   ```
   Or Anaconda:
   ```
   conda create -n intel-tf python=3.8
   conda activate intel-tf
   ```
   Then, from inside the activated virtualenv or conda environment run these steps:
   ```
   pip install --upgrade pip
   pip install -r ${TLK_REPO}/notebooks/tensorflow_requirements.txt
   pip install --no-deps tensorflow-text==2.8.2
   ```
3. Set environment variables for the path to the dataset folder and an output directory.
   The dataset and output directories can be empty. The notebook will download the dataset to
   the dataset directory, if it is empty. Subsequent runs will reuse the dataset.
   If the `DATASET_DIR` and `OUTPUT_DIR` variables are not defined, the notebooks will
   default to use `~/dataset` and `~/output`.
   ```
   export DATASET_DIR=<directory to download the dataset>
   export OUTPUT_DIR=<output directory for the saved model>

   mkdir -p $DATASET_DIR
   mkdir -p $OUTPUT_DIR
   ```
4. Navigate to the notebook directory in your clone of the model zoo repo, and then start the
   [notebook server](https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server):
   ```
   cd ${TLK_REPO}/notebooks
   PYTHONPATH=${TLK_REPO} jupyter notebook --port 8888
   ```
5. Copy and paste the URL from the terminal to your browser to view and run any of the
   TensorFlow notebooks.
