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
| [BERT Binary Text Classification with TF Hub using the Intel® Transfer Learning Tool](/notebooks/text_classification/tlt_api_tf_text_classification) | Text Classification | TensorFlow and the TLT API | Demonstrates how to use the TLT API to fine tune a BERT model from TF Hub using binary text classification datasets. |

## Computer Vision

| Notebook | Use Case |  Framework | Description |
| ---------| ---------|------------|-------------|
| [Image Classification with TF Hub](/notebooks/image_classification/tf_image_classification) | Image Classification | TensorFlow | Demonstrates transfer learning with multiple [TF Hub](https://tfhub.dev) image classifiers, TF datasets, and custom image datasets |
| [Image Classification with TensorFlow using Intel® Transfer Learning Tool](/notebooks/image_classification/tlt_api_tf_image_classification) | Image Classification | TensorFlow and the TLT API | Demonstrates how to use the TLT API to do transfer learning for image classification using a TensorFlow model. |
| [Image Classification with TensorFlow using Graph Optimization and Intel® Transfer Learning Tool](/notebooks/image_classification/tlt_api_tf_image_classification) | Image Classification | TensorFlow and the TLT API | Demonstrates how to use the TLT API to do transfer learning with graph optimization that increases throughput for image classification using a TensorFlow model. |
| [Image Classification with PyTorch & torchvision](/notebooks/image_classification/pytorch_image_classification) | Image Classification | PyTorch | Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) image classification models, torchvision datasets, and custom datasets |
| [Image Classification with PyTorch using Intel® Transfer Learning Tool](/notebooks/image_classification/tlt_api_pyt_image_classification) | Image Classification | PyTorch and the TLT API | Demonstrates how to use the TLT API to do transfer learning for image classification using a PyTorch model. |
| [Object Detection with PyTorch & torchvision](/notebooks/object_detection/pytorch_object_detection) | Object Detection | PyTorch |Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) object detection models, a public image dataset, and a customized torchvision dataset |
| [Video Classification with PyTorch & torchvision](/notebooks/video_classification/pytorch_video_classification) | Video Classification | PyTorch | Demonstrates transfer learning with multiple [torchvision](https://pytorch.org/vision/stable/index.html) video classification models using the HMBD51 torchvision compatible dataset |


## Environment setup and running the notebooks

Use the [setup instructions](setup.md) to install the dependencies required to run the
[PyTorch](setup.md#pytorch-environment) or [TensorFlow](setup.md#tensorflow-environment) notebooks.
