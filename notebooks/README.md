# Transfer Learning Notebooks

## Environment setup and running the notebooks

Use the [setup instructions](setup.md) to install the dependencies required to run the notebooks.

This directory has Jupyter notebooks that demonstrate transfer learning with
and without Intel® Transfer Learning Tool. All of the notebooks use models from public model repositories
and leverage optimized libraries [Intel-optimized TensorFlow](https://pypi.org/project/intel-tensorflow/)
and [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch).

## Intel® Transfer Learning Tool Tutorial Notebooks

| Notebook | Domain: Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [BERT Text Classification with TensorFlow using the Intel® Transfer Learning Tool](/notebooks/text_classification/tlt_api_tf_text_classification) | NLP: Text Classification | TensorFlow and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to fine tune a BERT model from Hugging Face using text classification datasets. |
| [BERT Text Classification with PyTorch using the Intel® Transfer Learning Tool](/notebooks/text_classification/tlt_api_pyt_text_classification) | NLP: Text Classification | PyTorch and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to fine tune a BERT model from Hugging Face using text classification datasets. |
| [Image Classification with TensorFlow using Intel® Transfer Learning Tool](/notebooks/image_classification/tlt_api_tf_image_classification) | CV: Image Classification | TensorFlow and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for image classification using a TensorFlow model. |
| [Image Classification with TensorFlow using Graph Optimization and Intel® Transfer Learning Tool](/notebooks/image_classification/tlt_api_tf_image_classification) | CV: Image Classification | TensorFlow and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning with graph optimization that increases throughput for image classification using a TensorFlow model. |
| [Image Classification with PyTorch using Intel® Transfer Learning Tool](/notebooks/image_classification/tlt_api_pyt_image_classification) | CV: Image Classification | PyTorch and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for image classification using a PyTorch model. |
| [Image Anomaly Detection with PyTorch using Intel® Transfer Learning Tool](/notebooks/image_anomaly_detection/tlt_api_pyt_anomaly_detection) | CV: Image Anomaly Detection| PyTorch and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do feature extraction and pca analysis using a single function for image anomaly detection using a Torchvision model. |

## Native Framework Transfer Learning Notebooks

| Notebook | Domain: Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [BERT SQuAD fine tuning with TF Hub](/notebooks/question_answering/tfhub_question_answering) | NLP: Question Answering | TensorFlow | Demonstrates BERT fine tuning using scripts from the [TensorFlow Model Garden](https://github.com/tensorflow/models) and the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The notebook allows for selecting a BERT large or BERT base model from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [BERT Text Classification with TF Hub](/notebooks/text_classification/tfhub_text_classification) | NLP: Text Classification | TensorFlow | Demonstrates BERT binary text classification fine tuning using the [IMDb movie review dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) and multiclass text classification fine tuning using the [AG News datasets](https://www.tensorflow.org/datasets/catalog/ag_news_subset) from [TensorFlow Datasets](https://www.tensorflow.org/datasets) or a custom dataset (for binary classification). The notebook allows for selecting a BERT encoder (BERT large, BERT base, or small BERT) to use along with a preprocessor from [TF Hub](https://tfhub.dev). The fine tuned model is evaluated and exported as a saved model. |
| [Text Classifier fine tuning with PyTorch & Hugging Face](/notebooks/text_classification/pytorch_text_classification) | NLP: Text Classification | PyTorch |Demonstrates fine tuning [Hugging Face models](https://huggingface.co/models) to do sentiment analysis using the [IMDb movie review dataset from Hugging Face Datasets](https://huggingface.co/datasets/imdb) or a custom dataset with [Intel® Extension for PyTorch*](https://github.com/intel/intel-extension-for-pytorch) |
| [Image Classification with TF Hub](/notebooks/image_classification/tf_image_classification) | CV: Image Classification | TensorFlow | Demonstrates transfer learning with multiple [TF Hub](https://tfhub.dev) image classifiers, TF datasets, and custom image datasets |
| [Image Classification with PyTorch & Torchvision](/notebooks/image_classification/pytorch_image_classification) | CV: Image Classification | PyTorch | Demonstrates transfer learning with multiple [Torchvision](https://pytorch.org/vision/stable/index.html) image classification models, Torchvision datasets, and custom datasets |

## Transfer Learning Tool End-to-End Pipelines

| Notebook | Domain: Use Case | Framework| Description |
| ---------| ---------|----------|-------------|
| [Document-Level Sentiment Analysis (SST2) using PyTorch and the Intel® Transfer Learning Tool API](/notebooks/e2e_workflows/Document_Level_Sentiment_Analysis.ipynb) | NLP: Text Classification | PyTorch and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for text classification using a PyTorch model from Hugging Face for a document-level sentiment analysis workflow. |
| [Medical Imaging Classification (Colorectal histology) using TensorFlow and the Intel® Transfer Learning Tool API](/notebooks/e2e_workflows/Medical_Imaging_Classification.ipynb) | CV: Image Classification | TensorFlow and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for image classification using a TensorFlow model for a medical imaging classification application. |
| [Remote Sensing Image Scene Classification (Resisc) using TensorFlow and the Intel® Transfer Learning Tool API](/notebooks/e2e_workflows/Remote_Sensing_Image_Scene_Classification.ipynb) | CV: Image Classification | TensorFlow and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for image classification using a TensorFlow model for a remote sensing image scene classification application. |
| [Multimodal Cancer Detection using TensorFlow, PyTorch, and the Intel® Transfer Learning Tool API](/notebooks/e2e_workflows/Multimodal_Cancer_Detection.ipynb) | CV: Image Classification<br>NLP: Text Classification | TensorFlow, PyTorch, and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for a late fusion multimodal ensemble application using both NLP and computer vision models from PyTorch and Tensorflow, respectively. |
| [Anomaly Detection with PyTorch using Intel® Transfer Learning Tool](/notebooks/e2e_workflows/Anomaly_Detection_MVTec.ipynb) | CV: Image Anomaly Detection | PyTorch and the Intel Transfer Learning Tool API | Demonstrates how to use the Intel Transfer Learning Tool API to do feature extraction and pca analysis using dedicated function calls for image anomaly detection using a Torchvision model. |

## Performance Comparison Notebooks

| Notebook | Domain: Use Case | Framework| Description |
| ---------| -----------------|----------|-------------|
| [Performance Comparison: Image Classification Transfer Learning with TensorFlow and the Intel Transfer Learning Tool](/notebooks/performance/tf_image_classification_performance.ipynb) | CV: Image Classification | TensorFlow and the Intel Transfer Learning Tool API | Compares training and evaluation metrics and performance for image classification transfer learning using TensorFlow libraries and the Intel Transfer Learning Tool. |
| [Performance Comparison: Text Classification Transfer Learning with Hugging Face and the Intel Transfer Learning Tool](/notebooks/performance/hf_text_classification_performance.ipynb) | NLP: Text Classification | Hugging Face, PyTorch, and the Intel Transfer Learning Tool API | Compares training and evaluation metrics for text classification transfer learning using the Hugging Face Trainer and the Intel Transfer Learning Tool. |
