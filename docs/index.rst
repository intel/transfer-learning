Intel® Transfer Learning Tool
=============================

Goals
-----

* To make transfer learning workflows easier for data scientists for a variety of AI use cases, frameworks, and public
  pretrained models
* To incorporate all available Intel optimizations and best practices for XPU

Features
--------

* Low-code API and no-code CLI for:
   * TensorFlow 2.9
      * Image Classification with 19 models from TFHub
      * Text Classification with 26 models from TFHub
   * PyTorch 1.11
      * Image Classification with 10 models from torchvision

.. figure:: images/features.png
   :scale: 50 %
   :alt: TLT Features

* Jupyter notebooks demonstrating:
   * 5 Computer Vision workflows
   * 3 Natural Language Processing workflows

.. csv-table::
   :header: "Notebook", "Use Case", "Framework"
   :widths: 60, 20, 20

   Image Classification with TF Hub, Image Classification, TensorFlow
   :doc:`Image Classification with TF using the Intel® Transfer Learning Tool API <notebooks/TLT_TF_Image_Classification_Transfer_Learning>`, Image Classification, TensorFlow & TLT
   Image Classification with PyTorch & torchvision, Image Classification, PyTorch
   :doc:`Image Classification with PyTorch using the Intel® Transfer Learning Tool API <notebooks/TLT_PyTorch_Image_Classification_Transfer_Learning>`, Image Classification, PyTorch & TLT
   Object Detection with PyTorch & torchvision, Object Detection, PyTorch
   BERT SQuAD fine tuning with TF Hub, Question Answering, TensorFlow
   BERT Binary Text Classification with TF Hub, Text Classification, TensorFlow
   :doc:`Text Classification with TF using the TLT API <notebooks/TLT_TF_Text_Classification_Transfer_Learning>`, Text Classification, TensorFlow & TLT
   Text Classifier fine tuning with PyTorch & Hugging Face, Text Classification, PyTorch

Models
------

.. csv-table::
   :header: TensorFlow,PyTorch
   :widths: 50, 50

   bert_en_uncased_L-12_H-768_A-12,densenet121
   bert_en_wwm_uncased_L-24_H-1024_A-16,densenet161
   efficientnet_b0,efficientnet_b0
   efficientnet_b1,efficientnet_b1
   efficientnet_b2,efficientnet_b2
   efficientnet_b3,googlenet
   efficientnet_b4,mobilenet_v2
   efficientnet_b5,resnet18
   efficientnet_b6,resnet50
   efficientnet_b7,shufflenet_v2_x1_0
   efficientnetv2-b0,
   efficientnetv2-b1,
   efficientnetv2-b2,
   efficientnetv2-b3,
   efficientnetv2-s,
   inception_v3,
   mobilenet_v2_100_224,
   nasnet_large,
   resnet_v1_50,
   resnet_v2_101,
   resnet_v2_50,
   small_bert/bert_en_uncased_L-10_H-128_A-2,
   small_bert/bert_en_uncased_L-10_H-256_A-4,
   small_bert/bert_en_uncased_L-10_H-512_A-8,
   small_bert/bert_en_uncased_L-10_H-768_A-12,
   small_bert/bert_en_uncased_L-12_H-128_A-2,
   small_bert/bert_en_uncased_L-12_H-256_A-4,
   small_bert/bert_en_uncased_L-12_H-512_A-8,
   small_bert/bert_en_uncased_L-12_H-768_A-12,
   small_bert/bert_en_uncased_L-2_H-128_A-2,
   small_bert/bert_en_uncased_L-2_H-256_A-4,
   small_bert/bert_en_uncased_L-2_H-512_A-8,
   small_bert/bert_en_uncased_L-2_H-768_A-12,
   small_bert/bert_en_uncased_L-4_H-128_A-2,
   small_bert/bert_en_uncased_L-4_H-256_A-4,
   small_bert/bert_en_uncased_L-4_H-512_A-8,
   small_bert/bert_en_uncased_L-4_H-768_A-12,
   small_bert/bert_en_uncased_L-6_H-128_A-2,
   small_bert/bert_en_uncased_L-6_H-256_A-4,
   small_bert/bert_en_uncased_L-6_H-512_A-8,
   small_bert/bert_en_uncased_L-6_H-768_A-12,
   small_bert/bert_en_uncased_L-8_H-128_A-2,
   small_bert/bert_en_uncased_L-8_H-256_A-4,
   small_bert/bert_en_uncased_L-8_H-512_A-8,
   small_bert/bert_en_uncased_L-8_H-768_A-12,

.. toctree::
   :maxdepth: 1
   :caption: Contents

   quickstart
   cli
   api
   notebooks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

