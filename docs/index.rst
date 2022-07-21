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
   * TensorFlow 2.8
      * Image Classification with 19 models from TFHub
   * PyTorch 1.11
      * Image Classification with 10 models from torchvision

.. figure:: images/features.png
   :scale: 50 %
   :alt: TLK Features

* Jupyter notebooks demonstrating:
   * 5 Computer Vision workflows
   * 3 Natural Language Processing workflows

.. csv-table::
   :header: "Notebook", "Use Case", "Framework"
   :widths: 60, 20, 20

   Image Classification with TF Hub, Image Classification, TensorFlow
   :doc:`Image Classification with TF using the TLK API <notebooks/TLK_TF_Image_Classification_Transfer_Learning>`, Image Classification, TensorFlow & TLK
   Image Classification with PyTorch & torchvision, Image Classification, PyTorch
   :doc:`Image Classification with PyTorch using the TLK API <notebooks/TLK_PyTorch_Image_Classification_Transfer_Learning>`, Image Classification, PyTorch & TLK
   Object Detection with PyTorch & torchvision, Object Detection, PyTorch
   BERT SQuAD fine tuning with TF Hub, Question Answering, TensorFlow
   BERT Binary Text Classification with TF Hub, Text Classification, TensorFlow
   Text Classifier fine tuning with PyTorch & Hugging Face, Text Classification, PyTorch

Models
------

.. csv-table::
   :header: TensorFlow,PyTorch
   :widths: 50, 50

   efficientnet_b0,densenet121
   efficientnet_b1,densenet161
   efficientnet_b2,efficientnet_b0
   efficientnet_b3,efficientnet_b1
   efficientnet_b4,efficientnet_b2
   efficientnet_b5,googlenet
   efficientnet_b6,mobilenet_v2
   efficientnet_b7,resnet18
   efficientnetv2-b0,resnet50
   efficientnetv2-b1,shufflenet_v2_x1_0
   efficientnetv2-b2,
   efficientnetv2-b3,
   efficientnetv2-s,
   inception_v3,
   mobilenet_v2_100_224,
   nasnet_large,
   resnet_v1_50,
   resnet_v2_101,
   resnet_v2_50,

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

