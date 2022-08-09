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
   * PyTorch 1.11
      * Image Classification with 56 models from torchvision

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
   Text Classifier fine tuning with PyTorch & Hugging Face, Text Classification, PyTorch

Models
------

.. csv-table::
   :header: TensorFlow,PyTorch
   :widths: 50, 50

   efficientnet_b0,alexnet
   efficientnet_b1,convnext_tiny
   efficientnet_b2,convnext_small
   efficientnet_b3,convnext_base
   efficientnet_b4,convnext_large
   efficientnet_b5,densenet121
   efficientnet_b6,densenet161
   efficientnet_b7,densenet169
   efficientnetv2-b0,efficientnet_b0
   efficientnetv2-b1,efficientnet_b1
   efficientnetv2-b2,efficientnet_b2
   efficientnetv2-b3,efficientnet_b3
   efficientnetv2-s,efficientnet_b4
   inception_v3,efficientnet_b5
   mobilenet_v2_100_224,efficientnet_b6
   nasnet_large,efficientnet_b7
   resnet_v1_50,googlenet
   resnet_v2_101,mnasnet0_5
   resnet_v2_50,mnasnet1_0
   ,mobilenet_v2
   ,mobilenet_v3_small
   ,mobilenet_v3_large
   ,resnet18
   ,resnet34
   ,resnet50
   ,resnet101
   ,resnet152
   ,resnext50_32x4d
   ,resnext101_32x8d
   ,regnet_x_400mf
   ,regnet_x_800mf
   ,regnet_x_1_6gf
   ,regnet_x_3_2gf
   ,regnet_x_8gf
   ,regnet_x_16gf
   ,regnet_x_32gf
   ,regnet_y_400mf
   ,regnet_y_800mf
   ,regnet_y_1_6gf
   ,regnet_y_3_2gf
   ,regnet_y_8gf
   ,regnet_y_16gf
   ,regnet_y_32gf
   ,shufflenet_v2_x0_5
   ,shufflenet_v2_x1_0
   ,vgg11
   ,vgg11_bn
   ,vgg13
   ,vgg13_bn
   ,vgg16
   ,vgg16_bn
   ,vgg19
   ,vgg19_bn
   ,wide_resnet50_2
   ,wide_resnet101_2

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

