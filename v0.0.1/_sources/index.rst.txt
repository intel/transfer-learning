.. tlk documentation master file, created by
   sphinx-quickstart on Wed Apr 27 16:19:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Transfer Learning Kit (TLK)
===========================

Introduction
------------

Transfer learning uses pretrained model weights as a starting point to train the model on a different dataset
for a new task. This allows for a smaller amount of training data and reduced training time, compared to training
from scratch. Transfer learning has been increasing in popularity (particularly for computer vision and natural
language processing) and there are many pretrained models available in public model hubs like TensorFlow Hub,
torchvision, and HuggingFace. As Intel, we want to utilize transfer learning on XPU to demonstrate that the training
times can be reasonable with transfer learning and fine tuning. To help do this, we provide a CLI and API to
reduce the complexity that can be involved with transfer learning.

Transfer learning involves several steps including preprocessing the dataset, finding and downloading an appropriate
pretrained model, manipulating the model’s layers, retraining, evaluation, and exporting the final model. There are
many transfer learning tutorials online, but those tutorials often cover a single use case, may gloss over certain
details like how the dataset needs to be formatted, and do not always explain how to apply the same method to other
tasks. These tutorials also do not utilize technologies that give the best performance on Intel hardware (like
Intel-optimized frameworks/extensions and pruning/quantization using the Intel Neural Compressor).

This transfer learning CLI and API provide a consistent interface for users to apply transfer learning to their own
tasks across various CV and NLP use cases. The tools also abstract out differences between frameworks and different
model hubs. The tools can be run on the command line, interactively in a low code environment with sample Jupyter
notebooks, and deployed as part of MLOps pipelines in containers.

Sequence Diagram
----------------

.. figure:: images/sequence_diagram.png
   :scale: 50 %
   :alt: TLK Sequence Diagram

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   cli
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`