API Reference
=============

Datasets
--------

.. currentmodule:: tlt.datasets

The simplest way to create datasets is with the dataset factory methods :meth:`load_dataset`, for using a
custom dataset, and :meth:`get_dataset`, for downloading and using a third-party dataset from a catalog such as TensorFlow
Datasets or Torchvision.

Factory Methods
***************

.. automodule:: tlt.datasets.dataset_factory
   :members: load_dataset, get_dataset

Class Reference
***************

Image Classification
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.datasets.image_classification

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    tfds_image_classification_dataset.TFDSImageClassificationDataset
    torchvision_image_classification_dataset.TorchvisionImageClassificationDataset
    tf_custom_image_classification_dataset.TFCustomImageClassificationDataset
    pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset
    image_classification_dataset.ImageClassificationDataset

Text Classification
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.datasets.text_classification

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    tfds_text_classification_dataset.TFDSTextClassificationDataset
    hf_text_classification_dataset.HFTextClassificationDataset
    tf_custom_text_classification_dataset.TFCustomTextClassificationDataset
    hf_custom_text_classification_dataset.HFCustomTextClassificationDataset
    text_classification_dataset.TextClassificationDataset

Base Classes
^^^^^^^^^^^^

.. note:: Users should rarely need to interact directly with these.

.. currentmodule:: tlt.datasets

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    pytorch_dataset.PyTorchDataset
    tf_dataset.TFDataset
    hf_dataset.HFDataset
    dataset.BaseDataset

Models
------

.. currentmodule:: tlt.models

Discover and work with available models by using model factory methods. The :meth:`get_model` function will download
third-party models, while the :meth:`load_model` function will load a custom model, from either a path location or a
model object in memory. The model discovery and inspection methods are :meth:`get_supported_models` and
:meth:`print_supported_models`.

Factory Methods
***************

.. automodule:: tlt.models.model_factory
   :members: get_model, load_model, get_supported_models, print_supported_models

Class Reference
***************

Image Classification
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.models.image_classification

.. autosummary::
  :toctree: _autosummary
  :nosignatures:

   tfhub_image_classification_model.TFHubImageClassificationModel
   tf_image_classification_model.TFImageClassificationModel
   keras_image_classification_model.KerasImageClassificationModel
   torchvision_image_classification_model.TorchvisionImageClassificationModel
   pytorch_image_classification_model.PyTorchImageClassificationModel
   pytorch_hub_image_classification_model.PyTorchHubImageClassificationModel
   image_classification_model.ImageClassificationModel

Text Classification
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.models.text_classification

.. autosummary::
  :toctree: _autosummary
  :nosignatures:

   tf_text_classification_model.TFTextClassificationModel
   pytorch_hf_text_classification_model.PyTorchHFTextClassificationModel
   tf_hf_text_classification_model.TFHFTextClassificationModel
   text_classification_model.TextClassificationModel

Base Classes
^^^^^^^^^^^^

.. note:: Users should rarely need to interact directly with these.

.. currentmodule:: tlt.models

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    pytorch_model.PyTorchModel
    tf_model.TFModel
    hf_model.HFModel
    model.BaseModel
