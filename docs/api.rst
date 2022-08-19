
API Reference
=============

Datasets
--------

.. currentmodule:: tlt.datasets

The simplest way to create datasets is with the dataset factory methods :meth:`load_dataset`, for using a
custom dataset, and :meth:`get_dataset`, for downloading and using a third-party dataset from a catalog such as TensorFlow
Datasets or torchvision.

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

    pytorch_custom_image_classification_dataset.PyTorchCustomImageClassificationDataset
    tf_custom_image_classification_dataset.TFCustomImageClassificationDataset
    tf_image_classification_dataset.TFImageClassificationDataset
    torchvision_image_classification_dataset.TorchvisionImageClassificationDataset
    image_classification_dataset.ImageClassificationDataset

Text Classification
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.datasets.text_classification

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    tfds_text_classification_dataset.TFDSTextClassificationDataset
    tf_custom_text_classification_dataset.TFCustomTextClassificationDataset
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
    dataset.BaseDataset

Models
------

.. currentmodule:: tlt.models

Discover and work with available models by using model factory methods. The :meth:`get_model`
function will download third-party models and provide a convenient interface for modifying, training, evaluating, and
so on. The model discovery and inspection methods are :meth:`get_supported_models` and :meth:`print_supported_models`.

Factory Methods
***************

.. automodule:: tlt.models.model_factory
   :members: get_model, get_supported_models, print_supported_models

Class Reference
***************

Image Classification
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.models.image_classification

.. autosummary::
  :toctree: _autosummary
  :nosignatures:

   tfhub_image_classification_model.TFHubImageClassificationModel
   torchvision_image_classification_model.TorchvisionImageClassificationModel
   image_classification_model.ImageClassificationModel

Text Classification
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.models.text_classification

.. autosummary::
  :toctree: _autosummary
  :nosignatures:

   tfhub_text_classification_model.TFHubTextClassificationModel
   text_classification_model.TextClassificationModel

Base Classes
^^^^^^^^^^^^

.. note:: Users should rarely need to interact directly with these.

.. currentmodule:: tlt.models

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    torchvision_model.TorchvisionModel
    tfhub_model.TFHubModel
    model.BaseModel
