Image Classification Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: tlt.models.image_classification

.. autosummary::
  :toctree: _autosummary
  :nosignatures:
   
   tfhub_image_classification_model.TFHubImageClassificationModel.train
   tfhub_image_classification_model.TFHubImageClassificationModel.quantize
   tfhub_image_classification_model.TFHubImageClassificationModel.get_inc_config

   tf_image_classification_model.TFImageClassificationModel.train
   tf_image_classification_model.TFImageClassificationModel.quantize
   tf_image_classification_model.TFImageClassificationModel.get_inc_config

   keras_image_classification_model.KerasImageClassificationModel.train
   keras_image_classification_model.KerasImageClassificationModel.quantize
   keras_image_classification_model.KerasImageClassificationModel.get_inc_config

   torchvision_image_classification_model.TorchvisionImageClassificationModel.train
   torchvision_image_classification_model.TorchvisionImageClassificationModel.quantize
   torchvision_image_classification_model.TorchvisionImageClassificationModel.get_inc_config

   pytorch_image_classification_model.PyTorchImageClassificationModel.train
   pytorch_image_classification_model.PyTorchImageClassificationModel.quantize
   pytorch_image_classification_model.PyTorchImageClassificationModel.get_inc_config

   pytorch_hub_image_classification_model.PyTorchHubImageClassificationModel.train
   pytorch_hub_image_classification_model.PyTorchHubImageClassificationModel.quantize
   pytorch_hub_image_classification_model.PyTorchHubImageClassificationModel.get_inc_config

   image_classification_model.ImageClassificationModel.train
   image_classification_model.ImageClassificationModel.quantize
   image_classification_model.ImageClassificationModel.get_inc_config

Text Classification Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: tlt.models.text_classification

.. autosummary::
  :toctree: _autosummary
  :nosignatures:
  :recursive:

   tf_text_classification_model.TFTextClassificationModel.train
   tf_text_classification_model.TFTextClassificationModel.quantize
   tf_text_classification_model.TFTextClassificationModel.write_inc_config_file

   pytorch_hf_text_classification_model.PyTorchHFTextClassificationModel.train
   pytorch_hf_text_classification_model.PyTorchHFTextClassificationModel.quantize
   pytorch_hf_text_classification_model.PyTorchHFTextClassificationModel.write_inc_config_file

   tf_hf_text_classification_model.TFHFTextClassificationModel.train
   tf_hf_text_classification_model.TFHFTextClassificationModel.quantize
   tf_hf_text_classification_model.TFHFTextClassificationModel.write_inc_config_file
   
   text_classification_model.TextClassificationModel.train
   text_classification_model.TextClassificationModel.quantize
   text_classification_model.TextClassificationModel.write_inc_config_file
