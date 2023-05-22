Image Classification Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: tlt.models.image_classification

.. autosummary::
  :toctree: _autosummary
  :nosignatures:
   
   tfhub_image_classification_model.TFHubImageClassificationModel.train
   tfhub_image_classification_model.TFHubImageClassificationModel.quantize
   tfhub_image_classification_model.TFHubImageClassificationModel.write_inc_config_file

   tf_image_classification_model.TFImageClassificationModel.train
   tf_image_classification_model.TFImageClassificationModel.quantize
   tf_image_classification_model.TFImageClassificationModel.write_inc_config_file

   keras_image_classification_model.KerasImageClassificationModel.train
   keras_image_classification_model.KerasImageClassificationModel.quantize
   keras_image_classification_model.KerasImageClassificationModel.write_inc_config_file

   torchvision_image_classification_model.TorchvisionImageClassificationModel.train
   torchvision_image_classification_model.TorchvisionImageClassificationModel.quantize
   torchvision_image_classification_model.TorchvisionImageClassificationModel.write_inc_config_file

   pytorch_image_classification_model.PyTorchImageClassificationModel.train
   pytorch_image_classification_model.PyTorchImageClassificationModel.quantize
   pytorch_image_classification_model.PyTorchImageClassificationModel.write_inc_config_file

   pytorch_hub_image_classification_model.PyTorchHubImageClassificationModel.train
   pytorch_hub_image_classification_model.PyTorchHubImageClassificationModel.quantize
   pytorch_hub_image_classification_model.PyTorchHubImageClassificationModel.write_inc_config_file

   image_classification_model.ImageClassificationModel.train
   image_classification_model.ImageClassificationModel.quantize
   image_classification_model.ImageClassificationModel.write_inc_config_file

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
