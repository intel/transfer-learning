Intel® Transfer Learning Tool API Notebook Examples
===================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   setup

API examples are demonstrated using Jupyter notebooks.

Prerequisites
*************

Before running these Jupyter notebook examples, use these :doc:`notebook setup
instructions<setup>` to install required dependencies.


Intel Transfer Learning Tool API Tutorial Notebooks
***************************************************

.. |imageClassPyTorch| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _imageClassPyTorch: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/image_classification/tlt_api_pyt_image_classification/TLT_PyTorch_Image_Classification_Transfer_Learning.ipynb

.. |imageClassTensorFlow| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _imageClassTensorflow: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/image_classification/tlt_api_tf_image_classification/TLT_TF_Image_Classification_Transfer_Learning.ipynb

.. |textClassPyTorch| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _textClassPyTorch: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/text_classification/tlt_api_pyt_text_classification/TLT_PYT_Text_Classification.ipynb

.. |textClassTensorFlow| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _textClassTensorflow: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/text_classification/tlt_api_tf_text_classification/TLT_TF_Text_Classification.ipynb

.. |imageAnomalyPyTorch| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _imageAnomalyPyTorch: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/image_anomaly_detection/tlt_api_pyt_anomaly_detection/Anomaly_Detection.ipynb

.. csv-table::
   :header: "Notebook Title", ".ipynb Link", "Use Case", "Framework"
   :widths: 30, 10, 20, 20

   :doc:`Image Classification with PyTorch <TLT_PyTorch_Image_Classification_Transfer_Learning>`, |imageClassPyTorch|_ , Image Classification, PyTorch & Intel Transfer Learning Tool
   :doc:`Image Classification with TensorFlow <TLT_TF_Image_Classification_Transfer_Learning>`, |imageClassTensorFlow|_ , Image Classification, TensorFlow & Intel Transfer Learning Tool
   :doc:`Text Classification with PyTorch <TLT_PyTorch_Text_Classification_Transfer_Learning>`, |textClassPyTorch|_ , Text Classification, PyTorch & Intel Transfer Learning Tool
   :doc:`Text Classification with TensorFlow <TLT_TF_Text_Classification_Transfer_Learning>`, |textClassTensorflow|_ , Text Classification, TensorFlow & Intel Transfer Learning Tool
   :doc:`Anomaly Detection using PyTorch <TLT_PyTorch_Anomly_Detection>`, |imageAnomalyPyTorch|_, Image Anomaly Detection, PyTorch & Intel Transfer Learning Tool

Intel Transfer Learning Tool API End-to-End Pipelines
*****************************************************

.. |imageClassMedical| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _imageClassMedical: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/e2e_workflows/Medical_Imaging_Classification.ipynb

.. |imageClassRemote| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _imageClassRemote: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/e2e_workflows/Remote_Sensing_Image_Scene_Classification.ipynb


.. csv-table::
   :header: "Notebook Title", ".ipynb Link", "Use Case", "Framework"
   :widths: 30, 10, 20, 20

   :doc:`Medical Imaging Classification (Colorectal histology) using TensorFlow <Medical_Imaging_Classification>`, |imageClassMedical|_ , Image Classification, TensorFlow & Intel Transfer Learning Tool
   :doc:`Remote Sensing Image Scene Classification (Resisc) using TensorFlow <Remote_Sensing_Image_Scene_Classification>`, |imageClassRemote|_ , Image Classification, TensorFlow & Intel Transfer Learning Tool

Intel Transfer Learning Tool Performance Comparison
*****************************************************

.. |imageClassTFPerf| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _imageClassTFPerf: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/performance/tf_image_classification_performance.ipynb

.. |textClassHFPerf| image:: /images/Jupyter_logo.svg
             :alt: Jupyter notebook .ipynb file
             :height: 35
.. _textClassHFPerf: https://github.com/IntelAI/transfer-learning/blob/main/notebooks/performance/hf_text_classification_performance.ipynb

.. csv-table::
   :header: "Notebook Title", ".ipynb Link", "Use Case", "Framework"
   :widths: 30, 10, 20, 20

   :doc:`Performance Comparison: Image Classification with TensorFlow <TLT_TF_Image_Classification_Performance>`, |imageClassTFPerf|_ , Image Classification, TensorFlow & Intel Transfer Learning Tool
   :doc:`Performance Comparison: Text Classification with Hugging Face <TLT_HF_Text_Classification_Performance>`, |textClassHFPerf|_ , Text Classification, "Hugging Face, PyTorch & Intel Transfer Learning Tool"
