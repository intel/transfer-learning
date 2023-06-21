# Transfer Learning for TensorFlow Image Classification using the Intel® Transfer Learning Tool API

These notebooks demonstrate how to use the Intel Transfer Learning Tool API to do transfer learning for
image classification using TensorFlow and then quantize or optimize the graph for inference.

`TLT_TF_Image_Classification_Transfer_Learning.ipynb`  performs the following steps:
1. Import dependencies and setup parameters
1. Get the model
1. Get the dataset
1. Prepare the dataset
1. Predict using the original model
1. Transfer learning
1. Predict
1. Export
1. Post-training quantization

`TLT_TF_Transfer_Learning_and_Graph_Optimization.ipynb`  performs the following steps:
1. Import dependencies and setup parameters
1. Get the model
1. Get the dataset
1. Prepare the dataset
1. Evaluate using the original model
1. Transfer learning
1. Export
1. Graph Optimization

## Running the notebooks

To run the notebooks, follow the instructions to setup the [TensorFlow notebook environment](/notebooks/setup.md).

## References

Dataset citations
```
@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }

@article{openimages,
  title={OpenImages: A public dataset for large-scale multi-label and multi-class image classification.},
  author={Krasin, Ivan and Duerig, Tom and Alldrin, Neil and Veit, Andreas and Abu-El-Haija, Sami
    and Belongie, Serge and Cai, David and Feng, Zheyun and Ferrari, Vittorio and Gomes, Victor
    and Gupta, Abhinav and Narayanan, Dhyanesh and Sun, Chen and Chechik, Gal and Murphy, Kevin},
  journal={Dataset available from https://github.com/openimages},
  year={2016}
}
```

