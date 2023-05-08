# Transfer Learning for PyTorch Image Classification using the Intel® Transfer Learning Tool API

This notebook demonstrates how to use the Intel Transfer Learning Tool API to do transfer learning for
image classification using PyTorch.

The notebook performs the following steps:
1. Import dependencies and setup parameters
1. Get the model
1. Get the dataset
1. Prepare the dataset
1. Predict using the original model
1. Transfer learning
1. Predict
1. Export

## Running the notebook

To run the notebook, follow the instructions to setup the [PyTorch notebook environment](/notebooks/setup.md).

## References

Dataset citations
```
@ONLINE {tfflowers,
author = "The TensorFlow Team",
title = "Flowers",
month = "jan",
year = "2019",
url = "http://download.tensorflow.org/example_images/flower_photos.tgz" }

@ONLINE {CIFAR10,
author = "Alex Krizhevsky",
title = "CIFAR-10",
year = "2009",
url = "http://www.cs.toronto.edu/~kriz/cifar.html" }

@article{openimages,
  title={OpenImages: A public dataset for large-scale multi-label and multi-class image classification.},
  author={Krasin, Ivan and Duerig, Tom and Alldrin, Neil and Veit, Andreas and Abu-El-Haija, Sami
    and Belongie, Serge and Cai, David and Feng, Zheyun and Ferrari, Vittorio and Gomes, Victor
    and Gupta, Abhinav and Narayanan, Dhyanesh and Sun, Chen and Chechik, Gal and Murphy, Kevin},
  journal={Dataset available from https://github.com/openimages},
  year={2016}
}
```
Model citations
```
@misc{yalniz2019billionscale,
    title={Billion-scale semi-supervised learning for image classification},
    author={I. Zeki Yalniz and Hervé Jégou and Kan Chen and Manohar Paluri and Dhruv Mahajan},
    year={2019},
    eprint={1905.00546},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

