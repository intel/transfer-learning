# Transfer Learning for Video Classification using PyTorch

This notebook uses video classification models from Torchvision that were originally trained
using Kinetic400 and does transfer learning with the HMDB51 dataset.

The notebook performs the following steps:

1. Import dependencies and setup parameters
2. Prepare the dataset
3. Predict using the original model
4. Transfer learning
5. Predict
6. Export the saved model

## Running the notebook

To run the notebook, follow the instructions to setup the [PyTorch notebook environment](/notebooks/setup.md).

## References

Dataset Citations:
```
@inproceedings{
  title = {HMDB: A Large Video Database for Human Motion Recognition},
  author = {H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre},
  year = {2011}
}
@ONLINE {HMDB,
author = {H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre},
title = "HMDB51",
year = "2011",
url = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar" }
```
