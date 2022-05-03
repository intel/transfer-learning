# Text Classifier fine tuning with TensorFlow

This notebook demonstrates fine tuning using various [BERT](https://arxiv.org/abs/1810.04805) models
from [TF Hub](https://tfhub.dev) using Intel® Optimization for TensorFlow for binary text classification.

The notebook performs the following steps:
1. Install dependencies and setup parameters
1. Prepare the dataset using either a TF dataset or your own images
1. Predict using the original model
1. Transfer learning
1. Evaluate the model
1. Export the saved model

## Running the notebook

To run the notebook, follow the instructions to setup the [TensorFlow notebook environment](/notebooks#tensorflow-environment).

## References

Dataset citations:
```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}

@misc{misc_sms_spam_collection_228,
  author       = {Almeida, Tiago},
  title        = {{SMS Spam Collection}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository}
}
```
