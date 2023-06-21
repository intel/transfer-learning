# Text Classifier fine tuning with PyTorch and the Intel® Transfer Learning Tool API

This notebook demonstrates how to use the Intel Transfer Learning Tool API to do fine tuning for
text classification using various [BERT](https://arxiv.org/abs/1810.04805) models
from [Hugging Face](https://huggingface.co/datasets).

The notebook performs the following steps:
1. Install dependencies and setup parameters
1. Get the model
1. Get the dataset
1. Prepare the dataset
1. Fine tuning
1. Predict
1. Export the saved model
1. Quantization

## Running the notebook

To run the notebook, follow the instructions to setup the [PyTorch notebook environment](/notebooks/setup.md).

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
Please see this dataset's applicable license for terms and conditions. Intel Corporation does not own the rights to this data set and does not confer any rights to it.
