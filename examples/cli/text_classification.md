# Text Classification Intel® Transfer Learning Tool CLI Example

## Fine Tuning Using Your Own Dataset

The example below shows how to fine tune a TensorFlow text classification model using your own
dataset in the .csv format. The .csv file is expected to have 2 columns: a numerical class label
and the text/sentence to classify. Note that although the TLT API is more flexible and allows for
providing map functions to translate string class names to numerical values and filtering which
columns are being used, the CLI only allows using .csv files in the expected format.

The `--dataset-dir` argument is the path to the directory where your dataset is located, and the
`--dataset-file` is the name of the .csv file to load from that directory. Use the `--class-names`
argument to specify a list of the classes and the `--delimiter` to specify the character that
separates the two columns. If no `--delimiter` is specified, the CLI will default to use a comma (`,`).

This example is downloading the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
dataset, which has a tab separated value file in the .zip file. This dataset has labeled SMS text
messages that are either being classified as `ham` or `spam`. The first column in the data file has
the label (`ham` or `spam`) and the second column is the text of the SMS message. The string class
labels are replaced with numerical values before training.
```bash
# Create dataset and output directories
DATASET_DIR=/tmp/data
OUTPUT_DIR=/tmp/output
mkdir -p ${DATASET_DIR}
mkdir -p ${OUTPUT_DIR}

# Download and extract the dataset
wget -P ${DATASET_DIR} https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
unzip ${DATASET_DIR}/sms+spam+collection.zip

# Make a copy of the .csv file with 'numerical' in the file name
DATASET_FILE=SMSSpamCollection_numerical.csv
cp ${DATASET_DIR}/SMSSpamCollection ${DATASET_DIR}/${DATASET_FILE}

# Replace string class labels with numerical values in the .csv file\
# The list numerical class labels passed as the --class-names during training and evaluation
sed -i 's/ham/0/g' ${DATASET_DIR}/${DATASET_FILE}
sed -i 's/spam/1/g' ${DATASET_DIR}/${DATASET_FILE}

# Train google/bert_uncased_L-10_H-256_A-4 using our dataset file, which has tab delimiters
tlt train \
    -f tensorflow \
    --model-name google/bert_uncased_L-10_H-256_A-4 \
    --output-dir ${OUTPUT_DIR} \
    --dataset-dir ${DATASET_DIR} \
    --dataset-file ${DATASET_FILE} \
    --epochs 2 \
    --class-names 0,1 \
    --delimiter $'\t'

# Evaluate the model exported after training
# Note that your --model-dir path may vary, since each training run creates a new directory
tlt eval \
    --model-dir ${OUTPUT_DIR}/google_bert_uncased_L-10_H-256_A-4/1 \
    --model-name google/bert_uncased_L-10_H-256_A-4 \
    --dataset-dir ${DATASET_DIR} \
    --dataset-file ${DATASET_FILE} \
    --class-names 0,1 \
    --delimiter $'\t'
```

## Fine Tuning Using a Dataset from the TFDS Catalog

This example demonstrates using the Intel Transfer Learning Tool CLI to fine tune a text classification model using a
dataset from the [TensorFlow Datasets (TFDS) catalog](https://www.tensorflow.org/datasets/catalog/overview).
Intel Transfer Learning Tool supports the following text classification datasets from TFDS:
[imdb_reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews),
[glue/sst2](https://www.tensorflow.org/datasets/catalog/imdb_reviews),
and [glue/cola](https://www.tensorflow.org/datasets/catalog/glue#gluecola_default_config).

```bash
# Create dataset and output directories
DATASET_DIR=/tmp/data
OUTPUT_DIR=/tmp/output
mkdir -p ${DATASET_DIR}
mkdir -p ${OUTPUT_DIR}

# Name of the dataset to use
DATASET_NAME=imdb_reviews

# Train google/bert_uncased_L-10_H-256_A-4 using the TFDS dataset
tlt train \
    -f tensorflow \
    --model-name google/bert_uncased_L-10_H-256_A-4 \
    --output-dir ${OUTPUT_DIR} \
    --dataset-dir ${DATASET_DIR} \
    --dataset-name ${DATASET_NAME} \
    --epochs 2

# Evaluate the model exported after training
# Note that your --model-dir path may vary, since each training run creates a new directory
tlt eval \
    --model-dir ${OUTPUT_DIR}/google_bert_uncased_L-10_H-256_A-4/2 \
    --model-name google/bert_uncased_L-10_H-256_A-4 \
    --dataset-dir ${DATASET_DIR} \
    --dataset-name ${DATASET_NAME}
```

## Distributed Transfer Learning Using a Dataset from Hugging Face
This example runs a distributed PyTorch training job using the TLT CLI. It fine tunes a text classification model
for document-level sentiment analysis using a dataset from the [Hugging Face catalog](https://huggingface.co/datasets).
Intel Transfer Learning Tool supports the following text classification datasets from Hugging Face:
* [imdb](https://huggingface.co/datasets/imdb)
* [tweet_eval](https://huggingface.co/datasets/tweet_eval)
* [rotten_tomatoes](https://huggingface.co/datasets/rotten_tomatoes)
* [ag_news](https://huggingface.co/datasets/ag_news)
* [sst2](https://huggingface.co/datasets/sst2)

Follow [these instructions](/tlt/distributed/README.md) to set up your machines for distributed training with PyTorch. This will
ensure your environment has the right prerequisites, package dependencies, and hostfile configuration. When
you have successfully run the sanity check, the following commands will fine-tune `bert-large-uncased` with sst2 for
one epoch using 2 nodes and 2 processes per node.

```bash
# Create dataset and output directories
DATASET_DIR=/tmp/data
OUTPUT_DIR=/tmp/output
mkdir -p ${DATASET_DIR}
mkdir -p ${OUTPUT_DIR}

# Name of the dataset to use
DATASET_NAME=sst2

# Train bert-large-uncased using the Hugging Face dataset sst2
tlt train \
    -f pytorch \
    --model_name bert-large-uncased \
    --dataset_name sst2 \
    --output_dir $OUTPUT_DIR \
    --dataset_dir $DATASET_DIR \
    --distributed \
    --hostfile hostfile \
    --nnodes 2 \
    --nproc_per_node 2
```

## Citations
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

@inproceedings{wang2019glue,
  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}

@misc{misc_sms_spam_collection_228,
  author       = {Almeida, Tiago},
  title        = {{SMS Spam Collection}},
  year         = {2012},
  howpublished = {UCI Machine Learning Repository}
}

@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard  and
      Perelygin, Alex  and
      Wu, Jean  and
      Chuang, Jason  and
      Manning, Christopher D.  and
      Ng, Andrew  and
      Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1170",
    pages = "1631--1642",
}
```
