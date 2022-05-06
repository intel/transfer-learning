Command Line Interface (CLI)
============================

Use `tlk --help` to see the list of CLI commands. More detailed information on each
command can be found using `tlk <command> --help` (like `tlk train --help`).

Examples
--------

List the available models::

    > tlk list models
    ------------------------------
    IMAGE CLASSIFICATION
    ------------------------------
    efficientnet_b0 (tensorflow)
    efficientnet_b1 (tensorflow)
    efficientnet_b2 (tensorflow)
    efficientnet_b3 (tensorflow)
    efficientnet_b4 (tensorflow)
    efficientnet_b5 (tensorflow)
    efficientnet_b6 (tensorflow)
    efficientnet_b7 (tensorflow)
    efficientnetv2-b0 (tensorflow)
    efficientnetv2-b1 (tensorflow)
    efficientnetv2-b2 (tensorflow)
    efficientnetv2-b3 (tensorflow)
    efficientnetv2-s (tensorflow)
    inception_v3 (tensorflow)
    mobilenet_v2_100_224 (tensorflow)
    nasnet_large (tensorflow)
    resnet_v1_50 (tensorflow)
    resnet_v2_101 (tensorflow)
    resnet_v2_50 (tensorflow)

    ------------------------------
    OBJECT DETECTION
    ------------------------------
    No object detection models are supported at this time

    ------------------------------
    TEXT CLASSIFICATION
    ------------------------------
    No text classification models are supported at this time

    ------------------------------
    QUESTION ANSWERING
    ------------------------------
    No question answering models are supported at this time

Train a model::

    > tlk train -f tensorflow --model-name efficientnet_b0 --dataset-dir /tmp/data --output-dir /tmp/output --dataset-name tf_flowers --dataset-catalog tf_datasets
    Model name: efficientnet_b0
    Framework: tensorflow
    Dataset name: tf_flowers
    Dataset catalog: tf_datasets
    Training epochs: 1
    Dataset dir: /tmp/data
    Output directory: /tmp/output
    2022-05-05 14:28:30.222567: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-05-05 14:28:30.229992: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting:
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
    keras_layer (KerasLayer)    (None, 1280)              4049564

    dense (Dense)               (None, 5)                 6405

    =================================================================
    Total params: 4,055,969
    Trainable params: 6,405
    Non-trainable params: 4,049,564
    _________________________________________________________________
    Checkpoint directory: /tmp/output/efficientnet_b0_checkpoints
    86/86 [==============================] - 23s 232ms/step - loss: 0.5875 - acc: 0.8125
    2022-05-05 14:28:57.169689: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    Saved model directory: /tmp/output/efficientnet_b0/1

Evaluate a trained model::

    > tlk eval --model-dir /tmp/output/efficientnet_b0/1 --dataset-dir /tmp/data --dataset-name tf_flowers --dataset-catalog tf_datasets
    Model directory: /tmp/output/efficientnet_b0/1
    Dataset directory: /tmp/data
    Dataset name: tf_flowers
    Dataset catalog: tf_datasets
    Model name: efficientnet_b0
    Loading model object for efficientnet_b0 using tensorflow
    Loading saved model from: /tmp/output/efficientnet_b0/1/saved_model.pb
    2022-05-05 14:33:06.090646: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-05-05 14:33:06.098042: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting:
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
    keras_layer (KerasLayer)    (None, 1280)              4049564

    dense (Dense)               (None, 5)                 6405

    =================================================================
    Total params: 4,055,969
    Trainable params: 6,405
    Non-trainable params: 4,049,564
    _________________________________________________________________
    29/29 [==============================] - 7s 222ms/step - loss: 0.4217 - acc: 0.8715

.. click:: tlk.tools.cli.main:cli_group
   :prog: tlk
   :nested: full